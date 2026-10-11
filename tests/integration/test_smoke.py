"""M1-D0 — smoke JAI-TRAIN : câblage complet, compteurs, reprise.

Une fixture module exécute `run_smoke` une fois ; chaque test reste
exécutable isolément (`pytest test_smoke.py::test_x` rejoue la fixture).
Config smoke §11.3 : 16 prompts, 2 rollouts, L_max=128, 5 pas.
LIMITATION DOCUMENTÉE : `response_cap=8` — toutes les réponses font
8 tokens ; l'effet du curriculum sur les longueurs n'est PAS mesuré ici
(seules les valeurs L_s calculées sont vérifiées). Aucun benchmark
qualité, aucune distillation (jeu synthétique, câblage uniquement).
Le diagnostic de compatibilité bloquée vit séparément :
`tests/numerical/test_teacher_extended_blocked.py` (xfail strict).
"""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.distill.curriculum import rollout_length
from jainius.distill.teacher import Teacher
from jainius.model.arch import DiffusionSpec, JainiusModel
from jainius.train.checkpoint import load_ckpt
from jainius.train.loop import SmokeConfig, param_fingerprint, run_smoke, train_step
from jainius.train.master import MasterOptimizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
CFG = SmokeConfig()


def _load_pair():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    student_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    )
    teacher_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    )
    return tok, JainiusModel(student_hf, DiffusionSpec()), Teacher(teacher_hf)


@pytest.fixture(scope="module")
def smoke_rep(tmp_path_factory):
    tok, student, teacher = _load_pair()
    rep = run_smoke(student, teacher, tok, CFG,
                    tmp_path_factory.mktemp("ckpt"))
    del tok, student, teacher
    gc.collect()
    torch.cuda.empty_cache()
    return rep


def test_smoke_wiring(smoke_rep):
    rep = smoke_rep
    assert rep["prompts_per_step"] == [16] * 5
    assert sorted(rep["trajectories"]) == [0, 1], "2 rollouts exigés"
    assert rep["updates"] == 5
    assert rep["lengths"] == [
        rollout_length(s, l_min=8, l_max=128, warmup_rollouts=10)
        for s in range(5)
    ], "progression curriculum inexacte"
    assert rep["student_train_fw"] == 16 * 5
    # Enseignant : exactement 16 forward `hidden` par pas (diagnostic
    # extended séparé, non compté ici).
    assert rep["teacher_fw"] == 16 * 5
    assert rep["rollout_fw"] == 16 * 7 * 2 + 7 * 2  # +1 rejouement/round
    assert len(rep["losses"]) == 5 and all(
        v == v and abs(v) != float("inf") for v in rep["losses"]
    )
    assert all(rep["params_changed"]) and all(rep["grads_finite"])
    assert len(rep["grad_norms"]) == 5 and all(
        g == g and abs(g) != float("inf") for g in rep["grad_norms"])
    # Isolation inter-rollouts vérifiée DANS run_smoke (poids identiques).


def test_smoke_resume(smoke_rep):
    rep = smoke_rep
    tok, student, teacher = _load_pair()
    optim = MasterOptimizer(student.model, lr=CFG.lr)
    student.model.train()
    manifest = load_ckpt(rep["saved"]["dir"], student.model, optim.optim,
                         {"smoke": True, "seed": CFG.seed, "lr": CFG.lr},
                         masters=optim.masters)
    assert manifest["step"] == 1
    assert manifest["code_sha"] and "has_masters" in manifest
    budget = manifest.get("budget", {})
    assert budget.get("curriculum_step") == 1, "curriculum non persisté"
    assert budget.get("updates") == 2, "compteurs non persistés"
    assert param_fingerprint(student.model) == rep["saved_fp"], (
        "reprise ≠ état sauvegardé"
    )
    trajs = rep["trajectories"][1]
    pids_list = [t.prompt_ids for t in trajs]
    train_step(student, teacher, optim, pids_list, trajs)
    assert param_fingerprint(student.model) == rep["params_fp"][2], (
        "reprise + 1 pas ≠ trajectoire de référence"
    )
    del tok, student, teacher
    gc.collect()
    torch.cuda.empty_cache()
