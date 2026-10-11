"""M1-D0 — smoke JAI-TRAIN : câblage complet, compteurs, reprise (F-004…F-008).

 wavefunction unique : wiring (1× run_smoke) puis reprise sur paire fraîche
(mémoire libérée entre les deux). Aucun benchmark qualité, aucune
distillation : jeu synthétique documenté, réponse plafonnée à 8 tokens
(L_max=128 respecté comme plafond, jamais atteint).
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
from jainius.decode.rollout import rollout
from jainius.decode.sampler import SamplerSpec

MODEL_ID = "Qwen/Qwen3.5-0.8B"
CFG = SmokeConfig()
REP = {}


def _load_pair():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    student_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    )
    teacher_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    )
    student = JainiusModel(student_hf, DiffusionSpec())
    teacher = Teacher(teacher_hf)
    return tok, student, teacher


def test_smoke_wiring(tmp_path):
    tok, student, teacher = _load_pair()
    rep = run_smoke(student, teacher, tok, CFG, tmp_path / "ckpt")
    # --- compteurs exacts ---
    assert rep["prompts_per_step"] == [16] * 5
    assert sorted(rep["trajectories"]) == [0, 1], "2 rollouts exigés"
    assert rep["updates"] == 5
    assert rep["lengths"] == [
        rollout_length(s, l_min=8, l_max=128, warmup_rollouts=10)
        for s in range(5)
    ], "progression curriculum inexacte"
    assert rep["student_train_fw"] == 16 * 5
    assert rep["teacher_fw"] == 16 * 5 + 5  # +1 extended/pas (prompt 0)
    assert rep["rollout_fw"] == 16 * 7 * 2 + 7 * 2  # +1 rejouement/round
    assert len(rep["losses"]) == 5 and all(v == v and abs(v) != float("inf") for v in rep["losses"])
    assert all(rep["params_changed"]) and all(rep["grads_finite"])
    assert all(diff == diff for diff in rep["compat_diffs"])
    # Isolation inter-rollouts vérifiée DANS run_smoke (poids identiques) :
    # rejouement du prompt 0 après les 15 autres ⇒ trajectoire identique.
    REP["rep"] = rep
    REP["ckpt_dir"] = rep["saved"]["dir"]
    import json
    from pathlib import Path
    out = Path("_artefacts/m1d0/smoke_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in rep.items() if k != "trajectories"}
    summary["trajectories_x0"] = {
        s: [t.x0 for t in trajs] for s, trajs in rep["trajectories"].items()
    }
    summary["trajectories_fw"] = {
        s: [t.n_forwards for t in trajs] for s, trajs in rep["trajectories"].items()
    }
    out.write_text(json.dumps(summary, indent=2))
    del tok, student, teacher, rep
    gc.collect()
    torch.cuda.empty_cache()


def test_smoke_resume():
    rep = REP["rep"]
    tok, student, teacher = _load_pair()
    optim = torch.optim.AdamW(student.model.parameters(), lr=CFG.lr)
    student.model.train()
    manifest = load_ckpt(REP["ckpt_dir"], student.model, optim,
                         {"smoke": True, "seed": CFG.seed})
    assert manifest["step"] == 1
    assert param_fingerprint(student.model) == rep["saved_fp"], (
        "reprise ≠ état sauvegardé"
    )
    # Rejoue exactement le pas 2 avec les trajectoires enregistrées.
    trajs = rep["trajectories"][1]
    pids_list = [t.prompt_ids for t in trajs]
    train_step(student, teacher, optim, pids_list, trajs)
    assert param_fingerprint(student.model) == rep["params_fp"][2], (
        "reprise + 1 pas ≠ trajectoire de référence"
    )
    del tok, student, teacher
    gc.collect()
    torch.cuda.empty_cache()


def test_compat_qwen35_extended():
    """Porte compatibilité (distincte du câblage) : voie simple ≡ extended
    sur le backbone hybride Qwen3.5. ÉCHEC = blocage expérimental D-04
    enregistré, pas un assouplissement."""
    rep = REP["rep"]
    worst = max(rep["compat_diffs"])
    assert worst <= CFG.compat_tol, (
        f"BLOCKED (D-04) : extended/simple divergent sur Qwen3.5 "
        f"(max {worst} > tol {CFG.compat_tol}) : {rep['compat_diffs']}"
    )
