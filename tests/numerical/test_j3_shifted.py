"""J3 — distillation on-policy Diffusion-Trust `shifted` (G1, G3).

Couverture :
- layout de supervision (pur CPU) : graines exclues, queue ignorée ;
- rollout shifted : blocs [graine, MASK×], graine jamais rééchantillonnée,
  déterminisme, compteur de forwards exact en statique 1 pas/bloc ;
- forward shifted : formes, états par bloc, chemin différentiable ;
- pas shifted : perte finie, enseignant gelé (aucun gradient), mise à jour ;
- boucle J3 : pertes finies, checkpoint exportable, reprise exacte.
"""

import pytest
import torch

from jainius.decode.rollout import rollout_shifted
from jainius.decode.sampler import SamplerSpec
from jainius.distill.teacher import Teacher
from jainius.model.twostream import TwoStreamRunner, shifted_supervision_mask
from jainius.train.loop import (
    DistillConfig,
    param_fingerprint,
    run_distill_loop,
    train_step_shifted,
)
from tests.conftest import MODEL_ID, chat_ids

MASK_ID = 248077
BB = 4
UNIT = BB - 1


@pytest.fixture(scope="module")
def teacher_hf_model():
    """Seconde instance 0,8B (l'enseignant gelé ne partage pas l'objet
    étudiant — comme en production : même checkpoint, instances distinctes)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval()
    yield model
    del model
    torch.cuda.empty_cache()


def _prompt_ids(tokenizer, text="Count to three."):
    return chat_ids(tokenizer, text)["input_ids"][0].tolist()


def _snapshot_params(model):
    return [p.detach().clone() for p in model.parameters()]


def _restore_params(model, saved):
    with torch.no_grad():
        for p, s in zip(model.parameters(), saved):
            p.copy_(s)
    model.zero_grad()


# ---------- 1. Layout de supervision (CPU) ----------

def test_shifted_supervision_layout_cpu():
    P, L = 5, 5 + 2 * BB
    sup = shifted_supervision_mask(L, P, BB)
    assert sup.shape == (L,) and sup.dtype == torch.bool
    assert not sup[:P].any(), "prompt jamais supervisé"
    # Bloc 0 : graine P exclue, [P+1, P+Bb) supervisés.
    assert sup[P] == False
    assert sup[P + 1: P + BB].all()
    # Bloc 1 : idem.
    assert sup[P + BB] == False
    assert sup[P + BB + 1: P + 2 * BB].all()
    assert int(sup.sum()) == 2 * UNIT


def test_shifted_supervision_tail_ignored_cpu():
    P, L = 5, 5 + BB + 2  # un bloc complet + queue de 2
    sup = shifted_supervision_mask(L, P, BB)
    assert int(sup.sum()) == UNIT
    assert not sup[P + BB:].any(), "queue ignorée"


def test_shifted_supervision_refusals_cpu():
    import pytest

    with pytest.raises(ValueError):
        shifted_supervision_mask(10, 5, 1)
    with pytest.raises(ValueError):
        shifted_supervision_mask(10, 0, BB)


# ---------- 2. Rollout shifted (GPU) ----------

def test_rollout_shifted_static_exact(model_bf16, tokenizer):
    runner = TwoStreamRunner(model_bf16)
    pids = _prompt_ids(tokenizer)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                       seed=11)
    traj = rollout_shifted(runner, pids, 2 * BB, spec, block_size=BB,
                           mask_id=MASK_ID)
    assert len(traj.x0) == 2 * BB
    assert len(traj.rounds) == 2, "statique 1 pas/bloc : 1 tour par bloc"
    for r in traj.rounds:
        assert len(r.ids) == BB and len(r.committed) == BB
        assert r.committed[0] == False, "graine jamais engagée"
        assert sum(r.committed) == UNIT, "dernier pas engage tout le résidu"
    # 1 prefill + 2 blocs × (1 denoise + 1 commit).
    assert traj.n_forwards == 1 + 2 * 2, traj.n_forwards
    assert traj.model_version.startswith("twostream-shifted-")
    assert traj.seed == 11 and not traj.eos_stop


def test_rollout_shifted_deterministic(model_bf16, tokenizer):
    runner = TwoStreamRunner(model_bf16)
    pids = _prompt_ids(tokenizer)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2,
                       seed=5)
    t1 = rollout_shifted(runner, pids, 2 * BB, spec, block_size=BB,
                         mask_id=MASK_ID)
    t2 = rollout_shifted(runner, pids, 2 * BB, spec, block_size=BB,
                         mask_id=MASK_ID)
    assert t1.x0 == t2.x0
    assert [(r.block, r.step, r.ids, r.committed) for r in t1.rounds] == \
        [(r.block, r.step, r.ids, r.committed) for r in t2.rounds]
    assert t1.n_forwards == t2.n_forwards


def test_rollout_shifted_dynamic_runs(model_bf16, tokenizer):
    runner = TwoStreamRunner(model_bf16)
    pids = _prompt_ids(tokenizer, "Say hello briefly.")
    spec = SamplerSpec(mode="dynamic", threshold=0.95, temperature=0.0,
                       steps_per_block=3, seed=3)
    traj = rollout_shifted(runner, pids, 2 * BB, spec, block_size=BB,
                           mask_id=MASK_ID)
    assert len(traj.x0) == 2 * BB
    assert all(r.committed[0] == False for r in traj.rounds)
    assert torch.isfinite(
        torch.tensor([float(t) for t in traj.x0])).all()


# ---------- 3. Forward shifted : formes + gradient (GPU) ----------

def test_forward_train_shifted_shapes_and_grad(model_bf16):
    runner = TwoStreamRunner(model_bf16)
    dev = next(model_bf16.parameters()).device
    P, R, L = 6, 2 * BB, 6 + 2 * BB
    x0 = torch.randint(0, 1000, (1, L), device=dev)
    sup = shifted_supervision_mask(L, P, BB).to(dev)
    xt = x0.clone()
    xt[0][sup] = MASK_ID
    h_x0, h_noisy, starts = runner.forward_train_shifted(
        x0, xt, block_size=BB, resp_start=P)
    assert h_x0.shape == (1, L, h_x0.shape[-1])
    assert h_noisy.shape == (1, 2, BB, h_x0.shape[-1])
    assert starts == [P, P + BB]
    assert torch.isfinite(h_noisy.float()).all()
    loss = h_noisy.square().mean()
    loss.backward()
    grads = [p.grad for p in model_bf16.parameters() if p.grad is not None]
    assert grads, "aucun gradient (chemin non différentiable)"
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    model_bf16.zero_grad()


# ---------- 4. Pas shifted : perte finie, enseignant gelé (GPU) ----------

def test_train_step_shifted_finite_frozen_teacher(
        model_bf16, teacher_hf_model, tokenizer):
    import torch.optim

    runner = TwoStreamRunner(model_bf16)
    teacher = Teacher(teacher_hf_model)
    assert teacher.is_frozen()
    pids = _prompt_ids(tokenizer)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                       seed=0)
    with torch.no_grad():
        traj = rollout_shifted(runner, pids, 2 * BB, spec, block_size=BB,
                               mask_id=MASK_ID)
    optim = torch.optim.SGD(model_bf16.parameters(), lr=1e-3)
    fp_before = param_fingerprint(model_bf16)
    saved = _snapshot_params(model_bf16)
    try:
        stats: dict = {}
        loss = train_step_shifted(
            model_bf16, teacher, optim, [pids], [traj], stats=stats,
            runner=runner, block_size=BB, mask_id=MASK_ID)
        assert loss == loss and abs(loss) != float("inf"), loss
        assert stats["n_supervised"] == 2 * UNIT, stats
        assert stats["grad_norm"] == stats["grad_norm"]  # fini
        assert all(p.grad is None for p in teacher_hf_model.parameters()), (
            "gradients côté enseignant (contrat gelé violé)")
        assert param_fingerprint(model_bf16) != fp_before, "aucune mise à jour"
    finally:
        _restore_params(model_bf16, saved)
    assert param_fingerprint(model_bf16) == fp_before


# ---------- 5. Boucle J3 : checkpoint exportable + reprise exacte (GPU) ----------

def test_j3_loop_ckpt_resume(model_bf16, teacher_hf_model, tokenizer,
                             tmp_path):
    import torch.optim

    from jainius.train.checkpoint import load_ckpt

    runner = TwoStreamRunner(model_bf16)
    teacher = Teacher(teacher_hf_model)
    cfg = DistillConfig(n_prompts=2, max_steps=2, l_max=2 * BB, l_min=BB,
                        warmup_rollouts=2, block_size=BB, steps_per_block=1,
                        sampler_mode="static", temperature=0.0, seed=0,
                        lr=1e-3)
    rep = None
    fp_orig = param_fingerprint(model_bf16)
    saved_orig = _snapshot_params(model_bf16)
    try:
        rep = run_distill_loop(model_bf16, teacher, tokenizer, runner, cfg,
                               tmp_path / "ckpt")
        assert rep["updates"] == 2
        assert rep["convention"] == "shifted" and rep["block_size"] == BB
        assert all(v == v and abs(v) != float("inf") for v in rep["losses"])
        assert rep["n_supervised"] == [2 * UNIT, 4 * UNIT], (
            rep["n_supervised"])
        assert rep["rollout_fw"] > 0 and rep["student_train_fw"] == 4
        saved_dir = tmp_path / "ckpt" / "step_000001"
        assert (saved_dir / "model.pt").exists()
        # Reprise exacte : perturber puis recharger → empreinte du checkpoint.
        saved_fp = rep["saved_fp"]
        assert saved_fp
        param = next(model_bf16.parameters())
        with torch.no_grad():
            param.add_(1.0)
        assert param_fingerprint(model_bf16) != saved_fp
        optim2 = torch.optim.AdamW([p for p in model_bf16.parameters()
                                    if p.requires_grad], lr=1e-3)
        manifest = load_ckpt(
            saved_dir, model_bf16, optim2,
            {"distill": True, "shifted": True, "seed": 0, "lr": 1e-3,
             "block_size": BB})
        assert manifest["step"] == 1
        assert manifest["budget"]["updates"] == 2, manifest["budget"]
        assert "curriculum_step" in manifest["budget"]
        assert param_fingerprint(model_bf16) == saved_fp, "reprise inexacte"
    finally:
        _restore_params(model_bf16, saved_orig)
    assert param_fingerprint(model_bf16) == fp_orig
