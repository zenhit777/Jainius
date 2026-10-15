"""J4 — pas batché : même mathématique que le pas unitaire (validation du débit)."""

import torch

from jainius.decode.rollout import rollout_shifted
from jainius.decode.sampler import SamplerSpec
from jainius.distill.teacher import Teacher
from jainius.model.twostream import TwoStreamRunner
from jainius.train.loop import (
    param_fingerprint,
    train_step_shifted,
    train_step_shifted_batched,
)
from tests.conftest import chat_ids

MASK_ID = 248077
BB = 4


def _ids(tokenizer, text):
    return chat_ids(tokenizer, text)["input_ids"][0].tolist()


def _same_length_pair(tokenizer):
    cands = ["Count to three.", "Name a planet.", "Solve x+3=7.",
             "Count to five.", "What is 9*8?", "What is H2O?",
             "Say hello.", "Define entropy.", "Who wrote Hamlet?",
             "Spell cat backwards."]
    seen: dict[int, list[int]] = {}
    for c in cands:
        ids = _ids(tokenizer, c)
        seen.setdefault(len(ids), []).append(ids)
    for _, group in seen.items():
        if len(group) >= 2:
            return group[0], group[1]
    raise AssertionError("aucune paire de prompts de même longueur")


def _grads(model):
    return [p.grad.detach().clone() if p.grad is not None else None
            for p in model.parameters()]


def test_batched_equals_unitary(model_fp32, tokenizer):
    """Équivalence batché ≡ moyenne des unitaires, validée en float32.

    Fait établi (J4) : en bf16, le seul changement de taille de batch
    déplace les résultats (~30 % relatif sur les hiddens, amplification
    de l'arrondi sur ~24 couches — le forward HF lui-même diverge entre
    B=1 et B=2 alors qu'en fp32 l'écart n'est que de 4,9e-5). L'équivalence
    mathématique se prouve donc en fp32 (tolérance 1e-3) ; en bf16, les
    deux voies restent chacune auto-cohérentes (déterminisme exact à
    forme égale).
    """
    import torch.optim

    from transformers import AutoModelForCausalLM

    from tests.conftest import MODEL_ID

    teacher_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cuda",
        trust_remote_code=False).eval()
    try:
        runner = TwoStreamRunner(model_fp32)
        teacher = Teacher(teacher_hf)
        p1, p2 = _same_length_pair(tokenizer)
        spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                           seed=0)
        with torch.no_grad():
            trajs = [rollout_shifted(runner, p, 1 * BB, spec, block_size=BB,
                                     mask_id=MASK_ID) for p in (p1, p2)]
        fp0 = param_fingerprint(model_fp32)
        saved = [p.detach().clone() for p in model_fp32.parameters()]
        try:
            opt = torch.optim.SGD(model_fp32.parameters(), lr=0.0)
            loss_u = train_step_shifted(
                model_fp32, teacher, opt, [p1, p2], trajs, runner=runner,
                block_size=BB, mask_id=MASK_ID)
            grads_u = _grads(model_fp32)
            with torch.no_grad():
                for p, s in zip(model_fp32.parameters(), saved):
                    p.copy_(s)
            model_fp32.zero_grad()
            opt_b = torch.optim.SGD(model_fp32.parameters(), lr=0.0)
            stats: dict = {}
            loss_b = train_step_shifted_batched(
                model_fp32, teacher, opt_b, [p1, p2], trajs, stats=stats,
                runner=runner, block_size=BB, mask_id=MASK_ID)
            grads_b = _grads(model_fp32)
            assert abs(loss_u - loss_b) <= 1e-3 * max(1.0, abs(loss_u)), (
                loss_u, loss_b)
            assert stats["n_supervised"] == 2 * 1 * (BB - 1)
            for gu, gb in zip(grads_u, grads_b):
                assert (gu is None) == (gb is None)
                if gu is not None:
                    assert torch.allclose(gu.float(), gb.float(),
                                          atol=1e-3, rtol=1e-3), (
                        (gu - gb).abs().max())
            assert param_fingerprint(model_fp32) == fp0, "lr=0 a modifié"
        finally:
            with torch.no_grad():
                for p, s in zip(model_fp32.parameters(), saved):
                    p.copy_(s)
            model_fp32.zero_grad()
        assert param_fingerprint(model_fp32) == fp0
    finally:
        del teacher_hf
        torch.cuda.empty_cache()


def test_batched_rejects_heterogeneous(model_bf16, tokenizer):
    import torch.optim
    from types import SimpleNamespace

    runner = TwoStreamRunner(model_bf16)
    teacher = SimpleNamespace(model=None)
    opt = torch.optim.SGD(model_bf16.parameters(), lr=0.0)
    t1 = SimpleNamespace(x0=[1] * 8)
    t2 = SimpleNamespace(x0=[1] * 4)
    import pytest

    with pytest.raises(ValueError, match="hétérogènes"):
        train_step_shifted_batched(
            model_bf16, teacher, opt, [[2] * 10, [3] * 12], [t1, t1],
            runner=runner, block_size=BB, mask_id=MASK_ID)
    with pytest.raises(ValueError, match="hétérogènes"):
        train_step_shifted_batched(
            model_bf16, teacher, opt, [[2] * 10, [2] * 10], [t1, t2],
            runner=runner, block_size=BB, mask_id=MASK_ID)
