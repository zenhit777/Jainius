"""T-STATE-02 — `denoise` laisse l'état et les caches inchangés (C-03).

Réalisé en M1-B via ``JainiusModel.denoise`` (copie de travail + garde
``StateLeakError``). Remplace le stub BLOCKED de ``test_state.py``.
"""

import torch

from jainius.decode.rollout import rollout
from jainius.decode.sampler import SamplerSpec
from jainius.model.arch import DiffusionSpec, JainiusModel
from jainius.model.state import cache_fingerprint
from tests.conftest import chat_ids


def _wrapped(model_bf16):
    return JainiusModel(model_bf16, DiffusionSpec())


def test_state_02_denoise_no_write(model_bf16, tokenizer):
    batch = chat_ids(tokenizer, "Hello, who are you?").to("cuda")
    model = _wrapped(model_bf16)
    with torch.no_grad():
        _, cache = model.prefill(batch["input_ids"])
    fp0 = cache_fingerprint(cache)
    probe = torch.full((1, 4), DiffusionSpec().mask_token_id, device="cuda")
    with torch.no_grad():
        logits = model.denoise(probe, cache)
    assert torch.isfinite(logits).all()
    assert cache_fingerprint(cache) == fp0, "denoise a écrit l'état (C-03 violé)"


def test_state_02_no_write_during_rollout(model_bf16, tokenizer):
    """Garde-fou trajectoire : l'état pré-bloc est préservé par le débruitage.

    Vérifie que deux rollouts partageant le même préfixe produisent le même
    premier bloc (les passes intermédiaires ne corrompent pas l'état).
    """
    batch = chat_ids(tokenizer, "Count to three.").to("cuda")
    prompt_ids = batch["input_ids"][0].tolist()
    model = _wrapped(model_bf16)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2, seed=7)
    with torch.no_grad():
        t4 = rollout(model, prompt_ids, 4, spec)
        t8 = rollout(model, prompt_ids, 8, spec)
    assert t4.x0 == t8.x0[:4], "le débruitage a corrompu l'état inter-blocs"
