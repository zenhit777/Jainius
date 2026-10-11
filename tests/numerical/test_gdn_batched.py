"""T-GDN-02 — indépendance conditionnelle des blocs bruités (M1-B).

Propriété [J] (§10.2) : les blocs bruités sont indépendants
conditionnellement à l'état propre de frontière. Test discriminant sur le
chemin de référence : l'ordre de débruitage (A puis B vs B puis A) ne change
ni les logits d'un bloc ni l'état partagé. Un échec isolerait une fuite
d'état inter-blocs (contradiction avec C-03 / FLARE Éq. 6).
"""

import torch

from jainius.model.arch import DiffusionSpec, JainiusModel
from jainius.model.state import cache_fingerprint
from tests.conftest import chat_ids


def test_gdn_02_block_order_invariance(model_bf16, tokenizer):
    batch = chat_ids(tokenizer, "List three colors.").to("cuda")
    model = JainiusModel(model_bf16, DiffusionSpec())
    mask_id = model.spec.mask_token_id
    a = torch.full((1, 4), mask_id + 1, device="cuda")  # bloc témoin non-MASK
    b = torch.full((1, 4), mask_id, device="cuda")  # bloc à débruiter
    with torch.no_grad():
        _, cache = model.prefill(batch["input_ids"])
        fp0 = cache_fingerprint(cache)
        d_a_first = model.denoise(a, cache)
        d_b_after_a = model.denoise(b, cache)
        d_b_first = model.denoise(b, cache)
        d_a_after_b = model.denoise(a, cache)
    assert torch.equal(d_a_first, d_a_after_b), "fuite d'état inter-blocs (A)"
    assert torch.equal(d_b_after_a, d_b_first), "fuite d'état inter-blocs (B)"
    assert cache_fingerprint(cache) == fp0, "l'état partagé a été muté"
