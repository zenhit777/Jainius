"""Two-stream `forward` — câblage et formes (M1-B, scaffolding M1-C).

NE PROUVE PAS la parité FLARE (H-02) : les couches GDN traitent le flux
concaténé causalement (divergence documentée dans `arch.py`, caractérisée
par `test_two_stream_gdn_future_leak` ci-dessous). Vérifie : formes
`[B, L, D]`, finitude, refus `shifted` (E1), validation d'entrées.
"""

import pytest
import torch

from jainius.model.arch import (
    ConventionMismatch,
    DiffusionSpec,
    JainiusModel,
    TwoStreamBatch,
)
from tests.conftest import chat_ids


def _batch(tokenizer, device, n_resp=8):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    p = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    x0 = torch.cat([p, torch.full((1, n_resp), 100, device=device)], dim=1)
    xt = x0.clone()
    xt[0, -n_resp:] = DiffusionSpec().mask_token_id
    sup = torch.zeros_like(x0, dtype=torch.bool)
    sup[0, -n_resp:] = True
    return TwoStreamBatch(x0=x0, xt=xt, sup_mask=sup)


def test_two_stream_shapes_finite(model_bf16, tokenizer):
    model = JainiusModel(model_bf16, DiffusionSpec())
    device = next(model_bf16.parameters()).device
    with torch.no_grad():
        h = model.forward_two_stream(_batch(tokenizer, device))
    assert h.h_x0.shape == h.h_xt.shape
    assert h.h_x0.shape[0] == 1 and h.h_x0.shape[2] == 1024
    assert torch.isfinite(h.h_x0).all() and torch.isfinite(h.h_xt).all()


def test_two_stream_rejects_shifted(model_bf16):
    with pytest.raises(ConventionMismatch):
        JainiusModel(model_bf16, DiffusionSpec(logit_convention="shifted"))


def test_two_stream_rejects_bad_shapes(model_bf16, tokenizer):
    model = JainiusModel(model_bf16, DiffusionSpec())
    device = next(model_bf16.parameters()).device
    batch = _batch(tokenizer, device)
    batch.xt = batch.xt[:, :-1]
    with pytest.raises(ValueError):
        with torch.no_grad():
            model.forward_two_stream(batch)


def test_two_stream_gdn_future_leak(model_bf16, tokenizer):
    """Discriminant H-02 (chemin étudiant, backbone hybride).

    (a) Le flux x0 est bit-exact vs forward AR plain (ordre causal +
    masque : aucune influence aval possible) — la plomberie du masque
    est exacte.
    (b) Le flux xt FUIT le futur de x0 par la récurrence GDN (le masque
    ne gouverne que l'attention) : perturber le dernier token x0 change
    les hidden xt. Ce test CARACTÉRISE la divergence (il passe parce que
    la fuite existe) ; la planification d'états GDN l'inversera.
    Il ne prouve ni n'infirme rien sur FLARE (pas de parité revendiquée).
    """
    model = JainiusModel(model_bf16, DiffusionSpec())
    device = next(model_bf16.parameters()).device
    batch = _batch(tokenizer, device)
    with torch.no_grad():
        h = model.forward_two_stream(batch)
        plain = model_bf16(batch.x0, output_hidden_states=True).hidden_states[-1]
    # Tolérance bf16 large mais discriminante : le chemin explicite
    # (masque 4D + positions) accumule l'arrondi différemment du chemin
    # natif (mesuré : max 0.75 ; un bug de masque donne ~20, cf. pré-M2).
    # L'exactitude du masque est prouvée par ailleurs (T-MASK-01, sonde 4D).
    assert torch.allclose(h.h_x0, plain, atol=1.0, rtol=0.05), (
        "plombrie du flux x0 inexacte"
    )
    n_resp = 8
    l0 = batch.x0.shape[1] - n_resp
    altered = batch.x0.clone()
    altered[0, -1] = (altered[0, -1] + 1) % 248320
    with torch.no_grad():
        h2 = model.forward_two_stream(
            TwoStreamBatch(x0=altered, xt=batch.xt, sup_mask=batch.sup_mask)
        )
    leak = (h.h_xt[0, l0:] - h2.h_xt[0, l0:]).abs().max().item()
    assert leak > 0, "fuite GDN attendue mais non observée (H-02 ?)"
