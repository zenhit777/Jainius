"""Two-stream `forward` — câblage et formes (M1-B, scaffolding M1-C).

NE PROUVE PAS la parité FLARE (H-02) : les couches GDN traitent le flux
concaténé causalement (divergence documentée dans `arch.py`). Vérifie :
formes `[B, L, D]`, finitude, refus `shifted` (E1), validation d'entrées.
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
