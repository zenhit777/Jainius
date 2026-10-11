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
    """Discriminant H-02 (chemin étudiant, backbone hybride) — BLOCKED
    pour tout pilote : la fuite est présente dans toutes les variantes.

    (a) Le flux x0 est proche de l'AR plain (tolérance bf16 large ;
    exactitude du masque prouvée par T-MASK-01 et la sonde 4D).
    (b) Le flux xt FUIT par la récurrence GDN (le masque 4D, exact au
    niveau booléen (T-MASK-01) et appliqué (sonde 4D fp32 à 2.2e-5), ne
    gouverne que l'attention) : perturber le dernier token x0 (futur de
    la réponse) change les hidden xt, ET perturber le préfixe prompt de
    xt lui-même les change aussi — la récurrence accumule tout le flux
    concaténé sans ré-initialisation par bloc. Les deux fuites sont
    caractérisées ici (présentes en bf16 ET fp32, à 1 comme à 2 blocs).
    Ce test CARACTÉRISE la divergence (il passe parce que les fuites
    existent) ; la planification d'états GDN (contrat M2 :
    ré-initialisation des blocs bruités depuis les frontières propres)
    l'inversera. Il ne prouve ni n'infirme rien sur FLARE.
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
    # Deuxième fuite (même mécanisme) : le préfixe prompt du flux xt
    # lui-même pollue les lignes réponse (aucune ré-initialisation par
    # bloc). Les deux voies d'attention sont prouvées exactes par ailleurs.
    xt_alt = batch.xt.clone()
    xt_alt[0, :l0] = (xt_alt[0, :l0] + 7) % 248320
    with torch.no_grad():
        h3 = model.forward_two_stream(
            TwoStreamBatch(x0=batch.x0, xt=xt_alt, sup_mask=batch.sup_mask)
        )
    leak_xt = (h.h_xt[0, l0:] - h3.h_xt[0, l0:]).abs().max().item()
    assert leak_xt > 0, "fuite intra-xt absente (inattendu, cf. H-02)"


def test_two_stream_gdn_leak_variants(tokenizer):
    """La fuite ne dépend ni de la précision ni du nombre de blocs
    (caractérisation B2 : fp32 + 1 bloc)."""
    from transformers import AutoModelForCausalLM

    model_fp32 = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.float32, device_map="cuda",
        trust_remote_code=False).eval()
    try:
        model = JainiusModel(model_fp32, DiffusionSpec())
        device = next(model_fp32.parameters()).device
        batch = _batch(tokenizer, device, n_resp=4)  # 1 seul bloc
        n_resp = 4
        l0 = batch.x0.shape[1] - n_resp
        with torch.no_grad():
            h = model.forward_two_stream(batch)
            altered = batch.x0.clone()
            altered[0, -1] = (altered[0, -1] + 1) % 248320
            h2 = model.forward_two_stream(
                TwoStreamBatch(x0=altered, xt=batch.xt, sup_mask=batch.sup_mask))
        leak = (h.h_xt[0, l0:] - h2.h_xt[0, l0:]).abs().max().item()
        assert leak > 0, "fuite absente en fp32/1 bloc (inattendu)"
    finally:
        del model_fp32
        torch.cuda.empty_cache()
