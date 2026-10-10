"""T-AR-01 — le chemin causal (avec cache) reproduit le forward AR de
``transformers`` du même checkpoint (A-002).

Protocole : forward complet vs préremplissage (tout sauf dernier token) +
un pas incrémental avec cache, modèle en float32 pour une tolérance stricte.
Critère : argmax identiques à 100 % et écart abs max ≤ 1e-4 (tolérance
float32, §13.5).
"""

import torch

from tests.conftest import chat_ids


def test_ar_01_full_vs_cached(model_fp32, tokenizer):
    torch.manual_seed(0)
    prompt = "Solve step by step: if x+3=7, what is x?"
    batch = chat_ids(tokenizer, prompt).to("cuda")
    input_ids = batch["input_ids"]
    assert input_ids.shape[1] >= 4

    with torch.no_grad():
        full = model_fp32(input_ids).logits
        pre = model_fp32(input_ids[:, :-1], use_cache=True)
        step = model_fp32(
            input_ids[:, -1:],
            past_key_values=pre.past_key_values,
            use_cache=True,
        ).logits

    assert full.shape[1] == input_ids.shape[1] and step.shape[1] == 1
    assert torch.equal(full[:, -1:, :].argmax(-1), step.argmax(-1)), (
        "argmax divergent"
    )
    diff = (full[:, -1:, :].float() - step.float()).abs().max().item()
    assert diff <= 1e-4, f"écart {diff} > 1e-4"
