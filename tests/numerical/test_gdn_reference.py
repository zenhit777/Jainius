"""T-GDN-01 — référence causale GDN via la pile HF (M1-A).

Vérifie : (1) la carte des couches du checkpoint (18 GDN + 6 attention
complète aux indices 3, 7, 11, 15, 19, 23, §3.3 du cahier technique) ;
(2) des sorties finies et déterministes en bf16 sur entrée contrôlée
(la pile utilise le repli PyTorch de référence, `fla` absent — correct
mais lent, cf. M0).

T-GDN-03 (parité vs noyau FLARE) : BLOCKED — requiert l'environnement
FLARE (torch 2.10 + fla 0.5.1), non installé. T-GDN-02/04/05 : BLOCKED —
requièrent le forward deux-flux (M1-B).
"""

import pytest
import torch

from tests.conftest import chat_ids

EXPECTED_FULL_ATTENTION = {3, 7, 11, 15, 19, 23}


def test_gdn_01_layer_map_and_determinism(model_bf16, tokenizer):
    cfg = model_bf16.config.get_text_config()
    layer_types = cfg.layer_types
    assert len(layer_types) == 24
    full = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    gdn = {i for i, t in enumerate(layer_types) if t == "linear_attention"}
    assert full == EXPECTED_FULL_ATTENTION, f"attention complète : {sorted(full)}"
    assert len(gdn) == 18, f"couches GDN : {len(gdn)}"

    batch = chat_ids(tokenizer, "What is 2+2?").to("cuda")
    with torch.no_grad():
        a = model_bf16(batch["input_ids"]).logits
        b = model_bf16(batch["input_ids"]).logits
    assert torch.isfinite(a).all(), "logits non finis (bf16)"
    assert torch.equal(a, b), "forward bf16 non déterministe"


@pytest.mark.skip(
    reason="BLOCKED : T-GDN-03 requiert l'environnement FLARE "
    "(torch 2.10 + fla 0.5.1) ; T-GDN-04 requiert JAI-STATE.reset ; "
    "T-GDN-05 requiert le graphe deux-flux avec gradient (M1-C). "
    "T-GDN-02 est testé pour de vrai dans test_gdn_batched.py."
)
def test_gdn_blocked_oracle_and_two_stream():
    raise AssertionError("ne doit pas s'exécuter")
