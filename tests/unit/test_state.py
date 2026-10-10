"""T-STATE-01/03 — capture et isolation de l'état récurrent (M1-A).

T-STATE-01 : snapshot bit-exact ; continuation depuis le snapshot ≡
continuation depuis le cache direct (prouve la capture complète des états
GDN récurrents + convolutifs et des caches KV).
T-STATE-03 : exécution batchée ≡ exécutions séparées (aucune contamination
inter-requêtes au niveau du forward HF).
T-STATE-02 (denoise sans écriture) : BLOCKED — requiert le chemin
`denoise` (M1-B, JAI-ARCH).
"""

import pytest
import torch

from jainius.model.state import (
    cache_fingerprint,
    reset_rows,
    snapshot_cache,
    states_equal,
)
from tests.conftest import chat_ids


def _forward_cache(model, input_ids):
    with torch.no_grad():
        return model(input_ids, use_cache=True)


def test_state_01_snapshot_restore_bitexact(model_bf16, tokenizer):
    torch.manual_seed(0)
    batch = chat_ids(tokenizer, "Hello, who are you?").to("cuda")
    ids = batch["input_ids"]

    out1 = _forward_cache(model_bf16, ids)
    snap = snapshot_cache(out1.past_key_values)
    fp_before = cache_fingerprint(snap)

    nxt = out1.logits[:, -1:].argmax(-1)
    with torch.no_grad():
        cont_direct = model_bf16(
            nxt, past_key_values=out1.past_key_values, use_cache=True
        ).logits
        cont_snap = model_bf16(
            nxt, past_key_values=snapshot_cache(snap), use_cache=True
        ).logits

    assert torch.equal(cont_direct, cont_snap), "snapshot incomplet"
    assert cache_fingerprint(snap) == fp_before, "snapshot muté"
    assert states_equal(snap, snapshot_cache(snap))


def test_state_03_batch_isolation(model_bf16, tokenizer):
    """T-STATE-03 : aucune contamination inter-requêtes (S-002, niveau HF).

    Critère correct : invariance au voisin de batch bit-exacte (la ligne 0
    ne dépend pas de la ligne 1). Mesuré le 2026-10-11 : le passage
    single→batch change le chemin de kernels bf16 (écart abs max 0.5625 à
    échelle de logits ~30, argmax stable à 100 %) — numéricité intra-ligne
    documentée, pas une fuite (voisin-invariance bit-exacte).
    """
    torch.manual_seed(0)
    b1 = chat_ids(tokenizer, "What is 2+2?").to("cuda")["input_ids"]
    b2 = chat_ids(tokenizer, "Name a planet.").to("cuda")["input_ids"]
    n = max(b1.shape[1], b2.shape[1])

    def pad(ids):
        if ids.shape[1] == n:
            return ids
        return torch.cat(
            [
                torch.full(
                    (1, n - ids.shape[1]),
                    tokenizer.pad_token_id,
                    device="cuda",
                ),
                ids,
            ],
            dim=1,
        )

    p1, p2 = pad(b1), pad(b2)
    with torch.no_grad():
        row_same = model_bf16(torch.cat([p1, p1])).logits[0:1]
        row_mix = model_bf16(torch.cat([p1, p2])).logits[0:1]
        row_single = model_bf16(p1).logits
    assert torch.equal(row_same, row_mix), "fuite inter-lignes"
    assert torch.equal(row_same.argmax(-1), row_single.argmax(-1)), (
        "instabilité argmax single→batch"
    )


def test_state_reset_rows_requires_jai_state():
    """reset par ligne : API HF globale uniquement → JAI-STATE (M1-B)."""
    with pytest.raises(NotImplementedError):
        reset_rows(object(), torch.tensor([True]))


@pytest.mark.skip(
    reason="BLOCKED (M1-A) : T-STATE-02 requiert le chemin `denoise` "
    "(JAI-ARCH, M1-B) pour vérifier l'absence d'écriture d'état."
)
def test_state_02_denoise_no_write():
    raise AssertionError("ne doit pas s'exécuter")
