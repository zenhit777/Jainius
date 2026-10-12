"""T-LOAD-01 — intégrité du checkpoint source (P5.1, CPU synthétique).

Couvre : chargement conforme, tenseur altéré, tenseur manquant /
supplémentaire, incompatibilité de forme, échec AVANT utilisation des
poids (aucune modification silencieuse). La validation du checkpoint de
production est instrumentée par `scripts/build_source_manifest.py`
(manifeste versionné sous `docs/audits/`) — statut dans le rapport.
"""

import pytest
import torch

from jainius.train.source_manifest import (
    FingerprintMismatch,
    build_manifest,
    verify_manifest,
)


def _toy_state():
    torch.manual_seed(0)
    return {"a.weight": torch.randn(4, 8, dtype=torch.float32),
            "a.bias": torch.zeros(4, dtype=torch.float32),
            "b.emb": torch.randint(0, 100, (6,)).to(torch.int64)}


def test_load_01_conform():
    sd = _toy_state()
    m = build_manifest(sd)
    assert m["n_tensors"] == 3
    assert verify_manifest(m, sd)["checked"] == 3
    # Les poids ne sont pas modifiés par la vérification.
    before = {k: v.clone() for k, v in sd.items()}
    verify_manifest(m, sd)
    for k in sd:
        assert torch.equal(sd[k], before[k])


def test_load_01_altered_rejected():
    sd = _toy_state()
    m = build_manifest(sd)
    bad = dict(sd)
    bad["a.weight"] = sd["a.weight"] + 1e-3
    with pytest.raises(FingerprintMismatch, match="empreinte"):
        verify_manifest(m, bad)


def test_load_01_missing_rejected():
    sd = _toy_state()
    m = build_manifest(sd)
    bad = {k: v for k, v in sd.items() if k != "a.bias"}
    with pytest.raises(FingerprintMismatch, match="absents"):
        verify_manifest(m, bad)


def test_load_01_extra_rejected():
    sd = _toy_state()
    m = build_manifest(sd)
    bad = dict(sd)
    bad["c.new"] = torch.zeros(2)
    with pytest.raises(FingerprintMismatch, match="supplémentaires"):
        verify_manifest(m, bad)


def test_load_01_shape_rejected():
    sd = _toy_state()
    m = build_manifest(sd)
    bad = dict(sd)
    bad["a.bias"] = torch.zeros(5, dtype=torch.float32)
    with pytest.raises(FingerprintMismatch, match="forme"):
        verify_manifest(m, bad)


def test_load_01_dtype_rejected():
    sd = _toy_state()
    m = build_manifest(sd)
    bad = dict(sd)
    bad["a.bias"] = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(FingerprintMismatch, match="dtype"):
        verify_manifest(m, bad)
