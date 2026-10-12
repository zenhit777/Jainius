"""JAI-LOAD — intégrité du checkpoint source (P5.1, T-LOAD-01).

Périmètre du tenseur haché (canonique, stable, documenté) : pour chaque
tenseur du `state_dict` — nom, forme (`tuple(shape)`), dtype (`str(dtype)`)
et octets des DONNÉES (vue CPU contiguë, `view(uint8)`, bit-exact y compris
bf16/fp16 — jamais `.numpy()` direct). Le hash du FICHIER de checkpoint
(safetensors) est distinct et non interchangeable (non couvert ici).

Création initiale du manifeste : opération EXPLICITE (`build_manifest` +
`write_manifest`) — jamais de génération silencieuse pour faire passer un
checkpoint différent. `verify_manifest` échoue explicitement (tenseur
altéré/absent/supplémentaire/forme incompatible) AVANT toute utilisation
des poids non vérifiés ; aucune modification silencieuse des poids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class FingerprintMismatch(ValueError):
    """Checkpoint source non conforme au manifeste de référence."""


def tensor_fingerprint(name: str, tensor: torch.Tensor) -> dict:
    """Empreinte canonique d'un tenseur : nom, forme, dtype, SHA-256."""
    t = tensor.detach().to("cpu").contiguous()
    view = _INT_VIEW[t.element_size()]
    digest = hashlib.sha256()
    digest.update(t.view(view).numpy().tobytes())
    return {"name": name, "shape": list(t.shape), "dtype": str(t.dtype),
            "sha256": digest.hexdigest()}


def build_manifest(state_dict: dict[str, torch.Tensor]) -> dict:
    """Manifeste de référence (opération explicite et traçable)."""
    tensors = [tensor_fingerprint(n, state_dict[n])
               for n in sorted(state_dict)]
    return {"version": 1, "n_tensors": len(tensors), "tensors": tensors}


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest: dict,
                    state_dict: dict[str, torch.Tensor]) -> dict:
    """Compare le manifeste de référence au checkpoint chargé.

    Lève `FingerprintMismatch` (altéré, absent, supplémentaire, forme ou
    dtype incompatible) avant toute utilisation des poids. Retourne un
    résumé `{"n_tensors": ..., "checked": ...}` en cas de succès.
    Les poids ne sont jamais modifiés.
    """
    ref = {t["name"]: t for t in manifest.get("tensors", [])}
    got_names = set(state_dict)
    ref_names = set(ref)
    if missing := sorted(ref_names - got_names):
        raise FingerprintMismatch(f"tenseurs absents : {missing[:5]}")
    if extra := sorted(got_names - ref_names):
        raise FingerprintMismatch(f"tenseurs supplémentaires : {extra[:5]}")
    for name in sorted(ref_names):
        r = ref[name]
        t = state_dict[name]
        if list(t.shape) != list(r["shape"]):
            raise FingerprintMismatch(
                f"{name} : forme {list(t.shape)} ≠ {r['shape']}")
        if str(t.dtype) != r["dtype"]:
            raise FingerprintMismatch(
                f"{name} : dtype {t.dtype} ≠ {r['dtype']}")
        if tensor_fingerprint(name, t)["sha256"] != r["sha256"]:
            raise FingerprintMismatch(f"{name} : empreinte différente")
    return {"n_tensors": len(ref), "checked": len(ref)}
