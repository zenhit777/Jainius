"""JAI-STATE (M1-A, partiel) — inspection et capture de l'état récurrent HF.

Périmètre M1-A : ``snapshot_cache`` (copie profonde bit-exacte),
``states_equal`` / ``cache_fingerprint`` (comparaison et hachage).
T-STATE-01 et T-STATE-03 sont testables à ce niveau.

Hors périmètre (M1-B, JAI-STATE complet) : ``denoise`` sans écriture
(T-STATE-02), ``reset`` par ligne (l'API HF ``DynamicCache.reset()`` est
globale uniquement — vérifié le 2026-10-11 sur transformers 5.17.0),
planification GDN deux-flux (H-02).
"""

from __future__ import annotations

import copy
import hashlib

import torch

_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _tensor_bytes(item: torch.Tensor) -> bytes:
    """Octets bit-exacts (numpy ne gère pas bf16 : vue entière même taille)."""
    view_dtype = _INT_VIEW[item.element_size()]
    return item.detach().to("cpu").contiguous().view(view_dtype).numpy().tobytes()


_STATE_TENSOR_ATTRS = (
    "conv_states",
    "recurrent_states",
    "keys",
    "values",
)


def _iter_state_tensors(cache) -> list[torch.Tensor]:
    """Collecte tous les tenseurs d'état (ordre stable : couches puis attrs)."""
    tensors: list[torch.Tensor] = []
    for layer in cache.layers:
        for attr in _STATE_TENSOR_ATTRS:
            value = getattr(layer, attr, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (list, tuple)):
                tensors.extend(v for v in value if isinstance(v, torch.Tensor))
    return tensors


def snapshot_cache(cache):
    """Copie profonde bit-exacte d'un cache HF (ne jamais muter le retour)."""
    snap = copy.deepcopy(cache)
    if not states_equal(cache, snap):
        raise RuntimeError("snapshot_cache : la copie ne reproduit pas l'état")
    return snap


def cache_fingerprint(cache) -> str:
    """SHA-256 sur (nom, forme, dtype, octets) de chaque tenseur d'état."""
    digest = hashlib.sha256()
    for layer_idx, layer in enumerate(cache.layers):
        for attr in _STATE_TENSOR_ATTRS:
            value = getattr(layer, attr, None)
            items = (
                value
                if isinstance(value, (list, tuple))
                else [value] if isinstance(value, torch.Tensor) else []
            )
            for item in items:
                if not isinstance(item, torch.Tensor):
                    continue
                digest.update(
                    f"{layer_idx}.{attr}.{tuple(item.shape)}.{item.dtype}."
                    f"{item.device.type}".encode()
                )
                digest.update(_tensor_bytes(item))
    return digest.hexdigest()


def states_equal(a, b) -> bool:
    """Égalité bit-exacte de deux caches (même structure et mêmes octets)."""
    ta, tb = _iter_state_tensors(a), _iter_state_tensors(b)
    if len(ta) != len(tb):
        return False
    return all(
        xa.shape == xb.shape and xa.dtype == xb.dtype and bool((xa == xb).all())
        for xa, xb in zip(ta, tb)
    )


def reset_rows(cache, rows: torch.Tensor) -> None:
    """Remise à zéro des lignes d'état (frontière de document / requête).

    NON IMPLÉMENTÉ en M1-A : ``DynamicCache.reset()`` est global uniquement.
    Requiert JAI-STATE (M1-B) : zéro par ligne de ``conv_states``,
    ``recurrent_states``, ``keys``/``values`` des 24 couches.
    """
    raise NotImplementedError(
        "reset_rows requiert JAI-STATE (M1-B) : l'API HF ne propose "
        "qu'un reset global (DynamicCache.reset, transformers 5.17.0)."
    )
