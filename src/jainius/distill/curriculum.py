"""Curriculum de longueur de rollout (JAI-TRAIN, M1-C micro-test).

Port de ``_get_scheduled_max_token`` (OPDLM ``rl.py:213-228``) :
``start + (end - start) * frac(progress)``, arrondi au ``block_size``,
avec ``progress = step / warmup`` saturé à 1. Formes : ``linear``,
``sin`` (``sin(π/2·p)``), ``cos`` (``1-cos(π/2·p)``). Défaut yaml OPDLM :
start=100, end=2000, ramp=20, cos (lanceurs : end=4000).
Les plafonds par domaine s'appliquent par ``min(L_s, plafond)`` côté données.
"""

from __future__ import annotations

import math


def rollout_length(
    step: int,
    *,
    l_min: int = 100,
    l_max: int = 4000,
    warmup_rollouts: int = 100,
    shape: str = "linear",
    block_size: int = 4,
) -> int:
    """Longueur de rollout au pas ``step`` (T-CURR-01)."""
    if step < 0:
        raise ValueError(f"step négatif : {step}")
    if warmup_rollouts <= 0:
        raise ValueError("warmup_rollouts doit être > 0")
    if l_min <= 0 or l_max < l_min:
        raise ValueError(f"bornes incohérentes : l_min={l_min}, l_max={l_max}")
    progress = min(1.0, step / warmup_rollouts)
    if shape == "linear":
        frac = progress
    elif shape == "sin":
        frac = math.sin(math.pi / 2 * progress)
    elif shape == "cos":
        frac = 1 - math.cos(math.pi / 2 * progress)
    else:
        raise ValueError(f"forme inconnue : {shape}")
    value = l_min + (l_max - l_min) * frac
    return max(block_size, int(round(value / block_size)) * block_size)
