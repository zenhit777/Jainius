"""JAI-ROLL — trajectoires de débruitage (M1-B)."""

from __future__ import annotations

from dataclasses import dataclass, field

from jainius.decode.sampler import SamplerSpec


@dataclass
class Round:
    block: int
    step: int
    ids: list[int]  # état d'entrée du tour (avec MASK)
    committed: list[bool]  # positions engagées à ce tour


@dataclass
class Trajectory:
    prompt_ids: list[int]
    x0: list[int]  # séquence terminale (réponse)
    rounds: list[Round] = field(default_factory=list)
    model_version: str = ""
    seed: int = 0
    sampler: SamplerSpec = field(default_factory=SamplerSpec)
    n_forwards: int = 0
    eos_stop: bool = False
