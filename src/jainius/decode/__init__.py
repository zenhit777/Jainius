"""JAI-ROLL / JAI-GEN — échantillonnage, trajectoires, génération (M1-B)."""

from jainius.decode.rollout import generate, rollout
from jainius.decode.sampler import SamplerSpec, sample_token
from jainius.decode.trajectory import Round, Trajectory

__all__ = [
    "Round",
    "SamplerSpec",
    "Trajectory",
    "generate",
    "rollout",
    "sample_token",
]
