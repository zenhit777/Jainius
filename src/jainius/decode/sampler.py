"""JAI-ROLL — échantillonneur par blocs (M1-B).

Confiance = probabilité du jeton échantillonné (H-08 résolue).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplerSpec:
    mode: str = "static"  # "static" | "dynamic"
    threshold: float = 0.9
    steps_per_block: int = 4
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0


def sample_token(logits: torch.Tensor, spec: SamplerSpec, rng) -> tuple[int, float]:
    """Échantillonne un jeton ; confiance = prob du jeton échantillonné."""
    row = logits.float()
    if spec.temperature > 0:
        row = row / spec.temperature
        if spec.top_k > 0:
            k = min(spec.top_k, row.shape[-1])
            vals, idx = torch.topk(row, k=k)
            masked = torch.full_like(row, float("-inf"))
            masked.scatter_(-1, idx, vals)
            row = masked
        if spec.top_p < 1.0:
            svals, sidx = torch.sort(row, descending=True)
            probs = svals.softmax(-1)
            keep = probs.cumsum(-1) <= spec.top_p
            keep[..., 0] = True
            svals = torch.where(keep, svals, torch.full_like(svals, float("-inf")))
            row = torch.full_like(row, float("-inf")).scatter_(-1, sidx, svals)
        probs = row.softmax(-1)
        token = int(torch.multinomial(probs, 1, generator=rng).item())
        return token, float(probs[token].item())
    token = int(row.argmax().item())
    return token, float(row.softmax(-1)[token].item())
