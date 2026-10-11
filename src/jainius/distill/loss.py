"""JAI-LOSS — KL enseignant→étudiant masquée (M1-C, micro-tests).

Référence : OPDLM ``rl_sdar.py:1366-1376`` (KL directe, terme NaN-safe :
les entrées ``-inf`` de l'enseignant contribuent 0) et normalisation
``:1427-1449`` (somme par ligne / max(1, n positions), puis moyenne batch).
Vocabulaire complet par morceaux (``chunk_tokens`` positions à la fois) ;
``top_k`` = repli mémoire type ``top_k_logits=16`` (sans renormalisation,
comme l'usage creux OPDLM — documenté, pas égal au vocabulaire complet).

Objectif OPDLM uniquement. Aucun terme FLARE (CE shiftée, L_AR) ici :
le mélange d'objectifs est interdit avant baseline (CdC §3.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


def _work_dtype(hidden: torch.Tensor) -> torch.dtype:
    """float32 de travail (stabilité bf16), sans jamais rétrograder float64."""
    return torch.float64 if hidden.dtype == torch.float64 else torch.float32


@dataclass
class LossOutput:
    loss: torch.Tensor  # scalaire différentiable
    per_row: torch.Tensor  # [B]
    n_supervised: int
    kl_mean: float


def kl_teacher_to_student(
    h_student: torch.Tensor,
    lm_student: Callable[[torch.Tensor], torch.Tensor],
    h_teacher: torch.Tensor,
    lm_teacher: Callable[[torch.Tensor], torch.Tensor],
    sup_mask: torch.Tensor,
    *,
    chunk_tokens: int = 1024,
    top_k: int | None = None,
) -> LossOutput:
    """KL(p_T ‖ p_S) masquée, normalisée par ligne puis par batch."""
    if h_student.shape != h_teacher.shape or h_student.dim() != 3:
        raise ValueError("formes caché étudiant/enseignant incohérentes")
    if sup_mask.shape != h_student.shape[:2] or sup_mask.dtype != torch.bool:
        raise ValueError("sup_mask doit être booléen [B, L]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens doit être > 0")
    bsz, length, _ = h_student.shape
    dev = h_student.device
    row_sums = torch.zeros(bsz, device=dev)
    row_counts = torch.zeros(bsz, device=dev)
    n_supervised = 0
    flat_s = h_student.reshape(-1, h_student.shape[-1])
    flat_t = h_teacher.reshape(-1, h_teacher.shape[-1])
    flat_m = sup_mask.reshape(-1)
    sup_idx = torch.where(flat_m)[0]
    n_supervised = int(sup_idx.numel())
    work = _work_dtype(h_student)
    for start in range(0, n_supervised, chunk_tokens):
        sl = slice(start, start + chunk_tokens)
        idx = sup_idx[sl]
        logp_t = F.log_softmax(lm_teacher(flat_t[idx]).to(work), dim=-1)
        logp_s = F.log_softmax(lm_student(flat_s[idx]).to(work), dim=-1)
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k doit être > 0 ou None")
            k = min(top_k, logp_t.shape[-1])
            vals, order = torch.topk(logp_t, k=k, dim=-1)
            logp_t = torch.gather(logp_t, -1, order)
            logp_s = torch.gather(logp_s, -1, order)
        finite = torch.isfinite(logp_t)
        terms = torch.where(
            finite,
            logp_t.exp() * (logp_t - logp_s),
            torch.zeros_like(logp_t),
        )
        kl = terms.sum(dim=-1)  # [n]
        rows = (idx // length).tolist()
        for r, v in zip(rows, kl):
            row_sums[r] += v.detach()
            row_counts[r] += 1
    per_row = row_sums / torch.clamp_min(row_counts, 1.0)
    # Voie différentiable : mêmes poids, graphe conservé (recalcule
    # vectorisé quand tout tient en un morceau, sinon accumulation).
    loss = _differentiable_loss(
        flat_s, flat_t, sup_idx, length, bsz, lm_student, lm_teacher,
        chunk_tokens, top_k,
    )
    return LossOutput(
        loss=loss,
        per_row=per_row,
        n_supervised=n_supervised,
        kl_mean=float(per_row.mean().item()),
    )


def _differentiable_loss(
    flat_s, flat_t, sup_idx, length, bsz, lm_student, lm_teacher,
    chunk_tokens, top_k,
) -> torch.Tensor:
    parts = []
    work = _work_dtype(flat_s)
    for start in range(0, int(sup_idx.numel()), chunk_tokens):
        idx = sup_idx[start : start + chunk_tokens]
        logp_t = F.log_softmax(lm_teacher(flat_t[idx]).to(work), dim=-1)
        logp_s = F.log_softmax(lm_student(flat_s[idx]).to(work), dim=-1)
        if top_k is not None:
            k = min(top_k, logp_t.shape[-1])
            _, order = torch.topk(logp_t.detach(), k=k, dim=-1)
            logp_t = torch.gather(logp_t, -1, order)
            logp_s = torch.gather(logp_s, -1, order)
        finite = torch.isfinite(logp_t)
        parts.append(
            torch.where(
                finite, logp_t.exp() * (logp_t - logp_s),
                torch.zeros_like(logp_t),
            ).sum(dim=-1)
        )
    kl_all = torch.cat(parts) if parts else torch.zeros(0)
    rows = sup_idx // length
    if kl_all.numel() == 0:
        per_row = torch.zeros(bsz)
    else:
        # index_add (pas d'assignation élément par élément) : préserve le
        # dtype (float64 en tests) et le graphe. L'assignation in-place dans
        # un accumulateur float32 tronquerait le gradient (bogue M1-C).
        dev = kl_all.device
        per_row = torch.zeros(bsz, dtype=kl_all.dtype, device=dev).index_add_(
            0, rows, kl_all
        )
    counts = torch.bincount(rows, minlength=bsz).clamp_min(1).to(per_row.dtype)
    return (per_row / counts).mean()
