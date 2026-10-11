"""JAI-LOSS — KL enseignant→étudiant masquée (M1-C, pré-pilote).

Référence : OPDLM ``rl_sdar.py:1366-1376`` (KL directe, terme NaN-safe :
les entrées ``-inf`` de l'enseignant contribuent 0) et normalisation
``:1427-1449`` (somme par ligne / max(1, n positions), puis moyenne batch).
Vocabulaire complet par morceaux ; ``top_k`` = repli mémoire type
``top_k_logits=16`` (sans renormalisation, comme l'usage creux OPDLM).

Deux voies (même mathématique, cf. test d'égalité) :
- `kl_teacher_to_student` : perte scalaire différentiable (petite échelle,
  tests). UNE SEULE boucle : têtes calculées une fois, métriques
  détachées + graphe dans la même passe (plus de double `lm_head`, plus
  de boucle Python par token — `index_add_` vectorisé).
- `kl_teacher_to_student_backward` : backward à mémoire bornée (grande
  échelle) : forwards enseignant (sans grad) et étudiant par morceaux,
  `backward()` immédiat par morceau. Pic ≈ 1 morceau au lieu de toute la
  séquence (mesuré : ~10,4 Mo/token supervisé en voie simple, soit
  ~340 Go à L=4096×8 lignes — OOM certain sans bornage). Retourne la
  perte DÉTACHÉE (backward déjà appliqué — ne pas rappeler `.backward()`).
  Égalité mathématique par linéarité ; arrondi d'accumulation d'ordre
  différent → test `allclose`, pas bit-exact.

`teacher_logits` (optionnel) : logits enseignant précalculés
(`Teacher.teacher_logits_for_loss`, roll inclus) — évite de recalculer
la tête enseignante et verrouille la cible unique B1.
Objectif OPDLM uniquement (CdC §3.2).
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
    loss: torch.Tensor  # scalaire (détachée si backward appliqué, voir fn)
    per_row: torch.Tensor  # [B]
    n_supervised: int
    kl_mean: float


def _apply_topk(logp_t, logp_s, top_k, detach_order=False):
    if top_k is None:
        return logp_t, logp_s
    if top_k <= 0:
        raise ValueError("top_k doit être > 0 ou None")
    k = min(top_k, logp_t.shape[-1])
    order_in = logp_t.detach() if detach_order else logp_t
    _, order = torch.topk(order_in, k=k, dim=-1)
    return torch.gather(logp_t, -1, order), torch.gather(logp_s, -1, order)


def _kl_terms(logp_t, logp_s):
    finite = torch.isfinite(logp_t)
    return torch.where(
        finite,
        logp_t.exp() * (logp_t - logp_s),
        torch.zeros_like(logp_t),
    ).sum(dim=-1)


def _resolve_teacher_chunk(lm_teacher, flat_t, idx, work, teacher_logits,
                           top_k):
    """Log-probs enseignant du morceau (précalculées ou tête, sans grad :
    contrat enseignant gelé — aucun gradient côté enseignant)."""
    if teacher_logits is not None:
        logp_t = teacher_logits[idx].to(work)
    else:
        with torch.no_grad():
            logp_t = F.log_softmax(lm_teacher(flat_t[idx]).to(work), dim=-1)
    return logp_t


def kl_teacher_to_student(
    h_student: torch.Tensor,
    lm_student: Callable[[torch.Tensor], torch.Tensor],
    h_teacher: torch.Tensor | None,
    lm_teacher: Callable[[torch.Tensor], torch.Tensor] | None,
    sup_mask: torch.Tensor,
    *,
    chunk_tokens: int = 1024,
    top_k: int | None = None,
    teacher_logits: torch.Tensor | None = None,
) -> LossOutput:
    """KL(p_T ‖ p_S) masquée, normalisée par ligne puis par batch.

    `teacher_logits` : logits enseignant FULL [B*L, V] précalculés
    (`Teacher.teacher_logits_for_loss`, roll B1 inclus). Quand fournis,
    `h_teacher`/`lm_teacher` sont ignorés pour le calcul (seule la forme
    de `h_student` est validée). Sinon `lm_teacher(h_teacher)`.
    """
    if h_student.dim() != 3:
        raise ValueError("h_student doit être [B, L, D]")
    if sup_mask.shape != h_student.shape[:2] or sup_mask.dtype != torch.bool:
        raise ValueError("sup_mask doit être booléen [B, L]")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens doit être > 0")
    if teacher_logits is None:
        if h_teacher is None or lm_teacher is None:
            raise ValueError("h_teacher/lm_teacher requis sans teacher_logits")
        if h_student.shape != h_teacher.shape:
            raise ValueError("formes caché étudiant/enseignant incohérentes")
        flat_t = h_teacher.reshape(-1, h_teacher.shape[-1])
    else:
        flat_t = None
        n_flat, v_dim = teacher_logits.shape
        if (n_flat, v_dim) != (h_student.shape[0] * h_student.shape[1],
                               teacher_logits.shape[-1]):
            raise ValueError("teacher_logits doit avoir la forme [B*L, V]")
    bsz, length, _ = h_student.shape
    dev = h_student.device
    work = _work_dtype(h_student)
    flat_s = h_student.reshape(-1, h_student.shape[-1])
    sup_idx = torch.where(sup_mask.reshape(-1))[0]
    n_supervised = int(sup_idx.numel())
    rows_all = sup_idx // length
    row_counts = torch.bincount(rows_all, minlength=bsz).clamp_min(1)
    # UNE SEULE boucle : têtes calculées une fois par côté (enseignant
    # sans grad — contrat : enseignant gelé), métriques détachées +
    # morceaux de graphe collectés pour la perte.
    per_row = torch.zeros(bsz, dtype=work, device=dev)
    graph_parts, graph_rows = [], []
    for start in range(0, n_supervised, chunk_tokens):
        idx = sup_idx[start : start + chunk_tokens]
        logp_t = _resolve_teacher_chunk(
            lm_teacher, flat_t, idx, work, teacher_logits, top_k)
        logp_s = F.log_softmax(lm_student(flat_s[idx]).to(work), dim=-1)
        if top_k is not None:
            logp_t, logp_s = _apply_topk(logp_t, logp_s, top_k,
                                         detach_order=True)
        kl = _kl_terms(logp_t, logp_s)
        graph_parts.append(kl)
        graph_rows.append(idx // length)
        with torch.no_grad():
            per_row.index_add_(0, idx // length, kl.detach())
    per_row = per_row / row_counts.to(work)
    if graph_parts:
        kl_all = torch.cat(graph_parts)
        rows = torch.cat(graph_rows)
        loss = (torch.zeros(bsz, dtype=kl_all.dtype, device=dev)
                .index_add_(0, rows, kl_all)
                / row_counts.to(kl_all.dtype)).mean()
    else:
        loss = torch.zeros((), device=dev)
    return LossOutput(
        loss=loss,
        per_row=per_row.detach(),
        n_supervised=n_supervised,
        kl_mean=float(per_row.mean().item()),
    )


def kl_teacher_to_student_backward(
    h_student: torch.Tensor,
    lm_student: Callable[[torch.Tensor], torch.Tensor],
    h_teacher: torch.Tensor | None,
    lm_teacher: Callable[[torch.Tensor], torch.Tensor] | None,
    sup_mask: torch.Tensor,
    *,
    chunk_tokens: int = 512,
    top_k: int | None = None,
    teacher_logits: torch.Tensor | None = None,
) -> LossOutput:
    """Backward à mémoire bornée (pic ≈ 1 morceau). Même mathématique que
    `kl_teacher_to_student` (test d'égalité) ; `loss` retournée DÉTACHÉE."""
    if h_student.dim() != 3:
        raise ValueError("h_student doit être [B, L, D]")
    if sup_mask.shape != h_student.shape[:2] or sup_mask.dtype != torch.bool:
        raise ValueError("sup_mask doit être booléen [B, L]")
    if teacher_logits is None and (h_teacher is None or lm_teacher is None):
        raise ValueError("h_teacher/lm_teacher requis sans teacher_logits")
    bsz, length, _ = h_student.shape
    dev = h_student.device
    work = _work_dtype(h_student)
    if not h_student.requires_grad:
        raise ValueError("h_student doit requérir le gradient (backward)")
    flat_s = h_student.reshape(-1, h_student.shape[-1])
    flat_t = (h_teacher.reshape(-1, h_teacher.shape[-1])
              if h_teacher is not None else None)
    sup_idx = torch.where(sup_mask.reshape(-1))[0]
    n_supervised = int(sup_idx.numel())
    rows_all = sup_idx // length
    row_counts = torch.bincount(rows_all, minlength=bsz).clamp_min(1)
    denom = (row_counts.to(work) * bsz)
    per_row = torch.zeros(bsz, dtype=work, device=dev)
    for start in range(0, n_supervised, chunk_tokens):
        idx = sup_idx[start : start + chunk_tokens]
        rows = rows_all[start : start + chunk_tokens]
        logp_t = _resolve_teacher_chunk(
            lm_teacher, flat_t, idx, work, teacher_logits, top_k)
        logp_s = F.log_softmax(lm_student(flat_s[idx]).to(work), dim=-1)
        if top_k is not None:
            logp_t, logp_s = _apply_topk(logp_t, logp_s, top_k,
                                         detach_order=True)
        kl = _kl_terms(logp_t, logp_s)
        (kl / denom[rows]).sum().backward()  # graphe libéré aussitôt
        with torch.no_grad():
            per_row.index_add_(0, rows, kl.detach())
    per_row = per_row / row_counts.to(work)
    loss_val = per_row.mean()
    return LossOutput(
        loss=loss_val.detach(),
        per_row=per_row.detach(),
        n_supervised=n_supervised,
        kl_mean=float(per_row.mean().item()),
    )
