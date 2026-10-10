"""JAI-MASK — masque softmax deux-flux ``[x0 | xt]`` (reference M1-A).

Convention (cahier technique §10.2) : sequence ``x0`` (propre) de longueur ``L``
suivie de ``xt`` (bruitee) de longueur ``L``. Indices ``[0, L)`` = x0,
``[L, 2L)`` = xt. ``blk = pos // unit_block`` avec ``pos`` la position
intra-flux. ``doc_ids`` optionnels (``None`` = un seul document).

Formule exacte (FLARE ``_block_diff_mask_x0_xt``) ::

    x0_mask    = (q∈x0) & (kv∈x0) & (q_pos >= kv_pos si causal_x0
                                     sinon q_blk >= kv_blk)
    xt_mask    = (q∈xt) & (kv∈xt) & (q_blk == kv_blk)
                 [& q_pos >= kv_pos si causal_xt]
    cross_mask = (q∈xt) & (kv∈x0) & (q_blk > kv_blk)
    mask       = (x0_mask | xt_mask | cross_mask) & (doc_ids[q] == doc_ids[kv])

Jainius fixe ``causal_x0=True``, ``causal_xt=False``. Les couches GDN ne sont
pas gouvernées par ce masque (planification d'états, JAI-STATE / H-02).
"""

from __future__ import annotations

import torch


def build_two_stream_mask(
    seq_len: int,
    unit_block: int,
    *,
    causal_x0: bool = True,
    causal_xt: bool = False,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construit le masque booléen ``[2L, 2L]`` (True = visible)."""
    if seq_len <= 0:
        raise ValueError(f"seq_len doit être > 0, reçu {seq_len}")
    if unit_block <= 0:
        raise ValueError(f"unit_block doit être > 0, reçu {unit_block}")
    if doc_ids is not None:
        if doc_ids.shape != (2 * seq_len,):
            raise ValueError(
                f"doc_ids doit avoir la forme (2L,)={(2 * seq_len,)}, "
                f"reçu {tuple(doc_ids.shape)}"
            )

    n = seq_len
    pos = torch.arange(2 * n)
    in_x0 = pos < n
    pos_in_stream = torch.where(in_x0, pos, pos - n)
    blk = pos_in_stream // unit_block

    q_in_x0 = in_x0[:, None]
    kv_in_x0 = in_x0[None, :]
    q_pos = pos_in_stream[:, None]
    kv_pos = pos_in_stream[None, :]
    q_blk = blk[:, None]
    kv_blk = blk[None, :]

    if causal_x0:
        x0_ok = q_pos >= kv_pos
    else:
        x0_ok = q_blk >= kv_blk
    x0_mask = q_in_x0 & kv_in_x0 & x0_ok

    xt_mask = (~q_in_x0) & (~kv_in_x0) & (q_blk == kv_blk)
    if causal_xt:
        xt_mask = xt_mask & (q_pos >= kv_pos)

    cross_mask = (~q_in_x0) & kv_in_x0 & (q_blk > kv_blk)

    mask = x0_mask | xt_mask | cross_mask
    if doc_ids is not None:
        mask = mask & (doc_ids[:, None] == doc_ids[None, :])
    return mask
