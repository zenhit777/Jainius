"""JAI-TEACH — enseignant autorégressif gelé (M1-C, micro-tests).

Voie simple (H-03) : forward AR ordinaire sur ``x0`` (prompt + réponse),
``roll(1)`` pour l'alignement next-token, ``log_softmax`` float32.
C'est le chemin « x0 only » commenté du dépôt OPDLM
(``rl_sdar.py:1328-1335``) : ``logits.roll(1)`` puis ``log_softmax``.

``teacher_logits_extended`` porte fidèlement la construction
``clean_fill`` (``rl_sdar.py:1297-1348`` : remplissage, masque de base par
blocs, double tril causal, slice, remplacement aux frontières de blocs,
``roll(1)``) pour T-TEACH-02. Sans pads, avec positions répétées
(``tok_idx_resp``) et masque de base exact, elle est mathématiquement
identique à la voie simple : base+tril se réduisent à un préfixe causal
de x0 (blocs passés + intra-bloc causal). Un tril seul, sans la base,
fuirait le futur de x0 vers xt.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def basic_block_attention(
    l0: int, l1: int, block_size: int, device
) -> torch.Tensor:
    """Port de ``make_basic_block_attention`` (OPDLM ``rl_sdar.py:513-557``).

    Retourne le masque booléen [1,1,N,N] (True = visible), N = L0+2L1.
    Sans ce masque de base, la construction enseignante fuirait le futur
    de x0 vers xt (contexte causal non garanti) — cf. T-TEACH-02.
    """
    n = l0 + 2 * l1
    bias = torch.zeros(1, 1, n, n, dtype=torch.bool, device=device)
    rows = torch.arange(l0 + l1, l0 + 2 * l1)
    rows_token = torch.arange(l0, l0 + l1)
    n_blocks = (l1 + block_size - 1) // block_size
    for bi in range(n_blocks):
        left_end = l0 + min(bi * block_size, l1)
        right_start = l0 + l1 + (left_end - l0)
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, l1)
        block_rows = rows[i_start:i_end]
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = True
        bias[:, :, block_rows.unsqueeze(-1), right_start : (right_start + block_size)] = True
        block_rows = rows_token[i_start:i_end]
        left_end = l0 + min((bi + 1) * block_size, l1)
        bias[:, :, block_rows.unsqueeze(-1), 0:left_end] = True
    if l0 > 0:
        num_blocks_pre = (l0 + block_size - 1) // block_size
        for bi in range(num_blocks_pre):
            row_end = max(l0 - bi * block_size, 0)
            row_start = max(l0 - (bi + 1) * block_size, 0)
            if row_end > row_start:
                block_rows = torch.arange(row_start, row_end)
                bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = True
    return bias


class Teacher:
    """Enseignant gelé : ``requires_grad=False``, ``eval()`` (T-TEACH-01)."""

    def __init__(self, hf_model):
        self.model = hf_model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.model.parameters())

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward AR causal ordinaire (sans gradient)."""
        self.model.eval()
        return self.model(input_ids).logits

    @torch.no_grad()
    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Dernier hidden AR causal (sans gradient) — cible JAI-LOSS."""
        self.model.eval()
        return self.model(input_ids, output_hidden_states=True).hidden_states[-1]

    @torch.no_grad()
    def hidden_for_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Hidden enseignant aligné `current`, consommé par la perte KL.

        Convention (OPDLM `rl_sdar.py:1344-1348`, `student_arm_shift=False`) :
        la position supervisée *i* porte la distribution qui prédit le
        jeton *i*. Le head AR brut prédit le jeton *i+1* à la position *i*,
        d'où le décalage `roll(1)` — bit-exact vs décalage des logits
        (tête linéaire par position). Début de séquence : la position 0
        reçoit le dernier hidden (déchet) ; elle n'est jamais supervisée
        car `sup_mask` ne couvre que la réponse (prompt non vide).
        Contexte : préfixe causal de la séquence terminale (H-03).
        """
        return self.hidden(x0).roll(1, 1)

    @torch.no_grad()
    def teacher_logits_for_loss(self, lm_head, x0: torch.Tensor) -> torch.Tensor:
        """Logits enseignant FULL [B, L, V] alignés `current`, consommés par
        la perte (`train_step` ET test de régression appellent CETTE
        fonction — tout décalage divergent échoue au test). Équivaut à
        `logprobs_simple` avant log_softmax (mêmes ops, même ordre)."""
        self.model.eval()
        return lm_head(self.hidden_for_loss(x0))

    @torch.no_grad()
    def logprobs_simple(
        self, x0: torch.Tensor, block_size: int = 4
    ) -> torch.Tensor:
        """Cibles voie simple : AR sur x0 puis ``roll(1)`` (``current``)."""
        _ = block_size  # pas de décalage sous `current` (T-TEACH-02 le prouve)
        logits = self.logits(x0)
        return F.log_softmax(logits.roll(dims=1, shifts=1).float(), dim=-1)

    @torch.no_grad()
    def logprobs_extended(
        self,
        prompt: torch.Tensor,
        x0_resp: torch.Tensor,
        xt: torch.Tensor,
        mask_token_id: int,
        block_size: int = 4,
    ) -> torch.Tensor:
        """Port fidèle de ``clean_fill`` (OPDLM ``rl_sdar.py:1297-1348``).

        ``prompt`` [B,L0], ``x0_resp``/``xt`` [B,L1]. Retour [B,L0+L1,V].
        Hypothèses du test : aucun pad, positions répétées
        (``tok_idx_resp``), ``teacher_fill="clean_fill"``,
        ``student_arm_shift=False``.
        """
        bsz, l0 = prompt.shape
        l1 = x0_resp.shape[1]
        assert xt.shape == (bsz, l1)
        device = prompt.device
        extended = torch.cat([prompt, x0_resp, xt], dim=1)  # [B,L0+2L1]
        # clean_fill : MASK de xt <- x0 (rl_sdar.py:1307-1310).
        x0_repeated = torch.cat([extended[:, : l0 + l1], extended[:, l0 : l0 + l1]], dim=1)
        sampled = extended.clone()
        is_mask = sampled == mask_token_id
        sampled[is_mask] = x0_repeated[is_mask]
        # Double tril causal (rl_sdar.py:1315-1323) combiné au masque de
        # base par ET (la base seule fuirait le futur de x0 vers xt).
        total = l0 + 2 * l1
        causal = basic_block_attention(l0, l1, block_size, device)
        win = torch.ones(1, 1, l0 + l1, l0 + l1, dtype=torch.bool, device=device)
        win = torch.tril(win, diagonal=0)
        causal[:, :, : l0 + l1, : l0 + l1] &= win
        xtw = torch.ones(1, 1, l1, l1, dtype=torch.bool, device=device)
        causal[:, :, l0 + l1 :, l0 + l1 :] &= torch.tril(xtw, diagonal=0)
        mdtype = self.model.dtype  # SDPA : masque au dtype des requêtes
        additive = torch.full(
            causal.shape, torch.finfo(mdtype).min, dtype=mdtype, device=device
        )
        additive[causal] = 0.0
        # Positions répétées (rl_sdar.py:850-855).
        pos = torch.arange(l0 + l1, device=device).unsqueeze(0).expand(bsz, -1)
        pos_ext = torch.cat([pos, pos[:, l0:]], dim=1)
        full = self.model(
            input_ids=sampled,
            attention_mask=additive,
            position_ids=pos_ext,
        ).logits
        teacher = torch.cat([full[:, :l0, :], full[:, l0 + l1 :, :]], dim=1)
        boundary = torch.arange(
            start=l0 + block_size - 1, end=l0 + l1, step=block_size
        )
        teacher[:, boundary] = full[:, boundary]
        teacher = teacher.roll(dims=1, shifts=1)
        return F.log_softmax(teacher.float(), dim=-1)
