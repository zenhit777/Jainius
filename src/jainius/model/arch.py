"""JAI-ARCH — modèle deux-flux de référence (M1-B).

Primitives causales (fidèles FLARE par construction, convention ``current``) :
``prefill`` (écrit l'état), ``denoise`` (lit une copie, n'écrit jamais),
``commit`` (forward causal d'un bloc finalisé, écrit l'état).

``forward_two_stream`` (chemin entraînement) : un forward concaténé
``[x0 | xt]`` avec le masque exact ``JAI-MASK`` appliqué aux couches
softmax (4D additif, accepté par la pile HF — sonde M1-B). **Divergence
documentée (H-02)** : les couches GDN traitent le flux concaténé de façon
causale ; elles ne réalisent PAS la lecture bidirectionnelle intra-bloc
depuis l'état de fin de bloc (noyaux FLARE, oracle T-GDN-03). Ce chemin
sert aux formes, au câblage et aux métadonnées — pas à la parité numérique.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import torch

from jainius.model.masks import build_two_stream_mask
from jainius.model.state import cache_fingerprint, snapshot_cache


class ConventionMismatch(ValueError):
    """Entrées incompatibles avec la convention de logits active."""


class StateLeakError(RuntimeError):
    """Un forward sans écriture a muté l'état récurrent."""


class LegacyLeakyError(RuntimeError):
    """Le chemin historique `legacy_leaky` (flux concaténé, fuite GDN H-02)
    est refusé en entraînement (P1). Utiliser `TwoStreamRunner.forward_train`
    (graphe autograd, sans fuite). Voie de caractérisation uniquement :
    `legacy_leaky_forward` / `forward_two_stream(allow_legacy_leaky=True)."""


@dataclass(frozen=True)
class DiffusionSpec:
    block_size: int = 4
    logit_convention: Literal["current", "shifted"] = "current"
    mask_token_id: int = 248077
    pad_token_id: int = 248044
    causal_x0: bool = True
    causal_xt: bool = False

    @property
    def unit_block(self) -> int:
        return self.block_size - 1 if self.logit_convention == "shifted" else self.block_size


@dataclass
class TwoStreamBatch:
    x0: torch.Tensor  # [B, L] séquence propre (prompt + réponse)
    xt: torch.Tensor  # [B, L] x0 avec mask_token_id aux positions masquées
    sup_mask: torch.Tensor  # [B, L] positions supervisées (réponse masquée)
    doc_ids: torch.Tensor | None = None  # [B, 2L] ou None
    prompt_len: torch.Tensor | None = None  # [B]


@dataclass
class TwoStreamHidden:
    h_x0: torch.Tensor  # [B, L, D]
    h_xt: torch.Tensor  # [B, L, D]


class JainiusModel:
    """Enveloppe causale + deux-flux autour du checkpoint HF (référence)."""

    def __init__(self, hf_model, spec: DiffusionSpec | None = None):
        self.model = hf_model
        self.spec = spec or DiffusionSpec()
        if self.spec.logit_convention != "current":
            raise ConventionMismatch(
                "M1-B : seul `current` est implémenté ; `shifted` requiert E1."
            )

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, past=None):
        """Forward causal ; ÉCRIT l'état. Retourne (logits, cache)."""
        out = self.model(ids, past_key_values=past, use_cache=True)
        return out.logits, out.past_key_values

    @torch.no_grad()
    def denoise(self, block_ids: torch.Tensor, cache) -> torch.Tensor:
        """Forward de débruitage ; LIT une copie de l'état, ne l'écrit jamais.

        Lève ``StateLeakError`` si le cache d'entrée a été muté.
        Garantie par construction (copie profonde isolée), vérifiée par
        empreinte avant/après.
        """
        fp_before = cache_fingerprint(cache)
        probe = snapshot_cache(cache)
        out = self.model(block_ids, past_key_values=probe, use_cache=True)
        if cache_fingerprint(cache) != fp_before:
            raise StateLeakError("le forward de débruitage a écrit l'état")
        logits = out.logits
        del out, probe
        return logits

    @torch.no_grad()
    def commit(self, block_ids: torch.Tensor, cache):
        """Forward causal du bloc finalisé ; ÉCRIT l'état. Retourne (logits, cache)."""
        return self.prefill(block_ids, past=cache)

    def forward_two_stream(self, batch: TwoStreamBatch,
                           *, allow_legacy_leaky: bool = False) -> TwoStreamHidden:
        """CHEMIN HISTORIQUE `legacy_leaky` (P1) — REFUSÉ par défaut.

        Flux concaténé ``[x0 | xt]`` avec masque JAI-MASK (attention).
        H-02 : les couches GDN restent causales sur le flux concaténé
        (divergence documentée vs lecture de fin de bloc FLARE) — FUITE
        du futur x0 vers xt, caractérisée par `test_two_stream_gdn_future_leak`.

        P1 : l'entraînement DOIT utiliser `TwoStreamRunner.forward_train`.
        Cet alias lève `LegacyLeakyError` sauf `allow_legacy_leaky=True`
        (voie de caractérisation/diagnostic isolée). Le test
        `test_legacy_leaky_refused_by_default` verrouille ce refus.
        """
        if not allow_legacy_leaky:
            raise LegacyLeakyError(
                "forward_two_stream est le chemin historique `legacy_leaky` "
                "(fuite H-02) — refusé en entraînement ; utiliser "
                "TwoStreamRunner.forward_train (P1). Caractérisation : "
                "forward_two_stream(batch, allow_legacy_leaky=True).")
        return self.legacy_leaky_forward(batch)

    def legacy_leaky_forward(self, batch: TwoStreamBatch) -> TwoStreamHidden:
        """Implémentation historique du flux concaténé (caractérisation)."""
        if self.spec.logit_convention != "current":
            raise ConventionMismatch("forward_two_stream exige `current` (M1-B)")
        x0, xt = batch.x0, batch.xt
        if x0.shape != xt.shape or x0.dim() != 2:
            raise ValueError(f"formes x0/xt incohérentes : {x0.shape} vs {xt.shape}")
        bsz, length = x0.shape
        concat = torch.cat([x0, xt], dim=1)  # [B, 2L]
        dtype = self.model.dtype  # SDPA exige le masque au dtype des requêtes
        neg = torch.finfo(dtype).min
        masks = []
        for b in range(bsz):
            doc = batch.doc_ids[b] if batch.doc_ids is not None else None
            m = build_two_stream_mask(
                length,
                self.spec.unit_block,
                causal_x0=self.spec.causal_x0,
                causal_xt=self.spec.causal_xt,
                doc_ids=doc,
            )
            masks.append(m)
        allow = torch.stack(masks)  # [B, 2L, 2L] booléen
        additive = torch.zeros(allow.shape, dtype=dtype, device=concat.device)
        additive[~allow] = torch.finfo(dtype).min
        additive = additive[:, None, :, :]  # [B, 1, 2L, 2L]
        # Positions répétées : xt reçoit les mêmes ids que x0 (cahier §10.2 ;
        # OPDLM `tok_idx_ext = cat([tok_idx, tok_idx_resp])`). Sans cela, le
        # forward étudiant diverge de la référence (test_target_alignment).
        pos = torch.arange(length, device=concat.device).unsqueeze(0)
        pos_ext = torch.cat([pos, pos], dim=1).expand(bsz, -1)
        out = self.model(
            concat, attention_mask=additive, position_ids=pos_ext,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        return TwoStreamHidden(h_x0=hidden[:, :length], h_xt=hidden[:, length:])

    def model_version(self) -> str:
        """Version courte du checkpoint (traçabilité des trajectoires)."""
        import transformers

        payload = (
            self.model.config.to_json_string()
            + str(self.model.dtype)
            + transformers.__version__
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
