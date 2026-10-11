"""Poids maîtres FP32 + AdamW (B4.3, JAI-TRAIN).

Constat (audit externe, reproduit) : AdamW direct sur paramètres bf16
purs à `lr=1e-5` ne modifie aucun paramètre dans le cas étudié
(ULP bf16 ~4e-3 à magnitude 1 ≫ update typique 1e-5). Les mises à jour
doivent s'accumuler en FP32 puis être recopiées (copie maître, §16.4).

`MasterOptimizer` : maîtres FP32 persistants + AdamW dessus ; les grads
du modèle (bf16) sont copiés en FP32 avant le pas, puis les maîtres sont
recopiés vers le modèle. Déterministe à entrées identiques (même ordre,
pas de RNG). États sauvegardables (maîtres + optimiseur) pour la reprise.
"""

from __future__ import annotations

import torch


class MasterOptimizer:
    """AdamW sur copies maîtres FP32 d'un modèle bf16 (API step/zero_grad)."""

    def __init__(self, model, lr: float = 1e-5, **adamw_kwargs):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.masters = [p.detach().float().clone().requires_grad_(True)
                        for p in self.params]
        self.optim = torch.optim.AdamW(self.masters, lr=lr, **adamw_kwargs)

    def zero_grad(self) -> None:
        self.optim.zero_grad()
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self, model=None) -> None:
        _ = model  # compatibilité d'appel avec train_step (non utilisé)
        for p, m in zip(self.params, self.masters):
            if p.grad is not None:
                if m.grad is None:
                    m.grad = p.grad.detach().float().clone()
                else:
                    m.grad.detach().copy_(p.grad.detach().float())
        self.optim.step()
        with torch.no_grad():
            for p, m in zip(self.params, self.masters):
                p.copy_(m.detach().to(p.dtype))
