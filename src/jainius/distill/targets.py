"""JAI-TEACH — calcul unique des cibles enseignant (A1, B1 non miroir).

Fonction unique de référence pour les cibles consommées par `train_step`
ET par le test de régression. Si `train_step` redevient non décalé, ce
test échoue : il compare la sortie de CETTE fonction avec la référence
indépendante `logprobs_simple`.

Convention `current` (OPDLM `rl_sdar.py:1344-1348`, `student_arm_shift=False`) :
la position supervisée *i* porte la distribution qui prédit le jeton *i*.
Le head AR brut prédit le jeton *i+1* à la position *i*, d'où le décalage
`roll(1)` — bit-exact vs décalage des logits (tête linéaire par position).

Hors périmètre : forward enseignant, gestion du cache, clean_fill.
Seul le calcul de cibles alignées `current`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_teacher_targets(
    x0: torch.Tensor,
    hidden_fn: torch.Callable[[torch.Tensor], torch.Tensor],
    lm_head: torch.Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Cibles enseignant alignées `current` pour la perte KL.

    Args:
        x0 : séquence propre [B, L].
        hidden_fn : callable(input_ids) → hidden [B, L, D] (AR causal).
            Doit être gelé / eval() de l'extérieur.
        lm_head : callable(hidden) → logits [B, L, V].

    Returns:
        log_softmax(logits.roll(1)) float32 [B, L, V].
        C'est la distribution cible de l'enseignant à chaque position
        supervisée (prédit le jeton *i* depuis les logits à *i*).
    """
    # Forward AR gelé sur x0 seul (H-03 : équivalent clean_fill).
    hidden = hidden_fn(x0)  # [B, L, D] — dernier hidden causal de x0
    logits = lm_head(hidden)  # [B, L, V] — head AR brut prédit i+1 à position i
    return F.log_softmax(logits.roll(dims=1, shifts=1).float(), dim=-1)
