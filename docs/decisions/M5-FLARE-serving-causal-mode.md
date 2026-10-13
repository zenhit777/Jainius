# Prérequis M5 — quel `causal_mode` le serving FLARE emploie-t-il à l'inférence ?

**Statut** : FAIT ÉTABLI (lecture seule du checkout FLARE `6ca547a`,
`_artefacts/oracles/FLARE` — aucun code copié dans `src/`, LIC-1 respecté).
**Date** : 2026-10-13 (clôture A5).
**Conséquence** : PRÉREQUIS du jalon M5 (serving). Bloque le transfert
entraînement → serving tant qu'il n'est pas levé.

## 1. Faits établis

### 1.1 Chemin d'entraînement FLARE (référence de T-GDN-03)

- `torchtitan/.../block_gated_delta_rule/chunk.py` :
  `ChunkBlockCausalGatedDeltaRuleFunction`, `causal_mode=0` = lecture
  de fin de bloc (toutes les positions du bloc lisent S̃_end).
- C'est la sémantique qu'implémente Jainius (`denoise` lit l'état de
  fin de bloc) — T-GDN-03 PASS (rel 2.23e-3 ≤ 5e-3, abs 1.33e-4).

### 1.2 Anomalie du helper torchtitan (rappel, déjà constatée en M1)

- `torchtitan/.../fused_recurrent.py` :
  `fused_recurrent_block_causal_gated_delta_rule(causal_mode=0)` NE
  reproduit PAS la lecture de fin de bloc (écart O(1) ~0.2–0.3),
  alors que son état final est exact (6e-8) et que son
  `causal_mode=1` est exact (2e-8).
- Anomalie côté oracle (helper entraînement/inférence), SANS EFFET
  sur T-GDN-03 qui fait foi sur le chemin chunk (§1.1).

### 1.3 Chemin de serving FLARE (`eval/sglang/`)

Conventions du backend (`srt/layers/attention/linear/gdn_dllm_backend.py:10`) :

- `0` = block-causal, `1` = token-causal, `2` = mixed ;
- défaut si le flag est absent : `1` (token-causal).

Le kernel de serving (`srt/layers/attention/block_gdn/fused_recurrent.py:543-545`)
documente :

- `causal_mode=0` : toutes les positions du bloc voient `h[block_end]`
  (= lecture fin de bloc, sémantique Jainius) ;
- `causal_mode=1` : chaque position voit `h[position]` (token-causal) ;
- `causal_mode=2` : positions `0..num_clean-1` en token-causal,
  positions `num_clean..block_end` en `h[block_end]` (mixte).

Modes effectivement posés par les algorithmes de serving :

| Algorithme | `dllm_gdn_causal_mode` posé |
|---|---|
| `LowConfidence`, `JointThreshold` | `int(causal_xt)`, défaut `causal_xt=False` → **0 (fin de bloc)** |
| `LowConfidenceShiftHybridDiffusion` (denoise) | **2 (mixte, `num_clean=1`)** |
| `LowConfidenceShiftHybridDiffusion` (commit) | **1 (token-causal)** |
| `HybridDiffusionSelfSpec` (variantes `BD_BIDIR*`) | **2 (mixte)** |
| `HybridDiffusionSelfSpec` (autres variantes) | **1 (token-causal)** |

Chemin de serving PAR DÉFAUT pour Qwen3.5 diffusion (`eval/scripts/serve.sh`,
`MODE=diffusion` → algorithme `LowConfidenceShiftHybridDiffusion`, config
`eval/configs/hybrid_diffusion_shift_b3_g1.yaml`, sans clé `causal_xt`) :
**denoise en mode 2 puis commit en mode 1 — jamais le mode 0.**

## 2. Réponse à la question A5

- La sémantique « lecture de fin de bloc » de Jainius est celle du
  chemin d'ENTRAÎNEMENT FLARE (chunk `causal_mode=0`) et celle des
  algorithmes `LowConfidence` / `JointThreshold` avec `causal_xt=False`.
- Elle N'EST PAS celle qu'applique le serving diffusion par défaut
  (`LowConfidenceShiftHybridDiffusion` : mixte puis token-causal).
- Le mode 0 existe et est documenté dans le kernel de serving, mais
  aucun algorithme de serving livré ne l'emploie sur le chemin
  diffusion Qwen3.5 par défaut.

## 3. Prérequis M5 (bloquant pour le transfert entraînement → serving)

Avant d'exporter/servir un checkpoint Jainius, démontrer au choix :

- (a) que le checkpoint entraîné en lecture fin de bloc produit les
  mêmes sorties (à tolérance explicite) sous le mode de serving retenu
  (2-puis-1), sur un jeu de prompts fixé ; ou
- (b) que le serving est configuré en mode 0 (fin de bloc) de bout en
  bout, avec mesure de non-régression vs le chemin d'entraînement ; ou
- (c) caractériser l'écart entre modes et en tenir compte dans
  l'évaluation (biais documenté, non silencieux).

Tant que ce point n'est pas tranché, aucune conclusion de qualité
mesurée côté entraînement (B1) ne peut être transférée telle quelle
au serving, et réciproquement aucune mesure serving ne valide le
chemin d'entraînement. Ce prérequis s'ajoute à H-11 (chargement du
checkpoint `smoke` par le fork).
