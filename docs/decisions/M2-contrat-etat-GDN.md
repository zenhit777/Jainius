# Contrat d'état GDN pour M2 — planification des blocs bruités

Statut : PROPOSITION (pré-M2, audit correctif B2). Aucune implémentation.
Références : FLARE Éq. (6) (planification), cahier technique §10.2 (H-02),
tests `test_two_stream_gdn_future_leak` (+ variantes).

## 1. Fait établi (caractérisé, pas supposé)

Sur Qwen3.5-0.8B, le forward concaténé `[x0|xt]` de référence pollue le
flux bruité par récurrence, malgré un masque d'attention exact
(T-MASK-01) et appliqué (sonde 4D fp32 : 2.2e-5 vs causal natif) :
- le futur de la réponse propre `x0` modifie les hidden `xt` ;
- le préfixe prompt de `xt` lui-même les modifie aussi ;
- en bf16 comme en fp32, à 1 comme à 2 blocs.
Le masque ne gouverne que les couches softmax ; les 18 couches GDN
accumulent tout le flux concaténé (états récurrents + contexte conv
noyau 4) sans ré-initialisation. Tout pilote qui calculerait la perte
sur ce forward entraînerait sur des cibles partiellement copiées :
**BLOCKED pour la distillation** (pas d'interprétation des pertes du
smoke antérieur comme apprentissage).

## 2. Contrat requis (M2)

- `S_clean(s_b)` : état récurrent + contexte conv après le préfixe causal
  propre jusqu'à la frontière de bloc `s_b` (forward causal, écrit).
- Chaque bloc bruité `b` démarre de `S_clean(s_b)` (snapshot immuable,
  jamais du flux bruité) ; forwards de débruitage : lecture seule.
- Commit : forward causal du bloc finalisé, seul écrivain.
- Isolation documents/requêtes : reset par ligne (JAI-STATE, pas
  `DynamicCache.reset()` global) + remise à zéro du contexte conv.
- Portes : T-GDN-03 (parité oracle FLARE `chunk_two_stream_wy`),
  T-GDN-04 (packing), T-GDN-05 (gradient float64 noisy→clean),
  inversion des asserts de fuite ci-dessus (fuite == 0.0 bit-exact).
- Hors contrat : kernels fusionnés (M6, selon LIC-1 déjà tranchée),
  objectif FLARE (jamais fusionné avant baseline, CdC §3.2).

## 3. Non-contrats explicites

- Aucun changement de masque d'attention ne suffit (récurrence non
  gouvernée) — B2.5 de la mission respecté.
- Aucun reset/restauration inventé ici : snapshot/restore par copie
  profonde bit-exacte déjà validée (T-STATE-01) ; le reset par ligne
  reste à implémenter (zéro par ligne des 24 couches).
