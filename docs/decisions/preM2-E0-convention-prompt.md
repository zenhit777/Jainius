# E0 — convention prompt pour la parité T-LOSS-05

Statut : DÉCISION CONSIGNÉE (pré-M2, audit correctif A7).
Référence : `test_prompt_context_delta_characterized` (T-LOSS-05/E0),
cahier technique §10.3 (H-01/H-03).

## Fait mesuré (pré-pilote, Qwen3-0.6B fp32)

Toutes choses égales par ailleurs (mêmes trajectoires, mêmes masques,
mêmes positions), le forward étudiant OPDLM (prompt bloc-bidirectionnel,
`make_basic_block_attention`) vs le forward Jainius (prompt causal,
`causal_x0=True`) diffère de **KL moyenne ~9,5 par position supervisée**
(test `test_prompt_context_delta_characterized`) — même ordre que la
perte elle-même.

## Conséquence

La parité numérique exacte avec le code officiel (`T-LOSS-05` : mêmes
trajectoires, mêmes masques ⇒ même perte) est IMPOSSIBLE si les deux
codes gardent chacun leur convention prompt : l'écart mesuré n'est pas
un bug d'implémentation mais une différence de modèle (contexte visible
des lignes prompt).

## Options (arbitrage M2)

- (a) Run de parité à convention commune : forcer temporairement le
  prompt causal des deux côtés (ou bloc-bidi des deux côtés) pour isoler
  la parité du reste (positions, slicing, rolls, normalisation).
- (b) Tolérance E0 ≥ effet mesuré (borne test : 30.0 ; mesuré ~9.5) —
  parité « à convention près », documentée comme telle.
- (c) Adopter le prompt bloc-bidirectionnel OPDLM côté Jainius :
  REJETÉ sauf décision architecturale explicite (casse `causal_x0=True`,
  requis par AR-Trust, la perte clean et la cohérence FLARE — D-10).

## Audit correctif A7

Le test de régression B1 (`test_train_step_targets_match_reference`)
vérifie que les cibles consommées par `train_step` (via l'unique fonction
`Teacher.teacher_logits_for_loss`, voir A1) coïncident avec la référence
`logprobs_simple`. Ce test échouera si `train_step` redevient non décalé.

`causal_x0=True` reste la référence (`current`, M0-D1). Aucun changement
de convention effectué ici.
