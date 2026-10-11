# B1 — convention d'alignement `current` (correctif pré-M2)

Référence : OPDLM `rl_sdar.py:1266-1272` (étudiant, `student_arm_shift=False`
par défaut, `rl_bd3lm.yaml:108`), `:1344-1348` (enseignant + `roll(1)`),
`:1425-1428` (positions supervisées), `:1428-1449` (normalisation) ;
cahier technique §10.1, table des conventions, ligne `current`
(« le logit de l'enseignant est décalé d'un cran (`roll`) »).

## Règle

- Position supervisée *i* : distribution qui prédit le jeton *i*.
- Étudiant : PAS de décalage (ses logits portent déjà le jeton *i* par
  construction de la perte ; le réalignement depuis un prior AR est
  l'objet même de la conversion).
- Enseignant : `roll(1)` OBLIGATOIRE (sa tête AR prédit le jeton *i+1*
  à la position *i*). Implémenté par `Teacher.hidden_for_loss`
  (roll du hidden pré-tête, bit-exact vs roll des logits, tête linéaire
  par position). Position 0 = déchet recyclé, jamais supervisée.
- Inférence (`rollout`) : jeton *i* ∼ `logits[i]` — cohérent avec
  l'entraînement uniquement si la règle ci-dessus est respectée.

## Antécédent (B1)

Version `8227424` : `train_step` consommait `teacher.hidden` sans roll
(cibles « jeton suivant », écart max ~25 mesuré) ; le test de
sensibilité de l'époque le validait par auto-similarité du prior AR.
Corrigé + verrouillé par `test_train_step_targets_match_reference`
(échoue sur l'ancien chemin) et `test_alignment_shift_trap_documented`.
