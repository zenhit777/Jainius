# Audit J3 — distillation on-policy Diffusion-Trust `shifted` (2026-10-15)

Objectifs servis : G1, G3. Note : `PROMPT_SESSION_J3.md` est vide (0 octet,
non versionné) ; la mission appliquée est le critère de sortie J3 du PRD §6.

## Résultat

| Exigence J3 (PRD §6) | Réponse (mesurée) |
|---|---|
| Rollouts par débruitage de l'étudiant (Diffusion-Trust) | `rollout_shifted` : blocs `[graine, MASK×(Bb-1)]`, échantillonnage depuis `logits[i-1]`, engagement seuil/top-1, commit causal + graine suivante — réplique de l'algorithme serving |
| États par bloc | `forward_train_shifted` : snapshots propres progressifs O(L), GDN mode 2 `num_clean=1`, masque graine-causale, sans écriture |
| KL | `train_step_shifted` : règle J2 des deux côtés, graines exclues, queue ignorée |
| Perte finie | pertes finies aux pas 1 et 2 de la boucle (assertées) |
| Enseignant gelé | instance distincte, `is_frozen()`, aucun gradient côté enseignant (asserté) |
| Reprise exacte | perturbation + `load_ckpt` → empreinte identique au checkpoint |
| Checkpoint exportable | `model.pt` + `optim.pt` + `master.pt` + RNG + `budget.json` + `manifest.json` |

**Tous les critères de sortie J3 sont vérifiés. J3 terminé.**

## 1. Fidélité au serving (E1)

Réplique d'entraînement de `LowConfidenceShiftHybridDiffusion.run`
(oracle lu tel quel, voir `SOURCES.md`) :
- préremplissage causal, graine 0 échantillonnée sur son dernier logit
  (température/top-k/top-p respectés via `sample_token`) ;
- passes `denoise_block_shifted` (J2 : GDN mode 2, graine causale) ;
- jeton en position i≥1 depuis `logits[i-1]`, `block_mask_index[0]=False`
  (graine jamais rééchantillonnée — asserté : `committed[0] == False`) ;
- mode `dynamic` ≡ serving (seuil, repli top-1, dernier pas tout le résidu) ;
- commit causal du bloc complet (GDN mode 1), graine suivante sur son
  dernier logit (comme `_capture_next_seeds`).
- Compteur exact en statique 1 pas/bloc : `1 + nb×(1 denoise + 1 commit)`.

Divergences training-side documentées (pas de portée serving) :
- mode `static` (1 jeton/pas) : choix d'entraînement, inexistant en serving
  (seuil 0,95 par défaut) ;
- arrêt EOS : tronque au bloc courant (queue < Bb ignorée côté perte) ;
  la boucle J3 ne passe pas d'`eos_id` (longueurs curriculum exactes) ;
- `Bb` = `spec.block_size` (4 par défaut → `[graine, M, M, M]`) ; le serving
  J1 utilise `block_size: 3` (voir PRD §4 / Q2 : la taille de bloc est un
  paramètre, départ 4).

Fait établi : la graine du bloc j+1 (échantillonnée sur le commit causal)
tombe exactement sur le début de slice du bloc j+1 — rollout et
entraînement partagent le même découpage (pas de décalage).

## 2. Forward d'entraînement (E2)

`TwoStreamRunner.forward_train_shifted` (ajout, `forward_train` historique
inchangé) : propre progressif avec snapshots après `[0, s)` (O(L), même
schéma que P3), blocs bruités `[s, e)` graine-en-tête, GDN mode 2,
masque `_serving_denoise_block_mask`, zéro si changement de document.
Retourne `(h_x0, h_noisy [B, nb, Bb, D], starts)` — `h_noisy[j, m]` est la
source du jeton `starts[j]+m`.
`shifted_supervision_mask` (pur CPU) : prompt exclu, graines exclues,
queue < Bb ignorée (testée sans GPU).

## 3. Pas et boucle (E3)

`train_step_shifted` : xt = x0 + MASK aux supervisées ; vues compactes par
bloc (rangs `[0, Bb)`, supervisés m≥1) ; enseignant
`compute_teacher_targets_shifted` (J2, non décalé, même indice source) ;
`kl_teacher_to_student` réutilisé tel quel (masque compact). Asserts :
perte finie, gradients étudiants finis et non vides, aucun gradient
enseignant, clipping 1.0, `n_supervised == nb×(Bb-1)`.
`run_distill_loop` (`DistillConfig`, prompts synthétiques comme le smoke —
pas un corpus, D-05) : curriculum `rollout_length` (arrondi au Bb),
empreinte poids par round + `assert_traj_fresh`-esprit (rejouement :
rejoué ≡ original, anti-fuite d'état), checkpoint au pas 1 avec budget.

## 4. Code, tests, limites

- Modifié (strict J3) : `model/twostream.py` (+~150 lignes : helper,
  `forward_train_shifted`, `_noisy_block_layers_shifted`),
  `decode/rollout.py` (+~110 : `rollout_shifted`, `rollout`/`generate`
  inchangés), `train/loop.py` (+~200 : `train_step_shifted`,
  `DistillConfig`, `run_distill_loop`, imports ; `run_smoke`/`train_step`
  `current` inchangés et gelés).
- Tests : `tests/numerical/test_j3_shifted.py` (9 tests, passent —
  `logs/j3_tests.log`). Les tests qui mutent le 0,8B restaurent ses poids
  (fixture partagée protégée, vérifié par empreinte).
- Non-régression : 125 passés + 15 intégration passés, 2 ignorés, 1 xfail
  (détail §5). Environnement : torch 2.11.0+cu128, transformers 5.17.0,
  RTX 5090 (`logs/versions.txt`, `logs/gpu.txt`).
- Limites : prompts synthétiques (le pilote J4 apportera de vrais prompts,
  prompts seuls) ; queue < Bb non supervisée ; pas de mesure de débit
  (J5, GPU non vérifié vide ici — résultats de correction uniquement) ;
  `temperature > 0` non exercée au-delà du câblage (tests à T=0).

## 5. Non-régression et fait environnemental

- `tests/numerical/test_gdn_oracle.py::test_gdn_03_oracle_kernel_parity`
  : NON EXÉCUTÉ ici — `ModuleNotFoundError: No module named 'tyro'`
  dans ce venv (preuve : `python -c "import tyro"` échoue). Pré-existant
  et indépendant de J3 (aucun import commun modifié) ; ce test relève du
  venv `flare-oracle` (`docs/environnement.md` §3). J2 l'avait validé
  (parité 3,0e-8, inchangée — `gdn.py` non touché par J3).
- Tout le reste passe à l'identique (voies `current` gelées intactes).
