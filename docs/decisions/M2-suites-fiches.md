# M2 — fiches de dépendances E2 / JAI-DATA / B5 (analyse, non implémentés)

Date : 2026-10-13. Statut : **NON COMMENCÉS** (aucun déclaré terminé).
Contexte : vectorisation GDN (étape D) et B0-plain mesuré (0.3205).

## E2 — sélection d'états `per_block`

- **Dépend de** : snapshots propres par frontière (disponibles :
  `TwoStreamRunner.forward_train` les calcule en un passage progressif) ;
  API de sélection par bloc (à écrire).
- **Critère de validation** : test comparant les états sélectionnés
  par bloc à la référence complète (égalité bit-exacte fp64 ou
  allclose strict documenté) + non-régression des tests de fuite.
- **Non résolu** : spécification exacte de la sélection (quels
  tenseurs par bloc : S, conv, KV ?) — à figer avant implémentation.

## JAI-DATA — pipeline de données d'entraînement

- **État réel** : aucun module de données ; `train/loop.py:34`
  documente la substitution (16 prompts synthétiques) ; le dataset
  OPDLM (`divelab/opdlm_train_data/opdlm_train.json`, 1,72 Go, LFS)
  n'est **ni téléchargé ni vérifié** (licence à contrôler avant usage).
- **Dépend de** : disponibilité + licence du dataset ; chargeur
  (batching, padding, masques, curriculum `L_max=512` maintenu) ;
  test T-DATA-01 (à écrire : déterminisme, découpage, supervision).
- **Critère de validation** : T-DATA-01 vert + reprise exacte sur
  curseurs (compatibilité `checkpoint.py`).
- **Ne pas télécharger** sans vérifier stockage (~2 Go + cache) et
  licence ; aucun pilote d'entraînement avant M2 validé.

## B5 — migration du rollout vers le runner

- **État réel** : `decode/rollout.py` utilise `JainiusModel`
  (prefill/denoise/commit sur cache HF) ; `TwoStreamRunner` expose
  `prefill`/`denoise_block`/`commit` sur `RunnerState` (testés :
  non-écriture, isolation, lecture S̃_end).
- **Dépend de** : réécriture de `rollout()` sur `RunnerState`
  (sémantique causale par blocs inchangée, T-ROLL-*) ; `Trajectory`
  et `SamplerSpec` inchangés.
- **Critère de validation** : équivalence des trajectoires
  (mêmes seeds → mêmes `x0`, test bit-exact) + isolation
  inter-rollouts via `RunnerState` (rejeu prompt 0 inchangé) +
  tous les tests `test_rollout.py` verts sur le nouveau chemin.
- **Non résolu** : le rollout courant est causal intra-bloc
  (documenté, pas du Diffusion-Trust) — le vrai débruitage
  bidirectionnel intra-bloc attend le contrat d'état GDN (M2).

## P3 — décision (rappel des mesures du 2026-10-13, GPU vide)

- L128 : 70.4 s → **10.2 s (×6.9)**, 9.37 Go ; train_step 9.4 s.
- L512 : 322 s → **~71 s (×4.5)**, 34.1 Go (paging WDDM, variance ±10 %).
- L1024 : OOM mesuré (73 Go demandés) — preuve archivée.
- **Décision** : `L_max=512` maintenu ; vectorisation logicielle
  suffisante pour M2 (pas de kernels ; LIC-1 non requise à ce stade).
  Réévaluer si le curriculum exige L>512 (kernels dédiés, arbitrage
  LIC-1 + T-GDN-03 élargi).
