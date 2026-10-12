# Rapport de suivi — validation et environnements oracle (`audit-8e4d9e3-20261013`)

Suite de `docs/audits/audit-8e4d9e3-20261012/REPORT.md` (référence initiale `8e4d9e39cf568595bec8df260378b9729ec6317e`).

## 1. Résumé exécutif

- **Dépôt inchangé** depuis l'audit précédent : même HEAD (`8e4d9e3`), même diff suivi (12 fichiers, +290/−97), aucun commit/push/merge, historique intact, sauvegarde non destructive sous `_artefacts/backup-audit-8e4d9e3-20261013/` (diff + inventaires, sans secrets ni poids).
- **Validation locale rejouée** : CPU 49 passed/1 skipped, P1-P3 GPU 4/4, GDN/runner 6/6 — tous conformes aux preuves archivées.
- **CI** : configuration vérifiée (YAML valide, déclencheurs PR/push-main/dispatch, 17/17 fichiers listés existants, pas de `continue-on-error`, `oracle-status` informatif). **CI distante NOT RUN** (aucun push autorisé — aucun résultat distant consulté ni prétendu).
- **T-GDN-03 — ENVIRONNEMENT LEVÉ, TEST EXÉCUTÉ, VERDICT FAIL (strict)** : venv WSL isolé (torch 2.10.0+cu128, triton 3.6.0, fla 0.5.1, RTX 5090). Noyaux Triton compilés et exécutés sur sm_120. Récurrence prouvée exacte (2e-8), état final exact (6e-8), deux voies oracle concordantes (1.3e-4) ; résidu abs 1.33e-4 / rel 2.23e-3 = bruit interne du noyau (prouvé côté noyau par référence fp64 identique), au-delà du critère prédéfini 1e-3/1e-3 — critère non modifié. Au passage : bug trouvé et corrigé dans MA formule de référence du test (somme sur V au lieu de K, silencieuse car K=V=32) ; anomalie documentée du helper d'inférence oracle en `causal_mode=0`.
- **T-LOSS-05 — BLOCKED (matériel)** : torch 2.6.0+cu124 et flash-attn 2.7.4.1 antérieurs à sm_120 (CUDA 12.4 < 12.8 requis) — inexécutables sur RTX 5090. Aucune installation tentée, aucune substitution.
- **M1 : terminé (confirmé). M2 : NON VALIDÉ** (T-GDN-03 FAIL strict, T-LOSS-05 BLOCKED, B0 NOT RUN, CI distante NOT RUN).

## 2. État Git avant/après

- Avant (= après) : `HEAD 8e4d9e3`, branche `main` suivie de `origin/main` sans divergence, `git diff --cached` vide, `git diff --check` vide (fichiers `git_*.txt` dans ce dossier).
- Seul ajout d'arborescence vs l'audit précédent : `tests/numerical/test_gdn_oracle.py` (non suivi, créé ici) ; scripts d'environnement sous `_artefacts/` (ignorés par Git) ; `docs/audits/audit-8e4d9e3-20261012/` intact (rien n'a été écrasé).
- **Aucun commit, push, merge ou publication.** Historique non modifié (`git log` : 8e4d9e3, 6e4c82a, 8227424).

## 3. Commandes exactes (toutes rejouées, logs bruts en `logs/`)

- CPU : `python -m pytest tests/unit/test_curriculum.py tests/unit/test_master.py tests/unit/test_sampler.py tests/unit/test_train_step.py tests/unit/test_p1_legacy_refusal.py tests/unit/test_jai_load.py tests/unit/test_eval_harness.py tests/integration/test_checkpoint.py tests/numerical/test_masks.py tests/numerical/test_loss.py tests/numerical/test_p2_gdn_grad.py tests/numerical/test_p4_strict.py -v` → **49 passed, 1 skipped** (`logs/cpu.log`, exit 0).
- GPU P1-P3 : `python -m pytest tests/numerical/test_p1_runner_train.py -v` → **4 passed** en 33,84 s (`logs/p1_runner_train.log`, exit 0).
- GPU GDN/runner : `... test_gdn_layer.py + 3 tests twostream_runner` → **6 passed** en 21,31 s (`logs/gdn_runner.log`, exit 0).
- CI : parsing YAML + contrôle des 17 fichiers (`logs/ci_validate.log`, exit 0).
- FLARE : `wsl bash _artefacts/run_t_gdn_03.sh` → `logs/t_gdn_03.log` (FAIL strict documenté, exit 1 du test — non masqué).
- Versions hôte : Python 3.13.15, torch 2.11.0+cu128, transformers 5.17.0, CUDA/RTX 5090.

## 4. Tableau P1–P5 (mis à jour)

| Porte | Statut | Évolution vs audit précédent |
|---|---|---|
| P1 (runner, refus legacy, fuite nulle) | **PASS** | Confirmé (rejoué 4/4 + refus CPU) |
| P2 (gradients GDN, 18/18 couches) | **PASS** | Confirmé (rejoué) |
| P3 (perf, ×6,9 à L=128) | **PASS** | Confirmé (JSON recopiés ici ; aucune mesure coûteuse relancée) |
| P4 (stricts fp64 ≤1e-5) | **PASS** | Confirmé (rejoué) |
| P5 (JAI-LOAD 321/321, harnais, freeze, CI, archives) | **PASS / B0 NOT RUN** | Confirmé ; manifeste non recopié (référence `../audit-8e4d9e3-20261012/source_manifest.json`) |
| T-GDN-03 (oracle FLARE) | **FAIL (strict)** — était BLOCKED | Environnement levé, test exécuté : sémantique prouvée, résidu bruit-noyau 1.33e-4/2.23e-3 > critère 1e-3 (inchangé) |
| T-LOSS-05 (parité OPDLM) | **BLOCKED** (inchangé) | Cause matérielle précise (sm_120 vs CUDA 12.4), préflight archivé |
| CI distante | **NOT RUN** (inchangé) | Config vérifiée ; push requis (autorisation nécessaire) |

## 5. CI locale et distante

- Locale : le sous-ensemble CPU exact du job `cpu-tests` passe (49 passed — les 12 fichiers du job + `test_train_step.py`, garde CUDA `skipif` OK).
- Config : 2 jobs (`cpu-tests`, `oracle-status`) ; déclencheurs push[main]/PR[main]/dispatch ; install CPU documentée (lock cu128 non installable sur runners linux — assumé, pas contourné) ; échecs pytest → statut non nul (aucun `|| true`/`continue-on-error`) ; `oracle-status` = 3 `echo` BLOCKED/NOT RUN + mention informative — ne peut pas passer pour une validation.
- Distante : **NOT RUN** — aucun commit/push (interdiction sans autorisation). Procédure préparée : autoriser publication de la branche → push → vérifier les 2 jobs sur l'onglet Actions (PASS exigé sur `cpu-tests`, `oracle-status` informatif) → consigner run-id et logs. Sans cela, aucune allégation distante.

## 6. FLARE (T-GDN-03) — détail

- Environnement dédié WSL (isolé, principal intact) : `/tmp/flare-oracle`, Python 3.13.16 (uv ; cp312 sans wheels torch 2.10.0), torch 2.10.0+cu128, triton 3.6.0, fla/fla-core 0.5.1, transformers 4.57.1, datasets 3.6.0, tyro, torchdata, torchao 0.18.0 (dégradé, import OK) — voir `flare_env_versions.json` (écarts documentés : numpy 2.5.4, Python 3.13, venv éphémère reconstructible).
- Oracle importé en lecture seule (`sys.path` vers le checkout, aucun code copié, aucun kernel écrit — LIC-1 respectée).
- Résultats : récurrence mode1 vs naïve **2e-8** ; état final **6e-8** ; chunk vs fused **1.3e-4** ; résidu vs critère : abs 1.33e-4 (OK) / rel 2.23e-3 (KO, critère 1e-3/1e-3 prédéfini, non modifié) ; fp64 prouve le résidu côté noyau (TF32/fast-math Triton, drapeau hôte sans effet).
- Corrections honnêtes en cours de route : bug de MA référence (somme sur K, pas V) ; constat que le helper `fused...causal_mode=0` ne fait pas de lecture fin-de-bloc (chemin d'entraînement chunk WY fait foi).
- Verdict : **FAIL strict** — parité sémantique démontrée, clôture H-02 en attente d'arbitrage de tolérance (à 1.3e-4 ≪ bruit bf16 modèle 0.25), pas d'un nouveau code.

## 7. OPDLM (T-LOSS-05) — détail

- **BLOCKED matériel** (`preflight_t_loss_05.json`) : station sm_120 (RTX 5090, compute 12.0) vs torch 2.6.0+cu124 (CUDA 12.4 < 12.8 minimum Blackwell, binaires sans sm_120) et flash-attn 2.7.4.post1 (pré-sm_120, build source). Code présent (`train/rl_sdar.py`, `configs/rl_bd3lm.yaml`), conventions intactes (non modifiées). Aucune installation, aucun substitut synthétique.

## 8. Limites matérielles et tests non exécutés

- P3 L≥1024/B0 : inchangés (NOT RUN/BLOCKED, cf. audit précédent — aucune extrapolation).
- `test_smoke.py` complet : NOT RUN (durée ; `run_smoke` rebranché runner non rejoué ici).
- CI distante : NOT RUN. T-GDN-03 : FAIL strict (exécuté). T-LOSS-05 : BLOCKED.
- venv FLARE éphémère (`/tmp` WSL) — reconstructible en ~10 min via scripts versionnés-ignorés cités dans `flare_env_versions.json`.

## 9. Décisions proposées (E2/JAI-DATA/B5, P3) — analyses, pas d'implémentation

- **P3** (mesures : 488→70 s à L=128 ; 322 s/34 Go à L=512) : recommander (i) `L_max=512` documenté pour tout travail pré-pilote + (ii) vectorisation logicielle des boucles GDN têtes/tokens en pur PyTorch (préserve sémantique/autograd, validable par les mêmes oracles), et seulement après arbitrage LIC-1 (iii) kernels dédiés. Ne pas trancher sur benchmark inexistant.
- **B5** (rollout→runner) : dépend de `TwoStreamRunner.denoise/commit` (présents) ; critère : équivalence trajectoires + isolation inter-rollouts sur RunnerState.
- **E2** (`per_block`) : dépend des snapshots par frontière (disponibles) ; critère : sélection/états par bloc == référence complète.
- **JAI-DATA** : dépend du dataset OPDLM 1,72 Go (LFS, non téléchargé) ; critère T-DATA-01. Aucun de ces trois n'est commencé ni déclaré terminé.

## 10. Prochaines actions (priorité)

1. **Autorisation de publication** → push + CI distante (seul moyen de passer NOT RUN → PASS/FAIL).
2. **Arbitrage tolérance T-GDN-03** (résidu 1.3e-4 prouvé bruit-noyau) et **confirmation LIC-1** avant kernels.
3. Station pré-Blackwell (ou conteneur CUDA 12.4+nvdia toolchain) pour T-LOSS-05.
4. Vectorisation GDN logicielle (P3-ii) puis `test_smoke` complet en CI GPU.
5. E2/JAI-DATA/B5 après M2.

## 11. Verdicts (inchangés, resserrés)

- **M1 : TERMINÉ** (preuves rejouées à l'identique).
- **M2 : NON VALIDÉ** — T-GDN-03 FAIL strict (sémantique OK, tolérance à arbitrer), T-LOSS-05 BLOCKED, B0 NOT RUN, CI distante NOT RUN.
- **Confirmation : aucun pilote d'entraînement, benchmark B0 ou entraînement de production lancé ; aucun kernel écrit ; aucun vendoring ; aucun commit/push/merge ; aucun secret exposé.**
