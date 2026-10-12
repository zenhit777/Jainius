# Rapport d'audit — passage M1 → M2 (`audit-8e4d9e3-20261012`)

## 1. Commits : référence et audité

- Commit de référence obligatoire : `8e4d9e39cf568595bec8df260378b9729ec6317e` — **conforme** (`git rev-parse HEAD` identique, arbre propre au démarrage : `git status --short` vide, aucun diff).
- Commit audité : `8e4d9e3` + modifications non commitées ci-dessous (aucun commit/push effectué, conformément à l'interdiction).

## 2. États Git initial et final

- Initial : `HEAD == 8e4d9e3`, `git diff --stat` vide, `git diff --cached --stat` vide. Python 3.13.15, pip 26.2.1, torch 2.11.0+cu128, transformers 5.17.0, CUDA disponible (RTX 5090, 32 Go, WDDM).
- Final : 12 fichiers suivis modifiés + 10 créations non suivies (détail §3). Aucune modification préexistante à préserver (arbre initial propre) ; rien n'a été détruit.

## 3. Inventaire des fichiers modifiés / créés

Modifiés (12) : `.github/workflows/ci.yml` (unifié), `.github/workflows/cpu.yml` (**supprimé**), `requirements-lock.txt` (vrai `pip freeze` UTF-8), `src/jainius/model/__init__.py`, `src/jainius/model/arch.py`, `src/jainius/model/gdn.py`, `src/jainius/model/twostream.py`, `src/jainius/train/loop.py`, `tests/numerical/test_target_alignment.py`, `tests/numerical/test_two_stream.py`, `tests/numerical/test_twostream_runner.py`, `tests/unit/test_train_step.py` (+290/−97, `git diff --stat`).
Créés : `src/jainius/train/source_manifest.py`, `src/jainius/eval/{__init__,harness}.py`, `scripts/{measure_p3,build_source_manifest}.py`, `tests/{unit/{test_p1_legacy_refusal,test_jai_load,test_eval_harness},numerical/{test_p1_runner_train,test_p2_gdn_grad,test_p4_strict}}.py`, `docs/audits/audit-8e4d9e3-20261012/` (REPORT, logs, manifests, P3 JSON, préflight copié).

## 4. Constats initiaux par porte (base 8e4d9e3)

- **P1 FAIL** : `train_step` (`src/jainius/train/loop.py:110`) utilisait `student.forward_two_stream` = flux concaténé à fuite GDN H-02 caractérisée ; `TwoStreamRunner.forward_train` (sans fuite, bit-exact) n'était branché nulle part dans l'entraînement.
- **P2 FAIL** : `gdn_step`/`gdn_run`/`GDNRef` (`src/jainius/model/gdn.py:32,49,316,356`) : `float(g[t])`, `float(beta[t])`, `math.exp(g)` — graphe autograd coupé ; de plus `GDNRefLayer` recopiait les poids en tenseurs détachés (aucun gradient vers les paramètres HF via le GDN).
- **P3 FAIL** : `forward_train` recalculait le préfixe propre depuis 0 à chaque bloc (`_clean_prefix_states`, O(n_blocs×L)) — mesuré §9.
- **P4 FAIL** : T-GDN-02/04 et fuite uniquement en bf16 (tolérances larges `atol 0.25–0.5`) ; aucune variante stricte fp32/fp64 à 1e-5.
- **P5 FAIL partiel** : pas de JAI-LOAD (empreintes tenseurs + T-LOAD-01) ; pas de harnais P-G versionné ; `requirements-lock.txt` en UTF-16 non-`freeze` (9 lignes) ; pas de `docs/audits/` ; deux workflows redondants (`ci.yml` + `cpu.yml`).

## 5. Corrections réalisées (sans changer seuils, hyperparamètres, conventions ou protocoles)

1. **P2/graphe** (`gdn.py`) : `gdn_step`/`gdn_run` en `torch.exp` + indexation native (plus aucun `float()`/`math.exp` sur le chemin) ; `GDNRefLayer` lit les poids VIVANTS HF (cast différentiable) — copies conservées pour formes/référence ; `conv_end` en `clone()` différentiable ; `GDNRef` : `float()` supprimés, poids en float64, `conv_w` redimensionnée (était incohérente, code mort jamais exécuté). Récurrence mathématique, paramètres, convention d'états, dimensions et ordre inchangés.
2. **P1/runner** (`arch.py`, `loop.py`) : chemin historique isolé sous `legacy_leaky_forward` ; `forward_two_stream(..., allow_legacy_leaky=False)` lève `LegacyLeakyError` par défaut ; `train_step(..., runner=None, allow_legacy_leaky=False)` utilise `TwoStreamRunner.forward_train` (brut → `final_norm` → `lm_head`) et refuse le legacy sans runner ; `run_smoke` construit et branche un runner. Tests de caractérisation mis à jour vers la voie explicite (`allow_legacy_leaky=True`) — caractérisation préservée, pas masquée.
3. **P3/perf** (`twostream.py`) : `forward_train` progressif (snapshots calculés en un passage O(L), réutilisés aux frontières) ; variante naïve conservée (`_forward_train_quadratic`) pour la mesure avant/après. Sémantique identique (équivalence testée, perte identique 0.205078125 à L=128).
4. **P4/strict** : nouveaux tests fp64 CPU (`test_p2_gdn_grad`, `test_p4_strict`, eps=1e-6 central, tol rel 1e-3 ; batched/packing/fuite tol 1e-5) ; variantes bf16 conservées comme non-régression.
5. **P5** : `source_manifest.py` + T-LOAD-01 (6 cas) + manifeste réel 321/321 ; `eval/harness.py` P-G/1.0 + tests fixtures ; `requirements-lock.txt` = `pip freeze` UTF-8 (65 lignes) ; CI unifiée (job CPU + job `oracle-status` informatif), `cpu.yml` supprimé sans perte de couverture (union des listes + 5 nouveaux fichiers CPU).

## 6. Commandes exactes et résultats bruts

- `python -m pytest tests/unit/test_curriculum.py tests/unit/test_master.py tests/unit/test_sampler.py tests/unit/test_train_step.py tests/unit/test_p1_legacy_refusal.py tests/unit/test_jai_load.py tests/unit/test_eval_harness.py tests/integration/test_checkpoint.py tests/numerical/test_masks.py tests/numerical/test_loss.py tests/numerical/test_p2_gdn_grad.py tests/numerical/test_p4_strict.py -v` → **49 passed, 1 skipped** (`logs/cpu.log`, exit 0).
- `python -m pytest tests/numerical/test_p1_runner_train.py -v` → **4 passed** en 34,44 s (`logs/p1_runner_train.log`, exit 0) : train_step runner + grads ; invariance fuite hidden + perte == 0.0 avec 3 dépendances légitimes > 0 ; 18/18 couches GDN grads finis non nuls (fp32, 162 paramètres vérifiés) ; équivalence quadratique/optimisé.
- `python -m pytest tests/numerical/test_gdn_layer.py "tests/numerical/test_twostream_runner.py::test_runner_no_leak_desired_property" "tests/numerical/test_twostream_runner.py::test_runner_no_leak_characterization" -v` → **5 passed** en 19,56 s (`logs/gdn_runner.log`, exit 0 ; T-GDN-01 non-régression OK après passage aux poids vivants).
- `python scripts/build_source_manifest.py --out docs/audits/.../source_manifest.json` → 321 tenseurs, 321 vérifiés (78 Ko).
- `python scripts/measure_p3.py --out ... --lengths 128 --variants optimized --reps 1 --warmup 0` → L128/opt : 70,43 s, 9,43 Go. `--variants quadratic` → L128/quad : 488,17 s, 54,11 Go. `--lengths 512 --variants optimized` → L512/opt : 322,44 s, 34,40 Go. Config : Qwen3.5-0.8B bf16, batch 1, unit_block 4, resp_start 0, 752 393 024 params, 24 couches, RTX 5090 (voir JSON).
- Répertoire : `D:\Intelligence_Artificielle\Projects\Jainius`, `PYTHONPATH=src` (scripts), `HF_HUB_OFFLINE=1` (mesures), Python 3.13.15 / torch 2.11.0+cu128 / transformers 5.17.0.

## 7. Statuts finaux P1–P5

| Porte | Statut | Preuve |
|---|---|---|
| P1 (runner branché, legacy refusé, fuite nulle, discriminant) | **PASS** | `logs/p1_runner_train.log` (4/4) + `logs/cpu.log` (refus) + `logs/gdn_runner.log` (fuite) |
| P2 (gradients GDN finis non nuls : finies + e2e 18 couches) | **PASS** | `logs/cpu.log` (`test_p2_gdn_grad`) + `logs/p1_runner_train.log` (e2e fp32, 0 manquant/non-fini/nul) |
| P3 (mesure + recalcul quadratique éliminé) | **PASS** | JSON P3 : speed-up 6,9×, mémoire ÷5,8 à L=128 ; L=512 OK ; L≥1024 BLOCKED (ressources, §9) ; têtes non vectorisées = limite documentée, pas de kernels (règle n°7) |
| P4 (variantes strictes fp32/fp64 ≤1e-5, bf16 conservées) | **PASS** | `logs/cpu.log` (`test_p4_strict`, `test_p2_gdn_grad`) ; bf16 non remplacées |
| P5.1 JAI-LOAD + T-LOAD-01 | **PASS** | 6/6 synthétique + manifeste réel 321/321 vérifié |
| P5.2 harnais P-G + B0 | **PASS (harnais) / NOT RUN (B0)** | 5/5 fixtures ; protocole P-G/1.0 + hash ; B0 non exécuté (interdiction pilote) |
| P5.3 env reproductible | **PASS** | `requirements-lock.txt` = `pip freeze` UTF-8, torch 2.11.0+cu128, transformers 5.17.0 |
| P5.4 archives | **PASS** | ce dossier (REPORT, 3 logs, source_manifest.json, 3 JSON P3, préflight copié) |
| P5.5 CI unifiée | **PASS (code, non exécutée à distance)** | `ci.yml` unique, `cpu.yml` supprimé, exclusions imprimées, oracles informatifs |

## 8. Tests bloqués (causes précises)

- **T-GDN-03 (oracle FLARE)** : BLOCKED — torch 2.11.0+cu128 ≠ 2.10.0 exigé, `fla` absent ≠ 0.5.1 (`preflight_t_gdn_03.json` copié ici) ; préflight existant conservé, aucune substitution naïve.
- **T-LOSS-05 (parité OPDLM)** : BLOCKED — requiert l'env OPDLM officiel (torch 2.6 + `rl_sdar` + GPU) ; prompt-convention documentée (`preM2-E0-convention-prompt.md`), conventions non modifiées pour faire passer le test.
- **P3 L=1024/4096** : BLOCKED (ressources) — extrapolé depuis mesures (L=512 : 322 s / 34,4 Go, WDDM avec pagination au-delà de 32 Go) : L=1024 ≈ 15 min/≈70 Go paginés, L=4096 = heures + centaines de Go ; aucune extrapolation présentée comme résultat.
- **B0 (7 benchmarks × 3 répétitions)** : NOT RUN — pilote interdit (règle absolue n°1) ; harnais + fixtures prêts, composite figé avant mesure.
- **CI distante** : NOT RUN — exécution GitHub non consultable depuis ce poste ; sous-ensemble CPU exécuté localement (49 passed).

## 9. Courbe temps/mémoire avant/après (P3)

| L | Avant (quadratique) | Après (optimisé) | Speed-up | Mémoire avant/après |
|---|---|---|---|---|
| 128 | 488,17 s / 54,11 Go | 70,43 s / 9,43 Go | **6,9×** | **÷5,8** |
| 512 | NOT RUN (prohibitif par construction) | 322,44 s / 34,40 Go | — | — |
| 1024 / 4096 | BLOCKED | BLOCKED | — | — |

Limites : boucles Python par tête GDN conservées (sémantique/autograd préservés) ; `.item()` sur `doc_ids` (non entraînables) conservés ; pas de kernels fusionnés/Triton (règle n°7). Propositions non appliquées : réduction documentée de `L_max` dans le curriculum, ou kernels dédiés après arbitrage LIC-1.

## 10. Versions et manifestes

Python 3.13.15, torch 2.11.0+cu128, transformers 5.17.0 (cf. `requirements-lock.txt`, 65 lignes, UTF-8) ; modèle Qwen3.5-0.8B (752 393 024 params entraînables, 24 couches : 18 GDN + 6 full-attention 3/7/11/15/19/23) ; empreinte source : `source_manifest.json` (321 tenseurs SHA-256, vérifiés 321/321) ; protocole P-G/1.0 + hash (`src/jainius/eval/harness.py`).

## 11. Décisions prises ; paramètres/tolérances modifiés

- Aucun seuil, hyperparamètre, convention mathématique ou protocole d'évaluation modifié : nouvelles tolérances documentées dans de nouveaux tests (finies rel 1e-3/eps 1e-6 justifiés ; strict 1e-5 ; équivalence P3 bf16 atol 0.5 héritée des mesures existantes) ; `requirements-lock.txt` remplacé (UTF-16→UTF-8 `pip freeze`, contenu réel) — signalé explicitement.
- Décisions : legacy isolé (`LegacyLeakyError`) ; poids GDN vivants ; snapshots progressifs ; `GDNRef` corrigée (float64, conv) mais reste hors chemin d'entraînement ; CI unifiée ; B0 NOT RUN.

## 12. Éléments restant à arbitrer

1. **LIC-1** : tranchée dans `docs/decisions/M0-arbitrages.md` (usage privé non commercial) mais kernels NON écrits par prudence (règle n°7 + P3 : proposition sans application) — confirmation responsable souhaitée avant tout vendoring/kernel.
2. **E2 (`per_block`) / JAI-DATA / B5 (rollout→runner)** : préparés conceptuellement (snapshots par frontière disponibles), ni implémentés ni testés — NON déclarés terminés.
3. **L_max du curriculum** : réduire documentairement vs kernels dédiés (P3) si L≥1024 requis.
4. **T-GDN-03 / T-LOSS-05** : lever les BLOCKED par environnements dédiés validés (pas de substitution).

## 13. Confirmation — aucun pilote lancé

**Fait établi** : aucun entraînement de production, aucun pilote d'entraînement, aucun benchmark complet de qualité du modèle. Seuls : tests unitaires/numériques ciblés, 1 pas `train_step` en test (optimiseur jouet, 4 tokens), mesures `forward_train+backward` instrumentales (L≤512, 1 répétition), forwards de caractérisation. `run_smoke` non exécuté.

## 14. Verdicts

- **M1 : TERMINÉ** — P1, P2, P4 PASS par preuves exécutées (fuite nulle bit-exacte + perte invariante, gradients 18/18 couches, strict fp64 ≤1e-5, non-régression T-GDN-01).
- **M2 : NON VALIDÉ** — portes obligatoires restantes non vertes : T-GDN-03 BLOCKED, T-LOSS-05 BLOCKED, B0 NOT RUN, P3 L≥1024 BLOCKED, CI distante NOT RUN. Le succès des tests synthétiques ne prouve pas la qualité de raisonnement.

## Résumé

Défauts corrigés : entraînement rebranché sur le runner sans fuite (legacy refusé) ; graphe autograd GDN rétabli (poids vivants, 18/18 couches avec grads) ; recalcul quadratique éliminé (×6,9 à L=128, −82 % mémoire) ; stricts fp64 ajoutés ; JAI-LOAD réel 321/321, harnais P-G/1.0, `pip freeze` UTF-8, CI unifiée.
Preuves : 49 passed CPU + 4/4 P1-P3 GPU + 5/5 GDN/runner, logs et manifests dans ce dossier.
Blocages : oracles FLARE/OPDLM (environnements), L≥1024/B0 (ressources + interdiction pilote), kernels (arbitrage), E2/JAI-DATA/B5 (non commencés).
