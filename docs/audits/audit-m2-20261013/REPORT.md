# Rapport d'audit — déblocage M2 (`audit-m2-20261013`)

Suite de `audit-8e4d9e3-20261012/13/14` et du spike
`audit-spike-sglang-20261012`. Référence initiale `8e4d9e3`
(dépôt à `2c95a18` en début de phase, arbre propre).

## 1. Résumé exécutif

- **T-GDN-03 PASS** (arbitrage utilisateur rel 5e-3 appliqué, re-exécuté
  PASS en environnement FLARE dédié). H-02 clos.
- **T-LOSS-05 PASS (E0-b)** (arbitrage utilisateur accepté : parité
  câblage au plancher de bruit, deltas E0 documentés). E0 clos.
- **Vectorisation GDN** : ×6.9 à L128 (70.4→10.2 s), ×4.5 à L512
  (322→~71 s), parité bit-exacte legacy, L1024 OOM (L_max=512 maintenu).
- **Smoke + intégration** : 8/8 PASS (wiring, reprise bit-exacte,
  rollout, generate).
- **B0-plain mesuré** : composite **0.3205** (7 benchmarks × 3
  répétitions, SGLang dev-cu13, hash P-G/1.0 conforme, 0 échec).
- **M1 : TERMINÉ. M2 : VALIDÉ** sous les deux arbitrages acceptés
  (décisions jointes) — voir matrice §2 et conditions §8.

## 2. Matrice des portes

| Porte | Statut | Critère | Mesuré | Preuve |
|---|---|---|---|---|
| P1 runner/legacy/fuite | PASS | train_step runner, refus legacy, fuite 0 bit-exact | inchangé, rejoué partiel. | audits précédents + `integration.log` (smoke voie runner) |
| P2 gradients GDN | PASS | finis non nuls, 18/18 couches | rejoué (`test_p1_runner_train` 4/4 dans `gpu_numerical.log` antérieur ; non régressé) | `logs/gpu_numerical*.log` (20261014) |
| P3 perf | PASS | amélioration mesurée ≥1 taille, GPU vide | L128 ×6.9, L512 ×4.5 (GPU vide, `check_env_p3remesure.log`) | `measure-p3-clean128/512.json` |
| P4 stricts fp64 | PASS | ≤1e-5 + bit-exact vectorisé | 4/4 parité bit-exacte (`test_gdn_vectorized`) | suite CPU 82 passed |
| P5 (LOAD/harnais/env/CI) | PASS | — | CI mise à jour (15 fichiers, 82 passed) | `ci.yml`, suite CPU |
| T-GDN-03 | **PASS** (arbitrage 5e-3) | abs ≤1e-3 ET rel ≤5e-3 | abs 1.33e-4, rel 2.23e-3 | `t_gdn_03_pass.log` (exit 0), `T-GDN-03-tolerance-arbitration.md` (ACCEPTÉE) |
| T-LOSS-05 | **PASS (E0-b)** | \|Δhybride\| ≤0.5 + deltas documentés | Δ0.020 (bruit FA 0.005–0.040) ; strict 1e-3 FAIL rapporté | `t_loss_05.json` (verdict double), `T-LOSS-05-E0-arbitration.md` (ACCEPTÉE) |
| Smoke/intégration | PASS | wiring, reprise, rollout, generate | **8/8** | `logs/integration.log` |
| B0-plain | **MESURÉ** | 7 bench. × 3 reps, composite figé | **0.3205**, hash P-G/1.0 OK, 0 échec gén. | `b0-results.json`, `stdout.log` |
| B0-MTP | NOT RUN (reporté) | — | décision M0-B0-plain-seul (WIP amont) | `M0-B0-plain-seul.md` |
| CI distante | NOT RUN | — | aucun push (interdit sans autorisation) | config vérifiée localement |
| E2/JAI-DATA/B5 | NOT RUN (non commencés) | — | fiches à jour | `M2-suites-fiches.md` |

## 3. Changements de code (depuis `2c95a18`)

- `src/jainius/model/gdn.py` : `gdn_run_batched`, `gdn_step`
  agnostique aux dims, `_core` vectorisé (+`_core_legacy`),
  `forward_noisy_block` vectorisé + einsum S̃_end
  (+`_forward_noisy_block_legacy`).
- `src/jainius/model/twostream.py` : `_doc_positions` 100 %
  tensoriel, `_doc_segments` vectorisé, snapshots sans clones
  redondants (sûreté autograd vérifiée).
- `tests/numerical/test_gdn_vectorized.py` (nouveau) : parité
  bit-exacte legacy/nouveau (sorties, états, grads) + positions
  et segments vs brute-force.
- `tests/numerical/test_gdn_oracle.py` : tolérance arbitrée
  (rel 5e-3, docstring).
- `tests/integration/test_smoke.py` : reprise via runner explicite.
- `scripts/measure_p3.py` : temps fwd/bwd séparés + train_step réel.
- `scripts/t_loss_05_parity.py` : verdict double (strict + E0-b),
  diagnostics (rerun, eager, fp32, layout).
- `.github/workflows/ci.yml` : +`test_b0_runner`,
  +`test_gdn_vectorized`, +`requests datasets` ; `oracle-status`
  actualisé (informatif, pointe les audits).
- Docs : `T-LOSS-05-E0-arbitration.md` (ACCEPTÉE),
  `T-GDN-03-tolerance-arbitration.md` (ACCEPTÉE),
  `M2-suites-fiches.md`.
- Logs `t_loss_05.json/log` (audit-20261014) : écrasés par le run
  final PASS(E0-b) — versions FAIL intermédiaires préservées par
  l'historique Git + copies `t_loss_05_runF/G.json`.

## 4. Résultats chiffrés

### GDN (Qwen3.5-0.8B bf16, batch 1, GPU vide 2 %/930 MiB)
| L | Avant (P3-20261012) | Après (vectorisé) | fwd | bwd | Pic |
|---|---|---|---|---|---|
| 128 | 70.4 s / 9.37 Go | **10.2 s / 9.37 Go** | 3.9 s | 6.4 s | identique |
| 128 train_step réel | — | **9.4 s / ~11–14 Go** | — | — | — |
| 512 | 322 s / 34.1 Go | **64–78 s / 34.1 Go** | ~21 s | 43–57 s | paging WDDM (±10 %) |
| 1024 | BLOCKED | **OOM mesuré** (73 Go) | — | — | preuve archivée |

### T-GDN-03 (FLARE dédié, RTX 5090)
Récurrence 2e-8, final 6e-8, chunk-vs-fused 1.3e-4,
abs 1.33e-4 / rel 2.23e-3 → PASS (5e-3 arbitré). Sensibilité
archivée (6 configs fp32 + fp16/bf16, erreur multiplicative).

### T-LOSS-05 (OPDLM exécuté verbatim, Qwen3-0.6B)
Teacher bit-exact (0.0), masque-xt bit-exact, masque-x0 12 cellules
E0, perte hybride Δ0.020 (bruit FA 0.005–0.040), strict 1e-3 FAIL
(Δ0.30 nominal) → PASS (E0-b ≤0.5).

### B0-plain (SGLang dev-cu13, C=8, temp 0)
| Benchmark | Lignes | mean (3 reps) | std |
|---|---|---|---|
| GSM8K | 1319 | 0.5219 | 1.3e-3 |
| MATH-500 | 500 | 0.3313 | 5.7e-3 |
| MMLU | 14042 | 0.4476 | 4.5e-4 |
| MMLU-Pro | 12032 | 0.1882 | 4.8e-4 |
| IFEval | 541 | 0.2181 | 1.5e-3 |
| HumanEval | 164 | 0.2825 | 2.9e-3 |
| MBPP | 500 | 0.2540 | 0.0 |
| **Composite** | — | **0.3205** | plancher ci-dessus |

Serveur : conteneur `sglang-b0` (préexistant, vérifié sain, arrêté
après mesure pour libérer le GPU — redémarrable `docker start`).
Aucune interférence GPU pendant B0 (vérifié : aucun travail
torch après le lancement ; P3/tolérance mesurés avant).

## 5. Reproduction

```bash
# CPU (référence, ~8 s)
python -m pytest tests/unit/test_curriculum.py tests/unit/test_master.py \
  tests/unit/test_sampler.py tests/unit/test_train_step.py \
  tests/unit/test_p1_legacy_refusal.py tests/unit/test_jai_load.py \
  tests/unit/test_eval_harness.py tests/unit/test_b0_runner.py \
  tests/integration/test_checkpoint.py tests/numerical/test_masks.py \
  tests/numerical/test_loss.py tests/numerical/test_p2_gdn_grad.py \
  tests/numerical/test_p4_strict.py tests/numerical/test_gdn_vectorized.py -q
# GPU : voir logs/ (commandes en tête de chaque log)
# T-GDN-03 : wsl bash _artefacts/run_t_gdn_03.sh  (venv ~/.venvs/flare-oracle)
# T-LOSS-05 : wsl bash _artefacts/run_t_loss_05.sh (venv ~/.venvs/opdlm-parity)
# P3 : PYTHONPATH=src python scripts/measure_p3.py --lengths 128 ...
# B0 : docker start sglang-b0 ; PYTHONPATH=src python -m jainius.eval.b0_runner \
#        --sglang-url http://127.0.0.1:30000 --out _artefacts/b0/b0-results.json
```

## 6. Limites et risques résiduels

- L512 sous paging WDDM (34 > 32 Go) : variance ±10 %, pas de
  mesure pic-fidèle au-delà du physique ; L1024 inaccessible.
- Bruit FA inter-process (±0.04) borne la précision des parités
  bf16 — constaté, pas contourné.
- B0 : IFEval contient des contraintes non implémentées (comptées
  échec, listées) ; plafonds relevés documentés (2048/1024) ;
  MMLU-Pro 0.188 et IFEval 0.218 bas — fidèles au protocole,
  à comparer au modèle converti (pas un jugement de qualité absolue).
- CI distante NOT RUN (aucun push) ; E2/JAI-DATA/B5 non commencés ;
  smoke couvre le câblage, pas un entraînement prolongé.
- WSL /tmp volatile (2 wipes constatés) : venvs sur `~/.venvs`
  (persistants), roue FA sauvegardée (`~/.venvs/wheels`).

## 7. Artefacts (ce dossier + 20261014)

`b0-results.json` (32 Ko, 3 reps, hash OK), `stdout.log`,
`b0-smoke.json`, `measure-p3-clean128/512.json`,
`measure-p3-vec*.json` (rappel), `logs/` (cpu=82,
integration=8, gpu_numerical 14+15, t_gdn_03_pass,
check_env ×2), `T-LOSS-05-E0-arbitration.md`,
`M2-suites-fiches.md` (docs/decisions).

## 8. Verdicts et conditions

- **M1 : TERMINÉ** (toutes portes M1 PASS avec preuves exécutées).
- **M2 : VALIDÉ** sous réserve des deux arbitrages acceptés le
  2026-10-13 (T-GDN-03 rel 5e-3, T-LOSS-05 E0-b ≤0.5), avec :
  P1–P5 PASS, smoke 8/8, B0-plain mesuré (MTP reporté par décision
  M0 acceptée), E2/JAI-DATA/B5 explicitement hors M2 (fiches).
- **Aucun pilote d'entraînement** lancé ; B0 = inférence du modèle
  source uniquement (pas de conversion évaluée) ; aucun kernel
  écrit ; aucun vendoring ; aucun secret exposé.
- Plus petit ensemble restant : CI distante (requiert autorisation
  de push), puis E2/JAI-DATA/B5 pour la suite du programme.
