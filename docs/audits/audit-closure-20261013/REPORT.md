# Rapport de clôture M0 / M1 / M2 — audit `audit-closure-20261013`

**SHA de départ** : `9ecab1a6743e40ba8d12a67a773edfd4a5c0e543`
**SHA de fin (code + logs + décisions)** : voir commit de clôture n°1
(§8). **Rapport présent** : commit de clôture n°2 (même section).
**Règles** : AGENTS.md lu en premier ; aucun pilote d'entraînement ;
aucun push ; aucun test/seuil/hyperparamètre modifié pour faire passer
(sauf resserrement A3, signalé) ; GPU vérifié vide avant chaque mesure
(< 2 Go : 861, 711, 698 MiB) avec `nvidia-smi` avant/après dans les
JSON (`gpu_state`) et les logs.

**Conclusion générale** : M1 est closable sous les arbitrages existants
+ A3/A4 proposés ; M0 et M2 NE SONT PAS clos (B2/B3/B4/B7, C2–C7
NOT RUN/BLOCKED). **Aucun jalon n'est déclaré terminé** : la décision
revient à l'auditeur et à l'utilisateur. **La clôture de M2
n'autorise pas le pilote (M3).**

## 1. Matrice des portes

| Porte | Statut | Commande exacte | Log brut (UTF-8) | Résultat |
|---|---|---|---|---|
| A1 suite complète | **PASS** (réserve documentée) | `python -m pytest tests/ -v` (Windows, PYTHONPATH=src) + `wsl bash _artefacts/run_t_gdn_03.sh <log>` (dédié) | `logs/full_final.log` (cmd+SHA+GPU+sortie+exit) ; `logs/t_gdn_03_dedicated.log` | 126 passed, 2 skipped (oracles absents), 1 xfail documenté ; 1 FAIL `test_gdn_03` sur Windows = `tyro` manquant (limite d'env, pas de régression) → **PASS re-vérifié en venv dédié (exit 0)**. Arbre == `9ecab1a` au moment du run : aucune régression. |
| A2 mutations GDN | **PASS** | `wsl` flare-oracle `pytest _artefacts/mutations/test_gdn_oracle_mut_*.py` | `logs/mut_gdn03.log` | 3/3 FAIL prouvé : token-causal abs 5.98e-02/rel 0.97 (60×/195× la tol.) ; g inversé abs 1.43 (1430×) ; beta ignoré abs 5.93e-02/rel 5.9e10. Pouvoir de détection réel à 5e-3. |
| A2 mutation LOSS | **FAIL (documenté)** | `wsl bash _artefacts/run_t_loss_05_mut.sh` (copie sans `roll(1)` enseignant, `src/` et script officiel intouchés) | `logs/mut_loss05_no_roll.log` + `logs/t_loss_05_mut_no_roll.json` | Diagnostic enseignant 0.0 → **22.6**, Δ hybride 0.020 → **0.319** (×16) MAIS verdict E0-b=0.5 inchangé (PASS). Le critère accepté ne détecte pas ce défaut → **justifie A3**. |
| A3 E0-b ≤ 0.1 | **PASS** (arbitrage requis) | `wsl bash _artefacts/run_t_loss_05_e0b01.sh` (`E0B_TOL=0.1` dans `scripts/t_loss_05_parity.py`, seul seuil modifié, signalé) | `logs/t_loss_05_e0b01.log` + `logs/t_loss_05_e0b01.json` | **d_loss_hybrid = 0.069 ≤ 0.1** (runs E/F/G : 0.020–0.076 ; 0.1 ≈ 1.3× max). Strict 1e-3 toujours FAIL (rapporté). Le run mutant A2 (0.319) serait rejeté à 0.1. Si futur FAIL : aucun assouplissement. |
| A4 argumentaire GDN | **PASS** | — (correction documentaire) | `docs/decisions/T-GDN-03-tolerance-arbitration.md` (diff, commit n°1) | Comparaison « ~100× bruit bf16 » retirée (absolu vs relatif, invalide). Nouvel argument : résidu **strictement multiplicatif** (relatif inchangé à amplitude ×8, `t_gdn_03_sens.log`), 0.12–0.22 % du même ordre que la résolution TF32 (≈0.39 %) ; `refmax` ≈ 5.9e-2 rapportée. Tolérance 5e-3 inchangée. Docstring du test corrigée pareillement. |
| A5 anomalie FLARE | **PASS** | lecture seule (`grep`/`sed` du checkout `6ca547a`, rien copié) | `docs/decisions/M5-FLARE-serving-causal-mode.md` | Chemin entraînement chunk `causal_mode=0` = fin de bloc = sémantique Jainius ✓. Serving diffusion par défaut (`LowConfidenceShiftHybridDiffusion`) : denoise **mode 2** puis commit **mode 1** — jamais le mode 0. Algos `LowConfidence`/`JointThreshold` (`causal_xt=False`) : mode 0 ✓. **Prérequis M5** inscrit (options a/b/c). |
| B1 cohérence B0 | **PASS** | `wsl bash _artefacts/run_ifeval_unimpl.sh` | `b1-coherence.md`, `logs/ifeval_unimpl.{json,log}` | Tableau vs fiche officielle (MMLU-Pro 18.8 vs 29.7, IFEval 21.8 vs 52.1 — écarts expliqués : génération-lettre vs logprobs, strict-only, temp 0 vs sampling). Métrique IFEval = **prompt-level strict** (arXiv:2311.07911 §2). Union contraintes non implémentées : **41/541 = 7.58 %** (10+31, disjointes) ; borne sup IFEval ≈ 0.30. |
| B2 IFEval (2 contraintes) | **NOT RUN** | — (décision d'abord) | — | `constrained_response` (kwargs vides : sémantique à établir) et `response_language` (nouvelle dépendance détection) : implémenter (P-G/1.1 + re-mesure IFEval) OU exclure IFEval du composite — **à arbitrer**. Gel P-G/1.0 respecté (aucun modèle converti : gel non rétroactif le moment venu). |
| B3 jeu de développement | **NOT RUN** | — (script à créer + serveur SGLang) | — | Protocole : hash-98/2 des splits d'entraînement GSM8K/MATH-3–5/MBPP + dev MMLU, IFEval exclu, B0 re-mesuré dessus. Non exécuté (budget de clôture). |
| B4 débit B0 C=1,4,8 | **NOT RUN** | `docker start sglang-b0` + harnais charge mixte (à écrire) | — | Spike indicatif existant (535 tok/s bs=1, 1507 tok/s C=4, `audit-spike-sglang-20261012`) — non grade-clôture (pas de p50/p95, req/s, mémoire max). Image/tag : `lmsysorg/sglang:dev-cu13`, conteneur `sglang-b0`. |
| B5 budgets | **PASS** (proposé, à arbitrer) | — | `docs/decisions/M0-budgets-A008-A009.md` | **A-008 : pic ≤ 28 Go** (L≤256 nominal ; L512 paging ; L1024 interdit sans checkpointing). **A-009 : pilote M3 ≤ 24 h-GPU et ≤ 1 passage dataset** ; conversion complète interdite avant C5 + checkpointing. |
| B6 H-06/H-07/H-10 | **PASS** (tableau ; H-06 écart, H-07 non clos, H-10 partiel) | — | `b6-h6910.md` | H-06 : écart 3.13/3.12 réel vs 3.11 D-13 documenté (**à arbitrer** : accepter ou reconstruire 3.11). H-07 : adaptateur d'état non lu, non clos. H-10 : licences relevées sans télécharger (compilation MIT ; DAPO Apache-2.0 ; Nemotron CC-BY-4.0 ± filtres ; TACO Apache-2.0 ; AceCode MIT ; ⚠️ KodCode subset à vérifier, CC-BY-NC possible). |
| B7 CI distante | **NOT RUN** (bloqué) | — (après push utilisateur : URL du run + statut) | — | Push interdit (§13). Config vérifiée localement uniquement. |
| C1 coût du pas | **BLOCKED** (pas livré, checkpointing manquant) | `python scripts/measure_p3.py --lengths 256 512 --reps 3 --train-step --out logs/measure-c1.json` | `logs/measure-c1.{json,log,err}` + `c1_gpu_{before,after}.txt` (GPU 711 MiB < 2 Go ✓) | `train_step` réel (rollout exclu), médianes 3 reps : **L256 20.6 s / 21.5 Go ; L512 105.9 s / 40.5 Go** (paging) ; quadratique OOM dès L256. Checkpointing : non implémenté → NOT RUN. **L_max = 512 maintenu par défaut** (1024 OOM prouvé ; tension : 60 % MATH-500 tronqués à 512 — à trancher en M3). |
| C2 JAI-DATA | **BLOCKED** (arbitrage H-10 requis) | `huggingface-cli download divelab/opdlm_train_data` (APRÈS arbitrage ; 786 Go libres vérifiés) | — | Chargeur déterministe + manifeste + T-DATA-01 : non implémentés. Règle : télécharger seulement si licence compatible (cf. B6 ⚠️ KodCode). |
| C3 `select_states` per_block | **NOT RUN** | — (implémentation + tests distribution/masque/non-régression fuite) | — | Spécifié (D-08, cahier) ; `train_step` consomme toujours l'état « all-mask ». |
| C4 rollout sur runner | **NOT RUN** | `python -m pytest tests/integration/test_rollout.py tests/unit/test_denoise_no_write.py -q` (verts en A1, chemin actuel) | — (couvert par `logs/full_final.log`) | Réécriture sur RunnerState/denoise_block/commit : travail d'architecture non entrepris ; déterminisme et isolation déjà prouvés sur le chemin actuel (A1 : 7/7 rollout + no-write verts). |
| C5 profil rollouts | **NOT RUN** (dépend C4) | — | — | jetons/s, forwards/bloc, part du pas, pic mémoire : à mesurer sur lot réel après C4. |
| C6 CLI minimale | **NOT RUN** | — (`jainius audit/train/resume/eval/generate`, erreurs cahier §15) | — | Aucun module CLI dans `src/` à ce jour. |
| C7 smoke réel | **NOT RUN** (dépend C2+C4 ; aucun pilote autorisé) | — | — | Perte finie, n_supervised > 0, enseignant figé, reprise exacte, export/rechargement : après C2+C4. |

## 2. Fichiers créés / modifiés (clôture)

- Modifiés : `scripts/t_loss_05_parity.py` (E0B_TOL=0.1 + `e0b_tol` au
  JSON, docstring), `tests/numerical/test_gdn_oracle.py` (docstring
  A4 uniquement — seuils inchangés), `docs/decisions/T-GDN-03-tolerance-arbitration.md` (A4).
- Créés : `docs/decisions/M5-FLARE-serving-causal-mode.md` (A5),
  `docs/decisions/M0-budgets-A008-A009.md` (B5),
  `docs/audits/audit-closure-20261013/` (`b1-coherence.md`,
  `b6-h6910.md`, `logs/` : 15 fichiers).
- Non versionnés (gitignorés) : `_artefacts/mutations/` (4 copies de
  test), `_artefacts/{run_t_loss_05_mut,run_t_loss_05_e0b01,run_ifeval_unimpl,merge_gpu_json,ifeval_unimpl}.sh/.py`
  (outillage reproductible, reconstructible).
- Aucun secret, aucun artefact lourd, aucun `push`.

## 3. Tests exécutés

- A1 : `python -m pytest tests/ -v` → 1 failed(env)/126 passed/2
  skipped/1 xfailed (258 s) ; T-GDN-03 dédié → PASS (6 s, exit 0,
  re-vérifié après retouche A4).
- A2 : 3 mutations GDN → 3 FAILED (13 s) ; 1 mutation LOSS → run
  complet (≈ verdict PASS E0-b, Δ 0.319).
- A3 : rerun pristine E0B_TOL=0.1 → PASS, d_loss_hybrid=0.069.
- C1 : `measure_p3.py --lengths 256 512 --reps 3 --train-step` → exit 0.
- B1 : `ifeval_unimpl.py` → 41/541 (exit 0).
- Non exécutés : tout ce qui exige serveur SGLang, téléchargement
  1.72 Go, CI distante, smoke/pilote (voir matrice).

## 4. Décisions nouvelles à arbitrer par l'utilisateur

1. **E0-b : accepter 0.1** (A3 PASS) en remplacement du 0.5 accepté ?
   (Le 0.5 reste la référence tant que non.)
2. **Budgets A-008/A-009 proposés** (B5) : adopter ?
3. **H-06** : accepter Python 3.12/3.13 ou reconstruire `env-train` 3.11 ?
4. **B2** : implémenter les 2 contraintes IFEval (P-G/1.1) ou exclure
   IFEval du composite ?
5. **H-10/C2** : télécharger malgré le point KodCode (NC possible) ?
   Exclure le subset ? Filtrer les échantillons Nemotron non-CC-BY-4.0 ?
6. **Prérequis M5** (A5) : option (a) équivalence, (b) serving mode 0,
   ou (c) écart caractérisé ?
7. **L_max = 512** maintenu (avec 60 % de troncature MATH-500) en
   attendant le checkpointing ?

## 5. Risques résiduels

- R-serving : le mode de serving par défaut (2-puis-1) diffère de la
  sémantique d'entraînement (fin de bloc) — aucun transfert de
  conclusion sans lever le prérequis M5.
- R-mesure : temps L512 gonflés par le paging WDDM/WSL ; re-mesurer
  sous Linux natif avant d'en faire des coûts de référence.
- R-B0 : écart IFEval −30 pts et MMLU-Pro −11 pts expliqués mais non
  résorbés ; composite 0.3205 établi sur P-G/1.0 uniquement.
- R-données : compilation MIT mais subsets NC/copyleft possibles
  (KodCode, Nemotron per-sample) — contamination juridique si C2
  sans filtre.
- R-sélection : pas de jeu de développement (B3) → tout futur choix
  de checkpoint serait optimiste (R-01).
- E0-b=0.5 toujours en vigueur formellement : T-LOSS-05 PASS actuel
  repose dessus ; le resserrement 0.1 est proposé, non accepté.

## 6. Rappel

**La clôture de M2 n'autorise pas le pilote (M3).** Aucune porte C
d'exécution (C2–C7) n'est verte ; le pilote reste interdit jusqu'à
leur clôture + arbitrages §4.

## 7. Reproduction

```bash
# A1 : python -m pytest tests/ -v   (ATTENTION : test_gdn_oracle exige le venv dédié)
# A1-dédié : wsl bash _artefacts/run_t_gdn_03.sh <log>
# A2-GDN : venv flare-oracle, pytest _artefacts/mutations/test_gdn_oracle_mut_*.py (attendu : 3 FAILED)
# A2-LOSS : wsl bash _artefacts/run_t_loss_05_mut.sh   (attendu : Δ≈0.32, verdict E0-b=0.5 inchangé)
# A3 : wsl bash _artefacts/run_t_loss_05_e0b01.sh      (attendu : PASS, d_loss_hybrid ≤ 0.1)
# B1 : wsl bash _artefacts/run_ifeval_unimpl.sh        (attendu : union 41/541)
# C1 : python scripts/measure_p3.py --lengths 256 512 --reps 3 --train-step --out <json>
```

## 8. Commits locaux (§13, aucun push)

- Commit n°1 (code + décisions + logs/données d'audit) : `a1d7af7d46083f37fa2cb7892fd133b124dc7a39`.
- Commit n°2 (le présent rapport) : voir `git log --oneline -1` (créé juste après n°1, aucun push).
