# Audit J4 — conversion pilote, premier checkpoint Jainius (2026-10-14/15)

Objectifs servis : G3 (entraînement très léger), G4 (rétention du
raisonnement). Configuration verrouillée : `shifted` + Diffusion-Trust
(J2/J3), bloc 4, fork tel quel (D-04). Données : `divelab/opdlm_train_data`
(licence MIT, 61 816 prompts seuls, usage privé D-01).

## 1. Run pilote (G3)

- 700 pas (0–699, aucun manquant), batch effectif 2 (paires de longueur
  exacte, pas de padding), curriculum R=32 (pas 0–99) puis R=64.
- Rollouts statiques T=1.0, 1 pas/bloc (écart OPDLM : 4 pas dynamiques —
  débit mesuré ~6,6 tok/s en pile de référence, voir `scripts/bench_j4.py`).
- Optimisation : MasterOptimizer, AdamW, LR cosine 1e-5→1e-6 (warmup 5),
  clip 1.0. LR finale 1e-6 au pas 699.
- Perte (moyenne par tranche de 100) : 2,39 → 1,77 → 1,77 → 1,68 → 1,73
  → 1,64 → 1,73. Finie partout, décroissance puis plateau. Stable.
- Interruptions : session coupée à 353 (pas de checkpoint entre 300 et
  353, recalculés), stall à 475 (processus fantôme, cause externe
  probable, pas d'exception/OOM au log), reprise exacte depuis
  `step_000299` puis `step_000399` (paires/seed/trajectoires
  déterministes). Fin 700/700, checkpoint `step_000699` complet
  (model/optim/master/RNG/budget/manifeste), empreinte de recharge
  identique (`c103d50e…`).
- Écart script (sans perte) : double sauvegarde du pas 699 (boucle puis
  finale) → `FileExistsError` rattrapée, sortie code 1, checkpoint intact.

## 2. Budgets réellement consommés (G3)

- Tokens réponse : 83 200 (0,000083 Md, plafond 0,1 Md — facteur ~1 200).
- Tokens supervisés : 62 400. Tokens séquence : ~152 000 (post-reprise ;
  pré-reprise non journalisée, même ordre de grandeur).
- h-GPU : ~3,3 h (pas uniques) + ~1,3 h (pas recalculés après
  interruptions) + évaluation/tests (à compléter) — plafond 24 h.
- Mémoire : pression allocateur constatée (VRAM 98,5 %, `train_s` ×2 à R
  constant, paging WDDM probable) — règles inscrites dans
  `docs/environnement.md` §9. Pas d'OOM, pas d'impact validité.

## 3. Rétention (G4)

- Protocole : P-G/1.0 identique à B0, moteur glouton local documenté
  (SGLang indisponible : 1 répétition T=0 au lieu de 3).
- Référence B0 : composite 7 datasets 0,3205 ; **sans IFEval 0,3376** →
  palier 1 ≥ 0,2363, palier 2 ≥ 0,3038.
- Comparabilité moteur : source locale (100 lignes) gsm8k 0,47 vs 0,522
  B0 (compatible à 1σ) ; mmlu 0,27 vs 0,448 = biais de sous-ensemble
  (100 premières lignes = sujets non représentatifs), pas un effet
  moteur. Sonde n=20 : source locale 8/20 vs B0 0,522 (biais moteur
  faible, dans le bruit).
- Résultat checkpoint (`eval_ckpt_<ds>.json`, 6 datasets complets) :

| Dataset | Source (B0-SGLang) | Checkpoint (local) | Retenu |
|---|---|---|---|
| gsm8k | 0,522 | 0,053 | 10 % |
| math500 | 0,331 | 0,056 | 17 % |
| mmlu | 0,448 | 0,319 | 71 % |
| mmlu_pro | 0,188 | 0,127 | 67 % |
| humaneval | 0,283 | 0,085 | 30 % |
| mbpp | 0,254 | 0,088 | 35 % |
| **Composite (sans IFEval)** | **0,3376** | **0,1213** | **35,9 %** |

- **Palier atteint : aucun (< 70 %).** Palier 1 (0,2363) non atteint.
- Mécanisme (sonde n=20) : le checkpoint ne produit **jamais** le
  marqueur final `####` (0/20 contre 6/20 source) et dérive vers un
  style verbeux à puces, parfois répétitif — la distillation sur
  prompts seuls ne supervise jamais la convention de clôture, et le
  style étudiant l'a perdue. Sorties fluentes et structurées : pas un
  effondrement génératif, une perte de suivi de consigne formatée.
  mmlu (format lettre unique, plus simple) est le moins touché (71 %).
- Réserves : moteur local vs SGLang (biais faible, § ci-dessus) ;
  1 répétition vs 3 (T=0 déterministe, variance nulle attendue).

## 4. Contrôles

- Tests : `pytest tests/numerical --ignore=test_gdn_oracle.py` →
  **65 passés, 2 ignorés, 1 xfail** (`pytest_night.log`, 142 s).
  `test_gdn_oracle` exclu : requiert le venv `flare-oracle` (`tyro`
  absent ici), déjà validé en J2 (parité 3,0e-8, `gdn.py` inchangé
  depuis). Voies `current` gelées intactes (suite verte à l'identique).
- GPU : pilote et évals sur carte dédiée ; mesures de correction
  uniquement (débit = J5).
- h-GPU totaux J4 ≈ 12 h (entraînement ~4,6 + évals ~7 + tests/sondes),
  plafond 24 h. Tokens inchangés : 0,000083 Md / 0,1 Md.

## 5. Fichiers
- `scripts/pilot_j4.py` (pilote), `scripts/eval_j4.py` (éval checkpoint),
  `scripts/bench_j4*.py`, `scripts/dbg_batch_gap.py` (diagnostics).
- `src/jainius/train/loop.py` : `train_step_shifted_batched` (équivalence
  fp32 vs unitaire, test `tests/numerical/test_j4_batched.py`).
- `src/jainius/eval/local_runner.py` (runner local P-G/1.0).
- `docs/environnement.md` §9 (mémoire WDDM) et §10 (stabilité runs longs).

## 6. Conclusion

Critères de sortie J4 (PRD §6 : budgets + rétention mesurée, palier
noté) : **formellement satisfaits** — 83 200 tokens / ~12 h-GPU, mesure
complète, palier noté. **Objectif G4 non atteint** : 35,9 %, sous le
palier 1 (70 %). Le pilote valide la boucle et l'instrumentation, pas
la conversion : le checkpoint ne suit plus les consignes formatées
(cause identifiée, remédiation proposée : consignes de format dans les
prompts seuls, toujours sans SFT — voir BRIEF_ARCHITECTE.md).
Défauts mineurs non traités ici (double sauvegarde finale,
compteurs non cumulatifs, env WSL à reconstruire) : renvoyés aux
prompts suivants, sans impact sur les résultats.
