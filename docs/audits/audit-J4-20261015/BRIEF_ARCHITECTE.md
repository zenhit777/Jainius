# Brief à l'architecte logiciel — étape J4 (conversion pilote)

Date : 2026-10-15. Statut J4 : run terminé (700/700 pas), évaluation de
rétention en cours, audit et commit à finaliser. Ce document transmet ce
que les prompts J4 ne couvraient pas et qui doit être intégré aux
prompts suivants.

## 1. Ce qui a été construit (rappel)

- `scripts/pilot_j4.py` : boucle on-policy `shifted` + Diffusion-Trust,
  batch effectif 2 (paires de longueur exacte, sans padding), curriculum
  R=32→64, MasterOptimizer + AdamW, LR cosine 1e-5→1e-6, 700 pas.
- `src/jainius/train/loop.py::train_step_shifted_batched` : même
  mathématique que le pas unitaire (équivalence prouvée en fp32,
  `tests/numerical/test_j4_batched.py`), un seul forward batché.
- `scripts/eval_j4.py` + `src/jainius/eval/local_runner.py` : protocole
  P-G/1.0 identique à B0, moteur glouton local (SGLang indisponible).
- `docs/environnement.md` §9 : gouvernance mémoire GPU (voir §3).

## 2. Faits établis utiles aux prochains prompts

- **Convergence** : perte 2,39 → ~1,7 (plateau), finie partout, 83 200
  tokens réponse (0,000083 Md, plafond 0,1 Md), ~4,6 h-GPU / 24 h.
- **Reprise exacte** : deux reprises (pas 299, 399) déterministes
  (paires, seed, générateur dédié par rollout) ; checkpoint final
  rechargeable, empreinte identique.
- **Biais de sous-ensemble** : `load_all(limit)` prend les N premières
  lignes — non représentatif (mmlu : sujets groupés, 0,27 local vs 0,448
  B0). Exiger échantillonnage stratifié ou lignes fixées pour toute
  comparaison sur sous-ensemble.
- **Biais moteur non quantifié** : local glouton vs SGLang T=0, 1 vs 3
  répétitions. La rétention J4 compare ckpt-local à B0-SGLang avec cette
  réserve ; une évaluation source-locale complète lèverait le doute.

## 3. Incident mémoire (WDDM) — à intégrer durablement

Symptôme : `train_s` ×2 à R constant, sans erreur, VRAM à 98,5 % pour
~16–18 Go de tenseurs. Cause probable : cache + fragmentation de
l'allocateur PyTorch, avec pagination silencieuse WDDM vers la RAM
partagée (pas d'OOM franche comme sur Linux). Un processus frais repart
vite → la pression croît avec la durée de vie du processus.

**Exigences proposées pour les futurs prompts d'entraînement :**
1. Tout script d'entraînement logue par pas `memory_allocated`,
   `reserved`, `max_allocated` + phase — aucun correctif mémoire sans
   attribution du pic.
2. Plafonner : `set_per_process_memory_fraction(~0,90)` (échec franc
   plutôt que ralentissement insidieux) + `expandable_segments:True` +
   `empty_cache()` après chaque `save_ckpt`.
3. Le contrôle d'admission **par taille de prompt** est hors sujet en
   entraînement (Ko vs Go) ; le réserver au serving/évaluation (J5).
4. Prévoir le « budget mémoire » comme le budget tokens/GPU : pic mesuré
   < plafond, avec marge, avant tout run long.

## 4. Défauts mineurs à corriger (prompts suivants)

- `pilot_j4.py` : double sauvegarde du dernier pas quand
  `steps % ckpt_every == 0` → `FileExistsError` rattrapée, sortie code 1,
  checkpoint intact. Garder la sauvegarde de boucle, rendre la finale
  conditionnelle (`if not exists`).
- Métriques : `tokens_seq` et compteurs ne couvrent que le processus
  courant (remis à zéro aux reprises) — les rendre cumulatifs via
  `budget.json` du checkpoint repris.

## 5. Environnement d'entraînement : kernels de référence (ajout validé 2026-10-15)

- `causal_conv1d` et `flash-linear-attention` absents du Python Windows
  natif → fallbacks PyTorch de référence (corrects, « much slower »).
  Pas de wheels Windows : installation = compilation source fragile.
- Une part du ralentissement des pas (`rollout_s`, `train_s`) leur est
  imputable, pas seulement à la pression mémoire (§3).
- Exigence : **avant tout run long, reconstruire l'environnement
  d'entraînement côté WSL/Linux avec kernels compilés** (gain estimé
  2–5×). Ne jamais changer la pile numérique pendant une mesure (la
  pile lente mais constante du pilote J4 a été conservée pour
  l'évaluation).

## 6. En attente (hors périmètre de ce brief)

Score de rétention vs B0 sans IFEval (0,3376 ; paliers 0,2363 / 0,3038),
tests de non-régression, audit final, commit unique.
