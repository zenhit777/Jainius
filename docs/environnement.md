# Environnement d'exécution Jainius

Inventaire des services, conteneurs, environnements Python et règles de
vérification. Document à tenir à jour : toute information d'environnement
donnée par l'utilisateur est écrite ici immédiatement.

Dernière vérification : 2026-10-12 (`scripts/check_env.sh`).

## 1. Matériel

- Station WSL2 (Linux 6.18, x64), NVIDIA RTX 5090 (32 Go, sm_120/Blackwell).
- Règle : toute mesure de performance se fait sur GPU vérifié vide
  (`nvidia-smi --query-compute-apps` sans process), sortie archivée.

## 2. Conteneurs Docker (WSL)

| Image | Usage | État (2026-10-12) |
|---|---|---|
| `lmsysorg/sglang:dev-cu13` (47.4 Go) | moteur de serving SGLang (cible JAI-SGLANG) | **spike 2026-10-12 : AR natif Qwen3.5-0.8B fonctionnel (535 tok/s bs=1, 1507 tok/s C=4) ; MTP FROZEN_KV_MTP WIP amont** |
| `lmsysorg/sglang:latest` (52.2 Go) | idem, tag stable | même verdict MTP (WIP amont, FIXME développeur) |
| `vllm/vllm-openai:latest` (29.9 Go) | moteur de serving vLLM (comparaison) | image présente, non testé |
| `qdrant/qdrant:latest` | base vectorielle docs | **en cours** (ports 6333-6334) |
| `nvidia/cuda:12.6.0-base-ubuntu22.04` | base CUDA | présente |
| (divers `llmd`) | serving distribué llmd | conteneurs arrêtés |

### Spike SGLang — Qwen3.5-0.8B (2026-10-12, `docs/audits/audit-spike-sglang-20261012/REPORT.md`)

- **B0-plain (AR) : PASS** — chargement natif (~55 s), couches
  récurrentes hybrides reconnues (`mamba usage`), CUDA graphs,
  `/v1/chat/completions` 200. Aucune installation ni fork requis.
- **B0-MTP (FROZEN_KV_MTP) : FAIL** — WIP amont sur dev-cu13 et
  latest : crash d'init `compute_num_reserved_tokens`
  (params EAGLE à `None`), timeout worker même contourné ;
  `spec_info.py` : « FIXME(kpham_sgl): Remove FROZEN_KV_MTP
  here once we have established support for it in the scheduler ».
- Lancement : `docker run --gpus all --rm
  -v /home/sjteb/.cache/huggingface:/root/.cache/huggingface
  -p 30000:30000 lmsysorg/sglang:dev-cu13
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-0.8B
  --host 0.0.0.0 --port 30000`.

**Règle** : vérifier `docker ps` / `docker images` AVANT d'installer tout
moteur de serving ou d'évaluation dans un venv. Une image existante prime
sur une installation pip.

## 3. Environnements Python

| Environnement | Python | torch | transformers | Usage |
|---|---|---|---|---|
| Système `python3` | 3.12.3 | 2.12.0+cu130 | 5.10.2 | usage général (pas de pytest) |
| `/home/sjteb/.venvs/opdlm-parity` | 3.13.16 | 2.11.0+cu128 | 4.52.4 (trop vieux pour `qwen3_5`) | parité OPDLM (T-LOSS-05) |
| `/home/sjteb/.venvs/flare-oracle` | — | 2.10.0+cu128 | 4.57.1 + tyro + fla 0.5.1 | oracle FLARE (T-GDN-03) |
| `/home/sjteb/.venvs/jainius-check` | 3.12.3 | 2.14.1+cu130 | 5.10.2 | vérifications GPU courantes (pytest, accelerate, tyro, datasets) — venv d'analyse, NON l'environnement de référence |
| `requirements-lock.txt` (référence ADR-003) | — | 2.11.0+cu128 | 5.17.0 | **environnement de référence à reconstruire pour les mesures définitives (B0)** |

## 4. Modèles en cache HF

- `Qwen/Qwen3.5-0.8B` (1.77 Go) — modèle source Jainius, en cache.
- Autres modèles Qwen en cache (3.5-27B, 3.6, etc.) — hors périmètre Jainius.

## 5. Accès réseau

- Hugging Face accessible (modèles et datasets, vérifié 2026-10-12).
- Les 7 datasets du protocole P-G/1.0 sont accessibles (200) :
  gsm8k, MATH-500, mmlu, MMLU-Pro, IFEval, `openai/openai_humaneval`,
  mbpp. Attention : `openai/humaneval` (sans préfixe de repo) est gated (401).

## 6. Commandes de vérification

```bash
scripts/check_env.sh            # état complet (GPU, Docker, venvs, cache)
docker ps                       # conteneurs en cours
docker images | grep -Ei 'sglang|vllm|llmd|qdrant'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv  # doit être vide
```

## 7. Leçons enregistrées

- L'information « SGLang disponible via Docker WSL » avait été donnée dans
  une session antérieure sans être persistée → elle a failli entraîner une
  installation pip inutile de SGLang. Désormais §11 AGENTS.md + ce document.
- Les audits des 2026-10-12/13/14 ne contiennent pas de preuve « GPU vide ».
  Leurs mesures de performance (P3) sont à re-mesurer sur GPU vérifié vide
  avant toute comparaison. Les résultats de correction numérique sont
  insensibles à la contention GPU et restent valides.

## 8. Jalon J1 — serving fork FLARE (terminé le 2026-10-14)

Rapport : `docs/audits/audit-J1-20261015/REPORT.md` (+ `logs/` : check_env
avant/après chaque série, journaux serveurs, JSON bruts).

- Checkout oracle `_artefacts/oracles/FLARE` (commit `6ca547a`, lecture seule) :
  le sous-arbre de serving est présent (`eval/sglang` 1604 fichiers Python,
  `eval/configs/hybrid_diffusion_shift_b3_g1.yaml`, `eval/third_party/flashinfer`) ;
  le lien de données de compilation FlashInfer (`.../flashinfer/data`) est absent
  (matérialisé par `setup_eval_env.sh`). Constat : l'index git du checkout porte
  des suppressions étagées (~3400 fichiers) alors que les fichiers de travail
  existent — on n'y touche pas (D-04 : fork utilisé tel quel).
- Images SGLang **amont** existantes (`dev-cu13`, `latest`) : PAS le fork.
  Image dédiée en construction : `jainius-flare-serve:j1`
  (`docker/Dockerfile.flare-serve`, base `nvidia/cuda:12.8.0-devel-ubuntu22.04`,
  Python 3.10, torch 2.9.1 cu128, flashinfer 0.6.7.post3, sglang-kernel 0.4.1,
  transformers 5.5.4 — versions épinglées de `setup_eval_env.sh`).
- Point d'attention : support sm_120 (RTX 5090 Blackwell) par torch 2.9.1 cu128 /
  sglang-kernel 0.4.1 à vérifier (torch 2.14.1+cu130 du venv `jainius-check`
  voit déjà le GPU et CUDA=True). Erreurs exactes consignées dans
  `docs/audits/audit-J1-20261015/REPORT.md`.
- Checkpoints publiés (collection HF `yuchen-zhu-zyc/hybrid-diffusion`,
  licence PolyForm Noncommercial 1.0.0 — usage privé D-01) :
  `HybridDiffusion-2B` (~4,8 Go, retenu), `-4B`, `-9B`. Référence
  autorégressive même taille : `Qwen/Qwen3.5-2B` (Apache 2.0, ~4,6 Go).
- Résultat : image `jainius-flare-serve:j1` (32,8 Go, base CUDA 12.9 —
  le sm_120 exige CUDA ≥ 12.9 côté FlashInfer JIT) ; serveur diffusion
  (bloc 3 + graine, seuil 0,95) prêt en ~60 s ; 5/5 fonctionnel ;
  C=1 : 66,7 tok/s diffusion vs 189,9 AR (0,35×, tpf ≈ 1,04 sur réponses
  courtes — G5 ≥ 2× non atteint dans ce régime) ; 0,8B source chargé et
  correct dans le fork. Conteneurs J1 supprimés, images amont intactes.

## 9. Gouvernance mémoire GPU — Windows/WDDM (2026-10-14, pilote J4)

Constat : sous Windows, le pilote WDDM **pagine silencieusement vers la
RAM partagée** quand la VRAM est pleine, au lieu de lever une OOM franche
(comme sur Linux). Symptôme observé : `train_s` ×2 à R constant, sans
erreur, VRAM à 98,5 % avec un besoin en tenseurs ~16–18 Go sur 32 Go.
`nvidia-smi` affiche le **réservé** (cache allocateur + fragmentation),
pas l'**alloué** — les deux doivent être distingués.

Règles (applicables à tout run d'entraînement local) :
- **Mesurer avant d'optimiser** : logger par pas `memory_allocated`,
  `memory_reserved`, `max_memory_allocated` + phase ; aucun correctif
  mémoire sans attribution du pic.
- **Plafonner** : `torch.cuda.set_per_process_memory_fraction(~0,90)` —
  l'échec devient franc (OOM Python) au lieu d'un ralentissement
  insidieux par pagination.
- **Défragmenter** : `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ;
  `empty_cache()` aux points connus (après chaque `save_ckpt`), pas à
  chaque pas.
- **Baisser le pic sous le plafond** : enseignant en micro-lignes ou sur
  CPU, checkpointing d'activations (DECISIONS C.2), sérialisation côté CPU.
- Le contrôle d'admission **par taille de prompt** n'est pas le levier
  pertinent en entraînement (un prompt = des Ko face à des Go d'états) ;
  il se justifie côté serving/évaluation (budget KV-cache, J5).

## 10. Stabilité des runs longs — reboots Windows (2026-10-14)

- 07:29 + 07:32 : reboots initiés par Windows Update (MoUsoCoreWorker,
  TrustedInstaller) → mort du pilote initial à 353/700. 15:03 : reboot
  initié via StartMenu (utilisateur) → fin de session.
- Règle : avant tout run de plus d'1 h, vérifier qu'aucun reboot n'est
  programmé et mettre Windows Update en pause (Paramètres → Windows
  Update → Suspendre). Un reboot tue processus + contexte CUDA sans
  laisser de trace dans les logs applicatifs — diagnostic via
  `Get-WinEvent` (System : 1074, 6006, 6005), jamais de mise en veille
  constatée (aucun événement 42/107 le 2026-10-14).
- Morts sans reboot ni trace (stall pilote à 475, eval complète) :
  cause non établie ; parade = jobs sans `timeout`, découpage par étape
  avec résultats persistés, contrôle vivant (processus + VRAM +
  horodatage du log, jamais le log seul).
