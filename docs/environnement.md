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
| `lmsysorg/sglang:dev-cu13` (47.4 Go) | moteur de serving SGLang (cible JAI-SGLANG) | image présente, non démarrée |
| `lmsysorg/sglang:latest` (52.2 Go) | idem, tag stable | image présente, non démarrée |
| `vllm/vllm-openai:latest` (29.9 Go) | moteur de serving vLLM (comparaison) | image présente |
| `qdrant/qdrant:latest` | base vectorielle docs | **en cours** (ports 6333-6334) |
| `nvidia/cuda:12.6.0-base-ubuntu22.04` | base CUDA | présente |
| (divers `llmd`) | serving distribué llmd | conteneurs arrêtés |

**Règle** : vérifier `docker ps` / `docker images` AVANT d'installer tout
moteur de serving ou d'évaluation dans un venv. Une image existante prime
sur une installation pip.

## 3. Environnements Python

| Environnement | Python | torch | transformers | Usage |
|---|---|---|---|---|
| Système `python3` | 3.12.3 | 2.12.0+cu130 | 5.10.2 | usage général (pas de pytest) |
| `/home/sjteb/.venvs/opdlm-parity` | 3.13.16 | 2.11.0+cu128 | 4.52.4 (trop vieux pour `qwen3_5`) | parité OPDLM (T-LOSS-05) |
| `/home/sjteb/.venvs/flare-oracle` | — | 2.10.0+cu128 | 4.57.1 + tyro + fla 0.5.1 | oracle FLARE (T-GDN-03) |
| `/home/sjteb/.venvs/jainius-check` | 3.12.3 | 2.12.0+cu130 | 5.10.2 | vérifications GPU courantes (pytest, accelerate, tyro, datasets) — venv d'analyse, NON l'environnement de référence |
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
