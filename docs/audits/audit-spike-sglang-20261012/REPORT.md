# Rapport de spike — SGLang amont + Qwen3.5-0.8B (`audit-spike-sglang-20261012`)

Objectif (M0, spike de serving §18) : vérifier si SGLang amont charge
Qwen3.5-0.8B en AR (B0-plain) et exploite ses têtes MTP (B0-MTP),
sans fork ni installation pip.

## Environnement

- Images Docker présentes : `lmsysorg/sglang:dev-cu13`
  (sglang 0.0.0.dev1+gf825d7293, torch 2.13.0+cu130, CUDA 13.0) et
  `lmsysorg/sglang:latest` (52,2 Go).
- Lancement : `docker run --gpus all -v
  /home/sjteb/.cache/huggingface:/root/.cache/huggingface -p 30000:30000
  lmsysorg/sglang:<tag> python -m sglang.launch_server --model-path
  Qwen/Qwen3.5-0.8B --host 0.0.0.0 --port 30000 [flags MTP]`.
- GPU vérifié vide avant mesures (`scripts/check_env.sh` →
  `_artefacts/env-check-20261012.txt`, aucun process compute).

## Résultats

### B0-plain (AR) — PASS

- Chargement natif : `Qwen3_5ForConditionalGeneration` reconnu,
  `load_weight=1.48 s`, `scheduler_e2e=35.29 s`, serveur prêt en ~55 s.
- Couches récurrentes hybrides reconnues (`mamba usage: 0.01` dans les
  logs de batch), CUDA graphs capturées (prefill 21.4 s / decode 2.1 s).
- `/v1/chat/completions` 200, génération cohérente (preuve par l'absurde
  de √2, 256 tokens).
- Débit mesuré (GPU vide, temperature 0) :
  - **535 tok/s** en bs=1 (1024 tokens en 1.92 s) ;
  - **1507 tok/s** agrégé à C=4 (4×512 tokens en 1.36 s).
  - Référence : 31 tok/s avec transformers 5.10.2 bs=1 (même machine)
    → SGLang amont ≈ **×17** en bs=1.

### B0-MTP (FROZEN_KV_MTP) — FAIL (WIP amont)

- Le checkpoint possède bien 15 poids MTP (`mtp.layers.0.*`) et SGLang
  amont contient un runner dédié `models/qwen3_5_mtp.py`
  (`EntryClass = [Qwen3_5ForCausalLMMTP]`) + `frozen_kv_mtp_utils.py`.
- `--speculative-algorithm FROZEN_KV_MTP` échoue sur **les deux images** :
  1. crash d'init `compute_num_reserved_tokens` :
     `speculative_eagle_topk * speculative_num_steps` sur `None`
     (aucune valeur par défaut pour MTP) ;
  2. contourné avec `--speculative-eagle-topk 1 --speculative-num-steps 1`
     : le serveur démarre mais les requêtes MTP **timeoutent**
     (`TimeoutError` dans `_wait_one_response`, HTTP 500) ;
  3. cause racine documentée par les développeurs eux-mêmes dans
     `speculative/spec_info.py` :
     `# FIXME(kpham_sgl): Remove FROZEN_KV_MTP here once we have
     established support for it in the scheduler` — `is_eagle()` inclut
     FROZEN_KV_MTP alors que le support scheduler n'est pas établi.
- Conclusion : le MTP natif de Qwen3.5 n'est **pas fonctionnel** dans
  les images SGLang publiques testées (dev-cu13, latest). Aucun fork
  Jainius n'a été créé, aucun kernel écrit.

## Verdict pour M0

- **B0-plain : exécutable dès maintenant** via l'image existante
  (aucune installation nécessaire) — le runner de benchmarks et les
  scorers restent à implémenter (`eval/harness.py` contient le protocole
  figé P-G/1.0 mais pas de runner de génération ni de scorers par
  benchmark).
- **B0-MTP : bloqué** sur SGLang amont (WIP). Options : attendre une
  version SGLang où FROZEN_KV_MTP est stabilisé, ou arbitrer un MTP
  alternatif (hors périmètre actuel).
- Le cahier technique §18 exige « B0-plain et B0-MTP à C = 1, 4, 8 sous
  SGLang amont » comme livrable M0 : avec B0-MTP bloqué, **M0 ne peut
  pas être clôturé sans arbitrage** (accepter B0-plain seul comme
  référence M2, ou différer M0/M2). Décision réservée au responsable.

## Reproduction

```bash
scripts/check_env.sh                                   # preuve GPU vide
docker run --gpus all --rm -v /home/sjteb/.cache/huggingface:/root/.cache/huggingface \
  -p 30000:30000 lmsysorg/sglang:dev-cu13 \
  python -m sglang.launch_server --model-path Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 --port 30000                        # B0-plain
# ajouter --speculative-algorithm FROZEN_KV_MTP --speculative-num-draft-tokens 1 \
#   --speculative-eagle-topk 1 --speculative-num-steps 1   # B0-MTP (échoue)
```
