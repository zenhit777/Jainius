#!/usr/bin/env bash
# Vérification d'environnement Jainius.
#
# À exécuter :
#   - en début de session de travail ;
#   - AVANT toute mesure de performance (preuve "GPU libre") ;
#   - AVANT toute installation de moteur de serving/évaluation.
#
# Sortie destinée à être archivée dans les rapports d'audit
# (docs/audits/<audit-id>/) comme preuve de l'état de l'environnement.
set -u

echo "== date (UTC) =="
date -u

echo "== GPU =="
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv 2>/dev/null || echo "nvidia-smi indisponible"

echo "== Processus GPU (doit être VIDE avant une mesure) =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
  || echo "aucun process GPU / nvidia-smi indisponible"

echo "== Conteneurs Docker =="
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null \
  || echo "docker indisponible"

echo "== Images Docker — moteurs de serving (SGLang, vLLM, llmd, Qdrant) =="
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>/dev/null \
  | grep -Ei 'sglang|vllm|llmd|qdrant' || echo "aucune image moteur trouvée"

echo "== venvs Python (/home/sjteb/.venvs) =="
for v in /home/sjteb/.venvs/*/bin/python; do
  if [ -x "$v" ]; then
    ver=$("$v" -c 'import torch, transformers; print(torch.__version__, transformers.__version__)' 2>/dev/null \
      || echo "sans torch/transformers")
    echo "$v : $ver"
  fi
done

echo "== Python système =="
python3 -c 'import torch, transformers; print(torch.__version__, transformers.__version__)' 2>/dev/null \
  || echo "sans torch/transformers"

echo "== Cache HF — modèles Jainius =="
ls -d ~/.cache/huggingface/hub/models--Qwen--* 2>/dev/null || echo "aucun modèle Qwen en cache"
