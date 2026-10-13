# Audit J1 — serving réel du fork FLARE sur la RTX 5090 (2026-10-14)

Jalon PRD §6 J1 (objectifs G2, G5). Fork utilisé tel quel (D-04) : lu, copié
dans l'image, jamais modifié. Usage privé non commercial (D-01).

## Résultat

| Question | Réponse |
|---|---|
| Le fork tourne sur la 5090 | **oui** (mode diffusion par défaut) |
| Accélération FLARE vs autorégressif à C=1 | **0,35×** (66,7 / 189,9 tok/s par utilisateur, invites courtes) |
| Le 0,8B se charge | **oui**, et génère correctement (« 2 + 2 equals 4 ») |

Critères de sortie J1 (PRD §6) : fork construit et lancé en image Docker,
checkpoint FLARE publié chargé et généré, débit de référence mesuré — **tous
vérifiés. J1 terminé.**

## 1. Environnement (versions exactes)

Image construite : `jainius-flare-serve:j1` (32,8 Go),
`docker/Dockerfile.flare-serve`, contexte `_artefacts/oracles/FLARE`
(commit `6ca547a`, sous-arbre `eval/` : sglang 1604 fichiers Python,
`configs/hybrid_diffusion_shift_b3_g1.yaml`, `third_party/flashinfer`).
Base `nvidia/cuda:12.9.1-devel-ubuntu22.04`, Python 3.10.12, torch 2.9.1+cu128,
flashinfer 0.6.7.post3, sglang-kernel 0.4.1, transformers 5.5.4, fla 0.4.2
(`setup_eval_env.sh` : « All installed packages are compatible »,
« source-overlay imports: PASS »). Journal : `logs/docker-build.log`.
Preuve fumée : torch voit CUDA=True, RTX 5090, capacité (12, 0) = sm_120.

Trois itérations de construction, toutes au niveau configuration
(versions, paquets système, base CUDA — aucun code du fork touché) :
1. `setup_eval_env.sh: line 2: set: pipefail: invalid option name` — cause :
   fins de ligne CRLF du checkout (disque Windows). Contournement :
   normalisation LF dans la COPIE image uniquement (`sed 's/\r$//'`).
2. `ImportError: libnuma.so.1` puis `sgl_kernel CRITICAL` (pas de binaire
   sm_120, seul `sm100/common_ops.abi3.so`) — contournement : `libnuma1` +
   `python3.10-dev` dans l'image. Le serveur démarre alors (plus de CRITICAL).
3. `RuntimeError: FlashInfer requires GPUs with sm75 or higher` à la capture
   des graphes CUDA — cause : `compilation_context.py` exige CUDA ≥ 12.9
   pour sm_120 (`SM 12.x requires CUDA >= 12.9`), or base 12.8.
   Contournement : base `nvidia/cuda:12.9.1-devel` (torch 2.9.1 cu128
   inchangé). Capture réussie en ~19 s.

## 2. Checkpoints (collection HF `yuchen-zhu-zyc/hybrid-diffusion`)

| Checkpoint | Poids | Licence | SHA snapshot vérifié |
|---|---|---|---|
| `yuchen-zhu-zyc/HybridDiffusion-2B` (retenu, 2,39 Md params BF16) | ~4,8 Go | PolyForm Noncommercial 1.0.0 (usage privé) | `04e16a06` |
| `Qwen/Qwen3.5-2B` (référence AR même taille) | ~4,6 Go | Apache 2.0 | `15852e8c` |
| `Qwen/Qwen3.5-0.8B` (test H-11, déjà en cache) | 1,77 Go | Apache 2.0 | — |

Espace vérifié avant téléchargement : 733 Go libres (D:), 586 Go (/).
Cache partagé : `/home/sjteb/.cache/huggingface`, monté dans les conteneurs.

## 3. Lancement (chemin par défaut `serve.sh`)

```bash
scripts/check_env.sh                                   # GPU vide vérifié avant chaque série
docker run --gpus all -d --name jainius-flare-j1 --shm-size 8g -p 31000:31000 \
  -v /home/sjteb/.cache/huggingface:/opt/flare-cache/huggingface \
  -e HF_HOME=/opt/flare-cache/huggingface -e PORT=31000 \
  jainius-flare-serve:j1 \
  bash /opt/flare/eval/scripts/serve.sh diffusion yuchen-zhu-zyc/HybridDiffusion-2B
```

Mode `diffusion` : `LowConfidenceShiftHybridDiffusion`,
`hybrid_diffusion_shift_b3_g1.yaml` (bloc 3 + graine, seuil 0,95),
TP=1, bf16, attention flashinfer, mem-fraction 0,8, requêtes max 32,
graphes CUDA [1, 2, 4, 8]. Prêt en ~60 s (`/v1/models` 200).
Chargement : `Qwen3_5DLLMForConditionalGeneration`, 4,27 Go, KV 2×8,71 Go,
GDN Triton + enveloppe dLLM. Référence AR : même commande sur
`lmsysorg/sglang:dev-cu13`, modèle `Qwen/Qwen3.5-2B`, port 31001.

## 4. Test fonctionnel (5/5 cohérentes, `logs/functional_5prompts_20261014.jsonl`)

Calcul « 84×3/2 » → `126` ; capitale → Paris ; fonction Python `square`
correcte ; liste numérotée de 3 planètes ; 2 phrases sur les récifs coralliens.

## 5. Mesures (mêmes 5 invites, max_tokens=128, température 0)

| C | Diffusion agrégé / par utilisateur | AR agrégé / par utilisateur | Latence p50/p95 diffusion | Pic GPU diffusion |
|---|---|---|---|---|
| 1 | 66,7 / **66,7** tok/s | 189,9 / **189,9** tok/s | 0,23 / 0,74 s | 27 334 MiB |
| 4 | 168,6 / 42,1 tok/s | 548,5 / 137,1 tok/s | 0,31 / 1,14 s | 27 490 MiB |
| 8 | 277,3 / 34,7 tok/s | 1 142,3 / 142,8 tok/s | 0,37 / 1,55 s | 28 027 MiB |

Compteurs `dllm_stats` (`/server_info`), par série (jetons / passes) :
C=1 : 117/113 (tpf ≈ 1,04) ; C=4 : 441/124 (≈ 3,56) ; C=8 : 942/153 (≈ 6,16).
Brut : `logs/measure_diffusion_20261014.json`, `logs/measure_ar_20261014.json`,
`check_env_*` avant/après chaque série.

Lecture : à C=1 le seuil 0,95 n'accepte qu'~1 jeton par passe sur ces
réponses courtes (~22 jetons/requête) ; la diffusion ne peut pas battre
l'autorégressif (1 jeton/passe par construction) dans ce régime.
Le seuil G5 minimal (≥ 2×, PRD §4) n'est **pas** atteint ici — attendu :
atteindre « niveau DiffusionGemma » suppose des générations longues et des
blocs plus grands (paramètre de configuration, après J4).

## 6. Test 0,8B (H-11)

`Qwen/Qwen3.5-0.8B` source chargé par le fork en mode diffusion sans erreur
(classe `Qwen3_5DLLMForConditionalGeneration`, 1,72 Go, caches GDN alloués,
graphes capturés), génération correcte. Journal : `logs/serve_08b_*.log`.

## 7. Limites

- Une seule passe par niveau, invites à réponses courtes (pas de série à
  réponses longues — proposée en J5).
- GPU « vide » = aucun process de calcul côté WSL ; ~1 Go WDDM (affichage)
  et le conteneur `qdrant` (CPU) restent actifs, sans contention mesurée.
- Avertissements non bloquants consignés : « No platform detected »,
  extensions C++ `fla` ignorées (torch < 2.11), `cutlass.cute.experimental`.
- Conteneurs J1 arrêtés et supprimés après figement des journaux ; GPU rendu
  libre (`logs/check_env_final_20261014.txt`). Images existantes intactes.
