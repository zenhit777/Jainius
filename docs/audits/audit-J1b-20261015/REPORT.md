# Audit J1b — le débit du fork FLARE dans le bon régime (2026-10-14/15)

Objectif servi : G5. Image réutilisée : `jainius-flare-serve:j1` (J1).
Fork tel quel (D-04), usage privé (D-01). Protocole : 30 invites longues
(~975–1 005 jetons/réponse, max_tokens=1024, T=0 sauf mention), C=1/4/8 ;
GSM8K : 50 premiers problèmes du test, consigne « Final answer: <nombre> »,
T=0. GPU vide vérifié avant chaque série (`check_env_j1b-*.txt`).

## Résultat

| Mode (FLARE-2B, C=1) | tok/s | tpf (jetons/passe) | vs AR même taille | GSM8K /50 |
|---|---|---|---|---|
| Diffusion défaut (seuil 0,95) | 95,0 | 1,16 | 0,32× | 28 (56 %) |
| Diffusion seuil 0,9 | 94,2 | 1,18 | 0,32× | 28 (56 %) |
| Diffusion seuil 0,8 | 94,8 | 1,21 | 0,32× | 23 (46 %) |
| Diffusion seuil 0,7 | 94,9 | 1,23 | 0,32× | 27 (54 %) |
| Diffusion 0,95 à T=1 (celle du papier) | 89,5 | 1,13 | 0,30× | — |
| AR-Trust self-spec b7_g4 publié | 270,6 | 2,03 | **0,92×** | 29 (58 %) |
| Référence AR Qwen3.5-2B (amont) | 294,5 | — | 1,00× | 33 (66 %) |

Concurrence (par utilisateur) : diffusion 95,0 → 71,8 → 56,9 ;
self-spec 270,6 → 172,8 → 168,6 ; AR 294,5 → 257,0 → 246,0.
Exactitudes à n=50 (±7 points à 1σ) : aucun écart significatif entre modes.

**Conclusion : non — le bloc de 4 de FLARE n'atteint pas 2× sur cette carte
en Diffusion-Trust (0,3×) ; seul l'AR-Trust s'approche de l'AR (0,9×).**

## Faits établis

- Le seuil de confiance est inopérant dans ce régime : 0,95 → 0,7 ne change
  ni le tpf (1,16 → 1,23) ni le débit (~95 tok/s), configs vérifiées une par
  une dans les journaux (`dllm_algorithm_config=/opt/flare-cfg/...`).
  Configs propres versionnées : `configs/hybrid_diffusion_shift_b3_g1_t*.yaml`.
- La température n'explique rien : T=1 donne 89,5 tok/s et tpf 1,13,
  identique à T=0. Le chemin Diffusion-Trust (bloc 3 + graine) commet
  ~1 jeton par passe ici, loin du ratio idéal B/S.
- Incident de mesure écarté : la première série « 0,9 » tournait contre
  l'ancien conteneur 0,95 (port occupé, START_EXIT=125) — détecté,
  conteneurs nettoyés avant chaque session, série rejouée (START_EXIT=0).
- Papier FLARE (§5.3, Fig. 9, Tab. 10, App. D.2.1) : débits sur **A100-80GB**,
  max_new_tokens=2048 + ignore_eos, T=1 — FLARE-2B GSM8K : 487 / 1 315 /
  2 087 tok/s à C=1/4/8 (mode d'échantillonnage non tagué dans Tab. 10,
  présenté comme débit de décodage parallèle). Notre écart à C=1 (~5×)
  s'explique par le GPU différent, le protocole différent (réponses ~1 000
  vs 2 048 forcés, API chat vs harnais fixe) et des noyaux moins optimisés
  sur sm_120 ; la mise à l'échelle C=1 → C=8 est comparable (×4,8 ici,
  ×4,3 papier). Référence qualité papier : GSM8K FLARE-2B 82,8–84,5 %
  (n=1 319) — notre contrôle n=50 n'a pas la puissance de détecter un écart.
- Bruts : `logs/measure_long_*.json`, `logs/gsm8k_*.json`,
  `logs/j1b-*_startup.log`, `long_prompts` et subset dans `_artefacts/`.

## Limites

- Une seule passe par série ; réponses ~1 000 jetons (pas 2 048 forcés,
  pas d'ignore_eos) ; T=0 sauf le point T=1.
- Self-spec mesuré en config publiée b7_g4 uniquement (variantes
  softmax_argmax/truncated_argmax non testées).
