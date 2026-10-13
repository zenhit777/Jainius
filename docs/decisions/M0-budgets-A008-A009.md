# Décision (PROPOSÉE) — Budgets A-008 (mémoire GPU) et A-009 (coût de conversion)

**Statut** : **PROPOSÉ** (clôture B5, 2026-10-13 — à arbitrer par
l'utilisateur avant toute expérience). Aucune expérience ne peut
dépasser ces plafonds sans ré-arbitrage.
**Base** : mesures P3/C1 sur RTX 5090 32 Go (31.8 utilisables), GPU
vérifié vide (< 1 Go), `logs/measure-c1.json`.

## A-008 — budget mémoire GPU

Mesuré (médianes 3 reps, batch 1, bf16, rollout exclu) :

- L=256 `train_step` : **20.6 s / pic 21.5 Go** ;
- L=512 `train_step` : **105.9 s / pic 40.5 Go** (dépasse la VRAM :
  paging WDDM sous WSL, temps non représentatif du coût réel) ;
- L=1024 : **OOM** (prouvé, 73 Go demandés) ;
- chemin quadratique legacy : OOM dès L=256 (référence, non utilisé).

Plafond fixé : **pic ≤ 28 Go** (marge 4 Go système/affichage sur 32 Go).

- L ≤ 256 : autorisé (21.5 Go < 28 Go).
- L = 512 : toléré uniquement avec paging (mesure non
  représentative) ou après re-mesure sous Linux natif ; le
  checkpointing d'activations (non implémenté, C1 NOT RUN) est
  requis pour en faire un régime nominal.
- L = 1024 : **interdit** sans checkpointing mesuré.
- Toute expérience dépassant 28 Go : arrêt obligatoire, réduire
  L/batch ou implémenter le checkpointing.

## A-009 — budget maximal de coût de conversion

- Données : 61 816 prompts (`divelab/opdlm_train_data`, 1.72 Go).
- Coût unitaire mesuré (sans rollouts) : ~21 s/pas (L256),
  ~106 s/pas (L512). Part des rollouts **inconnue localement**
  (C5 NOT RUN ; 82 % cité de NPD = hypothèse externe, pas mesure).
- Plafond pilote M3 fixé : **≤ 24 h-GPU** et **≤ 1 passage dataset**
  (≈ 6.2×10⁴ prompts ; tokens exacts chiffrés à C5, bornés par
  L_max=512 par séquence).
- Conversion complète : **interdite** avant C5 (profil rollouts) et
  checkpointing mesuré ; nouveau budget requis sur la base de ces
  mesures.

## Traçabilité

Toute consommation (h-GPU, tokens, passages dataset) est consignée
par run (compteurs JAI-OBS) et comparée à ces plafonds. Dépassement =
arrêt + arbitrage.
