# Audit J4-bis — remédiation format, phase 2 en cours (2026-10-15)

Objectifs servis : G3 (budgets), G4 (palier 1 ≥ 0,2363 = 70 % de 0,3376).
Décisions de session appliquées : prompts formatés (sans SFT), compteurs
cumulatifs + finale conditionnelle, gouvernance mémoire. Pile numérique
inchangée volontairement (§4). Un seul audit, un seul commit, pas de push.

## 1. Code (fait établi, vérifié)

- `src/jainius/train/remediation.py` (nouveau) : `FORMAT_SUFFIX` par domaine
  OPDLM (math → `####` + `\boxed{}` ; code → code seul ; science → lettre
  unique ou `####` ; chat → consigne générique), `build_training_text`
  (signature restreinte à question + domaine : réponses de référence
  structurellement inaccessibles), `merge_budgets`, `should_final_save`,
  `apply_memory_governance`, `mem_snapshot`.
- `scripts/pilot_j4b.py` (nouveau) : phase 2 depuis `_artefacts/j4/step_000699`
  (poids + optim + maîtres + RNG, `force=True` tracé), R=64 constant, warm
  restart 1e-5→1e-6, paires formatées, compteurs cumulés, finale
  conditionnelle, log mémoire par pas, `empty_cache()` après chaque
  `save_ckpt`, numérotation globale (700+).
- `scripts/pilot_j4.py` (J4, correctifs seuls, comportement J4 inchangé) :
  compteurs repris depuis `budget.json` (plus de remise à zéro),
  sauvegarde finale conditionnelle (plus de `FileExistsError`).
- Tests `tests/numerical/test_j4b_remediation.py` : **6 passés** (CPU).

## 2. Smoke GPU (résultat mesuré, `_artefacts/j4b/smoke`, 3 pas R=32)

- Reprise : empreinte de départ `c103d50e` = finale J4 (rapport J4) ;
  compteurs initiaux 83 200 / 28 800 / 152 232 / 300 (cumulatifs OK).
- Perte finie (1,64 / 2,07 / 2,10), gradients finis, empreinte modifiée
  (`ca1e336a`) : la boucle apprend. Checkpoint `step_000702` complet.
- Mémoire : alloué ~16,2 Go, réservé 24–25,6 Go / 32 Go, `train_s` stable
  (~5 s) : pas de pagination WDDM, plafond 90 % actif.
- Resmoke 1 pas : sortie 0, checkpoint complet. (Le code 1 du premier smoke
  venait du pipe PowerShell `Select-Object`, pas du script : `done` journalisé
  et checkpoint intact dans les deux cas.)

## 3. Environnement (fait établi)

- GPU : aucun process Python au lancement (seuls des processus d'affichage
  WDDM `[N/A]`, comme en J4 — même réserve).
- Pile Windows inchangée (torch 2.11.0+cu128, transformers 5.17.0,
  fallbacks de référence) : même réserve mémoire que J4, comparabilité
  J4 → J4-bis stricte (ablation sur les seuls prompts).
- WSL sondé (borne de temps) : `causal_conv1d` = sdist only (compilation
  CUDA requise), `flash-linear-attention` 0.5.2 = wheel pure-Python ;
  `flare-oracle` en transformers 4.57.1 (trop vieux pour `qwen3_5`).
  Reconstruction complète = compilation + revalidation : renvoyée en
  prérequis J5 (changement de pile interdit pendant une mesure).

## 4. Pilote phase 2 (en cours au commit)

Commande : `_artefacts/j4b/run_night.ps1` (pilot 300 pas R=64 seed 1,
ckpt/50 → éval 6 datasets sans IFEval → pytest), fond `sh_1406e23e10011`.
Prévision : +38 400 tokens réponse (total ~121 600, plafond 0,1 Md),
~2 h-GPU pilote. Dernier pas attendu `step_000999`.
Reprise si tué (reboot §10 non vérifiable ici) : relancer avec
`--from-ckpt` = dernier `step_*` de `_artefacts/j4b/phase2` (même commande,
`--steps` = restants ; config parente = `phase2/config.json`).
Écart assumé : rôle `system` absent à l'entraînement (comme J4) — la
remédiation ne porte que sur les consignes ; noté comme hypothèse
résiduelle si le palier manque encore.

## 5. Conclusion (au commit)

Scores J4-bis : **en attente** (pilote en cours) — palier 1 (0,2363)
ni atteint ni manqué à ce stade. Livré et validé : correctifs, prompts
formatés, gouvernance mémoire, pilote lancé avec chaîne d'évaluation
identique à J4 (comparaison ckpt-local vs J4-local sans biais moteur).
Prochaine étape : à la fin du run, comparer `eval_j4b.json` à J4 (0,1213)
et B0 (0,3376) ; si palier 1 → J5, sinon analyser l'écart résiduel.

## 6. Incident 16/10 ~01h40 : serveur relance, reprise depuis step_000899

- Fait etabli : le shell de fond a ete annule (redemarrage serveur), le
  processus python avec lui (aucun python actif, GPU libre, pas de checkpoint
  d urgence : kill sans exception). Dernier etat sain : step_000899 (19h31) ;
  pas 900-932 recalcules (deterministes : paires + seeds de rollout).
- Les 2 reboots Windows du 15/10 (13h14 utilisateur, 13h52 winlogon) sont
  ANTERIEURS au run (18h39+) : non lies a cet incident.
- Reprise : --from-ckpt step_000899 --steps 100 --pair-offset 200 (globals
  900-999, pas de rejeu des paires), meme seed/R/ckpt-every. Correctif
  --pair-offset trace dans pilot_j4b.py (commit suivant).

## 7. Incident 16/10 ~02h : double OOM CUDA, cause et correctif

- Fait etabli : ckpt step_000949 OK (01h58) ; puis OOM (12 Mio) dans le
  backward vers step 977 (fragmentation : 2,17 Gio libres mais non
  contigus), puis OOM (970 Mio) dans la sauvegarde d urgence.
  Dernier pas valide : 976. Pas d eval, pas de pytest.
- Cause : transitoire +3,2 Go de [m.detach().clone() for m in masters]
  dans save_ckpt + absence de purge (J4 survivait par pagination WDDM ;
  le plafond 90 % rend l echec franc, comme prevu au §9).
- Correctif : sauvegarde des maitres SANS clone (detach suffit, roundtrip
  couvert par test_ckpt_*), empty_cache() avant/apres chaque save et tous
  les 10 pas. Valide : 13 tests passent (dont roundtrip GPU avec maitres).
- Reprise : --from-ckpt step_000949 --steps 23 --pair-offset 277 (globals
  977-999, paires 277-299, sans rejeu ni trou).

## 8. Correction 16/10 : pas de fuite, pic transitoire ; erreur d offset rattrapee

- Sonde GPU : rollouts seuls stables (5,97 Go), pas complets stables a
  15,81 Go (P court). Pas de fuite : le run vit a ~99 % du plafond 90 %
  et les prompts les plus longs (paires triees croissantes) font
  deborder le transitoire backward. J4 ne survivait que par pagination.
- Correctif : cibles enseignant calculees AVANT le forward etudiant
  (pic = max au lieu de la somme, ~plusieurs Go gagnes ; seul t_full
  fp32 ~0,3 Go survit). Reordonancement seul, valeurs bit-identiques :
  suite numerique complete verte (71 passes, 2 ignores, 1 xfail).
- Comptabilite paires (seed 1) : persiste base[0..249] + [277..278]
  (urgence 952). Le runmort a consomme base[277..278] pour globals
  950-951 (la note du §7 annoncait a tort 977-999 : corrige ici, sans
  effet persiste au-dela du calcul perdu). Reprise depuis 952 :
  --pair-offset 250 --steps 47 (globals 953-999, 2 repetitions
  documentees base[277..278], negligeables).
