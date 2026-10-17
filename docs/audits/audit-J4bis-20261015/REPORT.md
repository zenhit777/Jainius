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

## 9. OOM confirme sur prompts longs ; evaluation du ckpt 979 (279 updates)

- Fait etabli : 3e OOM (12 Mio, 28,42 Go alloues) vers step 980 = paire
  base[277] (la plus longue). Le reorder ne suffit pas : le transitoire
  backward etudiant B=2 L~256 depasse seul le plafond. Motif stable sur
  3 incidents : prompts les plus longs en fin de curriculum trie.
- Decision : echec franc accepte (plafond 90 % a joue son role), queue
  de 20 pas abandonnee. Le ckpt 979 (279 updates phase 2, budgets
  ~119k tokens reponse) est un resultat de pilote exploitable : les
  paliers se jugent sur la mesure, pas sur 300 pile.
- Evaluation lancee sur step_000979 (6 datasets, sans IFEval) + pytest.
  Travail memoire restant (micro-batch exact ou checkpointing
  d activations C.2) renvoye en prerequis J5 : l entrainement actuel
  ne passe plus les prompts >= ~180 tokens sous ce plafond.

## 10. Eval monolithe tuee sans trace ; relance par dataset (16/10)

- Fait etabli : l eval monolithe est morte apres chargement des poids,
  sans traceback ni reboot (mort sans trace, cf. environnement §10).
  Diagnostic foreground --limit 4 : EXIT 0 en 74 s -> pas un bug,
  un kill externe ( meme motif que le run pilote).
- Parade : eval par dataset (6 fichiers indep., sans porte &&) +
  scripts/aggr_eval.py (meme formule composite que local_runner).
  Chaine relancee en fond : 6 evals -> agregation -> pytest.

## 11. Decisions Q1/Q2 + eval 979 relancee sur GPU libre (16/10 ~23h30)

- D-09 (G5 : minimal ≥ 2×, objectif ≥ 3×) et D-10 (bloc 4, puis 8/16)
  validees par l'utilisateur, inscrites en DECISIONS.md (commit bd755d2).
- Chaine aggr_eval.py validee sur CPU : reagregation des 6 evals J4
  existantes -> composite sans IFEval 0,1213 a l'identique.
- GPU libere par l'utilisateur (llama-server/LM Studio quitte) : 1209 Mio
  utilises, aucun process compute (`check_env_eval979.log` archive).
  Test foreground `--limit 4` gsm8k sur ckpt 979 : EXIT 0, 87 s,
  score 0,0 — sorties degenerees repetitives confirmées sur GPU
  (`diag_gpu979.json` : boucles "1.1.1.1…", jamais `####`).
- Chaine complete relancee en fond (`run_eval_split.ps1`, shell
  sh_146a456ad0016ynfD5YrmEwJgx) : 6 evals -> agregation -> pytest.
  Verdict palier 1 (0,2363) a la fin du run : si atteint -> J5, sinon
  analyse de l'ecart residuel (4 lignes non conclusives).

## 12. Chaine fond tuee en 18 s (3e mort sans trace) ; relance detachee (17/10 ~01h05)

- Fait etabli : `night.log` fige a 23:35:56 (poids 100 % + fallbacks,
  aucun traceback, aucun `eval_j4b_*.json`), aucun python/powershell
  actif au controle. Pas de reboot ni de veille (journal System vide
  sur 8-12 h), GPU libre (1365 Mio). Meme signature que les 2 morts
  precedentes, mais le test foreground identique (meme ckpt, 2 min
  avant) sortait 0 : la taille du dataset est hors cause (mort avant
  toute generation), OOM improbable (1,6 Go de poids sur 32 Go libres).
  Cause non etablie.
- Parade : relance DETACHEE via `Start-Process` (survit a la session
  agent, en-tete horodate dans `night.log`) : python PID 35580, GPU
  24 % / 5,5 Go a +45 s — fenetre des 18 s depassee en generation.

## 13. Correctif structurel : chaine supervisee hors serveur d'agent (17/10 ~01h25)

- Cause identifiee des 4 morts sans trace : processus enfants du shell de
  l'agent, tues avec lui (redemarrages serveur — cf. §6 ; meme signature
  pour les autres, aucune trace OS). Le run relance a 01h04
  (`Start-Process`, python 35580) etait sain a 01h18 (CPU 821 s, GPU
  actif) mais restait dans l'arbre du shell : remplace par precaution.
- Correctif : `scripts/run_eval_chain.ps1` lance par la tache planifiee
  Windows `Jainius-EvalJ4bis` — totalement detachee du serveur d'agent
  (chaine parente verifiee : python -> powershell -> svchost/services ->
  wininit). Heartbeat 60 s (PID + GPU + dataset), idempotence par dataset
  (JSON valide saute), faulthandler, stdout/stderr par tentative,
  post-mortem evenements Windows, RestartCount 10 x 5 min, reprise au
  logon, duree illimitee. Dry-run valide avant deploiement.
- Verifications 01h25-01h30 : tache Running, python 43548 (gsm8k), GPU
  ~7,6 Go / 22-24 %, heartbeat regulier, `chain.log` alimente.
- L'ancien run (01h04) a ete arrete volontairement pour basculer sur la
  chaine supervisee ; gsm8k relance de zero (~20 min perdues, aucun
  dataset n'avait abouti).
- Controle utilisateur : arret/reprise sans perte documentes
  (`docs/environnement.md` §12) ; regle permanente `AGENTS.md` §11.1.

## 14. Resultats finaux du ckpt 979 : palier 1 non atteint (17/10)

- Chaine supervisee (§13) terminee seule, sans aucune mort : 6/6 datasets
  exit=0 (01h25:46 -> 08h30:46 ; gsm8k 3h23, math500 1h16, mmlu 1h08,
  mmlu_pro 1h04, humaneval 4 min, mbpp 10 min), agregation exit=0, pytest
  71 passed / 2 skipped / 1 xfailed (151 s). `LastTaskResult` 0 ;
  heartbeat fige a 08h33:13 ; aucun python residuel ; GPU libre. Duree
  totale 7h08. Premier run long de bout en bout depuis le 15/10 : le
  detachement (AGENTS.md §11.1) a tenu.
- Comparabilite stricte : protocole P-G/1.0, hash `8bd3da29` identique a
  la re-agregation J4 ; T=0, 1 repetition ; moteur ckpt-local.
- Scores (resultat mesure, `_artefacts/j4b/eval_j4b.json`) :

  | dataset | J4 (699) | J4-bis (979) |
  | --- | --- | --- |
  | gsm8k | 0,0531 | 0,0106 |
  | math500 | 0,0560 | 0,0140 |
  | mmlu | 0,3188 | 0,2219 |
  | mmlu_pro | 0,1267 | 0,0767 |
  | humaneval | 0,0854 | 0,0305 |
  | mbpp | 0,0880 | 0,0400 |
  | composite sans IFEval | 0,1213 | **0,0656** |

- Verdict : palier 1 (0,2363) NON atteint. Le 979 est inferieur a J4
  (-0,0557) et a B0-plain (0,3376) sur les six datasets. J5 non demarre ;
  palier 2 (0,3038) inchange.
- Nature de l'ecart (fait etabli, echantillons d'eval) : J4 sortait des
  reponses structurees mais sans marqueur final (`####`, lettre) ; le 979
  degenere en boucles repetitives (`1.1.1.1…`, `0000…`, `2222…`, puces
  vides, commentaires de code dupliques), motif retrouve sur les six
  datasets y compris les QCM. La perte d'entrainement baissait pendant ce
  temps (moyenne par fenetres de 50 pas : 2,10 -> ~1,45-1,35 ; lr encore
  5,8e-06 au pas 978) : l'effondrement est qualitatif, invisible sur la
  courbe de perte du pilote.
- Fait etabli (log pilote + relecture du code) : le lr est re-rechauffe
  a CHAQUE reprise. `pilot_j4b.py` (l.200-203) boucle `range(args.steps)`
  avec `lr_schedule(k, args.steps)` : schedule local a l'invocation, donc
  warmup 5 pas rejoue a chaque relance. Sequences 2 -> 10e-06 observees
  aux pas 700, 900, 950 et 953. Les incidents (§6-9) ont decoupe la
  phase 2 en 4 runs courts : AUCUN ckpt n'est anneale — lr au pas sauve :
  9,51e-06 (749), 7,93e-06 (799), 5,67e-06 (849), 3,36e-06 (899),
  6,02e-06 (949), 5,57e-06 (952), 5,84e-06 (978). En regard,
  `pilot_j4.py` (l.176-179) reprend `range(start_step, args.steps)` sur
  k GLOBAL : le schedule J4 est continu a travers ses reprises (300/400 —
  verifie : 6,56e-06 et 4,54e-06 suivent la cosine de total 700) et
  step_000699 est anneale a 1,00e-06. La comparaison J4/979 oppose donc
  un ckpt en fin d'annealing a des ckpt pris en plein regime (~6-9e-06) :
  l'ecart de protocole est etabli (distinct de l'effet des prompts).
- Hypotheses (a tester, non etablies) : H1 (principale) la degradation
  du 979 reflete des poids en milieu d'annealing (aucun ckpt anneale en
  phase 2) ; H2 ecart train (suffixes de format, sans role `system`) vs
  eval (consignes systeme), deja present en J4 ; H3 R=64 constant.
  Discrimination sans nouvel entrainement : generation courte par ckpt
  (749/799/849/899/949/952) — correlation boucles vs lr de sauvegarde
  (899, le plus froid, doit etre le moins degenere si H1) ; puis, si H1
  se confirme, reprise courte depuis le 979 avec schedule global continu
  terminant a 1e-6 (paires courtes, marge OOM) avant re-evaluation.
- Post-run : tache planifiee `Jainius-EvalJ4bis` desactivee (plus de
  relance au logon) ; reprise possible par `Enable-ScheduledTask` (les
  datasets valides seraient sautes, l'agregation/pytest seraient refaits).
