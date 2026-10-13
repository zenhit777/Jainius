# DECISIONS — registre des décisions de Jainius

Source d'autorité n°2 (après `docs/PRD.md`). **Une décision VERROUILLÉE ne se rouvre pas et ne se redemande pas.**
Avant toute question à l'utilisateur, chercher ici. Toute réponse de l'utilisateur est écrite ici immédiatement.

Statuts : **VERROUILLÉE** (dite par l'utilisateur) · **PAR DÉFAUT** (recommandation appliquée tant que l'utilisateur ne dit pas le contraire ; réversible) · **CONSERVÉE** (décision antérieure du dépôt, toujours valable).

## A. Décisions verrouillées

| ID | Décision | Remplace |
|---|---|---|
| **D-01** | Usage **privé et non commercial**. Les checkpoints sont privés. La reprise du code FLARE (kernels, fork SGLang) est autorisée sous PolyForm Noncommercial 1.0.0 : conserver `LICENSE` et `Required Notice` dans tout répertoire contenant du code FLARE. Toute publication du dépôt ou d'un checkpoint exige un réexamen préalable. Les licences des jeux de données ne bloquent pas (usage privé). | LIC-1 « ouverte », M0-D2, AGENTS §4.2 (« vérification préalable de la licence ») |
| **D-02** | **Un seul mécanisme de diffusion : celui de FLARE** (double flux clean/noisy, masques, planification des états GDN, convention de logits `shifted` avec jeton graine, décodage Diffusion-Trust). La convention `current` est **gelée** (code conservé, plus étendu), pas supprimée. | M0-D1, AGENTS §4.3, cahier technique D-04/D-05/D-06 |
| **D-03** | **Un seul mécanisme d'entraînement : la distillation on-policy d'OPDLM** (rollouts de l'étudiant, états intermédiaires, cibles de l'enseignant AR gelé, KL enseignant → étudiant, curriculum de longueur, prompts seuls). On ne reprend ni l'objectif `L_AR + L_diff` de FLARE, ni la conversion d'attention d'OPDLM. | AGENTS §4.2 (« OPDLM voie privilégiée pour la conversion ») |
| **D-04** | Cible de serving : **le fork SGLang de FLARE**, utilisé tel quel. Pas de portage sur SGLang amont, pas de serveur propre. | cahier technique §14.3 (options A/B/C) |
| **D-05** | Entraînement **très léger**, dans l'ordre de grandeur d'OPDLM (≈ 0,05–0,08 Md tokens). Aucun SFT sur corpus. | — |
| **D-06** | Débit **comparable à DiffusionGemma**, défini en accélération par rapport au modèle source servi en autorégressif (PRD §4). | A-006 seul (≥ 1,5×) |
| **D-07** | Les STM 1, 4, 5, 6, 7 sont **hors périmètre**. d-OPD peut être étudié après J4, uniquement sur une mesure. | cahier technique §10.9 |
| **D-08** | Modèle de développement : **Qwen3.5-0.8B**. Qwen3-0.6B reste un banc d'essai auxiliaire. Matériel : RTX 5090 32 Go, WSL2, Docker (voir `docs/environnement.md`). | — |

## B. Décisions antérieures conservées

- Sémantique d'état GDN du flux noisy (`docs/decisions/M2-contrat-etat-GDN.md`) : c'est celle de FLARE, validée contre son kernel (T-GDN-03, PASS).
- B0 = Qwen3.5-0.8B servi en autorégressif, sans MTP (`M0-B0-plain-seul.md`). MTP est en développement amont (FROZEN_KV_MTP, échec du spike).
- Règles Docker/WSL, GPU vérifié vide avant toute mesure, aucun `git push` (AGENTS §11, §13).

## C. Arbitrages en attente — réponse PAR DÉFAUT appliquée

À partir de ce registre, ces 7 questions du compte rendu du 13 octobre sont **closes**. L'utilisateur peut contredire n'importe laquelle ; sinon elles ne reviennent pas.

| # | Question | Réponse par défaut | Raison |
|---|---|---|---|
| 1 | Seuil E0-b : 0,5 ou 0,1 ? | **0,1**, puis T-LOSS-05 **gelé** (plus aucun travail dessus) | Le seuil de 0,1 passe déjà (0,069) et détecte le défaut `roll(1)` que 0,5 laissait passer. Le test compare à l'architecture OPDLM, qui n'est plus la cible (D-02). |
| 2 | Budgets A-008 / A-009 | **Adoptés** : pic mémoire ≤ 28 Go (carte de 32 Go) ; pilote ≤ 24 h-GPU, ≤ 1 passage de données | Cohérents avec G3. Le checkpointing d'activations est à implémenter avant toute séquence > 256 (au plus tard en J3). |
| 3 | Version de Python (H-06) | **3.12 / 3.13 acceptées** pour l'entraînement tant que les tests passent. Le serving FLARE s'exécute dans son propre environnement isolé (Docker). | Reconstruire en 3.11 ne sert aucun objectif. |
| 4 | IFEval : implémenter 2 contraintes ou exclure ? | **Exclure IFEval du composite** pour l'instant ; recalculer B0 sans lui à partir des résultats existants | 41 / 541 invites impossibles à satisfaire aujourd'hui ; l'implémentation ne sert aucun objectif G1–G5. À rouvrir seulement si G4 l'exige. |
| 5 | Téléchargement des données OPDLM malgré KodCode (licence NC possible) ? | **Oui**, télécharger (1,72 Go ; 786 Go libres). Consigner les licences dans le manifeste, sans filtrage. | D-01 : usage privé. |
| 6 | Prérequis serving (lecture de l'état GDN : mode 0, 1 ou 2) | **Remplacé par le jalon J2** : on ne choisit pas un mode, on prouve la parité avec le serving **natif** du fork (mode 2 puis 1) sur un checkpoint FLARE publié | Le fork est la cible (D-04) : c'est à notre forward de s'y conformer. |
| 7 | `L_max` = 512 ? | **512 pour le pilote**, tant que le checkpointing n'est pas implémenté ; relever ensuite | Mesuré : 1024 → mémoire insuffisante. |

## D. Questions réellement ouvertes (l'utilisateur doit répondre)

| # | Question | Recommandation |
|---|---|---|
| Q1 | Valider les seuils de débit G5 : minimal ≥ 2×, objectif ≥ 3× (PRD §4) | Valider |
| Q2 | Taille de bloc de départ | 4 (éprouvée par FLARE), puis essayer 8 et 16 après J4 |
