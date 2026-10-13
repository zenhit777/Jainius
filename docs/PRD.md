# PRD — Jainius

Version 1.0 · 14 octobre 2026 · **Source d'autorité n°1 du projet.**
En cas de conflit avec tout autre document, ce fichier l'emporte.

## 1. Objet

Jainius est une application qui **convertit un LLM autorégressif hybride (Qwen3.5 : couches softmax + Gated DeltaNet) en modèle de diffusion par blocs (dLLM)**, en conservant son raisonnement, avec un entraînement très léger, et dont le checkpoint est **servi nativement par le fork SGLang de FLARE** avec un débit de génération nettement supérieur à celui du modèle d'origine.

Usage : **privé, non commercial**. Les checkpoints produits ne sont pas distribués.

## 2. Principe : deux mécanismes, et rien d'autre

| | On prend | On ne prend pas |
|---|---|---|
| **FLARE** (STM 2) — *tout ce qui concerne la structure du modèle et son exécution* | Double flux clean/noisy · masque par blocs et isolation des documents · planification des états récurrents GDN (lecture de fin de bloc, état noisy initialisé depuis l'état clean) · convention de logits décalée (`shifted`, jeton « graine ») · décodage Diffusion-Trust (AR-Trust en option) · kernels GDN deux flux · fork SGLang (modèle `qwen3_5_dllm`, algorithmes de décodage) | Son objectif d'entraînement `L_AR + L_diff` (SFT sur ≈ 10 Md tokens de corpus Nemotron) |
| **OPDLM** (STM 3) — *tout ce qui concerne l'apprentissage* | Distillation on-policy : l'étudiant génère ses propres trajectoires de débruitage · états intermédiaires tirés dans ces trajectoires · cibles = distribution de l'enseignant AR gelé (même checkpoint) · perte KL(enseignant ‖ étudiant) sur les positions masquées · curriculum de longueur de rollout · prompts seuls, pas de réponses de référence | Sa conversion d'attention (bloc-causale), sa convention `current`, son moteur de rollout JetEngine |

Il y a **un seul mécanisme de diffusion** (celui de FLARE) et **un seul mécanisme d'entraînement** (la distillation d'OPDLM). Les STM 1, 4, 5, 6, 7 sont **hors périmètre**. Seul d-OPD (STM 7) pourra être étudié plus tard, uniquement si une mesure montre un défaut qu'il corrige.

Conséquence de conception : le décodage de FLARE est utilisé **à la fois** pour générer les rollouts d'entraînement et pour servir le modèle. L'étudiant est donc entraîné sur les états qu'il rencontrera à l'inférence.

## 3. Objectifs et critères de réussite

| ID | Objectif | Critère mesurable |
|---|---|---|
| **G1** | Convertir un Qwen3.5 hybride | Qwen3.5-0.8B converti de bout en bout (modèle de développement). Un autre Qwen3.5 hybride se déclare par configuration, sans réécriture. |
| **G2** | Compatibilité native avec le fork SGLang de FLARE | Le checkpoint se charge et se sert avec `launch_server` du fork, **sans modifier le fork**, avec son algorithme de décodage par défaut. Les logits du forward d'entraînement et ceux du serving concordent (test de parité). Sorties correctes à concurrence 1, 4 et 8. |
| **G3** | Entraînement très léger | Conversion pilote ≤ 24 h-GPU sur la RTX 5090 et ≤ 0,1 Md tokens (OPDLM : 0,048 à 0,076 Md tokens, STM 3). Aucun corpus SFT. |
| **G4** | Rétention du raisonnement | Palier 1 ≥ 70 % du score composite du modèle source · palier 2 ≥ 90 % · palier 3 (98 %) = objectif final. Même protocole d'évaluation pour source et converti. |
| **G5** | Débit comparable à DiffusionGemma | Voir §4. |

## 4. Cible de débit (G5) — à valider

**Ce que DiffusionGemma annonce** (Google, 10 juin 2026 ; fiche HF) : jusqu'à 4× plus rapide que la génération autorégressive ; 15 à 20 tokens par passe avant (canevas de 256 tokens, 48 pas maximum, arrêt adaptatif) ; plus de 700 tokens/s sur RTX 5090, plus de 1 100 tokens/s sur H100 (FP8) à faible concurrence. Ce sont des annonces « jusqu'à », sur un modèle MoE de 26 B (3,8 B actifs), avec un protocole que nous ne maîtrisons pas.

**Pourquoi la comparaison absolue n'a pas de sens** : notre modèle fait 0,8 B ; son décodage autorégressif atteint déjà 535 tokens/s à concurrence 1 sur la RTX 5090 (spike du 12 octobre 2026). Le chiffre à comparer à DiffusionGemma est donc **l'accélération par rapport à notre propre référence autorégressive**, à protocole identique.

**Cible proposée (G5)**, mesurée sur la RTX 5090, GPU vérifié vide, concurrence 1, mêmes prompts, comparée au modèle source servi en autorégressif (B0-plain) :

- seuil minimal : **≥ 2×** en tokens/s par utilisateur ;
- objectif : **≥ 3×** ;
- référence DiffusionGemma : **jusqu'à 4×**, hors de portée avec des blocs de 4 tokens (voir ci-dessous).
- métrique complémentaire : tokens par passe avant (DiffusionGemma : 15–20).

**Limite structurelle à connaître.** Avec un bloc de 4 tokens (réglage éprouvé de FLARE), on ne peut pas dépasser 4 tokens par passe ; en pratique 2 à 3. Atteindre « niveau DiffusionGemma » suppose des blocs plus grands (8, 16 ou plus), au prix d'une rétention plus faible dans les publications (STM 3 : OPDLM-8B, 55,9 → 43,1 de moyenne en passant du bloc 4 au bloc 16 ; STM 7 : 46,9 avec d-OPD). La taille de bloc est donc un **paramètre de configuration** : on démarre à 4, on mesure tokens/passe, rétention et débit, puis on décide d'essayer 8 ou 16.

## 5. Non-objectifs

- Pas de pré-entraînement, pas de SFT, pas de RL.
- Pas de modification du tokenizer ni de l'architecture (hors ligne de masque).
- Pas de multimodal.
- Pas de nouveau serveur d'inférence : on utilise le fork de FLARE.
- Pas d'interface utilisateur.
- Pas de seuil ni de test sans lien avec G1–G5.

## 6. Jalons

Les anciens jalons M0–M6 sont retirés. Jalons dans l'ordre ; un jalon n'est « terminé » que si **tous** ses critères de sortie sont vérifiés.

| Jalon | But | Critère de sortie | Objectif servi |
|---|---|---|---|
| **J1 — Serving réel** | Prouver que le fork FLARE tourne sur la RTX 5090 (Blackwell) | Le fork se construit ou se lance (image Docker ou environnement isolé), charge un checkpoint FLARE publié (à vérifier : collection HF `yuchen-zhu-zyc/flare`), génère. Débit de référence mesuré. | G2, G5 |
| **J2 — Parité entraînement ↔ serving** | Lever le doute sur la lecture de l'état GDN (modes 0 / 1 / 2) | Notre forward deux flux de référence reproduit les logits du serving FLARE sur ce checkpoint publié, à tolérance explicite. Règle d'alignement des cibles sous `shifted` établie et validée numériquement (la KL enseignant/étudiant calculée sur le checkpoint FLARE publié doit être nettement plus faible avec le bon alignement qu'avec un mauvais). | G2, G4 |
| **J3 — Distillation on-policy réelle** | Rollouts par débruitage de l'étudiant (Diffusion-Trust), états par bloc, KL | Boucle complète sur 0,8B : perte finie, enseignant gelé, reprise exacte, checkpoint exportable. | G1, G3 |
| **J4 — Conversion pilote** | Premier checkpoint Jainius | ≤ 24 h-GPU, ≤ 0,1 Md tokens ; rétention mesurée contre B0 (palier atteint noté). | G3, G4 |
| **J5 — Serving du checkpoint Jainius** | Servir et mesurer | Critères G2 complets ; débit vs B0-plain (G5), tokens par passe. | G2, G5 |

## 7. Bilan de M0–M2 contre ce PRD (au 14 octobre 2026)

Constaté dans le dépôt (code lu), pas dans les comptes rendus.

| Acquis | Statut |
|---|---|
| Chargement du checkpoint, empreintes, tokenizer, harnais d'évaluation, B0-plain mesuré (0,3205), environnement Docker/WSL, spike SGLang AR (535 tok/s) | **Réutilisable tel quel** |
| Référence GDN deux flux (vectorisée) validée contre le kernel FLARE (T-GDN-03), masques, états, tests de fuite | **Cœur du projet — à conserver** |
| Perte KL par morceaux, enseignant gelé, curriculum, checkpoint/reprise | **Réutilisable** (l'alignement des cibles est à refaire sous `shifted`) |
| Convention `current` codée en dur (modèle, cibles avec `roll(1)`, boucle d'entraînement). `shifted` n'est pas implémentée (lève une erreur). | **À remplacer** |
| Rollout : décodage causal par blocs, **pas de débruitage** ; états d'entraînement = « tout masqué » dégénéré | **La distillation on-policy n'existe pas encore** (J3) |
| Parité avec le code OPDLM officiel (T-LOSS-05, E0) | **Gelée** : valide la plomberie de la perte, pas l'architecture visée |
| Fork FLARE jamais lancé ; le spike a testé SGLang amont | **Risque majeur non levé** (J1) |

## 8. Hiérarchie documentaire

1. `docs/PRD.md` (ce fichier)
2. `docs/DECISIONS.md` (décisions verrouillées)
3. `AGENTS.md` (règles de fonctionnement)
4. `docs/decisions/*.md` (décisions de détail ; un en-tête « OBSOLÈTE » fait foi)
5. `docs/ARCHITECTURE.md` (à produire après J3 : description de l'existant, vérifiée sur le code)
6. `docs/archive/` : l'ancien cahier des charges et l'ancien cahier technique, conservés pour l'historique et la traçabilité des tests. **Jamais une justification.**
