# Cahier des charges technique — Jainius

Version 1.0 · 9 octobre 2026 · Document de référence pour la conception, l'implémentation et la validation

## 1. Objet et vision du projet

Jainius a pour objectif de convertir un modèle de langage hybride autorégressif de la famille Qwen3.5 en un modèle de langage à diffusion (diffusion language model, dLLM), en conservant autant que possible ses capacités de raisonnement, tout en réduisant le coût de conversion et en permettant une exécution native dans SGLang.

Le projet ne consiste pas à réentraîner intégralement un nouveau modèle. Il consiste à transformer le mode d'apprentissage et de génération d'un checkpoint existant, puis à démontrer que les performances obtenues justifient cette transformation.

Le présent cahier des charges s'appuie sur le recueil des sept Spécifications Techniques Minimales (STM) fourni pour Jainius. Il transforme ces références scientifiques indépendantes en une proposition d'architecture, un plan d'implémentation et des critères d'acceptation communs. Les résultats publiés dans les STM sont des résultats expérimentaux, et non des garanties de performance pour Jainius.&#x20;

STM_7_PUBLICATIONS.md



### 1.1 Objectifs prioritaires

1. Préserver le raisonnement. Maintenir les performances du modèle source en mathématiques, en raisonnement général, en suivi d'instructions et en programmation.
2. Limiter l'entraînement. Éviter un préentraînement complet et privilégier une conversion et une distillation sur un budget de tokens réduit.
3. Exploiter la diffusion par blocs. Permettre de prédire et de valider plusieurs tokens en parallèle, au lieu de dépendre exclusivement de la génération token par token.
4. Préserver l'architecture hybride. Traiter correctement les couches d'attention softmax et les couches récurrentes de type Gated DeltaNet, lorsque celles-ci sont présentes dans le checkpoint retenu.
5. Garantir la compatibilité SGLang. Intégrer le modèle dans le moteur de serving sans créer un serveur de génération parallèle indépendant.
6. Mesurer le bénéfice réel. Comparer la qualité, la latence, le débit, la mémoire et le coût total à ceux du modèle autorégressif d'origine.

### 1.2 Principes directeurs

- Réutiliser les poids du modèle source.
- Privilégier OPDLM comme méthode de conversion et de distillation.
- Réutiliser les mécanismes FLARE compatibles avec l'architecture Qwen3.5 et SGLang.
- Introduire les méthodes complémentaires uniquement lorsqu'un gain est mesuré sur une baseline contrôlée.
- Séparer la correction fonctionnelle, la qualité du modèle et les performances de serving.
- Rendre chaque expérience reproductible et chaque décision technique traçable.

## 2. Périmètre et limites

Inclus dans le périmètre

- Conversion AR vers diffusion par blocs.
- Distillation sur les trajectoires du modèle étudiant.
- Adaptation des mécanismes de visibilité des tokens et de l'état récurrent.
- Entraînement, évaluation, checkpoints et reproductibilité.
- Intégration et optimisation du serving SGLang.

Hors périmètre initial

- Préentraînement complet du modèle.
- Modification arbitraire de l'architecture ou du tokenizer.
- Entraînement RL à grande échelle avant validation de la conversion.
- Prise en charge simultanée de toutes les variantes Qwen.
- Garantie d'une supériorité sur toutes les tâches et tous les matériels.

Le premier livrable doit être un prototype fonctionnel sur une seule taille de modèle. L'extension à d'autres tailles intervient après validation de la chaîne complète.

## 3. Base scientifique et décisions d'architecture

Les sept publications ne décrivent pas une méthode unique. Jainius doit donc sélectionner leurs mécanismes selon leur fonction, et non les empiler indistinctement.

| Référence     | Apport pour Jainius                                              | Décision                                             |
| ------------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| STM 1 — NPD   | Génération asynchrone, filtrage et packing                       | Option d'optimisation du pipeline de données         |
| STM 2 — FLARE | Double flux causal/diffusion, traitement hybride et serving      | Référence principale pour l'architecture d'exécution |
| STM 3 — OPDLM | Conversion AR→diffusion avec distillation on-policy              | Méthode prioritaire de conversion                    |
| STM 4 — TOPD  | Distillation des décisions présentes dans la trace de débruitage | Extension conditionnelle                             |
| STM 5 — TSD   | Auto-distillation temporelle et pondération par vérificateur     | Optimisation conditionnelle                          |
| STM 6 — HOPD  | Supervision des positions hésitantes ou non engagées             | Extension conditionnelle                             |
| STM 7 — d-OPD | Correction de la cible AR tenant compte du futur visible         | Extension conditionnelle                             |

STM_7_PUBLICATIONS.md

### 3.1 Décision principale : OPDLM pour la conversion

OPDLM est retenu comme point de départ, car il traite directement la conversion d'un modèle autorégressif en modèle de diffusion, avec un enseignant autorégressif gelé et des trajectoires produites par l'étudiant.

Le mécanisme de base est le suivant :

1. Initialiser l'étudiant à partir du checkpoint autorégressif source.
2. Activer la visibilité bidirectionnelle à l'intérieur des blocs de diffusion, conformément à l'algorithme retenu.
3. Générer des trajectoires de débruitage avec l'étudiant.
4. Prélever des états intermédiaires sur ces trajectoires.
5. Calculer, pour les positions masquées, la distribution cible de l'enseignant autorégressif gelé.
6. Optimiser l'étudiant par divergence KL directe entre la cible de l'enseignant et la distribution de l'étudiant.
7. Augmenter progressivement la longueur des trajectoires, selon un curriculum, afin de stabiliser l'entraînement.

L'intérêt de cette approche est de limiter la conversion à un post-entraînement et d'aligner l'apprentissage sur les états rencontrés à l'inférence. Les réductions de volume de tokens rapportées dans la publication OPDLM restent à vérifier sur le modèle et les données effectivement retenus pour Jainius.&#x20;

STM_7_PUBLICATIONS.md



### 3.2 Décision complémentaire : FLARE pour l'architecture et le serving

FLARE apporte les mécanismes structurels nécessaires pour convertir un backbone hybride sans ignorer le fonctionnement des couches récurrentes.

La conception cible doit prévoir :

- un flux causal propre (clean) pour préserver la prédiction autorégressive ;
- un flux de diffusion bruité (noisy) pour le débruitage bidirectionnel par blocs ;
- une gestion explicite des frontières de blocs et de documents ;
- une planification des états récurrents pour les couches Gated DeltaNet ;
- des mises à jour d'état uniquement lorsque la séquence ou le bloc finalisé doit être validé ;
- une interface de serving commune aux modes autorégressif et diffusion, si les mécanismes retenus le permettent.

L'objectif FLARE \\(L = L\_{\mathrm{AR}} + L\_{\mathrm{diff}}\\) et son schéma de double flux ne doivent pas être confondus avec la loss OPDLM. Ce sont deux formulations d'entraînement distinctes. La combinaison de leurs objectifs est une hypothèse de recherche, pas une propriété établie des publications. Elle ne sera testée qu'après établissement de la baseline OPDLM.&#x20;

STM_7_PUBLICATIONS.md


## 4. Architecture technique cible

L'architecture doit séparer cinq responsabilités : le modèle source, la conversion, la distillation, le moteur de diffusion et le serving. Cette séparation permet de tester chaque composant indépendamment et de revenir au checkpoint initial en cas de régression.

A. Checkpoint Qwen3.5 source

Poids, tokenizer, configuration, état récurrent et référence de qualité

B. Adaptateur architectural

Masques par blocs · visibilité clean/noisy · planification des états récurrents

C. Distillation OPDLM

Trajectoires étudiant · cibles AR gelées · KL directe · curriculum de longueur

D. Checkpoint Jainius

Poids convertis · configuration diffusion · paramètres de génération · métadonnées de validation

E. Serving natif SGLang

Ordonnancement des blocs · kernels compatibles · gestion des états · API et métriques

### 4.1 Composants logiciels à développer

| Identifiant | Composant                     | Responsabilité                                                                                              |
| ----------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| JAI-ARCH    | Adaptateur de modèle          | Adapter le forward pass et la visibilité des tokens aux blocs de diffusion sans altérer les poids initiaux. |
| JAI-MASK    | Gestionnaire de masques       | Construire les vues masquées, les frontières de blocs et les masques d'isolation des documents.             |
| JAI-STATE   | Gestionnaire d'état récurrent | Sauvegarder, restaurer, réinitialiser et valider les états des couches récurrentes.                         |
| JAI-ROLL    | Générateur de trajectoires    | Produire les trajectoires de débruitage et enregistrer les états utilisés par la distillation.              |
| JAI-TEACH   | Adaptateur enseignant         | Calculer les distributions de référence à partir du modèle autorégressif gelé.                              |
| JAI-LOSS    | Moteur de distillation        | Calculer la KL, appliquer les masques de perte et gérer les positions supervisées.                          |
| JAI-TRAIN   | Orchestrateur d'entraînement  | Gérer les rollouts, les mises à jour, le curriculum, les checkpoints et la reprise après incident.          |
| JAI-EVAL    | Banc d'évaluation             | Exécuter les tests de qualité, de cohérence, de latence et de débit.                                        |
| JAI-SGLANG  | Intégration serveur           | Brancher le modèle et ses mécanismes de diffusion dans la pile SGLang retenue.                              |

Ces identifiants sont des noms de composants proposés pour Jainius ; ils ne correspondent pas à des modules déjà existants dans les publications.

### 4.2 Compatibilité avec le backbone hybride

Le point le plus risqué n'est pas le calcul de la divergence KL : c'est la correction de la visibilité des tokens dans les couches hybrides.

Pour les couches d'attention softmax, le mécanisme de masque doit autoriser l'accès au contexte antérieur et aux tokens visibles du bloc actif, sans permettre aux tokens d'autres documents de fuiter dans la séquence.

Pour les couches Gated DeltaNet ou équivalentes, une matrice de masque ne suffit pas à imposer la visibilité souhaitée. L'implémentation doit contrôler les transitions d'état, les écritures, les lectures et les réinitialisations. Cette partie nécessitera des adaptations spécifiques du forward pass et, potentiellement, des kernels fusionnés.

Décision : implémenter d'abord un chemin de référence correct, même s'il est lent, puis optimiser le chemin d'exécution. Ne pas développer des kernels Triton ou CUDA complexes avant d'avoir vérifié la correction mathématique et fonctionnelle du modèle.

### 4.3 Modes de génération

Deux modes doivent être envisagés, mais ils ne doivent pas être considérés comme équivalents.

[Free Autoregressive Generation Simulator: How GPT and LLMs Generate Text Step by Step | Simulations4All](https://images.openai.com/static-rsc-4/T00tKBSj3DLRCWsjvJMYS6Oe_30tb9NI8I0NGjaBSg04ZetlyDY7Q79bEOokm1xdUCEO2wE4ZRvACWDg92m21HGrvQLypSSoehT7dKf95e-fyBB2WRP-9uZlN6IcCAJjqQu-NHWROmc41tXjt9nOH9vx06dGsHFqgZH2j62DNY0?purpose=inline)

Mode de référence : AR-Trust

Le modèle produit des propositions de tokens en parallèle dans un bloc, puis les vérifie avec son flux causal. Les propositions non acceptées sont remplacées ou recalculées. Ce mode sert de point de comparaison et de solution de repli pour la qualité.

[Diffusion Language Models — A Visual Walkthrough](https://images.openai.com/static-rsc-4/7VvVkrCvLtdB-srk0WDF2vIrWdhqH-UErmM5wct_sF4oy7P97i6JqB8cGN19BQV4wD1SfR0ORbAPSqgdGl0zOw_Xy3dQ8ZOLlVIjMb3TybZgC0N_-a7i-f9naF9AED0Yf35fGo2bD0vA0mRdFHRX6I6VLr_lG5fTijntyfIVv6c?purpose=inline)

Mode cible : Diffusion-Trust

Le modèle débruite plusieurs positions d'un bloc, engage les positions jugées suffisamment fiables et répète l'opération sur les positions restantes. L'état récurrent est mis à jour à partir du bloc finalisé, et non lors des forwards provisoires de débruitage.

Les deux modes issus de FLARE sont des options techniques à étudier pour Jainius. La première version doit sélectionner le mode qui respecte le mieux les critères de qualité et de compatibilité. Le second mode ne sera activé par défaut qu'après validation de sa qualité et de son débit.

## 5. Spécifications fonctionnelles

Les exigences suivantes constituent la base de développement. Les seuils quantitatifs qui ne proviennent pas explicitement des publications sont identifiés comme des propositions d'acceptation pour Jainius.

| ID    | Exigence                                         | Critère d'acceptation                                                                                                                 |
| ----- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| F-001 | Charger le checkpoint source sans perte de poids | Empreintes et tenseurs identiques avant conversion, hors modifications explicitement documentées.                                     |
| F-002 | Préserver le tokenizer                           | Vocabulaire, identifiants, tokens spéciaux et template de conversation compatibles.                                                   |
| F-003 | Construire des blocs de diffusion                | Les masques correspondent exactement aux positions visibles et masquées définies par l'algorithme.                                    |
| F-004 | Produire des trajectoires d'étudiant             | Chaque trajectoire contient les états, positions et versions du modèle nécessaires à sa reproduction.                                 |
| F-005 | Calculer les cibles enseignant                   | Le modèle enseignant est gelé ; ses distributions sont calculées sur les contextes définis par OPDLM.                                 |
| F-006 | Optimiser l'étudiant                             | La loss KL est finie, correctement masquée et différentiable sur les positions retenues.                                              |
| F-007 | Gérer le curriculum                              | La longueur des rollouts évolue selon une politique enregistrée et reproductible.                                                     |
| F-008 | Sauvegarder et reprendre                         | Un entraînement interrompu peut reprendre à partir d'un checkpoint sans perte silencieuse de l'état de l'optimiseur ou du curriculum. |
| F-009 | Générer en diffusion                             | Le modèle produit une séquence complète à partir de blocs progressivement débruités.                                                  |
| F-010 | Servir via SGLang                                | Le modèle est accessible par une interface de serving documentée, sans serveur d'inférence externe obligatoire.                       |
| F-011 | Isoler les modes de génération                   | Les paramètres, métriques et résultats AR-Trust et Diffusion-Trust sont séparément identifiables.                                     |
| F-012 | Reproduire une expérience                        | Une configuration versionnée permet de reconstruire le même protocole d'entraînement et d'évaluation.                                 |

## 6. Spécifications de l'entraînement

### 6.1 Pipeline nominal

1. Initialisation. Charger le modèle Qwen3.5 retenu et conserver une copie immuable du checkpoint de référence.
2. Préparation des données. Normaliser les conversations, contrôler les tokens spéciaux, dédupliquer les exemples et constituer les ensembles d'entraînement et de validation.
3. Génération on-policy. Produire des trajectoires à partir du modèle étudiant courant. Associer chaque échantillon à la version du checkpoint qui l'a généré.
4. Échantillonnage des états. Extraire les états intermédiaires du processus de débruitage et les positions masquées à superviser.
5. Annotation enseignant. Calculer les distributions cibles du modèle AR gelé dans le contexte prescrit par OPDLM.
6. Optimisation. Calculer la KL directe et mettre à jour les paramètres de l'étudiant.
7. Curriculum. Augmenter progressivement la longueur des rollouts selon des critères enregistrés et mesurables.
8. Validation périodique. Évaluer le checkpoint, mesurer la dérive de qualité et conserver le meilleur modèle selon le protocole défini.

### 6.2 Fonction objectif de référence

Pour OPDLM, la loss principale est la divergence KL directe :

\\[ \mathcal{L}\_{\mathrm{OPDLM}} = D\_{\mathrm{KL}} \left( p\_{\mathrm{AR}}\\;\\|\\;p\_{\theta} \right) \\]

où :

- \\(p\_{\mathrm{AR}}\\) est la distribution de l'enseignant autorégressif gelé, conditionnée par le préfixe causal de la séquence terminale de la trajectoire ;
- \\(p\_{\theta}\\) est la distribution de l'étudiant sur l'état partiellement débruité ;
- la loss est calculée sur les positions supervisées, avec une normalisation à fixer et à documenter.

Cette définition décrit le mécanisme central de la publication OPDLM. La normalisation exacte, les détails du sampler et certains paramètres d'optimisation ne sont pas suffisamment spécifiés dans le recueil pour être considérés comme définitivement déterminés. Ils doivent être établis lors de l'implémentation et validés par des tests numériques.&#x20;

STM_7_PUBLICATIONS.md



### 6.3 Paramètres de départ

Les valeurs ci-dessous constituent une configuration expérimentale initiale proposée pour le prototype. Elles ne doivent pas être interprétées comme les hyperparamètres officiels d'OPDLM.

| Paramètre        | Décision initiale                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| Modèle source    | Un checkpoint Qwen3.5 dense, de préférence le plus petit qui permette de tester l'architecture complète |
| Enseignant       | Checkpoint source gelé                                                                                  |
| Étudiant         | Copie du checkpoint source, avec adaptation du mécanisme de diffusion                                   |
| Taille des blocs | 4 tokens comme point de départ expérimental, à confirmer par benchmark                                  |
| Loss             | KL directe OPDLM                                                                                        |
| Optimiseur       | AdamW comme candidat initial                                                                            |
| Précision        | BF16 si le matériel et les kernels le permettent, sinon précision de référence adaptée                  |
| Curriculum       | Court au démarrage, puis augmentation progressive des longueurs                                         |
| Données          | Mélange contrôlé de raisonnement, mathématiques, programmation et suivi d'instructions                  |
| Validation       | À chaque intervalle de checkpoints prédéfini                                                            |
| Arrêt            | Budget de tokens prédéfini et absence de régression significative selon les critères d'acceptation      |

La taille de bloc 4 est également celle utilisée dans les expériences FLARE rapportées pour les modèles finaux. Elle n'est toutefois pas une exigence intrinsèque d'OPDLM. La longueur optimale pour Jainius devra être mesurée.

### 6.4 Budget d'entraînement

Le projet doit être piloté en tokens et en heures GPU, et non uniquement en nombre d'époques.

Le suivi doit distinguer :

- tokens de données présentés à l'étudiant ;
- tokens consommés lors des rollouts ;
- tokens traités par l'enseignant ;
- tokens effectivement utilisés dans la loss ;
- nombre de mises à jour ;
- coût de chaque phase ;
- coût cumulé et coût de l'évaluation.

La métrique principale de rendement sera le nombre de tokens d'entraînement requis pour atteindre un niveau de qualité donné, comparé au coût d'un entraînement de référence documenté.

Les réductions de 15× à 7 000× annoncées dans OPDLM ne sont pas un objectif contractuel tant que le protocole de comparaison et le périmètre des coûts n'ont pas été reproduits.

## 7. Intégration de SGLang

La compatibilité avec SGLang est une exigence structurante. Jainius ne doit pas être conçu comme un modèle de diffusion fonctionnant uniquement dans un prototype de recherche.

### 7.1 Contrat d'intégration

Le serveur doit prendre en charge :

- le chargement du checkpoint et de sa configuration de diffusion ;
- le préremplissage du prompt et l'initialisation correcte des états récurrents ;
- l'ordonnancement des blocs et des passes de débruitage ;
- la gestion des positions engagées, masquées et restant à résoudre ;
- le maintien de l'état récurrent entre les blocs finalisés ;
- la libération correcte des ressources à la fin d'une requête ;
- les requêtes concurrentes sans mélange des états de deux sessions ;
- l'arrêt sur EOS, les limites de longueur et l'annulation d'une requête ;
- la remontée des métriques de latence, de débit et d'utilisation mémoire.

### 7.2 Réutilisation des composants FLARE

La publication FLARE décrit des mécanismes spécifiques de serving, notamment la planification d'états récurrents, les kernels GDR et ShortConv, la vérification fusionnée et la gestion des graphes CUDA.

Jainius doit les considérer dans cet ordre :

1. Réutiliser les interfaces et mécanismes de gestion des états déjà disponibles dans la version de SGLang retenue.
2. Adapter les chemins de génération et de vérification à la sémantique des blocs de diffusion.
3. Réutiliser les kernels existants lorsque leur contrat d'entrée et de sortie correspond exactement à celui de Jainius.
4. Développer des kernels spécifiques seulement lorsque les profils démontrent un goulot d'étranglement mesurable.
5. Maintenir un chemin de référence non fusionné pour vérifier la correction des kernels optimisés.

La présence de Gated DeltaNet dans un modèle source ne suffit pas à garantir que les kernels FLARE sont directement compatibles. La version exacte de Qwen3.5, la structure des couches, les API SGLang et les interfaces de cache doivent être contrôlées avant toute réutilisation.

### 7.3 Exigences de performance du serveur

| ID    | Exigence                | Validation                                                                                   |
| ----- | ----------------------- | -------------------------------------------------------------------------------------------- |
| S-001 | Chargement du modèle    | Chargement sans erreur, avec vérification des poids et de la configuration                   |
| S-002 | Gestion des états       | Aucun état périmé ou partagé entre requêtes indépendantes                                    |
| S-003 | Cohérence du décodage   | Le chemin optimisé correspond au chemin de référence dans les tolérances numériques définies |
| S-004 | Concurrence             | Résultats corrects à concurrence 1, 4 et 8, puis à la concurrence cible                      |
| S-005 | Gestion des erreurs     | Une requête annulée ne corrompt pas les autres requêtes                                      |
| S-006 | Mesure des performances | Latence, débit, mémoire et nombre de passes de débruitage enregistrés                        |
| S-007 | Régression              | Comparaison systématique avec le modèle source servi dans le même environnement              |

Les niveaux de concurrence ci-dessus sont des points de test proposés, et non des limites imposées par les publications.

## 8. Méthodes complémentaires et conditions d'intégration

Les méthodes autres qu'OPDLM et FLARE doivent être introduites selon une logique expérimentale. Chacune doit résoudre un problème précis de la baseline, et son coût doit être justifié par un gain mesuré.

Étape 2

TOPD — supervision des traces

STM 4 · arXiv:2607.16872

TOPD supervise les décisions qui participent à la réponse finale, au moyen d'un enseignant diffusion gelé et d'une KL inverse. Il ne doit pas être ajouté à la première conversion, car il suppose un enseignant diffusion compatible et un protocole de distillation distinct.

Condition d'intégration : disposer d'un enseignant diffusion adapté et démontrer un gain de qualité sur des tâches de raisonnement difficiles.

Étape 2

d-OPD — correction future-aware

STM 7 · arXiv:2609.35362

d-OPD corrige la distribution cible de l'enseignant AR afin de prendre en compte le contexte futur visible par l'étudiant dans le bloc actif. La méthode vise à réduire le décalage entre les contextes des deux modèles.

Condition d'intégration : vérifier que le décalage de contexte est une source significative d'erreurs dans OPDLM et que le vérificateur de correction est fiable.

Étape 3

HOPD — supervision des hésitations

STM 6 · arXiv:2609.33301

HOPD étend la supervision aux positions masquées non engagées pendant le débruitage, avec des poids calculés a posteriori sur la trajectoire.

Condition d'intégration : démontrer que les positions non engagées contiennent un signal d'apprentissage utile et que le coût supplémentaire de calcul des cibles reste acceptable.

Étape 3

TSD — auto-distillation temporelle

STM 5 · arXiv:2609.15177

TSD utilise une auto-distillation entre états temporels, avec pondération par vérificateur. Elle cible l'efficacité de l'inférence et la qualité des décisions de débruitage.

Condition d'intégration : identifier une faiblesse de la trajectoire de débruitage et mesurer un gain sur la qualité ou le nombre de passes nécessaires.

Option

NPD — efficacité du pipeline

STM 1 · arXiv:2605.05940

NPD utilise la génération asynchrone, le filtrage ΔIFD, le packing et des mises à jour clairsemées.

Condition d'intégration : réutiliser uniquement les éléments compatibles avec les trajectoires diffusion et la loss retenue. NPD ne doit pas être transplanté tel quel, car il est spécifié pour des étudiants autorégressifs.

Cette hiérarchie est une proposition d'ingénierie pour Jainius, pas un ordre de supériorité scientifique établi entre les publications.&#x20;

## 9. Données et préparation du corpus

La qualité du corpus est un déterminant majeur de la conservation des capacités du modèle lors de la conversion. FLARE rapporte que l'adéquation des données de transfert a un effet important sur les performances et qu'un objectif d'entraînement amélioré ne suffit pas à compenser une mauvaise distribution.

### 9.1 Composition fonctionnelle

Le corpus doit couvrir les catégories suivantes :

| Domaine                   | Finalité                                                            | Mesures de validation                                 |
| ------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------- |
| Raisonnement mathématique | Préserver les chaînes de raisonnement et la résolution de problèmes | GSM8K, MATH-500, AIME selon le niveau cible           |
| Raisonnement scientifique | Évaluer l'inférence et la connaissance spécialisée                  | GPQA-Diamond                                          |
| Programmation             | Préserver la génération et la correction de code                    | HumanEval, MBPP et LiveCodeBench                      |
| Suivi d'instructions      | Évaluer le respect des contraintes explicites                       | IFEval                                                |
| Connaissances générales   | Mesurer les pertes de connaissances lors de la conversion           | MMLU et MMLU-Pro                                      |
| Conversations générales   | Préserver la capacité de dialogue                                   | Jeu de validation indépendant, versionné et documenté |

Les jeux publics cités dans les publications sont des candidats pour les tests. Leur présence dans le corpus d'entraînement doit être contrôlée afin d'éviter la contamination des évaluations.

### 9.2 Prétraitement obligatoire

Le pipeline de données doit :

- conserver la structure des conversations et les marqueurs de raisonnement ;
- éliminer les exemples vides, mal formés et tronqués de façon incohérente ;
- dédupliquer les exemples et identifier les doublons entre entraînement et validation ;
- documenter la provenance, la licence et les restrictions d'usage ;
- enregistrer les règles de filtrage et les statistiques avant/après traitement ;
- empêcher le franchissement des frontières de documents lors du packing ;
- préserver les tokens spéciaux et le template attendu par Qwen3.5 ;
- séparer les données d'entraînement, de validation et de test.

### 9.3 Budget de données

Le volume de données sera choisi en fonction du budget de calcul et de la convergence, plutôt que repris mécaniquement des expériences FLARE ou OPDLM.

La stratégie est de commencer par un corpus pilote réduit, représentatif des domaines cibles, puis d'augmenter progressivement le budget uniquement si les résultats montrent une amélioration mesurable.

## 10. Protocole d'évaluation et critères d'acceptation

L'acceptation de Jainius repose sur trois dimensions distinctes : la correction fonctionnelle, la conservation de la qualité et l'amélioration du serving.

### 10.1 Références de comparaison

Trois références sont obligatoires :

- B0 — Source AR : le checkpoint Qwen3.5 initial, sans modification.
- B1 — Conversion seule : le modèle converti sans les optimisations expérimentales complémentaires, lorsque cette comparaison est applicable.
- B2 — Jainius final : le checkpoint converti et distillé, avec les optimisations validées.

Tous les modèles doivent être évalués avec le même matériel, les mêmes prompts, les mêmes limites de génération et des protocoles explicitement documentés.

Les scores obtenus avec des températures ou des règles de décodage différentes ne doivent pas être comparés directement. Le recueil des sept STM signale précisément que leurs protocoles d'évaluation diffèrent.&#x20;

STM_7_PUBLICATIONS.md



### 10.2 Seuils proposés

Les seuils ci-dessous sont des objectifs d'acceptation proposés pour Jainius. Ils ne sont pas des résultats garantis par les publications.

| ID    | Critère                    | Seuil de décision                                                                                                       |
| ----- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| A-001 | Intégrité du checkpoint    | Aucun écart non expliqué dans les poids et la configuration de référence                                                |
| A-002 | Correction du forward pass | Tests numériques et tests de masques réussis                                                                            |
| A-003 | Qualité globale            | Au moins 98 % du score composite du modèle source                                                                       |
| A-004 | Régression par domaine     | Aucune baisse supérieure à 5 points de pourcentage sur un benchmark prioritaire, sauf dérogation documentée             |
| A-005 | Reproductibilité           | Résultats reproductibles sur un protocole fixe, avec plusieurs exécutions lorsque le caractère stochastique le justifie |
| A-006 | Débit                      | Amélioration d'au moins 1,5× à concurrence 8 sur une charge représentative                                              |
| A-007 | Latence                    | Gain mesuré de latence p50 ou p95, sans régression significative sur les requêtes courtes                               |
| A-008 | Mémoire                    | Pas de dépassement du budget GPU fixé pour le prototype                                                                 |
| A-009 | Coût de conversion         | Coût total documenté et inférieur au budget maximal défini avant l'expérience                                           |
| A-010 | SGLang                     | Aucun défaut bloquant sur le chargement, la concurrence, les arrêts et les états récurrents                             |

Le score composite doit être défini avant les expériences : ensemble de benchmarks, pondérations, protocole d'agrégation et règles de gestion des résultats manquants.

Le seuil de débit de 1,5× constitue un objectif de décision initial, pas une estimation de la vitesse réelle de Jainius. Si l'amélioration du débit se fait au prix d'une perte de raisonnement, le modèle n'est pas accepté en l'état.

### 10.3 Métriques de qualité

- Exactitude ou pass\@1 sur les tâches de mathématiques et de programmation.
- Scores de raisonnement scientifique.
- Respect des instructions et contraintes de sortie.
- Score composite global.
- Variance entre générations pour les tâches stochastiques.
- Taux de réponses tronquées, invalides ou interrompues.
- Taux de régression par rapport au modèle source.

### 10.4 Métriques de serving

- Latence de préremplissage du prompt.
- Latence totale de génération et temps jusqu'au premier token.
- Débit agrégé en tokens/s.
- Latence p50 et p95.
- Nombre moyen de passes de débruitage par bloc.
- Taux de positions engagées par passe.
- Mémoire maximale utilisée.
- Débit en fonction de la concurrence.
- Coût de calcul par réponse valide.

Le débit doit être mesuré à la fois sur les tokens de sortie et sur les requêtes terminées, car une génération qui produit rapidement de nombreux tokens mais requiert des réponses plus longues peut être moins efficace en pratique.

## 11. Plan de réalisation

Le projet sera conduit par jalons avec des critères de sortie explicites. Il ne faut pas commencer par une conversion complète suivie d'une intégration tardive dans SGLang : les deux doivent être développés avec des contrats d'interface communs, et la compatibilité du backbone doit être vérifiée dès le début.

## M0

Audit

Audit du modèle et de l'environnement

- Identifier précisément le checkpoint Qwen3.5, ses couches, son tokenizer et ses dépendances.
- Vérifier les versions de PyTorch, CUDA, Triton et SGLang.
- Établir la baseline de qualité et de performances.
- Vérifier la disponibilité et les licences des composants réutilisés.

Livrable : rapport de compatibilité et environnement reproductible.

## M1

Prototype

Forward pass de diffusion

- Implémenter les masques et les frontières de blocs.
- Vérifier les couches récurrentes et les états.
- Comparer le chemin de référence avec les optimisations éventuelles.

Livrable : prototype correct sur des séquences synthétiques et réelles.

## M2

Conversion

Baseline OPDLM

- Générer les trajectoires d'étudiant.
- Calculer les distributions de l'enseignant.
- Vérifier la loss KL, les gradients et le curriculum.
- Effectuer un entraînement pilote à budget limité.

Livrable : premier checkpoint Jainius et rapport de conservation des capacités.

## M3

Serving

Intégration SGLang

- Implémenter le cycle de vie des blocs et des états.
- Ajouter le mode de génération retenu.
- Tester les requêtes concurrentes, les annulations et les arrêts.
- Profiler les goulots d'étranglement.

Livrable : serveur de démonstration compatible avec l'interface SGLang retenue.

## M4

Validation

Optimisation et acceptation

- Comparer qualité et débit à protocole constant.
- Tester les optimisations FLARE compatibles.
- N'introduire TOPD, d-OPD, HOPD ou TSD qu'en cas de besoin démontré.
- Vérifier tous les critères d'acceptation.

Livrable : modèle validé, rapport de benchmark et décision de mise en production.

Les jalons sont des phases logiques, pas des estimations calendaires. La durée totale ne peut être estimée sérieusement qu'après l'audit du matériel, du checkpoint et des dépendances.

## 12. Risques techniques et mesures de maîtrise

| Risque                                      | Gravité  | Mesure de maîtrise                                                                 |
| ------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| Perte de raisonnement lors de la conversion | Critique | Évaluation régulière, comparaison à la source et arrêt des expériences régressives |
| Mauvaise gestion des états récurrents       | Critique | Tests d'état, de restauration et d'isolation par requête                           |
| Incompatibilité avec SGLang                 | Critique | Prototype de serving avant l'entraînement à grande échelle                         |
| Cibles enseignant mal alignées              | Élevée   | Tests de distributions, de contextes et de masques                                 |
| Coût des rollouts trop élevé                | Élevée   | Profilage précoce et réduction des longueurs de trajectoire                        |
| Faible accélération réelle                  | Élevée   | Mesures de bout en bout, et non uniquement de débit de kernels                     |
| Dérive des données de transfert             | Élevée   | Corpus versionné, validation indépendante et analyses par domaine                  |
| Instabilité des optimisations               | Élevée   | Chemin de référence et tests différentiels systématiques                           |
| Dépendances ou licences incompatibles       | Élevée   | Inventaire logiciel et audit de provenance avant diffusion                         |

### 12.1 Règle de décision en cas d'échec

Si la conversion dégrade fortement le raisonnement, il faut d'abord examiner l'alignement des contextes, les masques, les cibles enseignant et le curriculum OPDLM. Il ne faut pas compenser automatiquement une erreur structurelle en augmentant le volume d'entraînement.

Si la qualité est satisfaisante mais que le débit est insuffisant, la priorité passe à l'analyse du serving : passes de débruitage, gestion des états, kernels, coût des vérifications et ordonnancement des requêtes.

Si le débit s'améliore mais que la qualité échoue aux critères d'acceptation, le checkpoint n'est pas prêt pour la production.

## 13. Livrables obligatoires

À la fin du projet, le dépôt Jainius doit contenir :

1. Code source de l'adaptateur architectural, de la distillation, de l'entraînement, de l'évaluation et de l'intégration SGLang.
2. Configuration versionnée décrivant le modèle, les blocs, les masques, les pertes, les hyperparamètres et les paramètres de génération.
3. Checkpoint final, avec les métadonnées nécessaires à son chargement et à son utilisation.
4. Manifestes de données, contenant les sources, les versions, les licences, les règles de filtrage et les statistiques.
5. Suite de tests automatisés, couvrant les masques, les états récurrents, les gradients, la reprise et le serving.
6. Rapport comparatif avec le modèle source : qualité, coût, mémoire, latence, débit et résultats par domaine.
7. Documentation d'exploitation, incluant les dépendances, les commandes d'entraînement et de lancement, les paramètres de génération et les limites connues.
8. Registre des expériences, retraçant les hypothèses, les résultats, les ablations et les raisons de retenir ou d'écarter une méthode.

## 14. Gouvernance scientifique et traçabilité

Chaque affirmation technique du projet doit être classée dans l'une des catégories suivantes :

- [SOURCE] : explicitement établie dans une publication ou une documentation technique vérifiée ;
- [DÉDUCTION] : conséquence technique justifiée à partir des sources ;
- [PROPOSITION] : choix de conception propre à Jainius ;
- [À VÉRIFIER] : point nécessitant un test, une reproduction ou une vérification de compatibilité.

Chaque expérience doit enregistrer au minimum l'identifiant du modèle, la version du code, les données, les hyperparamètres, la seed, le budget de tokens, le matériel, les métriques et le checkpoint obtenu.

Les résultats des sept publications ne doivent pas être amalgamés en une moyenne unique : les modèles, les tâches, les objectifs et les protocoles sont différents. La décision d'architecture doit reposer sur les expériences Jainius elles-mêmes, tout en conservant la traçabilité vers les méthodes originales.&#x20;

## 15. Décisions finales

La stratégie de réalisation retenue est la suivante :

- Conversion : OPDLM comme méthode de référence.
- Architecture hybride : réutilisation sélective des mécanismes FLARE, notamment pour les flux clean/noisy et la gestion des états récurrents.
- Entraînement : une baseline reproductible avant toute combinaison d'objectifs.
- Serving : intégration native à SGLang, avec un chemin de référence correct et des optimisations ajoutées progressivement.
- Distillations complémentaires : TOPD, d-OPD, HOPD et TSD activés uniquement à partir d'une hypothèse testable et d'un gain mesuré.
- Acceptation : qualité proche de la source, comportement correct, débit amélioré et coût de conversion documenté.

Le premier objectif n'est donc pas de produire immédiatement le modèle le plus rapide possible. 
Il est d'établir une chaîne de conversion correcte et mesurable, capable de conserver les capacités de Qwen3.5. La vitesse sera ensuite optimisée sans compromettre ces capacités.

Critère de réussite global : Jainius ne sera considéré comme réussi que si le gain de serving est démontré sur une charge représentative, que la qualité de raisonnement satisfait les seuils convenus et que la chaîne complète peut être reproduite à partir des livrables du projet.