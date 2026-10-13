# AGENTS.md — Jainius

## 0. Autorité des documents et décisions verrouillées (lire en premier)

Hiérarchie, du plus au moins prioritaire :
1. `docs/PRD.md` : objectif, exigences, critères de réussite, jalons.
2. `docs/DECISIONS.md` : décisions verrouillées et arbitrages déjà rendus.
3. `AGENTS.md` : règles de fonctionnement.
4. `docs/decisions/*.md` : décisions de détail, valables tant qu'elles ne contredisent pas 1 et 2 (un en-tête « OBSOLÈTE » fait foi).
5. `docs/ARCHITECTURE.md` (à produire après le jalon J3) : description de l'existant, vérifiée sur le code.
6. `docs/archive/` : ancien cahier des charges, ancien cahier technique. Historique uniquement. **Ne jamais citer ces documents pour justifier un choix de conception.**

Règles permanentes :
- **R0.1 Début de session** : lire le PRD et DECISIONS.md. Annoncer en cinq lignes maximum l'objectif du projet et l'exigence du PRD (G1 à G5, jalon J1 à J5) que la tâche sert.
- **R0.2 Décisions verrouillées** : ne jamais les rouvrir, les contourner ni leur substituer une variante. Si l'agent estime l'une d'elles infaisable, il écrit dans `docs/PROPOSITIONS.md` le fait établi, la preuve et une alternative, puis continue sur ce qui n'est pas concerné.
- **R0.3 Avant toute question à l'utilisateur** : chercher dans `docs/DECISIONS.md`. Ne jamais redemander une décision déjà rendue.
- **R0.4 Arbitrages** : au plus 3 par compte rendu, chacun avec une recommandation par défaut. Si le choix est réversible, l'agent applique le défaut, l'inscrit dans DECISIONS.md (statut PAR DÉFAUT) et continue. Toute réponse de l'utilisateur est inscrite dans DECISIONS.md immédiatement.
- **R0.5 Lien avec le PRD** : toute tâche doit servir une exigence du PRD. Un travail sans lien (durcissement de tests périphériques, métriques annexes, audits de clôture répétés) n'est fait que sur demande explicite. Au plus un audit de clôture par jalon.
- **R0.6 Contradiction** : le document de rang supérieur gagne sans arbitrage. L'agent annote le document de rang inférieur dans la même intervention.
- **R0.7 Compte rendu** : français simple, sans identifiant non expliqué, selon ce plan : TERMINÉ / PAS TERMINÉ / DÉCISIONS DEMANDÉES (3 maximum) / PROCHAINE ÉTAPE. Le détail technique va dans le fichier d'audit, pas dans le message. Un jalon n'est déclaré terminé que si tous ses critères de sortie du PRD sont vérifiés.

## 1. Mission et rôle

Jainius convertit un LLM autorégressif hybride (Qwen3.5) en modèle de diffusion par blocs, selon `docs/PRD.md` : architecture double flux de FLARE, distillation on-policy d'OPDLM, serving natif par le fork SGLang de FLARE, entraînement très léger, débit comparable à DiffusionGemma, raisonnement conservé.

L'agent agit comme architecte d'implémentation et ingénieur logiciel au service de cet objectif. Il prend les décisions d'implémentation nécessaires dans le périmètre confié, sans déléguer à l'utilisateur les investigations techniques relevant de son expertise.

L'agent doit privilégier :
- les preuves expérimentales plutôt que les suppositions ;
- les changements minimaux, maîtrisés et réversibles ;
- la reproductibilité des résultats ;
- la traçabilité des décisions et des expériences ;
- la compatibilité avec le fork SGLang de FLARE et l'environnement d'exécution cible.

La progression apparente du projet ne constitue jamais une priorité supérieure à la validité technique des résultats.

## 2. Documents de référence

Avant toute intervention, l'agent lit `docs/PRD.md` et `docs/DECISIONS.md`, puis les documents pertinents pour la tâche :

- `docs/decisions/` : décisions de détail et audits (`docs/audits/`).
- `docs/environnement.md` : matériel, conteneurs, environnements Python.
- `docs/arxiv/` : publications originales au format PDF.
- `docs/STM_7_PUBLICATIONS.md` : synthèse documentaire des sept publications (aide à la lecture, pas une source de décision).
- `docs/archive/` : historique (voir R0 ci-dessus).

L'agent ne suppose pas qu'un fichier existe ou est à jour sans l'avoir vérifié. Il distingue une affirmation de synthèse, une publication, du code lu et un résultat mesuré localement.

Contradiction entre documents : appliquer R0.6 ; en cas de doute entre deux documents de même rang, identifier les passages, préserver les travaux existants, proposer une résolution et demander un arbitrage seulement si nécessaire. Ne jamais résoudre silencieusement une contradiction susceptible de modifier l'architecture, les résultats ou les critères d'acceptation.

## 3. Portée des instructions et autonomie

Le prompt de la tâche définit le périmètre de l'intervention en cours : objectif, phase, livrables, opérations autorisées et éventuelles restrictions supplémentaires.

Le présent fichier définit les règles permanentes applicables à toutes les interventions.

L'agent doit :
- lire le prompt courant et identifier explicitement le résultat attendu ;
- déterminer les prérequis, les risques et les dépendances ;
- inspecter l'état réel du dépôt avant de modifier des fichiers ;
- exécuter les investigations et validations relevant de son périmètre ;
- signaler les blocages réels et les informations manquantes ;
- produire les livrables et preuves correspondant à la demande.

L'agent peut prendre de manière autonome les décisions d'implémentation réversibles et conformes aux spécifications.

Il ne doit pas, de sa propre initiative :
- modifier un objectif ou une exigence validée ;
- changer une décision architecturale structurante ;
- modifier une convention expérimentale validée ;
- élargir substantiellement le périmètre de la tâche ;
- engager une opération destructive ou difficilement réversible ;
- effectuer une opération externe ayant des conséquences non autorisées ;
- commencer une phase ultérieure lorsque le prompt limite explicitement le travail à une phase précédente.

Lorsqu'une ambiguïté n'empêche pas de progresser sans risque, l'agent adopte l'interprétation la plus prudente, la documente et poursuit. Il pose une question à l'utilisateur seulement si aucune solution raisonnable et conforme ne permet d'avancer.

## 4. Décisions techniques validées

Le détail et l'historique sont dans `docs/DECISIONS.md`. Rappel de ce qui structure le code :

### 4.1 Modèle cible

- Le modèle principal est Qwen3.5-0.8B. Qwen3-0.6B peut servir de banc d'essai auxiliaire.
- Un résultat sur un modèle auxiliaire ne remplace pas la validation sur Qwen3.5-0.8B ; toute différence architecturale est prise en compte avant de transférer une conclusion.

### 4.2 FLARE et OPDLM (D-01, D-02, D-03)

- **Diffusion : FLARE seul.** Double flux, masques, planification des états GDN, convention de logits `shifted`, décodage Diffusion-Trust, fork SGLang.
- **Entraînement : distillation on-policy d'OPDLM seule.** Rollouts de l'étudiant, états intermédiaires, cibles de l'enseignant AR gelé, KL enseignant → étudiant, curriculum de longueur.
- Ne reprendre ni l'objectif `L_AR + L_diff` de FLARE, ni la conversion d'attention d'OPDLM.
- Usage privé non commercial : le code FLARE peut être repris. Conserver `LICENSE` et `Required Notice` dans tout répertoire qui en contient ; ne rien distribuer sans réexamen. Distinguer dans les rapports le code repris, le code lu et le code réécrit.

### 4.3 Convention des logits

- La convention de production est `shifted` (D-02). La convention `current` est gelée : son code reste en place, il n'est plus étendu.
- La règle d'alignement des cibles sous `shifted` doit être établie et validée numériquement (jalon J2) avant toute comparaison de qualité.

### 4.4 État récurrent et GDN

- La sémantique de l'état récurrent est une propriété critique. Une génération plausible ne prouve pas sa correction.
- Toute divergence numérique non expliquée sur un chemin critique est résolue ou explicitement acceptée avant de poursuivre ce qui en dépend.

### 4.5 Serving (D-04)

- La cible est le fork SGLang de FLARE, utilisé tel quel. Notre forward d'entraînement s'aligne sur son comportement natif, et non l'inverse.
- Une mesure de débit n'est valable que sur GPU vérifié vide (§11).

Ces décisions n'autorisent pas à elles seules un entraînement ou une conversion : le périmètre de chaque intervention est fixé par le prompt courant.

## 5. Qualité du code et architecture

L'agent doit produire un code lisible, modulaire, testable et cohérent avec l'architecture existante.

Principes obligatoires :

- Examiner les abstractions et conventions déjà utilisées avant d'en introduire de nouvelles.
- Préférer une correction ciblée à une réécriture étendue.
- Éviter les abstractions génériques sans besoin démontré.
- Séparer autant que nécessaire la configuration, les données, les algorithmes, les interfaces et les tests.
- Rendre les paramètres importants explicites et reproductibles.
- Valider les entrées aux frontières des composants.
- Fournir des erreurs explicites plutôt que masquer une configuration invalide.
- Documenter les hypothèses numériques et les conventions de tenseurs.
- Préserver la compatibilité avec les interfaces déjà définies, sauf modification autorisée.
- Ne pas ajouter une dépendance uniquement par commodité lorsqu'une solution existante, plus simple et fiable, convient.

Toute modification d'interface publique ou de format de données doit être identifiée dans le compte rendu. Les migrations nécessaires doivent être incluses dans le périmètre de la tâche ou signalées explicitement.

## 6. Tests et preuves de correction

Les tests doivent correspondre aux risques réels du changement.

Selon le composant concerné, l'agent doit privilégier :
- les tests unitaires ;
- les tests d'intégration ;
- les tests de régression ;
- les tests numériques de référence ;
- les tests de reproductibilité ;
- les tests de compatibilité avec les versions et environnements visés ;
- les tests de performance lorsque celle-ci constitue une exigence.

Pour les composants numériques et les modèles :
- expliciter les formes, types, appareils et conventions des tenseurs ;
- justifier les tolérances numériques ;
- comparer aux références appropriées ;
- distinguer les erreurs d'implémentation des différences attendues liées à la précision ou à l'environnement ;
- vérifier les frontières de blocs, les états et les chemins de gradient lorsqu'ils sont concernés.

L'agent doit rapporter honnêtement :
- les tests exécutés et leurs commandes ;
- les tests réussis et échoués ;
- les tests non exécutés ;
- les limites de ressources ou d'environnement ;
- les résultats non concluants ;
- les éventuelles divergences restantes.

Ne jamais affirmer qu'un test a réussi s'il n'a pas été exécuté. Un test qui passe ne démontre que les propriétés qu'il vérifie effectivement.

## 7. Recherche scientifique et gestion des sources

Les publications scientifiques et leurs implémentations doivent être examinées avec rigueur.

Lorsqu'une conclusion dépend d'une publication :
- identifier la source et, lorsque c'est possible, la section, la figure, l'équation ou la page pertinente ;
- distinguer le résultat publié de son interprétation ;
- vérifier les détails d'implémentation dans le code officiel lorsque cela est nécessaire et possible ;
- identifier les écarts entre la publication, le code disponible et l'implémentation Jainius ;
- ne pas inventer de résultats, de références ou de capacités.

Toute utilisation de code ou de poids externes doit respecter les licences applicables et les restrictions du projet.

L'absence d'information dans une source ne doit pas être compensée par une affirmation non vérifiée. Les incertitudes doivent être indiquées.

## 8. Documents PDF, extraction et recherche documentaire

Les PDF originaux de référence sont conservés dans `docs/arxiv/`.

L'agent ne doit pas modifier les originaux pour faciliter leur lecture, leur extraction ou leur indexation.

Les extractions de texte, fichiers intermédiaires, découpages en passages, index, caches et résultats de recherche doivent être placés dans `_artefacts/`, sauf décision explicite contraire.

Toute extraction destinée à soutenir une décision technique doit préserver autant que possible :
- le nom du fichier source ;
- la version ou l'empreinte du fichier ;
- les numéros de page ;
- les titres de sections ;
- les équations et tableaux pertinents ;
- les références nécessaires à la vérification.

Une extraction imparfaite ne doit pas être traitée comme une reproduction fidèle du PDF. En cas de doute, consulter le document original.

La recherche lexicale, la recherche vectorielle et toute solution RAG sont des moyens d'accès aux sources, pas des sources d'autorité en elles-mêmes. Les affirmations importantes doivent être vérifiables dans le document original ou dans une autre source primaire fiable.

## 9. Configuration, environnement et secrets

Les paramètres dépendant de la machine doivent être configurés sans imposer des valeurs spécifiques à un poste lorsque cela peut être évité.

Le fichier `.env`, situé à la racine du dépôt, est réservé à la configuration locale et aux secrets nécessaires. Il peut rester vide tant qu'aucune variable n'est requise.

Règles obligatoires :
- ne jamais inscrire de secret dans le code, les tests, les journaux ou les rapports versionnés ;
- ne jamais afficher la valeur d'un secret dans un compte rendu ;
- ne pas supposer qu'une variable existe parce qu'elle est mentionnée dans un document ;
- vérifier la configuration avant de lancer une opération qui en dépend ;
- fournir, si nécessaire, un modèle `.env.example` dépourvu de secrets ;
- documenter les variables nécessaires et leurs valeurs autorisées sans exposer de données sensibles.

Les versions des dépendances, les paramètres d'exécution et les contraintes matérielles doivent être consignés lorsqu'ils influencent la reproductibilité.

## 10. Gestion des fichiers et artefacts

Le dossier `_artefacts/`, situé à la racine du workspace, est destiné aux résultats locaux qui ne doivent pas être publiés dans le dépôt Git.

Y placer notamment, selon les besoins :
- les logs et sorties de commandes volumineuses ;
- les extractions et découpages de PDF ;
- les index de recherche et données dérivées ;
- les caches ;
- les résultats temporaires de tests et de benchmarks ;
- les checkpoints et poids générés localement ;
- les données intermédiaires ;
- les rapports de travail qui ne sont pas destinés à être versionnés.

Les scripts, tests, configurations génériques, documentation de référence et fichiers source nécessaires à la reconstruction des artefacts doivent rester versionnables lorsqu'ils font partie du projet.

Règles de gestion :
- ne jamais supprimer un artefact existant sans avoir vérifié sa nature et son utilité ;
- ne pas écraser un résultat de référence sans sauvegarde ou autorisation ;
- utiliser des noms explicites et, lorsque cela est pertinent, des sous-dossiers par tâche ou expérience ;
- conserver les informations nécessaires pour reproduire les résultats ;
- vérifier le contenu de `_artefacts/` avant toute opération Git susceptible d'inclure des fichiers ;
- ne pas déplacer un document de référence dans `_artefacts/` simplement parce qu'il est volumineux.

Le `.gitignore` doit exclure `.env` et `_artefacts/`. L'agent doit vérifier l'état réel de Git : un fichier déjà suivi peut continuer à être versionné malgré une règle d'exclusion.

## 11. Services locaux (Qdrant, moteurs de serving sous Docker)

Des conteneurs Docker peuvent être disponibles dans l'environnement WSL
(Qdrant, moteurs de serving tels que SGLang, vLLM, llmd). Leur présence,
leur état, leurs ports et leur accessibilité doivent être vérifiés AVANT
toute utilisation — via `scripts/check_env.sh` ou `docker ps` / `docker images`.

Règle d'installation : **ne jamais installer un moteur de serving ou
d'évaluation dans un venv Python tant qu'une image Docker correspondante
existe sur la machine**. Vérifier d'abord `docker images`. Toute
information d'environnement donnée par l'utilisateur (services, conteneurs,
matériel) doit être écrite IMMÉDIATEMENT dans `docs/environnement.md`
(ou AGENTS.md si elle est permanente) — une information non persistée
dans une session est une information perdue pour les sessions suivantes.

L'agent ne doit pas :
- supposer qu'un service ou un conteneur est démarré ;
- démarrer, arrêter, recréer ou supprimer un conteneur sans nécessité liée à la tâche et sans vérifier les conséquences ;
- effacer une collection ou des données existantes sans autorisation explicite ;
- exposer un service local sur une interface réseau plus large que nécessaire ;
- enregistrer les identifiants ou les secrets dans les rapports.

Les index, caches, extractions et autres données reconstructibles restent dans `_artefacts/` ou dans le stockage local prévu à cet effet.

Le code du projet ne doit pas dépendre de Qdrant pour les fonctions qui peuvent raisonnablement fonctionner sans lui. Toute fonctionnalité de recherche documentaire doit gérer proprement l'indisponibilité du service lorsque le contrat de la tâche exige ce comportement.

L'utilisation de Qdrant, d'un moteur de serving ou d'une architecture RAG doit répondre à un besoin démontré. Les choix de modèles, de découpage et de recherche doivent être documentés et évaluables.

Mesures de performance : exécuter `scripts/check_env.sh` avant chaque mesure et archiver sa sortie avec les résultats — une mesure de débit/latence n'est valable que sur GPU vérifié vide (aucun process compute).

## 12. Sécurité et opérations à risque

Avant toute opération destructive ou difficilement réversible, l'agent doit identifier précisément les fichiers, données, services ou ressources affectés.

Sont notamment concernées :
- la suppression ou l'écrasement de fichiers et checkpoints ;
- les réinitialisations Git ;
- les changements de branche affectant des travaux existants ;
- les suppressions de collections ou volumes Docker ;
- les changements de configuration de services ;
- les opérations distantes ;
- les modifications de permissions ou de secrets.

L'agent doit préserver les changements de l'utilisateur et les résultats existants.

En l'absence d'autorisation explicite, il doit préférer une opération non destructive ou s'arrêter avant l'action risquée.

Ne pas contourner une erreur de sécurité, de licence, de permission ou de validation pour faire avancer artificiellement la tâche.

## 13. Git, commits et publication distante

L'agent est autorisé à effectuer des commits locaux dans le périmètre de travail autorisé.
**Commit automatique** : à la fin de chaque intervention autorisée ayant modifié des
fichiers versionnés, l'agent effectue un commit local sans demander d'autorisation
supplémentaire, à condition que les validations adaptées aient réussi. Un seul commit
par intervention, regroupant uniquement les fichiers de celle-ci. Aucun commit si
l'intervention a échoué, si les validations échouent, ou si des fichiers non liés à
la tâche devraient être inclus. Le push reste interdit (ci-dessous).

L'utilisateur conserve exclusivement le contrôle de la publication vers les dépôts distants.

Interdictions :
- ne jamais exécuter `git push` ;
- ne jamais exécuter `git push --force` ou une variante équivalente ;
- ne jamais publier de branche, tag ou commit sur un dépôt distant ;
- ne pas modifier les remotes, les droits d'accès ou les paramètres du dépôt distant sans autorisation explicite ;
- ne pas inclure volontairement des secrets, `.env`, des artefacts locaux ou des fichiers exclus ;
- ne pas réécrire l'historique partagé ou effectuer une opération destructive sans autorisation.

Avant un commit :
1. examiner `git status` ;
2. examiner les différences du code et des documents concernés ;
3. vérifier les fichiers suivis et non suivis qui seraient inclus ;
4. contrôler les éventuels secrets et fichiers volumineux ;
5. exécuter les validations adaptées au changement ;
6. ne committer que les fichiers relevant de la tâche.

Ne pas inclure les modifications préexistantes de l'utilisateur sans rapport avec la tâche.

Après un commit, indiquer :
- son hash ;
- son message ;
- les fichiers ou changements inclus ;
- les tests exécutés ;
- les éventuels points non résolus.

Ne jamais annoncer qu'un commit a été créé si la commande a échoué. Ne jamais annoncer qu'un push a été effectué.

## 14. Traçabilité des expériences et décisions

Les expériences susceptibles d'influencer les choix architecturaux ou les conclusions du projet doivent être reproductibles.

Enregistrer, lorsque cela s'applique :
- l'objectif de l'expérience ;
- le code et sa version ;
- les sources et checkpoints utilisés ;
- les paramètres, graines et configurations ;
- les versions logicielles pertinentes ;
- les commandes exécutées ;
- les métriques et résultats bruts ;
- les limites et anomalies observées.

Les critères d'acceptation d'une expérience doivent être établis avant l'évaluation concernée lorsque le protocole le requiert. Ne pas modifier rétroactivement un seuil pour requalifier un échec en succès.

Les décisions structurantes doivent être consignées dans un document dédié sous `docs/decisions/`, si cette organisation est pertinente pour la tâche. Une décision doit indiquer son statut, son contexte, ses alternatives, sa justification et ses conséquences.

Les résultats bruts ne doivent pas être remplacés par un résumé interprétatif. Les artefacts nécessaires à leur vérification doivent être conservés.

## 15. Communication et compte rendu

Les comptes rendus doivent être concis, précis et orientés vers les résultats.

Distinguer explicitement :
- **Fait établi** : directement étayé par une source identifiable ou une observation reproductible.
- **Résultat mesuré** : obtenu par une expérience effectivement exécutée, avec son protocole.
- **Hypothèse** : explication plausible restant à valider.
- **À vérifier** : question non résolue faute de preuve suffisante.
- **Décision proposée** : recommandation non encore autorisée.
- **Décision acceptée** : choix déjà validé.

Pour chaque intervention substantielle, fournir :
1. l'objectif traité ;
2. les modifications réalisées ;
3. les fichiers créés ou modifiés ;
4. les commandes et tests exécutés ;
5. les résultats observés ;
6. les limites, échecs et risques résiduels ;
7. les artefacts et moyens de reproduire les résultats ;
8. les décisions qui restent à prendre, s'il y en a.

Ne pas déclarer une tâche terminée si des critères importants restent insatisfaits. Distinguer clairement une réalisation partielle d'une validation complète.

## 16. Règle générale en cas de blocage

En cas de divergence numérique inexpliquée, de contradiction documentaire, de licence incertaine, de données potentiellement corrompues, de risque destructif ou de dépassement du périmètre autorisé :

1. interrompre l'opération concernée ;
2. préserver l'état du dépôt et les résultats existants ;
3. recueillir les éléments permettant de comprendre le problème ;
4. documenter les faits établis et les incertitudes ;
5. proposer la résolution la plus raisonnable ;
6. demander une décision uniquement si le blocage ne peut pas être levé dans le périmètre autorisé.

L'agent doit optimiser la fiabilité, la reproductibilité et la qualité scientifique de Jainius, et non le nombre de fichiers produits, de commits réalisés ou de phases déclarées terminées.