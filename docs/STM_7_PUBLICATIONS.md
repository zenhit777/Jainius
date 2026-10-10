# Recueil des 7 Spécifications Techniques Minimales

## Introduction

### Objectif du document

Ce fichier regroupe **7 STM indépendantes**, une par publication PDF du répertoire de travail. Chaque STM décrit **uniquement** ce que sa publication spécifie techniquement, signale ce qu'elle ne spécifie pas, et distingue faits, déductions et propositions. Le fichier est un conteneur : **aucune STM commune n'est produite**.

### Règles d'interprétation

| Étiquette | Sens |
|---|---|
| **[SOURCE]** | Information explicitement présente dans la publication |
| **[DÉDUCTION]** | Déduction directe et justifiable, non formulée par les auteurs |
| **[PROPOSITION]** | Suggestion d'implémentation de l'agent, jamais une caractéristique de la publication |
| **[NON SPÉCIFIÉ]** | Information absente de la publication (équivaut à « [NON SPÉCIFIÉ DANS LA PUBLICATION] ») |

- Les paramètres expérimentaux (learning rate, batch size, steps, température, tailles de modèle, GPU, coûts, métriques, seuils) sont des **paramètres expérimentaux**, sauf si la publication les déclare nécessaires à la méthode.
- Un résultat expérimental n'est pas une exigence technique.
- Chaque STM est autonome : aucune ne renvoie à une autre. Aucune hypothèse, valeur ou définition n'est transférée d'une publication à une autre.
- Les valeurs lues uniquement sur des **figures** sont marquées « ≈ ».
- Traçabilité : `REQ-<n°STM>-<nnn>` pointe vers section / équation / tableau / figure / algorithme / annexe de la publication concernée.
- L'ordre des STM suit l'ordre alphabétique des noms de fichiers (identifiant arXiv croissant) ; **il n'a aucune signification scientifique**.

### Corpus analysé

| STM | Fichier source | arXiv | Méthode |
|---|---|---|---|
| 1 | `2605_05940v1_-_Near-Policy__Accelerating_On-Policy_Distillation_via_Asynchronous_Generation_and_Selective_Packing_.pdf` | 2605.05940v1 | NPD (Near-Policy Distillation) |
| 2 | `2606_01774v2_-_FLARE__Diffusion_for_Hybrid_Language_Model_.pdf` | 2606.01774v2 | FLARE |
| 3 | `2606_06712v1_-_OPDLM__Data-Efficient_Autoregressive-to-Diffusion_Language_Models_via_On-Policy_Distillation_.pdf` | 2606.06712v1 | OPDLM |
| 4 | `2607_16872v1_-_TOPD__Trace-Based_On-Policy_Distillation_for_Masked_Diffusion_Language_Models_.pdf` | 2607.16872v1 | TOPD |
| 5 | `2609_15177v1_-_TSD__Temporal_Self-Distillation_-_Faster_Inference_in_Discrete_Diffusion_Language_Models_.pdf` | 2609.15177v1 | TSD |
| 6 | `2609_33301v1_-_HOPD__Hesitation-Aware_On-Policy_Distillation_for_Diffusion_Language_Models_.pdf` | 2609.33301v1 | HOPD |
| 7 | `2609_35362v1_-_d-OPD__Future-Aware_On-Policy_Distillation_for_Block_Diffusion_Language_Models_.pdf` | 2609.35362v1 | d-OPD |

**Note de lecture.** Les 7 PDF ont été lus via leur couche texte. Pour FLARE (52 pages), le corps principal, les ablations, les annexes de serving (B), de données (C) et d'hyperparamètres (D) ont été lus ; les dérivations mathématiques détaillées des kernels (annexe A, Algorithmes 3–10) n'ont été lues que de façon partielle.

---

# STM 1 — Near-Policy: Accelerating On-Policy Distillation via Asynchronous Generation and Selective Packing

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | Near-Policy: Accelerating On-Policy Distillation via Asynchronous Generation and Selective Packing |
| Auteurs | Miao Rang\*, Zhenni Bi\*, Hang Zhou, Kai Han, Xuechun Wang, An Xiao, Xinghao Chen, Yunhe Wang, Hanting Chen† (Huawei Technologies ; Tianjin University) — \* contribution égale, † auteur correspondant |
| Date / version | arXiv:2605.05940v1 [cs.LG], 7 mai 2026 ; « Preprint » |
| URL arXiv | [NON SPÉCIFIÉ] (seul l'identifiant figure dans le PDF) |
| Code | « Codes will be released soon » — aucune URL |
| Méthode | Near-Policy Distillation (NPD) |
| Objectif principal | Obtenir les bénéfices de la distillation on-policy avec l'efficacité d'un entraînement SFT avec sequence packing |

## 2. Résumé technique

[SOURCE] NPD découple la génération des réponses de l'étudiant de la mise à jour du gradient. Les réponses étudiant sont pré-générées, filtrées par un score ΔIFD, packées en séquences de longueur fixe ; l'enseignant calcule les top-k logits par un prefill parallèle ; l'étudiant est entraîné dans une architecture SFT avec une loss composite (1−λ)·CE + λ·KD. La politique de génération n'est rafraîchie que rarement (« sparse student updates »). Les auteurs rapportent un gain de vitesse de 8,1× par rapport à DistiLLM-2 et +8,09 points sur la moyenne de 11 benchmarks par rapport au SFT.

## 3. Problème traité

- [SOURCE] La distillation classique (KL sur corpus fixe) souffre d'un décalage de distribution : entraînement sur séquences forcées par l'enseignant, génération autoregressive à l'inférence.
- [SOURCE] Les méthodes on-policy corrigent ce décalage mais reposent sur des cadres synchrones de type RL, qui ne peuvent pas utiliser le sequence packing et ont une faible utilisation des tokens.
- [SOURCE] L'asynchronie introduit un « policy lag » : les échantillons peuvent devenir bruités ou dériver vers l'off-policy.

## 4. Objectif de la méthode

[SOURCE] Rapprocher la distillation des performances on-policy tout en conservant le débit du SFT avec packing ; limiter le policy lag par des mises à jour clairsemées et un filtre ΔIFD ; fournir une meilleure initialisation pour un RL ultérieur.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| Les sorties générées par l'étudiant s'alignent avec sa distribution mais contiennent du bruit de qualité | [SOURCE] | §3.2 |
| Un ΔIFD négatif signale des réponses courtes/triviales que l'enseignant juge probables et que l'étudiant génère avec incertitude | [SOURCE] | §3.2, zone dégénérée |
| Un ΔIFD > τ signale une incohérence enseignant/étudiant dommageable à l'optimisation | [SOURCE] | §3.2 |
| Le régime « near-policy » est défini par D_KL(π_courant ‖ π_gen) ≤ ε, avec ε ≈ 0,10 observé | [SOURCE] | App. C.1 |
| ΔIFD est un heuristique (« heuristic sample selection ») | [SOURCE] | §3.2, C.2 |
| Un déclenchement dynamique des mises à jour par seuil KL serait théoriquement possible, mais une allocation statique de 1–2 mises à jour suffit | [SOURCE] | App. B.3 |

## 6. Périmètre de la méthode

[SOURCE] Distillation d'étudiants **autoregressifs** (openPangu-Embedded, Qwen2.5-Math/Coder), phase de distillation puis, optionnellement, RL (GRPO). Non traité : étudiants non autoregressifs [DÉDUCTION : aucun n'est mentionné dans les extraits], enseignants multiples, distillation hors langage.

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Jeu de prompts D_x | Ensemble d'instructions | [SOURCE] Alg. 1 |
| Modèle étudiant θ_S | Initial | [SOURCE] |
| Modèle enseignant p_T | Gelé | [SOURCE] Alg. 1 |
| Longueur de pack L | Séquences packées | [SOURCE] Alg. 1 |
| Hyperparamètres η, K (intervalle de refresh), λ | — | [SOURCE] Alg. 1 |
| Seuil τ du filtre ΔIFD, k du top-k | — | [SOURCE] §3.2, App. B.4 |

## 8. Sorties

[SOURCE] Étudiant optimisé θ_S (Alg. 1, « Return Optimized θ_S »). Optionnel : étudiant après GRPO (§4.6). [NON SPÉCIFIÉ] Format de sortie/checkpoint.

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Moteur de génération batch (ex. vLLM) | Générer les réponses étudiant | [SOURCE] §3.1 |
| Calcul IFD enseignant et étudiant | Filtrage ΔIFD | [SOURCE] §3.2 |
| Packer de séquences | Concaténation en séquences fixes | [SOURCE] §3.1 |
| Enseignant en prefill parallèle | Calcul des top-k logits | [SOURCE] §3.1 |
| Entraîneur SFT avec loss CE + KD | Mise à jour de l'étudiant | [SOURCE] §3.1 |
| Buffer D_train de (séquence, logits enseignant) | Réutilisation entre epochs | [SOURCE] Alg. 1 |
| Moniteur de dérive KL | Mesure de policy lag | [SOURCE] App. C.1 (mesure) ; utilisation en entraînement : [NON SPÉCIFIÉ] |

## 10. Architecture / organisation de la méthode

```text
Prompts D_x ─► Étudiant (génération batch) ─► ΔIFD (zones) ─► Packing
                                                         │
          ◄── mise à jour θ_S ◄── Entraîneur (CE + KD) ◄── Enseignant (prefill, top-k logits)
                    │
                    └── refresh de la génération à l'epoch (e−1) mod K = 0  (Fig. 1, Alg. 1)
```
[SOURCE] Fig. 1 : trois phases — Generation & Filtering ; Teacher Annotation ; Student Updates.

## 11. Flux d'exécution

1. [SOURCE] Génération de D_S = {(x,y) | y ∼ p_θ(·|x)}.
2. [SOURCE] Filtrage ΔIFD → D_select.
3. [SOURCE] Packing → S_pack.
4. [SOURCE] Calcul des logits enseignant (sans gradient) → O_T ; D_train.
5. [SOURCE] Pour chaque minibatch : logits étudiant, probabilités, L_KD, L_total, mise à jour.
6. [SOURCE] Retour à l'étape 1 aux epochs de refresh.

## 12. Algorithme

[SOURCE] Algorithm 1 (Annexe A) — « Iterative Near-Policy Distillation (NPD) with ΔIFD » : pour e = 1…E ; si (e−1) mod K = 0 : génération, filtrage ΔIFD (« is valid »), packing, annotation enseignant, mise à jour du buffer ; puis boucle d'optimisation sur D_train : OS ← p_θ(S) ; pT ← Softmax(OT), pS ← Softmax(OS) ; L_KD ← Σ_{c∈V_top-k} pT(c) log(pT(c)/pS(c)) ; L_total ← (1−λ)L_CE(S) + λ L_KD ; θ_S ← θ_S − η∇L_total.

[NON SPÉCIFIÉ] Le pseudo-code ne précise pas si Softmax est appliqué sur le vocabulaire complet ou renormalisé sur le support top-k.

## 13. Équations et fonctions objectif

| Équation | Expression | Rôle | Variables |
|---|---|---|---|
| (1) | L_total = (1−λ)·L_CE + λ·L_KD | Loss composite | λ ∈ [0,1] |
| (2) | L_KD = Σ_{c∈V_top-k} p_T(c) log(p_T(c)/p_S(c)) | KL sur le support top-k de l'enseignant | V_top-k : indices des k plus grands logits enseignant |
| (3) | IFD(Q,A) = L_θ(A\|Q) / L_θ(A) | Difficulté de suivi d'instruction | L_θ : cross-entropy ; Q instruction, A réponse |
| (4) | ΔIFD = IFD_Teacher − IFD_Student | Écart de difficulté | — |
| App. C.1 | D_KL(π_current^(t) ‖ π_gen^(t)) ≤ ε | Définition du régime near-policy | ε ≈ 0,10 |

[SOURCE] La notation L_θ(·) de l'éq. (3) est ambiguë : le modèle sous lequel la cross-entropy est calculée n'est pas précisé pour IFD_Teacher / IFD_Student. [DÉDUCTION] L_θ est évalué avec les paramètres de l'enseignant, respectivement de l'étudiant.

## 14. Données

| Jeu | Usage | Taille | Source | Classe |
|---|---|---|---|---|
| Sous-ensemble « MetaQA » (cité avec la référence MetaMath, Yu et al. 2023) | Comparaison avec distillation sample-wise | 50k instances | §4.2 | [SOURCE] |
| Jeu SFT industriel interne | Comparaison à grande échelle, distillation principale | 1,2 million d'échantillons | §4.2 ; App. B.1, E | [SOURCE] |
| Jeu « 12M proprietary dataset » | Ratios de zones (Tab. 13) | « 12M » | App. C.2 | [SOURCE] — incohérent avec 1,2 M ailleurs |
| 7 000 problèmes de mathématiques | RL (GRPO) | 7 000 | §4.6 | [SOURCE] |
| WizardCoder | Code | [NON SPÉCIFIÉ] | App. D.3 | [SOURCE] (nom seul) |

[NON SPÉCIFIÉ] Splits train/validation/test des jeux d'entraînement, contenu et accessibilité du jeu interne, format, preprocessing.

## 15. Génération / préparation des données

- [SOURCE] Données étudiant : générées par l'étudiant courant, par inférence batch ; réponse gloutonne dans la configuration de base ; optionnel : K=3 trajectoires (1 gloutonne + 2 échantillonnées) (App. D.4).
- [SOURCE] Filtrage ΔIFD : conservation de 0 ≤ ΔIFD ≤ τ ; Tab. 13 : ≈ 96 % conservé, 2,60 % → 3,77 % dégénéré, 1,20 % → 0,04 % déconnexion cognitive (début → fin).
- [SOURCE] Packing : plusieurs réponses concaténées en séquences de longueur fixe (max 32 768 tokens) ; l'enseignant les traite en un passage parallèle.
- [SOURCE] Données enseignant : top-k logits uniquement, pas le vocabulaire complet.
- [NON SPÉCIFIÉ] Algorithme de packing, gestion des frontières entre réponses, nombre de réponses générées par prompt en base, température de génération.

## 16. Modèles

| Rôle | Modèle | Taille | Classe |
|---|---|---|---|
| Enseignant (principal) | openPangu-Embedded-7B | 7B | [SOURCE] §4.1 |
| Étudiant (principal) | openPangu-Embedded-1B | 1B | [SOURCE] |
| Enseignant / étudiant (comparaison sample-wise) | Qwen2.5-Math-7B (texte) / Qwen2.5-Math-7B-Inst (Tab. 1) ; Qwen2.5-Math-1.5B | 7B / 1,5B | [SOURCE] §4.2, Tab. 1 |
| Enseignant / étudiant (scalabilité) | 72B / openPangu-7B | 72B / 7B | [SOURCE] App. D.1 |
| Enseignant / étudiant (thinking) | openPangu-7B-think / openPangu-1B-think | 7B / 1B | [SOURCE] App. D.2 |
| Enseignant / étudiant (code) | Qwen2.5-Coder-7B-Inst / Qwen2.5-Coder-1.5B | 7B / 1,5B | [SOURCE] App. D.3 |
| Modèle de récompense | Aucun | — | [DÉDUCTION] (RL via GRPO ; fonction de récompense [NON SPÉCIFIÉ]) |

[SOURCE] Note : Tab. 1 mentionne Qwen2.5-Math-7B-Inst comme enseignant ; le texte §4.2 mentionne Qwen2.5-Math-7B.

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] gelé ; « pure evaluator » ; calcule les logits sur des séquences packées par prefill ; [SOURCE] fournit aussi IFD_Teacher.
- Étudiant : [SOURCE] génère ses propres réponses ; reçoit les mises à jour ; fournit IFD_Student.
- Modèle de référence : Non applicable à cette publication [DÉDUCTION].

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Rôle | Source |
|---|---|---|---|
| λ (poids KD) | 0,9 par défaut | Équilibre CE/KD | [SOURCE] App. B.4 (Tab. 11 : essai 0,5–1,0) |
| top-k | 10 par défaut | Support de la KD | [SOURCE] App. B.4 (Tab. 12 : 5/10/15/20) |
| τ (seuil ΔIFD) | 0,8 optimal (0,7 / 0,8 / 0,9) | Borne haute de zone proximale | [SOURCE] Tab. 5, §4.4 |
| Zone retenue | 0 ≤ ΔIFD ≤ 1 dans Tab. 3, 9 ; ≤ τ dans Tab. 5 | Filtre | [SOURCE] |
| Intervalle de refresh K / fréquence de mise à jour | 1 refresh par cycle de 10 epochs (à l'epoch 5) optimal | Policy sync | [SOURCE] Tab. 4, §4.5 |
| ε (borne near-policy) | ≈ 0,10 | Définition empirique | [SOURCE] App. C.1 |

**Paramètres expérimentaux**

| Paramètre | Valeur | Source |
|---|---|---|
| Epochs | 10 | [SOURCE] §4.1 |
| Optimiseur | AdamW, weight decay 0,1, gradient clipping 1,0 | [SOURCE] §4.1 |
| Learning rate | cosine decay de 1×10⁻⁵ à 1×10⁻⁶ | [SOURCE] §4.1 |
| Longueur max de séquence packée | 32 768 tokens | [SOURCE] §4.1 |
| Batch global | 2 M tokens | [SOURCE] §4.1 |
| GRPO | 300 pas ; 7 000 problèmes | [SOURCE] §4.6 |
| Contexte (thinking) | 32K tokens | [SOURCE] App. D.2 |
| Évaluation (Tab. 11–12) | décodage glouton, longueur 8K ; moyenne de 8 benchmarks | [SOURCE] |
| Matériel | nœud 8-NPU | [SOURCE] App. E |

**Paramètres non spécifiés** : [NON SPÉCIFIÉ] valeur de τ utilisée pour le résultat final 64,37 ; positions des refresh pour une fréquence 2 ; température de génération ; hyperparamètres GRPO (LR, taille de groupe, récompense) ; seed ; nombre de réponses par prompt ; k et λ pour les expériences de la Tab. 1 (« unless otherwise specified » renvoie aux défauts).

## 19. Boucle d'entraînement

[SOURCE] Alg. 1 : boucle sur E epochs ; rafraîchissement du jeu d'entraînement aux epochs où (e−1) mod K = 0 ; boucle sur minibatchs de D_train ; une mise à jour de gradient par minibatch. [SOURCE] « single-rollout-multiple-updates » : une génération sert plusieurs epochs (App. E).

## 20. Inférence / génération

[SOURCE] Génération d'entraînement par moteur batch ; évaluation avec décodage glouton, longueur 8K pour les ablations KD (Tab. 11–12). [NON SPÉCIFIÉ] Protocole d'inférence des benchmarks principaux (Tab. 6–8).

## 21. Fonction de perte

[SOURCE] Éq. (1)–(2). Le terme CE est calculé sur les séquences packées (S) [DÉDUCTION]. [NON SPÉCIFIÉ] Normalisation (par token / par séquence), masquage des tokens de prompt.

## 22. Optimisation

[SOURCE] AdamW ; WD 0,1 ; clip 1,0 ; cosine 1e-5 → 1e-6 ; descente de gradient θ_S ← θ_S − η∇L_total (Alg. 1). [NON SPÉCIFIÉ] β, ε d'Adam, warmup, précision numérique.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Filtre ΔIFD en trois zones (Fig. 2) : dégénérée (ΔIFD < 0, rejet ; exemples : réponses courtes/vides), proximale (0 ≤ ΔIFD ≤ τ, conservée ; IFD_Teacher ≤ 1), déconnexion cognitive (ΔIFD > τ, rejet). Comparaisons : PPL « weighted », IFD_Teacher[0,1], IFD_Student[0,1], ΔIFD à une borne (Tab. 3). [SOURCE] Surcoût du filtrage : 1,37 % du temps (calcul ΔIFD) + 0,01 % (filtrage) (Tab. 19).

## 24. Critères d'arrêt

[SOURCE] Nombre d'epochs fixe (10). [NON SPÉCIFIÉ] Critère d'arrêt adaptatif, early stopping.

## 25. Évaluation

[SOURCE] 11 benchmarks en 4 domaines : général (MMLU, CMMLU, C-Eval, IF-Eval, CLUEWSC) ; mathématiques (GSM8K, MATH-500) ; raisonnement (DROP, GPQA-Diamond) ; code (MBPP, HumanEval) (§4.1). Benchmarks supplémentaires : AIME24, AIME25, LiveCodeBench (App. D.1). Comparaisons : taux de précision, temps d'entraînement (NPU-heures/epoch), courbes de KL.

## 26. Métriques

[SOURCE] Précision/Pass@1/F1 (DROP) par benchmark ; moyenne arithmétique sur 11 benchmarks (« AVG ») ; NPU-heures ; KL(learner‖generator) ; loss d'entraînement. [NON SPÉCIFIÉ] Définition exacte de l'« AVG » (pondération), métrique de IF-Eval (« Prompt Strict » dans Tab. 7).

## 27. Protocole expérimental

[SOURCE] (a) Comparaison sample-wise : sous-ensemble 50k, 1 epoch, Qwen2.5-Math 7B → 1,5B, baselines KD supervisée, SeqKD, GKD, DistiLLM, DistiLLM-2 ; résultats † repris de DistiLLM-2. (b) Comparaison packée : jeu 1,2 M, 10 epochs, openPangu 7B → 1B. (c) Efficacité vs on-policy standard à pas fixés et à temps fixé (8 NPU-heures, Fig. 3). (d) Ablations. (e) RL aval (GRPO 300 pas). [NON SPÉCIFIÉ] Nombre de runs/seeds, intervalles de confiance, sélection de checkpoint.

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences.)

| Expérience | Résultat | Trace |
|---|---|---|
| Tab. 1 (50k, 1 epoch) : NPD GSM8K 82,87 ; MATH500 76,60 ; temps 1,48 NPU-h/epoch | DistiLLM-2 : 81,27 ; 74,20 ; 12,00 → « 8,1× » | Tab. 1 |
| Tab. 1 autres | KD sup. 78,62/75,00/1,17 ; SeqKD 82,41/72,60/2,06 ; GKD 80,17/73,60/6,54 ; DistiLLM 81,15/73,80/12,00 ; enseignant 89,31†/80,00 ; étudiant 77,33†/62,00 | Tab. 1 |
| Tab. 2/8 (1,2 M, AVG 11 benchmarks) | SFT 56,28 ; KD sup. 58,79 ; SeqKD 59,55 ; NPD 64,37 | Tab. 2, 8 |
| Après GRPO 300 pas | SFT 59,86 (+3,58) ; NPD 68,73 (+4,36) ; Qwen3-1.7B : 63,69 | Tab. 6, 7 |
| 72B→7B | SFT 38,32 ; NPD 41,73 (AIME25 : 35,00 → 33,12, seul recul) | Tab. 15 |
| Thinking 1B (32K) | 66,43 → 73,52 | Tab. 16 |
| Code (Qwen2.5-Coder 7B→1,5B) | NPD 43,58 / 63,01 (HumanEval/MBPP) ; DistiLLM-2 42,24 / 62,70 | Tab. 17 |
| Coût (1,2 M, 10 epochs, 8-NPU) | rollout 300 h (82,01 %) ; annotation 18 h (4,92 %) ; ΔIFD 5 h (1,37 %) ; filtrage 0,04 h ; packing 0,75 h ; entraînement 42 h (11,48 %) | Tab. 19 |

## 29. Ablations

| Ablation | Résultat (AVG) | Trace |
|---|---|---|
| Composants (cumulatif) | KD standard 58,79 → packed distillation (rollouts étudiant) 61,63 (+2,84) → +ΔIFD 62,78 (+1,15) → +sparse update 64,37 (+1,59) | Tab. 9 |
| Filtre | Non filtré 61,63 ; PPL 61,69 ; IFD_Teacher 62,69 ; IFD_Student 62,60 ; ΔIFD (0,∞) 61,77 ; (−∞,1) 61,76 ; [0,1] 62,78 | Tab. 3 |
| τ | 0,9 : 62,86 ; 0,8 : 63,05 ; 0,7 : 62,60 | Tab. 5 |
| Fréquence de refresh | 0 : 62,78 (335 NPU-h) ; 1 : 64,37 (659) ; 2 : 64,17 (983) ; sans ΔIFD : 61,63 / 63,26 / 63,43 / 63,12 (0/1/2/3) | Tab. 4, 10 |
| λ_KD | 0,5 : 53,94 ; 0,6 : 54,07 ; 0,7 : 53,88 ; 0,8 : 54,43 ; 0,9 : 55,29 ; 1,0 : 54,44 | Tab. 11 |
| top-k | 5 : 54,78 ; 10 : 55,29 ; 15 : 54,24 ; 20 : 54,40 | Tab. 12 |
| ΔIFD et RL | NPD sans ΔIFD 63,43 → 65,91 (+2,48) ; NPD 64,37 → 68,73 (+4,36) | Tab. 14 |
| Multi-chemin | 61,63 (335 h) ; + multi-chemin 62,82 (1000 h) ; + ΔIFD 63,29 (998 h) | Tab. 18 |

## 30. Contraintes techniques

- [SOURCE] L'enseignant doit pouvoir calculer des logits par prefill parallèle sur séquences packées.
- [SOURCE] La politique de génération et d'apprentissage doivent rester proches (KL ≲ 0,10 observé).
- [DÉDUCTION] Compatibilité de tokenizers entre enseignant et étudiant (les distributions sont comparées token à token).
- [NON SPÉCIFIÉ] Contraintes de mémoire de stockage des top-k logits.

## 31. Dépendances

[SOURCE] Moteur d'inférence batch (« e.g., vLLM ») ; NPU (nœud 8-NPU) ; openPangu-Embedded, Qwen2.5-Math/Coder ; GRPO (Guo et al. 2025) ; benchmarks cités §4.1. [NON SPÉCIFIÉ] Framework d'entraînement, versions logicielles, marque des NPU.

## 32. Coûts / ressources

[SOURCE] Nœud 8-NPU ; NPU-heures : voir Tab. 1, 4, 18, 19 (335 h pour 0 refresh ; 659 h pour 1 refresh ; 983 h pour 2) ; [SOURCE] Tab. 19 : somme des phases 365,79 h, alors que Tab. 4/18 rapportent 335 h pour la configuration sans refresh (non rapproché par les auteurs dans les extraits lus).

## 33. Limites identifiées par les auteurs

1. [SOURCE] L'efficacité de l'étudiant reste contrainte par la distribution de l'enseignant ; biais et logits sous-optimaux se propagent (§6).
2. [SOURCE] ΔIFD est une heuristique.
3. [SOURCE] Le rollout étudiant domine le coût (82,01 %).
4. [SOURCE] Impact statement : l'efficacité peut accélérer le déploiement ; gouvernance des données et transparence du filtrage nécessaires (§7).

## 34. Points non spécifiés

Voir la table GAP ci-dessous et §18.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Étudiant/enseignant (openPangu ou Qwen2.5), pipeline de génération batch, calcul IFD, packing 32 768, prefill enseignant top-k=10, loss avec λ=0,9, AdamW, cosine 1e-5→1e-6, 10 epochs, 1 refresh à l'epoch 5, filtre 0 ≤ ΔIFD ≤ τ.

## 36. Éléments impossibles à déterminer à partir de la publication

Contenu du jeu industriel interne ; code ; procédure précise de calcul d'IFD ; algorithme de packing ; hyperparamètres GRPO ; résolution des incohérences numériques internes (1,2 M vs 12M ; 335 h vs 365,79 h ; CLUEWSC 82,17 en Tab. 6 vs 82,87 en Tab. 7 ; τ=0,8 vs zone [0,1] dans le cumul de la Tab. 9).

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Calcul exact d'IFD (passe inconditionnelle, modèle utilisé) | Haute | Éq. (3)–(4) |
| GAP-002 | Renormalisation de p_T/p_S sur le top-k | Haute | Éq. (2) ; Alg. 1 l. 17–18 |
| GAP-003 | τ utilisé dans le résultat final 64,37 | Haute | Tab. 5 vs Tab. 9 |
| GAP-004 | Contenu/accès du jeu interne 1,2 M | Haute | §4.2 |
| GAP-005 | Algorithme de packing | Moyenne | §3.1 |
| GAP-006 | Hyperparamètres GRPO | Moyenne | §4.6 |
| GAP-007 | Seeds, variance | Moyenne | Tab. 1–19 |
| GAP-008 | Position des refresh pour fréquence 2 | Basse | Tab. 4 |
| GAP-009 | Température de génération des rollouts | Moyenne | §3.1 |
| GAP-010 | Disponibilité du code | Haute | Page 1 |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-1-001 | Loss composite (1−λ)CE + λKD | [SOURCE] | §3.1, Éq. (1) |
| REQ-1-002 | KD restreinte au top-k enseignant | [SOURCE] | §3.1, Éq. (2) |
| REQ-1-003 | Prefill enseignant sur séquences packées | [SOURCE] | §3.1 |
| REQ-1-004 | Filtre ΔIFD (3 zones) | [SOURCE] | §3.2, Éq. (3)–(4), Fig. 2 |
| REQ-1-005 | Refresh clairsemé | [SOURCE] | §4.5 ; Alg. 1 ; Tab. 4 |
| REQ-1-006 | Borne near-policy ε≈0,10 | [SOURCE] | App. C.1 ; Fig. 5 |
| REQ-1-007 | Paramètres d'entraînement | [SOURCE] | §4.1 |
| REQ-1-008 | Initialisation RL | [SOURCE] | §3.3, §4.6 |
| REQ-1-009 | Compatibilité de tokenizers | [DÉDUCTION] | — |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Moyenne** (pseudo-code présent ; détails de normalisation et d'IFD absents)
- Architecture : **Moyenne** (pipeline en 3 phases décrit ; composants logiciels non précisés)
- Paramètres : **Moyenne** (tables nombreuses ; plusieurs incohérences internes)
- Reproductibilité : **Faible** (jeu de données interne, code non disponible, détails manquants)

---

# STM 2 — FLARE: Diffusion for Hybrid Language Model

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | FLARE: Diffusion for Hybrid Language Model |
| Auteurs | Yuchen Zhu\*, Jing Shi\*, Chongjian Ge, Hao Tan, Yiran Xu, Wanrong Zhu, Jason Kuen, Koustava Goswami, Rajiv Jain, Yongxin Chen, Molei Tao, Jiuxiang Gu\*† (Adobe Research ; Georgia Institute of Technology) — \* contributeurs principaux, † chef de projet |
| Date / version | arXiv:2606.01774v2 [cs.LG], 4 août 2026 (tampon arXiv) ; en-tête « Date: June 2, 2026 » |
| URL | Page projet : tokflare.github.io ; code : github.com/yuchen-zhu-zyc/FLARE ; checkpoints : huggingface.co/collections/yuchen-zhu-zyc/flare [SOURCE] |
| Méthode | FLARE : conversion d'un LLM autoregressif à attention hybride en dLLM par blocs |
| Objectif principal | Convertir des LLM hybrides AR en dLLM capables et efficaces en serving, avec un budget d'entraînement réduit |

## 2. Résumé technique

[SOURCE] FLARE est une recette de conversion supervisée (SFT, une seule étape) : (1) un diagnostic identifiant la qualité/distribution des données de transfert comme déterminant principal ; (2) un objectif token-équilibré L = L_AR + L_diff avec flux « clean » causal et flux « noisy » bidirectionnel par bloc ; (3) une réalisation du masque sur couches à attention linéaire (Gated DeltaNet) par planification d'états récurrents et kernels dédiés ; (4) une inférence unifiée où un même checkpoint supporte AR-Trust (brouillons noisy vérifiés par le flux clean) et Diffusion-Trust (engagement par confiance). La méthode **n'est pas une distillation on-policy** ; aucun modèle enseignant n'intervient dans l'entraînement de FLARE [DÉDUCTION : les données de transfert sont des corpus SFT publics].

## 3. Problème traité

- [SOURCE] Le décodage AR séquentiel est un goulot de latence ; les backbones hybrides réduisent le coût par invocation, les dLLM réduisent le nombre de pas sériels ; les combiner est difficile.
- [SOURCE] Les conversions AR→dLLM échouent souvent à préserver les capacités du checkpoint source ; sur backbones hybrides, états récurrents et contraintes de masque compliquent entraînement et serving.
- [SOURCE] Plus de tokens par passe avant n'implique pas plus de tokens par seconde (préfixe, cache KV/état récurrent, surcoût du décodage parallèle).

## 4. Objectif de la méthode

[SOURCE] Convertir des LLM hybrides forts en dLLM qui conservent la capacité de la source, atteignent une efficacité d'entraînement proche de l'AR, et supportent deux régimes de décodage depuis un seul checkpoint.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| La qualité et l'adéquation de distribution des données de transfert dominent la préservation de capacité, devant la loss et le masque | [SOURCE] | Résumé ; §1 ; §4.1 |
| La capacité est gouvernée par l'alignement du flux clean (causal) avec la prédiction AR ; logit shift et échantillonnage de masque sont marginaux en précision | [SOURCE] | §4.2 |
| L'AR fine-tuning est un proxy peu coûteux pour sélectionner les mélanges de données | [SOURCE] | §4.1 |
| Le même schéma de planification d'états s'étend à d'autres couches linéaires (Mamba-2, Kimi Delta Attention) en substituant transition/écriture/lecture | [SOURCE] (affirmation) | §3.2 |
| Un backbone hybride doit implémenter la visibilité non causale par des mécanismes différents selon le type de couche | [SOURCE] | §3.1–3.2 |
| Exactitude d'AR-Trust : q_i doit être la loi qui a échantillonné d_i | [SOURCE] | §3.3 |

## 6. Périmètre de la méthode

[SOURCE] Checkpoints denses < 10B (Qwen3-1.7B pour l'étude contrôlée ; Qwen3.5-2B/4B/9B pour les modèles finaux), une étape SFT, taille de bloc B=4 pour les modèles finaux. Non testé [SOURCE §6] : MoE, post-entraînement au-delà du SFT (RL).

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Checkpoint AR source | Qwen3.5-2B/4B/9B post-entraînés ; Qwen3-1.7B (étude) | [SOURCE] §4, §5 |
| Données de transfert | Mélanges SFT publics (Long-CoT, Math, IF, Short-CoT) | [SOURCE] §4.1, App. C |
| Format SFT | Prompt = contexte clean sans perte ; réponse = blocs de taille B avec L_AR et L_diff | [SOURCE] §4 |
| Taille de bloc B | 4 | [SOURCE] §5 |
| (Inférence) prompt, longueur, seuils γ_s, (T, k, p), horizon N | — | [SOURCE] §3.3 ; App. B |

## 8. Sorties

[SOURCE] Un checkpoint qui produit deux jeux de logits dans un même passage : logits du flux clean (prédiction AR causale) et logits du flux noisy (débruitage de bloc). À l'inférence : texte généré en AR-Trust ou Diffusion-Trust.

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Masque clean/noisy packé par document | Visibilité : clean causal ; noisy bidirectionnel intra-bloc voyant le clean précédent ; isolation par document | [SOURCE] §3.2, Fig. 3 |
| Planification d'états récurrents (GDN) | Réaliser le masque sur couches linéaires : état noisy initialisé à l'état clean de frontière, lecture de l'état de fin de bloc, reset entre blocs | [SOURCE] §3.2, Éq. (6) |
| Kernels GDR et ShortConv Route II (fused two-stream) | Entraînement efficace à petit B | [SOURCE] §5.2, App. A |
| Pipeline de curation/packing de données | Mélanges, packing source-local 4 096 tokens | [SOURCE] App. C |
| Serveur SGLang modifié | AR-Trust, Diffusion-Trust, commit d'état récurrent, modes de masque, vérification fusionnée, CUDA graphs | [SOURCE] §5.3, App. B |

## 10. Architecture / organisation de la méthode

```text
Checkpoint AR (hybride softmax + GDN)
        ↓  SFT 1 étape (≈10 Md tokens), L = L_AR + L_diff, masque clean/noisy packé
Checkpoint FLARE (un seul)
        ├── AR-Trust : brouillons noisy → vérification par logits clean (rejet min{1,p/q})
        └── Diffusion-Trust : débruitage de bloc en ≤ S pas, engagement par confiance ≥ γ_s
                 + replay causal du bloc finalisé pour mettre à jour l'état récurrent
```
[SOURCE] Fig. 3 (masque), Fig. 18 (motifs d'attention de décodage).

## 11. Flux d'exécution

**Entraînement.** [SOURCE] Packing → pour chaque bloc : sous-ensemble masqué M_b et complément M_b^c (deux vues noisy complémentaires) → passage unique avec flux clean + noisy → L_AR + L_diff → mise à jour.
**Inférence Diffusion-Trust.** [SOURCE] Prefill causal du prompt (écriture d'état) → par bloc : graine clean, boucle de débruitage (forward bidirectionnel sans écriture d'état), engagement par seuil, forward causal de remise à jour d'état, graine suivante (Alg. 12).
**Inférence AR-Trust.** [SOURCE] Bloc actif de largeur 2N−1 (ancre, K brouillons, N−1 masques) → un forward → vérification gauche-droite → rewind de l'état récurrent à S(r) → nouveaux brouillons (Alg. 11).

## 12. Algorithme

[SOURCE] Algorithm 1 (Diffusion-Trust), Algorithm 2 (AR-Trust) dans le corps ; Algorithms 11 et 12 (détails de serving) en annexe B ; Algorithms 3–10 (kernels GDR et ShortConv, Route I/II) en annexe A. Le pseudo-code de la boucle d'entraînement elle-même est [NON SPÉCIFIÉ] (la méthode est décrite par les Éq. (4)–(6) et le texte).

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1) | o_t = Σ_{i∈V_t} softmax(q_tᵀk_i/√d_k) v_i | Attention softmax avec ensemble visible V_t |
| (2)–(3) | S_t = S_{t−1} + k_t v_tᵀ ; Gated Delta Rule : S_t = (α_t I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ, o_t = S_tᵀ q_t/√d_k | Attention linéaire / GDN |
| (4) | L_FLARE(θ) = L_AR(θ) + L_diff(θ) | Objectif total |
| (5) | L_AR = Σ_ℓ log p_θ(x^ℓ \| x_{<ℓ}) ; L_diff = Σ_b [Σ_{ℓ∈M_b} log p_θ(x^ℓ \| x̃_b, x_{<b}) + Σ_{ℓ∈M_b^c} log p_θ(x^ℓ \| x̃_b^c, x_{<b})] | Termes clean et noisy (deux vues complémentaires) |
| (6) | État clean : S_t selon (3) ; bloc noisy b : S̃_{e(b−1)} := S_{e(b−1)}, puis S̃_ℓ = (α̃_ℓ I − β̃_ℓ k̃_ℓ k̃_ℓᵀ) S̃_{ℓ−1} + β̃_ℓ k̃_ℓ ṽ_ℓᵀ ; lecture clean causale ; lecture noisy depuis l'état de fin de bloc S̃_{e(b)} | Réalisation du masque dans une couche GDN |
| (7) | A_s = {i∈R_s : c_i ≥ γ_s} si s<S ; A_S = R_S ; mise à jour de y et R_{s+1} | Diffusion-Trust |
| (8) | a_i ∼ Bern(min{1, p_i(d_i)/q_i(d_i)}) ; y_i = d_i si a_i = 1, sinon y* ∼ NormV([p_i − q_i]₊) | AR-Trust |
| (21), (22) | Politiques Softmax-Argmax et Truncated-Argmax (règles d'acceptation modifiées) | App. B.2 |

[SOURCE] Chaque token contribue un signal AR et un signal diffusion à poids unitaire car M_b et M_b^c partitionnent le bloc. [SOURCE] Un décalage de logits (logit shift) est appliqué aux termes noisy ; la formule exacte du décalage n'est pas donnée dans les extraits lus [NON SPÉCIFIÉ].

## 14. Données

| Jeu | Description | Taille | Source |
|---|---|---|---|
| Llama-Nemotron-Post-Training-Dataset (SFT) | Traces long-CoT ; ≈4,2 M échantillons (math 2,22 M ; code 953 K ; science 708 K ; chat 349 K) | 114 Go | [SOURCE] App. C.1 |
| Nemotron-Post-Training-Dataset-v2 | Corpus général ; splits anglais : 1 397 187 échantillons ; CoT court | 92 Go | [SOURCE] |
| Nemotron-Instruction-Following-Chat-v1 / v2 | 320 K / 1,99 M échantillons | 6,4 / 14,4 Go | [SOURCE] |
| Nemotron-Science-v1 | 226 K | 2,3 Go | [SOURCE] |
| Nemotron-Math-Proofs-v1 | 925 K | 27 Go | [SOURCE] |
| Competitive-Programming-v2, Nemotron-Cascade-1/2-SFT-Data (IF) | Utilisés dans Long-CoT et IF | — | [SOURCE] App. C.3 |

**Mélanges** [SOURCE] Tab. 7 : Mix 1 Long-CoT (1,0) ; Mix 2 Short-CoT+Math (0,5/0,5) ; Mix 3 Long-CoT+Math (0,5/0,5) ; Mix 4 Long-CoT+Math+IF (0,4/0,4/0,2). Mix 1 : filtre ≥ 3 000 tokens, 4 588 156 échantillons ≈ 18,3 Md tokens (16,8 Md tokens d'assistant) ; pool IF : 591 718 échantillons, ≈ 0,66 Md tokens. Poids de mélange arrivés « empiriquement » (§4.1).
[NON SPÉCIFIÉ] Splits de validation/test pour l'entraînement ; les benchmarks d'évaluation sont listés en §25.

## 15. Génération / préparation des données

- [SOURCE] Préprocessing commun (App. C.2) : fusion d'un éventuel champ reasoning_content dans le message assistant ; filtre sur `<think>` non fermé ; déduplication par hash (≈11 K doublons retirés du pool IF) ; filtre de longueur par estimation de tokens (préfiltre en caractères au double du seuil).
- [SOURCE] Packing source-local en séquences de 4 096 tokens par bin-packing glouton, ChatML Qwen3, labels −100 sur prompt/système, document-IDs pour masque document-causal ; troncature des échantillons > 4 096 ; tirage multinomial par séquence packée selon les poids du mélange.
- [SOURCE] Données de réponses : générées par des modèles externes (DeepSeek-R1 et variantes, Qwen3, GPT-OSS-120B selon les sources) — Aucune donnée n'est produite par le modèle en cours d'entraînement.
- [SOURCE] Pipeline automatique de sélection IFD (App. C.5) explorée mais **non utilisée** pour les modèles FLARE finaux.

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Initialisation de l'étude contrôlée | Qwen3-1.7B | [SOURCE] §4 |
| Modèles convertis | Qwen3.5-2B / 4B / 9B → FLARE-2B/4B/9B | [SOURCE] §5 |
| Références AR | Qwen3.5 publiés ; Qwen3-1.7B (référence à 8K en App. C.5) | [SOURCE] |
| Baselines dLLM | Mercury-2 (API), LLaDA-2.0/2.1 mini et flash, SDAR (1.7B, 4B, 8B, 30B-A3B) | [SOURCE] §5 |
| Teacher / reward model | Non applicable | [DÉDUCTION] |
| Évaluation | Pile d'inférence SGLang (checkpoints Qwen3.5 et FLARE) ; LLaDA et SDAR repris de leurs rapports techniques | [SOURCE] §5 |

## 17. Teacher / Student / Reference Model

Non applicable à cette publication (pas de distillation). [SOURCE] Le checkpoint AR source joue le rôle de point de départ et de référence de comparaison ; la vérification en AR-Trust utilise les logits du flux clean du même modèle.

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Source |
|---|---|---|
| Taille de bloc B | 4 | [SOURCE] §5 |
| Poids de L_AR et L_diff | 1 et 1 (token-balanced) | [SOURCE] §3.2 |
| Vues noisy | Masque aléatoire M_b et complément M_b^c | [SOURCE] §3.2 |
| Logit shift sur termes noisy | Oui | [SOURCE] §3.2 |
| Distribution/fraction de masquage M_b | [NON SPÉCIFIÉ] dans les extraits lus (Tab. 13 : « random mask sampling ») | — |
| Horizon N, budget S, seuils γ_s | [NON SPÉCIFIÉ] valeurs numériques | §3.3, App. B |

**Paramètres expérimentaux**

| Paramètre | Valeur | Source |
|---|---|---|
| Longueur de séquence | 4 096 | [SOURCE] §4 ; App. D.1 |
| Batch global | 256 | [SOURCE] |
| Pas d'optimisation | 9 000 (≈10 Md tokens) | [SOURCE] |
| Optimiseur | AdamW ; LR 1×10⁻⁵ ; warm-up 500 pas ; pas de décroissance | [SOURCE] App. D.1 |
| Matériel | 32 A100 80 Go (Qwen3), 32 H100 80 Go (Qwen3.5) ; 12 à 36 h par expérience | [SOURCE] App. D.1 |
| Inférence (benchmarks) | SGLang ; longueur max 32 768 ; température 1 ; top-p 0,95 ; top-k 50 | [SOURCE] App. D.1 |
| Débit | 1×A100 80 Go, bf16, max_new_tokens 2 048, ignore_eos, concurrence C ∈ {1,4,8} | [SOURCE] Fig. 9, Tab. 10 |

## 19. Boucle d'entraînement

[SOURCE] Une étape SFT unique de 9 000 pas, batch global 256 séquences packées de 4 096 tokens, perte L_FLARE sur les tokens de réponse (le prompt est visible mais exclu de la perte). [NON SPÉCIFIÉ] Pseudo-code de la boucle, précision numérique de l'entraînement (bf16 mentionné pour le débit), accumulation de gradient, checkpointing.

## 20. Inférence / génération

[SOURCE] Deux modes depuis un même checkpoint. Diffusion-Trust : blocs débruités sur au plus S pas ; engagement des positions de confiance c_i ≥ γ_s ; confiance = max_v p_θ(y_i = v|·) (App. B.1) ; aucune vérification clean ; replay causal pour valider l'état. AR-Trust : brouillons issus des logits noisy, vérification par logits clean ; politiques de brouillon **Exact-Truncated**, **Softmax-Argmax**, **Truncated-Argmax** (App. B.2) ; Exact-Truncated stocke qi utilisé au tirage ; les deux autres l'approximent (le texte rapporte l'absence de dégradation significative observée). Mécanismes de serving : commit d'état récurrent (enregistrement des états intermédiaires et gather-scatter fusionné), modes de masque natifs, kernels de vérification et top-k fusionnés, sécurité de rejeu des graphes CUDA (§5.3).

## 21. Fonction de perte

[SOURCE] Éq. (4)–(5). Variantes comparées en ablation : AM-Causal (noisy « all-mask » purement causal), AM-Bidir (all-mask bidirectionnel intra-bloc), diffusion de bloc pur, ajout du flux clean causal, ajout de la perte NTP clean, ajout du logit shift (Tab. 12–14).

## 22. Optimisation

[SOURCE] AdamW, LR 1e-5, warm-up 500, pas de decay (App. D.1). [NON SPÉCIFIÉ] β, weight decay, gradient clipping.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Sélection par longueur et par poids de source (Long-CoT : ≥ 3 000 tokens). Échantillonnage multinomial par séquence packée. Pipeline IFD (App. C.5) : explorée mais non utilisée dans les runs finaux ; comparaison unfiltered vs auto-selected sur Qwen3-1.7B (Tab. 9) : GSM8K 82,79 → 85,14 ; MATH-500 84,40 → 79,80 ; IFEval 67,10 → 63,59 ; HumanEval 66,46 → 70,12 ; MBPP 58,37 → 66,54 ; ARC-C 82,51 → 85,58 (évaluation à 16K).

## 24. Critères d'arrêt

[SOURCE] 9 000 pas fixes ; checkpoint final. [NON SPÉCIFIÉ] Critère de sélection de checkpoint ; critère d'arrêt adaptatif.

## 25. Évaluation

[SOURCE] Suite de 12 benchmarks : GSM8K, MATH-500, AIME 24/25, ARC-Challenge, GPQA-Diamond ; MMLU, MMLU-Pro, IFEval ; HumanEval, MBPP, LiveCodeBench v6. Débit en tokens/s sur GSM8K, HumanEval, GPQA-Diamond (Tab. 10). MFU d'entraînement (Fig. 8).

## 26. Métriques

[SOURCE] Précision / pass-rate par benchmark ; scores groupés (Math + Reasoning, Knowledge + IF, Code) pour l'étude contrôlée ; débit (tokens/s) ; MFU. [NON SPÉCIFIÉ] Mode de calcul des moyennes de groupes ; nombre de seeds d'évaluation.

## 27. Protocole expérimental

[SOURCE] (a) Étude contrôlée sur Qwen3-1.7B, 4 mélanges × 4 recettes (AR-SFT, AM-Causal, AM-Bidir, FLARE) ; (b) ablations d'objectif/masque sur Mix 1 ; (c) conversion de Qwen3.5-2B/4B/9B sur Mix 4, mêmes réglages ; (d) microbenchmarks de kernels (Qwen3.5-2B) ; (e) MFU sur 8×A100-80 Go ; (f) débit en serving. [SOURCE] Les lignes LLaDA et SDAR proviennent de leurs rapports techniques ; Mercury-2 via API ; Qwen3.5 et FLARE mesurés dans la pile SGLang des auteurs. [SOURCE] Dans les tableaux détaillés de l'annexe, les checkpoints non-AR sont décodés en mode AR-Trust sauf mention contraire.

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences ; « \* » : potentiellement sous-estimé selon les auteurs.)

| Modèle / mode | Résultats principaux | Trace |
|---|---|---|
| FLARE-9B, AR-Trust / Diffusion-Trust (source Qwen3.5-9B) | MATH-500 95,20 / 93,60 (96,60) ; AIME-24 63,33 / 60,00 (65,56) ; MMLU-Pro 77,39 / 74,73 (81,39) ; GPQA-D 71,21 / 64,65 (80,30) ; MBPP 91,05 / 82,10\* (89,11) ; LCB v6 49,71 / 5,71\* (49,71) | Tab. 1 |
| FLARE-4B | MATH-500 94,20 / 91,60 (95,40) ; AIME-24 58,89 / 55,56 (63,33) ; GPQA-D 63,64 / 64,65 (80,30) | Tab. 2 |
| FLARE-2B | MATH-500 84,40 / 82,20 (72,20\*) ; AIME-24 31,11 / 31,11 | Tab. 2 |
| MFU FLARE-2B (B=4, 8×A100) | 13,80 % → 17,93 % → 21,40 % → 24,81 % (référence AR 24,04 %) | Fig. 8 |
| Kernel GDR, B=1 | latence 135,10 ms → 37,69 ms ; mémoire 18,14 GiB → 0,45 GiB | Fig. 7 |
| Débit C=8 sur 1×A100 | FLARE-2B GSM8K 2 087 tok/s (LLaDA-2.1-mini 963 ; SDAR-1.7B 438) ; GPQA-D 1 441 tok/s ; FLARE-4B 1 293 / 990 ; FLARE-9B 1 096 / 786 | Fig. 9 |
| Comparaison AR-SFT / FLARE (Tab. 11) | p. ex. 9B : AR-SFT MATH-500 95,00 ; FLARE 95,20 ; IFEval 76,52 vs 71,35 | Tab. 11 |

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| Recette cumulative (Mix 1, Qwen3-1.7B ; Math+Reasoning / Knowledge+IF / Code) | (a) AR-SFT 51,93 / 57,29 / 54,81 ; (b) block diffusion 40,98 / 31,29 / 26,47 ; (c) + contexte causal 43,09 / 48,42 / 49,19 ; (d) + perte NTP 52,70 / 54,53 / 51,41 ; (e) + logit shift (FLARE) 51,00 / 56,91 / 51,93 | Tab. 12 |
| Échantillonnage du masque et shift | AM sans shift 52,83 / 55,67 / 51,59 ; AM avec shift 52,05 / 56,44 / 52,45 ; D sans shift 52,70 / 54,53 / 51,41 ; D avec shift (FLARE) 51,00 / 56,91 / 51,93 | Tab. 13 |
| Données × recette | Qwen3-1.7B référence : 64,70 / 66,74 / 65,60 ; Mix 4 : AR-SFT 53,31 / 59,35 / 54,43 ; FLARE 53,09 / 59,19 / 52,66 ; AM-Causal 47,88 / 57,57 / 50,35 ; AM-Bidir 49,09 / 59,14 / 52,06 (autres mélanges : Tab. 14) | Tab. 14 |
| Texte | « −21,8 points » en moyenne pour la diffusion pure vs AR-SFT ; « +14,0 » en restaurant le flux clean causal | §4.2 |
| Route I vs II | Route II réduit latence et mémoire à petit B ; Route I gagne à B=16 pour GDR | Fig. 7 ; Tab. 4–5 |
| Pipeline IFD | Voir §23 | Tab. 9 |

## 30. Contraintes techniques

- [SOURCE] Sur couches linéaires, la visibilité clean/noisy ne peut pas être imposée par un masque dense ; elle devient un calendrier d'écritures, de propagations et de resets d'état.
- [SOURCE] À petit B (B=1,4), Route II est une exigence structurelle de mémoire (« structural requirement rather than an optimization »).
- [SOURCE] En Diffusion-Trust, les forwards de débruitage ne doivent pas écrire l'état récurrent ; l'écriture se fait par un forward causal sur le bloc finalisé.
- [SOURCE] La validité distributionnelle d'AR-Trust exige que q_i soit la loi réellement utilisée pour tirer d_i.
- [SOURCE] La réutilisation de graphes CUDA doit correspondre à la taille de bloc, au mode de masque, au mode de mise à jour d'état et au mode de sortie des logits.
- [SOURCE] L'entraînement concatène une vue clean et une vue noisy en une entrée de longueur 2L.

## 31. Dépendances

[SOURCE] Qwen3 / Qwen3.5 ; Gated DeltaNet ; Triton (kernels de vérification, mentions) ; SGLang ; FlashAttention (référence de style de calcul) ; corpus Nemotron ; GPU A100/H100 80 Go ; baselines LLaDA-2.x, SDAR, Mercury-2 (API). [NON SPÉCIFIÉ] Versions logicielles et licences.

## 32. Coûts / ressources

[SOURCE] 32 GPU A100 (Qwen3) / 32 H100 (Qwen3.5), 12–36 h par expérience ; ≈ 10 Md tokens, 9 000 pas ; coût d'entraînement ≈ 2× un entraînement AR de même taille (limite i) ; MFU 24,81 % ; débit : voir Tab. 10. Mémoire kernel : 0,45 à 18,14 GiB selon la route (GDR B=1).

## 33. Limites identifiées par les auteurs

1. [SOURCE] Surcoût d'entraînement ≈ 2× vs AR (entrée de longueur 2L) ; amplifié sur hybrides ; non éliminé par les kernels.
2. [SOURCE] Écart résiduel à la source : dérive de distribution des données de transfert ; le filtrage IFD automatique ne comble pas l'écart.
3. [SOURCE] Validation limitée aux denses < 10B, une étape SFT ; MoE et RL non testés.
4. [SOURCE] Diffusion-Trust plus faible sur le code (extraction de réponse fragile, troncatures, absence de vérification gauche-droite).

## 34. Points non spécifiés

Voir table GAP ci-dessous.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Checkpoints Qwen3.5 ; mélanges Mix 1–4 (sources, filtres, poids) ; SFT 9 000 pas, batch 256, séquence 4 096, AdamW 1e-5, warm-up 500 ; objectif L_FLARE avec B=4 ; masque clean/noisy packé ; kernels Route II ; SGLang modifié ; seuils γ_s et paramètres de décodage.

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Distribution du taux de masquage de M_b (random mask) | Haute | §3.2 ; Tab. 13 |
| GAP-002 | Formule exacte du logit shift | Haute | §3.2 |
| GAP-003 | Valeurs de γ_s, S, N utilisées par benchmark | Haute | §3.3 ; App. B |
| GAP-004 | β, weight decay, clipping d'AdamW ; précision d'entraînement | Moyenne | App. D.1 |
| GAP-005 | Pseudo-code de la boucle d'entraînement | Moyenne | §3.2 |
| GAP-006 | Critère de sélection de checkpoint | Moyenne | §5 |
| GAP-007 | Nombre de seeds d'évaluation | Moyenne | §4–5 |
| GAP-008 | Dérivations complètes des kernels (annexe A) : lecture partielle par l'agent | Moyenne | App. A |
| GAP-009 | Licences des corpus | Basse | App. C |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-2-001 | Objectif L_AR + L_diff, deux vues complémentaires | [SOURCE] | §3.2, Éq. (4)–(5) |
| REQ-2-002 | Masque clean/noisy packé par document | [SOURCE] | §3.2, Fig. 3 |
| REQ-2-003 | Planification d'états GDN | [SOURCE] | §3.2, Éq. (6) |
| REQ-2-004 | Diffusion-Trust | [SOURCE] | §3.3, Alg. 1, Éq. (7), App. B.3 |
| REQ-2-005 | AR-Trust et politiques de brouillon | [SOURCE] | §3.3, Alg. 2, Éq. (8), App. B.2 |
| REQ-2-006 | Kernels Route I/II | [SOURCE] | §5.2, App. A |
| REQ-2-007 | Mécanismes de serving | [SOURCE] | §5.3, App. B.4 |
| REQ-2-008 | Mélanges de données | [SOURCE] | §4.1, App. C, Tab. 7 |
| REQ-2-009 | Hyperparamètres d'entraînement | [SOURCE] | App. D.1 |
| REQ-2-010 | Exactitude AR-Trust | [SOURCE] | §3.3 |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Moyenne** (objectif et décodage clairs ; kernels lus partiellement ; loi de masquage et logit shift non précis)
- Architecture : **Moyenne** (hybride softmax/GDN et deux flux décrits ; détails kernels partiels)
- Paramètres : **Moyenne** (entraînement bien documenté ; paramètres de décodage manquants)
- Reproductibilité : **Moyenne** (code et checkpoints annoncés ; plusieurs paramètres critiques absents du texte lu)

---

# STM 3 — Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation (OPDLM)

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation |
| Auteurs | Xingyu Su\*, Jacob Helwig\*, Shubham Parashar\*, Atharv Chagi, Lakshmi Jotsna, Degui Zhi, James Caverlee, Dileep Kalathil, Shuiwang Ji (Texas A&M University ; UT Health Science Center at Houston) — \* contribution égale |
| Date / version | arXiv:2606.06712v1 [cs.CL], tampon arXiv 4 juin 2026 ; en-tête « June 8, 2026 » |
| URL | Homepage : opdlm.vercel.app ; GitHub : github.com/divelab/OPDLM ; HuggingFace : huggingface.co/collections/divelab/opdlm [SOURCE] |
| Méthode | OPDLM (On-Policy Diffusion Language Model), self-OPD |
| Objectif principal | Convertir un modèle autoregressif (ARLM) en modèle de langage à diffusion (DLM) par distillation on-policy, avec très peu de tokens d'entraînement |

## 2. Résumé technique

[SOURCE] L'étudiant est un ARLM dont l'attention est rendue bidirectionnelle par blocs (initialisé du même checkpoint que l'enseignant) ; il génère ses propres trajectoires de débruitage inverse. Un état partiellement débruité est tiré dans chaque trajectoire. Pour chaque position masquée, la cible est la distribution de l'ARLM d'origine, gelé, conditionnée sur le préfixe causal de la **séquence terminale** de la trajectoire. La perte est une KL directe D_KL(p_ARLM ‖ p_θ), avec poids w(t)=1. Un curriculum de longueur de rollout stabilise les premières itérations. Les auteurs rapportent 15× à 7 000× moins de tokens que des baselines DLM.

## 3. Problème traité

- [SOURCE] Deux décalages dans la conversion ARLM→DLM classique : (1) transition de l'objectif next-token vers l'objectif DLM, qui peut faire perdre des connaissances ; (2) décalage train/inférence : perte définie sur séquences masquées aléatoirement vs trajectoires d'inférence par décodage fondé sur la confiance.
- [SOURCE] Standard OPD suppose que l'enseignant et l'étudiant partagent l'espace d'états ; un enseignant DLM serait requis pour fournir des cibles sur des états masqués, ce que la méthode veut éviter.

## 4. Objectif de la méthode

[SOURCE] Formuler la conversion ARLM→DLM comme un post-entraînement ; entraîner l'étudiant sur ses propres états d'inférence ; conserver les connaissances de l'ARLM par distillation de ses distributions ; produire des DLM généralistes ou spécialisés.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| La perte DLM standard est évaluée sur des états forward-masking uniformes, différents des états inverses de l'inférence | [SOURCE] | §3 |
| L'ARLM peut fournir des cibles token-level sur des états partiellement masqués via le préfixe causal de la séquence terminale | [SOURCE] | §4.2 |
| w(t)=1 suffit (en suivant la référence [60] de la publication) | [SOURCE] | §4.2 |
| Les rollouts longs précoces sont instables après conversion, d'où le curriculum | [SOURCE] | §4.3 |
| Les données on-policy sont le principal moteur ; la trajectoire de masquage a un effet limité (bloc 4) | [SOURCE] | §5.3 |
| Même taille d'enseignant suffit pour les étudiants 4B+ ; un enseignant plus grand aide pour 0,6B (hypothèse des auteurs) | [SOURCE] | App. A.2 |

## 6. Périmètre de la méthode

[SOURCE] Famille Qwen3 (0,6B–8B), diffusion par blocs (taille 4 pour les résultats principaux ; 8 et 16 pour l'étude de parallélisme), formats non-thinking (principal) et thinking (OPDLM-MATH-Thinking). Non traité [SOURCE App. C] : distillation de raisonnement à grande échelle sur données générales ; cross-famille ; étude systématique de la taille d'enseignant.

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| ARLM enseignant gelé p_ARLM | Checkpoint Qwen3 d'origine | [SOURCE] §4.2 |
| DLM étudiant p_θ | Même checkpoint, attention bloc-causale | [SOURCE] §5.1 |
| Sampler S | Décodage inverse (remasquage dynamique, seuil 0,9) | [SOURCE] Alg. 1 ; Tab. 13 |
| Prompts | « Since OPDLM is an on-policy method, we only keep the prompts » | [SOURCE] §5.1 |
| Hyperparamètres | Tab. 13 | [SOURCE] |

## 8. Sorties

[SOURCE] Un DLM par blocs (checkpoints OPDLM-0.6B/1.7B/4B/8B ; OPDLM-MATH-4B/8B ; variantes Thinking). [SOURCE] Alg. 1 retourne la perte L d'un pas d'entraînement.

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Sampler de trajectoires inverses | Générer τ = (x_{t_T}, …, x_{t_0}) | [SOURCE] §3, Alg. 1 |
| Sélecteur d'état | Tirer k ∼ U{1,…,T} | [SOURCE] Alg. 1 |
| Enseignant ARLM avec logit shifting | Cibles sur préfixe causal | [SOURCE] §5.1 |
| Étudiant à attention bloc-causale | Prédictions aux positions masquées | [SOURCE] §5.1 |
| Calcul de perte KL par bloc et position | — | [SOURCE] Alg. 1 |
| Contrôleur de curriculum | Longueur max de rollout | [SOURCE] §4.3 |
| Enregistrement des instants de dénoisage | Reconstruire les états visités | [DÉDUCTION] (τ est « viewed as a map on realized times ») |

## 10. Architecture / organisation de la méthode

```text
Prompt ─► Étudiant DLM (rollout inverse, sampler dynamique 0.9) ─► τ = (x_tT … x_t0)
                                                              │
                                  x⁰ = terminal ; k ~ U{1..T} ; x_t = τ(t_k)
                                                              │
   ARLM gelé : p_T(·| x⁰_{<b}, x⁰_{b,<i}) ───────►  D_KL(p_T ‖ p_S)  ◄─── étudiant p_θ^{b,i}(·| x⁰_{<b}, x_t^b)
                                                              │
                                                   mise à jour de l'étudiant
```
[SOURCE] Fig. 2 (cadre d'entraînement, un seul bloc de taille 4 représenté).

## 11. Flux d'exécution

1. [SOURCE] Échantillonner une trajectoire inverse.
2. [SOURCE] Fixer la séquence terminale x⁰ ; tirer k ; fixer x_t = τ(t_k).
3. [SOURCE] Pour chaque bloc b et chaque position masquée i : cible = p_ARLM(·|x⁰_{<b}, x⁰_{b,<i}) ; prédiction étudiant = p_θ^{b,i}(·|x⁰_{<b}, x_t^b) ; accumuler D_KL(p_T ∥ p_S).
4. [SOURCE] Mettre à jour l'étudiant ; augmenter la longueur max de rollout selon le curriculum.

## 12. Algorithme

[SOURCE] Algorithm 1 « OPDLM Training Step » : entrée ARLM gelé, DLM étudiant, sampler S ; rollout inverse ; x⁰ ← τ(t_0) ; k ∼ U({1,…,T}) ; t ← t_k ; x_t ← τ(t_k) ; L ← 0 ; pour chaque bloc b et position masquée i : p_T ← p_ARLM(·|x⁰_{<b}, x⁰_{b,<i}) ; p_S ← p_θ^{b,i}(·|x⁰_{<b}, x_t^b) ; L ← L + D_KL(p_T ∥ p_S) ; sortie L. [NON SPÉCIFIÉ] Normalisation de L (somme vs moyenne), traitement du batch de trajectoires, étape d'optimisation (non écrite dans l'algorithme).

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1) | L_MDLM(θ) = −E[(1/t) Σ_{i∈M(x_t)} log p_θ(x⁰_i \| x_t)] | Perte MDLM standard (forward masking), temps t ∼ U(0,1) |
| (2) | L_block(θ) = −E[Σ_b (1/t) Σ_{i∈M(x_t)} log p_θ(x⁰_{b,i} \| x⁰_{<b}, x_t^b)] | Perte de diffusion par blocs |
| (3) | L_OPD(θ) = E_{x̂∼p_θ}[(1/\|x̂\|) Σ_i D_KL(p_φ(·\|x̂_{<i}) ∥ p_θ(·\|x̂_{<i}))] | OPD autoregressif (rappel) |
| (4) | L_{Γ,ν,φ}(θ) = E_{τ∼Γ, t∼ν(·\|τ)}[Σ_b Σ_{i∈M(x_t^b)} w(t) D_KL(φ^{b,i}(·\|x⁰) ∥ p_θ^{b,i}(·\|x⁰_{<b}, x_t^b))] | Objectif généralisé ; recouvre (2) pour Γ=forward, ν=U(0,1), w=1/t, φ=cibles dures |
| (5) | L_on(θ) : Γ = Γ_θ^rev, ν = ν^rev | Objectif on-policy |
| (OPDLM) | L_OPDLM(θ) = E[Σ_b Σ_i w(t) D_KL(p_ARLM(·\|x⁰_{<b}, x⁰_{b,<i}) ∥ p_θ^{b,i}(·\|x⁰_{<b}, x_t^b))] | Objectif OPDLM (équation non numérotée, §4.2) |
| (6) | L_OPDLMoff : x⁰ ∼ p_θ, t ∼ U(0,1), x_t ∼ q_{t\|0}(·\|x⁰) | Variante d'ablation (masquage aléatoire sur séquence terminale) |
| Curriculum | L_s = min(L_max, L_min + ⌊(s/S_warm)(L_max − L_min)⌋) | Longueur de rollout au pas s |
| (7) | FLOPs ≈ 2 N_teacher · T + 6 N_student · T | Estimation de coût |

Variables : M(x) = positions masquées ; T = nombre de pas du sampler (ici nombre d'états du rollout) ; w(t) = 1 ; φ^{b,i} = distribution cible. [SOURCE] Dans (1), le facteur 1/t vient du schedule linéaire α_t = 1−t. [SOURCE] Dans (7), T = nombre total de tokens d'entraînement (échantillons × longueur de génération, sommé sur les étapes du curriculum) ; les auteurs notent que la formule surestime légèrement le coût OPDLM (≈1,5 token/pas en mode bloc vs 1 supposé).

## 14. Données

| Jeu | Domaine | Taille | Source | Classe |
|---|---|---|---|---|
| Corpus d'entraînement général | Math 20 222 ; code 21 594 ; science 10 000 ; chat 10 000 (total 61 816, « ∼60K ») | 61 816 prompts | DAPO, Nemotron-v2-Math ; TACO, KodCode-Light-RL, AceCode ; Nemotron-v2-STEM ; Nemotron-v2-Chat | [SOURCE] §5.1 ; Tab. 12 |
| Données OPDLM-MATH | Problèmes MATH niveaux 3–5 | 8K | MATH training set, mêmes données que le baseline comparé | [SOURCE] §5.5 |
| Évaluation | Voir §25 | — | — | [SOURCE] App. B.2 |

[SOURCE] Seuls les prompts sont conservés (les réponses de ces jeux ne sont pas utilisées). [NON SPÉCIFIÉ] Prétraitement/déduplication/filtrage autre que la mention « curated corpus » ; splits de validation ; format de stockage. [SOURCE] Les sections de la publication désignent ce corpus par « pretraining dataset » (Tab. 12, B.1) alors qu'il s'agit de données de post-entraînement (prompts).

## 15. Génération / préparation des données

[SOURCE] Données générées par l'étudiant : trajectoires inverses échantillonnées avec température 1,0 et remasquage dynamique (seuil 0,9), bloc 4. Données enseignant : distributions du modèle gelé sur préfixes causaux de la séquence terminale. Pas de données synthétiques produites par l'enseignant utilisées comme cibles dures (cibles = distributions complètes) [DÉDUCTION]. [SOURCE] Ablation « Off-Policy Distillation » : génération par l'enseignant + masquage aléatoire (voir §29). [NON SPÉCIFIÉ] Nombre de rollouts par prompt, format de stockage.

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Enseignant (principal) | Qwen3 (0,6B/1,7B/4B/8B) de même taille que l'étudiant, causal, avec logit shifting | [SOURCE] §5.1 |
| Étudiant | Même checkpoint, attention bloc-causale | [SOURCE] §5.1 |
| Enseignants de taille supérieure (ablation) | Qwen-32B (non-thinking) pour 4B/8B ; Qwen3-4B pour 0,6B | [SOURCE] App. A.2 |
| Modèle de référence / récompense | Aucun (« no rewards ») | [SOURCE] §5.5 |
| Baselines | LLaDA-8B, Dream-7B, SDAR (1.7B/4B/8B), Fast-dLLM-v2 (1.5B/7B), Simple-dLLM (0,6B), TraDo (comparaison OPDLM-MATH) | [SOURCE] §5.1, Tab. 1, 5, 6 |
| Modèle d'évaluation | Aucun modèle juge mentionné | [DÉDUCTION] |

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] ARLM original gelé, attention causale, avec logit shifting ; interrogé en inférence (forward uniquement) ; fournit p_ARLM(·|préfixe causal de x⁰).
- Étudiant : [SOURCE] DLM initialisé du même checkpoint, attention bloc-causale ; génère ses trajectoires et reçoit le gradient.
- Modèle de référence : Non applicable à cette publication.
- [SOURCE] « The teacher and student are the same models … differing only in use of logit shifting for the ARLM and their attention mask. »

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Rôle | Source |
|---|---|---|---|
| Divergence | KL directe D_KL(p_ARLM ∥ p_θ) | Perte | [SOURCE] Tab. 13 |
| w(t) | 1 | Pondération temporelle | [SOURCE] §4.2 |
| État par trajectoire | 1 (indice k uniforme sur états non terminaux) | Sélection | [SOURCE] Alg. 1 |
| Taille de bloc | 4 (résultats principaux) | Rollout et évaluation | [SOURCE] Tab. 13 |
| Curriculum | L_min = 100, L_max = 4 000, « Rollouts to reach final generation length » = 10 | Stabilité | [SOURCE] Tab. 13 |
| Remasquage | Dynamique, seuil de confiance 0,9 | Rollout | [SOURCE] Tab. 13 |

**Paramètres expérimentaux**

| Paramètre | Valeur | Source |
|---|---|---|
| Époques de données | 1 | [SOURCE] Tab. 13 |
| Tâches par rollout | 128 | [SOURCE] |
| Batch d'entraînement effectif | 8 | [SOURCE] |
| Learning rate | 1×10⁻⁵, final 1×10⁻⁶, cosine avec warmup de 5 pas | [SOURCE] |
| Température de rollout | 1,0 | [SOURCE] |
| OPDLM-MATH (Tab. 14) | 10 époques ; longueur finale 2 000 (non-thinking) / 8 000 (thinking) ; 10 / 20 rollouts pour atteindre la longueur finale | [SOURCE] |
| Matériel | H200 (modèles finaux) ; A6000 et A100 (ablations) ; < 4B entraînables sur 2 A6000 | [SOURCE] App. B.5 |
| Évaluation | Décodage glouton statique (1 token/pas), bloc 4, sauf mention | [SOURCE] §5.1 |
| Seeds d'évaluation | AIME : 32 ; GPQA-Diamond : 8 | [SOURCE] §5.2 ; Tab. 9 |

**Paramètres non spécifiés** : [NON SPÉCIFIÉ] optimiseur (type, β, weight decay), gradient clipping, précision numérique, S_warm en pas (hors « 10 rollouts »), top-p/top-k du rollout, longueur max d'évaluation, nombre de GPU du run final.

## 19. Boucle d'entraînement

[SOURCE] 1 époque sur ≈ 62K prompts, 128 prompts par rollout (soit ≈ 483 rollouts, [DÉDUCTION] 61 816/128), batch effectif 8, LR cosine avec warmup 5 pas, curriculum de longueur sur 10 rollouts. [NON SPÉCIFIÉ] Nombre exact de pas d'optimisation ; relation rollout/pas de gradient.

## 20. Inférence / génération

[SOURCE] Décodage par blocs ; statique (un token/pas) pour la plupart des résultats ; dynamique avec seuil γ pour l'étude de parallélisme (γ de 0,8 à 1,0). [SOURCE] « think@eval » : le modèle est invité à raisonner entre balises `<think></think>` à l'évaluation seulement. Template de chat officiel Qwen3 ; instructions ajoutées (math : « Please reason step by step, and put your final answer within \boxed{} » ; choix multiples : réponse en lettre).

## 21. Fonction de perte

[SOURCE] KL directe, Éq. OPDLM (§13), sommée sur blocs et positions masquées de l'état tiré ; poids 1. [NON SPÉCIFIÉ] Normalisation.

## 22. Optimisation

[SOURCE] LR 1e-5 → 1e-6 (cosine), warmup 5, batch effectif 8. [NON SPÉCIFIÉ] reste (voir §18).

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Sélection d'état : indice k uniforme sur les états non terminaux (équivalent à un ν^rev uniforme sur {t_1,…,t_T}). Pas de filtrage de qualité des rollouts ni de pondération mentionnés [DÉDUCTION]. Curriculum de longueur : voir §13.

## 24. Critères d'arrêt

[SOURCE] Fin de l'époque (1 pour le modèle général ; 10 pour OPDLM-MATH). [NON SPÉCIFIÉ] Critère de sélection de checkpoint.

## 25. Évaluation

[SOURCE] Connaissances générales et instructions : MMLU, MMLU-Pro, GPQA-Diamond, IFEval, CEval, LiveBench ; mathématiques et raisonnement : GSM8K, MATH-500, AIME-24, AIME-25, LMB-Hard, ZebraLogic ; code : HumanEval, MBPP, LiveCodeBench-v6, Codeforces ; multilingue : MMMLU-lite, INCLUDE-lite, MT-AIME2024, MLogiQA.

## 26. Métriques

[SOURCE] Précision par benchmark ; tokens d'entraînement ; FLOPs d'entraînement (Éq. 7) ; tokens par pas de débruitage ; moyenne ± écart-type sur seeds (GPQA-D, AIME). [NON SPÉCIFIÉ] Métrique exacte de LiveBench, LMB-Hard, ZebraLogic, Codeforces dans les extraits lus.

## 27. Protocole expérimental

[SOURCE] (a) Résultats généraux (Tab. 1, 6) : OPDLM vs baselines (chiffres des baselines issus de leurs publications [DÉDUCTION], non précisé pour toutes) ; (b) ablation données/objectif à données et budget appariés (1 époque) : Off-Policy, OPDLM_off, OPDLM (Tab. 4) ; (c) SFT avec la perte BD3LM sur données générées par l'ARLM (1 réponse par prompt, 1 époque, bloc 4) (App. A.5) ; (d) décodage multi-token (seuil, taille de bloc 4/8/16) ; (e) OPDLM-MATH vs TraDo (mêmes 8K problèmes) ; (f) taille d'enseignant ; (g) multi-seed. [SOURCE] Les auteurs précisent que la comparaison au baseline SDAR est confondue (≈55 Md tokens de données non divulguées vs ≈66–76 M tokens).

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences.)

| Expérience | Résultat | Trace |
|---|---|---|
| Coût | OPDLM-8B : 0,066 Md tokens, 4,2×10¹⁸ FLOPs ; OPDLM-4B : 0,076 Md, 2,4×10¹⁸ ; 0,6B : 0,048 Md, 0,23×10¹⁸ ; 1,7B : 0,072 Md, 0,98×10¹⁸ ; réduction « 15× à 7 000× » | Tab. 1, 6 ; Fig. 1 |
| OPDLM-8B | MMLU 70,9 ; MMLU-Pro 53,7 ; GPQA-D 36,1 ; IFEval 50,1 ; GSM8K 87,1 ; MATH-500 71,2 ; AIME-24 14,7 ; AIME-25 12,4 ; HumanEval 59,8 ; MBPP 48,7 ; LCB-v6 9,7 | Tab. 1 |
| OPDLM-4B | MMLU 65,5 ; MMLU-Pro 46,3 ; GSM8K 87,6 ; MATH-500 72,8 ; AIME-24 14,4 ; HumanEval 56,1 ; MBPP 57,7 | Tab. 1 |
| Baselines SDAR-8B / SDAR-4B | MMLU 78,6 / 74,9 ; MATH-500 78,6 / 72,8 ; HumanEval 82,3 / 76,8 | Tab. 1 |
| Petites échelles | OPDLM-0,6B : GSM8K 42,1 ; MATH500 28,0 ; OPDLM-1,7B : GSM8K 71,0 ; MATH500 53,0 ; en retrait sur le code | Tab. 6 |
| Zero-shot thinking (think@eval) | 8B : AIME-24 14,7 → 18,6 ; AIME-25 12,4 → 19,4 ; 4B : MATH-500 72,8 → 75,0 | Tab. 2 |
| Multilingue | OPDLM-8B : MMMLU-lite 56,0 ; INCLUDE-lite 51,9 ; MT-AIME2024 7,9 ; MLogiQA 42,0 | Tab. 3 |
| Parallélisme (MATH-500, bloc 4) | ≈ 1 token/pas à γ=1 → > 2 tokens/pas à γ=0,8 ; à γ=0,9, bloc 4 → 16 : ≈ 2 → ≈ 3,5 tokens/pas (lecture de figure) | Fig. 3 ; §5.4 |
| OPDLM-MATH vs TraDo | 4B : OPDLM-MATH GSM8K 83,8, MATH500 75,8, AIME24 10,0 ; TraDo-4B-Instruct 91,2 / 75,6 / 8,3 ; 8B : 86,2 / 76,6 / 23,3 vs 92,3 / 78,5 / 13,3 ; Thinking 8B : 93,8 / 92,4 / 50,0 | Tab. 5 |
| Multi-seed | OPDLM-4B AIME-24 14,4 ± 3,7 ; AIME-25 12,6 ± 3,6 ; GPQA-D 29,1 ± 2,7 ; OPDLM-8B AIME-24 14,7 ± 3,4 ; AIME-25 12,4 ± 5,0 ; GPQA-D 36,1 ± 2,2 | Tab. 9 |

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| Off-Policy vs OPDLM_off vs OPDLM (4B ; 8B) | MMLU : 66,3 / 66,7 / 65,5 ; 65,5 / 69,1 / 70,9 ; MATH500 : 73,2 / 75,8 / 72,8 ; 69,2 / 73,8 / 71,2 ; HumanEval : 48,8 / 37,8 / 56,1 ; 54,9 / 54,9 / 59,8 (autres lignes : Tab. 4) | Tab. 4 |
| Lecture des auteurs | Les données on-policy sont le moteur principal de performance ; la trajectoire de masquage a un effet limité (étudié à bloc 4 seulement) | §5.3 |
| Taille d'enseignant (OPDLM-MATH) | 4B : depuis Qwen-4B 83,8 / 75,8 / 10,0 ; depuis Qwen-32B 84,9 / 72,2 / 10,0 ; 8B : 86,2 / 76,6 / 23,3 vs 86,4 / 71,0 / 10,0 | Tab. 7 |
| Taille d'enseignant (0,6B) | depuis 0,6B : GSM8K 42,1, MATH500 28,0 ; depuis Qwen3-4B : 47,9 et 36,2 | Tab. 8 |
| SFT (BD3LM) vs OPDLM (4B) | SFT : GSM8K 86,8 ; MATH500 75,4 ; HumanEval 65,9 ; IFEval 44,7 ; OPDLM : 87,6 ; 72,8 ; 56,1 ; 53,8 | Tab. 10 |
| Décodage dynamique (4B) | OPDLM statique → dynamique : GSM8K 87,6 → 86,0 ; MATH500 72,8 → 71,6 ; SFT : 86,8 → 83,9 ; 75,4 → 71,2 ; HumanEval SFT 65,9 → 59,8 | Tab. 11 |

[SOURCE] Remarque : certaines valeurs diffèrent entre tableaux pour OPDLM-4B (HumanEval 56,1 en Tab. 1 et 10 vs 53,0 en Tab. 11 « static » ; MBPP 57,7 vs 58,7 ; MBPP SFT 58,7 en Tab. 10 vs 59,0 en Tab. 11) — non expliqué dans les extraits lus.

## 30. Contraintes techniques

- [SOURCE] L'enseignant doit appliquer un décalage de logits (logit shifting) et utiliser une attention causale ; l'étudiant utilise l'attention bloc-causale.
- [SOURCE] La cible utilise la séquence terminale de la trajectoire pour construire le préfixe causal de chaque position masquée.
- [SOURCE] Le sampler d'entraînement est le même type de décodage dynamique que celui visé à l'inférence (seuil 0,9).
- [SOURCE] Les modèles < 4B peuvent être entraînés sur 2 GPU A6000.
- [NON SPÉCIFIÉ] Contraintes de mémoire pour les rollouts de 4 000 tokens.

## 31. Dépendances

[SOURCE] Famille Qwen3 ; chat template Qwen3 ; corpus DAPO, Nemotron-v2 (Math, STEM, Chat), TACO, KodCode-Light-RL, AceCode ; benchmarks listés en §25 ; baselines LLaDA, Dream, SDAR, Fast-dLLM-v2, Simple-dLLM, TraDo ; GPU H200/A100/A6000. [NON SPÉCIFIÉ] Frameworks logiciels, moteur d'inférence du rollout, versions.

## 32. Coûts / ressources

[SOURCE] Tokens et FLOPs (Tab. 1, 6) ; formule Éq. (7) ; GPU H200 pour les checkpoints finaux ; ablations sur A6000/A100. [NON SPÉCIFIÉ] Temps mural, nombre de GPU du run final, mémoire.

## 33. Limites identifiées par les auteurs

1. [SOURCE] Marge d'amélioration sur certaines tâches vs les meilleures baselines (code notamment) ; qualité des données = axe d'amélioration.
2. [SOURCE] Comparaisons contrôlées difficiles (données de pré-entraînement des baselines non publiques).
3. [SOURCE] Distillation de raisonnement explorée seulement dans le cadre mathématique spécialisé.
4. [SOURCE] Expériences en auto-distillation (même taille, même famille) ; étude cross-taille/cross-famille incomplète ; l'ablation de taille d'enseignant est « preliminary ».
5. [SOURCE] Étude de l'effet du masquage limitée au bloc 4.
6. [SOURCE] Risques génériques des systèmes génératifs (biais) ; atténuations recommandées (audit, filtrage, alignement).

## 34. Points non spécifiés

Voir table GAP.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Checkpoints Qwen3 ; corpus de 61 816 prompts (sources de la Tab. 12) ; Alg. 1 ; Tab. 13 ; sampler dynamique 0,9 ; curriculum 100 → 4 000 sur 10 rollouts ; protocole d'évaluation de l'App. B.2 ; code et checkpoints annoncés (liens du §1).

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Optimiseur, clipping, précision | Haute | Tab. 13 |
| GAP-002 | Normalisation de la perte (somme/moyenne), gestion du batch de trajectoires | Haute | Alg. 1 |
| GAP-003 | Détail du sampler dynamique (définition de la confiance, repli) | Haute | Tab. 13 ; §3 |
| GAP-004 | Pas d'optimisation exacts et S_warm en pas | Moyenne | §4.3 ; Tab. 13 |
| GAP-005 | Procédure de déduplication/filtrage du corpus de 61 816 prompts | Moyenne | §5.1 ; App. B.1 |
| GAP-006 | Sélection de checkpoint | Moyenne | §5 |
| GAP-007 | Origine des chiffres de baselines (reproduits ou repris) | Moyenne | Tab. 1, 6 |
| GAP-008 | Incohérences entre Tab. 1/10 et Tab. 11 | Basse | App. A.5 |
| GAP-009 | Moteur d'inférence du rollout | Moyenne | — |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-3-001 | Objectif trajectoire généralisé | [SOURCE] | §4, Éq. (4) |
| REQ-3-002 | Cible ARLM sur préfixe causal de x⁰ | [SOURCE] | §4.2 |
| REQ-3-003 | Un état par trajectoire, k uniforme | [SOURCE] | §4.1 ; Alg. 1 |
| REQ-3-004 | Curriculum de longueur | [SOURCE] | §4.3 |
| REQ-3-005 | Hyperparamètres généraux | [SOURCE] | App. B.3, Tab. 13 |
| REQ-3-006 | Hyperparamètres OPDLM-MATH | [SOURCE] | Tab. 14 |
| REQ-3-007 | Ablation Off-Policy / OPDLM_off | [SOURCE] | §5.3, Tab. 4 |
| REQ-3-008 | FLOPs | [SOURCE] | App. B.4, Éq. (7) |
| REQ-3-009 | Données | [SOURCE] | §5.1, Tab. 12 |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Élevée** (Alg. 1 complet ; optimisation non détaillée)
- Architecture : **Moyenne** (méthode de formulation ; composants logiciels non précisés)
- Paramètres : **Moyenne** (Tab. 13–14 ; optimiseur absent)
- Reproductibilité : **Moyenne** (code/checkpoints annoncés ; plusieurs paramètres d'optimisation absents du texte)

---

# STM 4 — Trace-Based On-Policy Distillation for Masked Diffusion Language Models (TOPD)

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | Trace-Based On-Policy Distillation for Masked Diffusion Language Models |
| Auteurs | Haolin Ren\*, Ziyang Huang\*, Chenhao Yuan, Jun Zhao, Kang Liu (Institute of Automation, Chinese Academy of Sciences ; University of Chinese Academy of Sciences) — \* contribution égale ; contact : renhaolin2026@ia.ac.cn |
| Date / version | arXiv:2607.16872v1 [cs.CL], 18 juillet 2026 |
| URL / code | [NON SPÉCIFIÉ] (aucune URL de code dans le PDF) |
| Méthode | TOPD (Trace-Based On-Policy Distillation) |
| Objectif principal | Transférer une capacité de raisonnement d'un dLLM enseignant vers un dLLM étudiant, sans estimation de récompense |

## 2. Résumé technique

[SOURCE] L'étudiant dLLM échantillonne sa propre trajectoire de débruitage ; seules les décisions de token qui apparaissent dans la réponse finale (« trace ») reçoivent une supervision ; un enseignant dLLM gelé fournit sa distribution de tokens sur le même état partiellement débruité et la même position ; l'étudiant est mis à jour avec une divergence KL inverse token-level (estimateur de score échantillonné décrit en §3.4). Pas de modèle de récompense, de valeur ni de ratio de politique.

## 3. Problème traité

- [SOURCE] Le SFT des dLLM entraîne sur des états masqués aléatoires hors-politique ; le RL repose sur des récompenses éparses, retardées, à forte variance.
- [SOURCE] « Random-mask mismatch » : des états construits par masquage aléatoire peuvent exposer des tokens aval en cachant des variables amont (Fig. 2), créant des dépendances conditionnelles incohérentes avec le débruitage à l'inférence.
- [SOURCE] En dLLM, un pas opère sur une séquence partiellement masquée et peut proposer plusieurs tokens en parallèle : il faut décider quels états et quelles positions reçoivent le retour de l'enseignant.

## 4. Objectif de la méthode

[SOURCE] Distillation on-policy « trace-aware » : supervision dense à l'état de l'étudiant, alignée sur les décisions qui forment la réponse finale.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| Le principe de construction de trace de TraceRL est adopté : on ne garde que les décisions présentes dans la réponse finale | [SOURCE] | §3.2 ; App. B |
| L'ordre de révélation et les plannings d'inférence affectent le comportement des dLLM masqués | [SOURCE] (citant des travaux externes) | §1 |
| La KL inverse concentre l'étudiant sur les modes préférés de l'enseignant, utile dans des contextes bruités | [SOURCE] | §3.4 |
| Les propositions provisoires peuvent être révisées ou écrasées par un débruitage ultérieur | [SOURCE] | §3.2 |
| [DÉDUCTION] Pour les configurations décrites (blocs de 4 tokens, positions engagées figées), le filtre « x_j = s_T,j » ne retire rien en pratique ; la publication ne le démontre pas. | [DÉDUCTION] | §3.2 ; App. A.3 |

## 6. Périmètre de la méthode

[SOURCE] dLLM masqués : block-attention (SDAR) en cadre principal ; full-attention (LLaDA) comme généralisation. Évaluation limitée aux mathématiques (MATH500, AIME2024, GSM8K). Non couvert [SOURCE Limitations] : code, usage d'outils, instructions ouvertes ; enseignants plus grands ou plus capables.

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Étudiant π_θ | SDAR-4B-Chat (principal) ; LLaDA-8B-Instruct (généralisation) | [SOURCE] §4.1, §4.3 |
| Enseignant π_tea gelé | TraDo-8B-Instruct ; LLaDA-8B-Instruct-ESPO | [SOURCE] |
| Prompts | 8K problèmes MATH niveaux 3–5 | [SOURCE] §4.1 |
| Planning de décodage | Blocs de 4, 4 pas/bloc, dynamique τ=0,9 | [SOURCE] Tab. 5 |

## 8. Sorties

[SOURCE] Étudiant mis à jour (checkpoints évalués tous les 5 pas de rollout). [SOURCE] Lignes d'entraînement (q, ŝ_t, j, x_j) (Alg. 1).

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Moteur de rollout dLLM | Échantillonner τ avec les poids courants | [SOURCE] §3.1 ; App. C.1 (JetEngine pour SDAR, Fast-dLLM étendu pour LLaDA) |
| Extracteur de trace | ã_t = {(j,x_j) ∈ a_t : x_j = s_T,j} | [SOURCE] Éq. (4) |
| Reconstructeur d'état | ŝ_t avec contraintes de planning | [SOURCE] App. B |
| Enseignant gelé | p_tea(·\|q, s_t, j) ; logits calculés une fois par état reconstruit et réutilisés | [SOURCE] §3.3 ; App. B |
| Module de perte KL inverse + estimateur | Éq. (7)–(9) | [SOURCE] |
| Banc d'évaluation statique/dynamique | — | [SOURCE] App. C.5 |

## 10. Architecture / organisation de la méthode

```text
Prompt ─► Étudiant dLLM (rollout dynamique, T=1.0) ─► τ = (s₀, a₀, …, s_T)
                        │
              trace : ã_t = {(j,x_j) ∈ a_t : x_j = s_T,j}
                        │
     Enseignant gelé p_tea(·| q, s_t, j)   ◄── même état/position ──►   Étudiant p_θ(·| q, s_t, j)
                        │
                    KL inverse (token-level)  ─► mise à jour de l'étudiant
```
[SOURCE] Fig. 3.

## 11. Flux d'exécution

[SOURCE] Alg. 1 : pour chaque round : échantillonner N prompts ; pour chaque prompt, G réponses (G=1 en pratique) ; construire les lignes de trace ; mises à jour par minibatchs avec supervision de l'enseignant gelé ; resynchroniser le moteur.

## 12. Algorithme

[SOURCE] Algorithm 1 (App. B) : (1) échantillonnage de trajectoires : s₀ complètement masqué de longueur L_max ; pour t = 0..T−1 : a_t ∼ π_θ(·|q, s_t) ; s_{t+1} ← unmask(s_t, a_t) ; (2) lignes de trace : pour chaque t et chaque (j,x_j) ∈ ã_t, ajouter (q_i, ŝ_t, j, x_j) à D_TOPD ; (3) mises à jour : pour chaque minibatch M de taille m : p_tea ← π_tea(·|q, ŝ, j), p_θ ← π_θ(·|q, ŝ, j), ℓ ← D_KL(p_θ ‖ p_tea) ; L_M = (1/|M|) Σ ℓ ; θ ← θ − η∇_θ L_M.
[SOURCE] Note : Alg. 1 écrit une KL pleine par ligne ; le texte §3.4 décrit l'estimateur échantillonné ; App. B (Éq. 22) donne l'estimateur de gradient.

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1) | L_SFT(θ) = E[Σ_{i∈M(ỹ)} −log π_θ(y_i \| q, ỹ)] | SFT par masquage (rappel) |
| (2) | J_RL(θ) = E_{q, τ∼P_θ(·\|q)}[R(q, s_T)] | RL (rappel) |
| (3) | τ = (s₀, a₀, s₁, a₁, …, s_{T−1}, a_{T−1}, s_T) | Trajectoire |
| (4) | ã_t = {(j, x_j) ∈ a_t : x_j = s_{T,j}} | Trace |
| (5)–(6) | p^tea_{t,j}(·) = π_tea(·\|q, s_t, j) ; p^θ_{t,j}(·) = π_θ(·\|q, s_t, j) | Distributions appariées |
| (7) | L_TOPD(θ) = (1/BG) Σ_{i=1}^{BG} Σ_{t:\|ã_{t,i}\|>0} (1/\|ã_{t,i}\|) Σ_{(j,x_j)∈ã_{t,i}} D_KL(p^θ_{t,j} ∥ p^tea_{t,j}) | Perte (normalisation par pas puis par trajectoire) |
| (8) | r_{t,j} = sg(log p^tea_{t,j}(x_j) − log p^θ_{t,j}(x_j)) | Coefficient d'estimateur |
| (9) | ∇_θ J_TOPD ≈ (1/BG) Σ Σ (1/\|ã_{t,i}\|) Σ ∇_θ log p^θ_{t,j}(x_j) · r_{t,j} | Estimateur de gradient (ascension) |
| (15)–(17) | c_{t,j} = max_v π_θ(v\|q, s_t, j) ; U_t^static = TopK(c_{t,j}; m_t) ; U_t^dynamic = {j ∈ M_t : c_{t,j} ≥ τ}, repli sur ≥ 1 position | Règles de décodage |
| (18) | L_Random-Mask = (1/M) Σ_r (1/\|R^{(r)}\|) Σ_{j∈R^{(r)}} D(p^θ_{r,j} ∥ p^tea_{r,j}) | Ablation masque aléatoire |
| (19)–(21) | D_RKL(p,q) = D_KL(p∥q) ; D_FKL(p,q) = D_KL(q∥p) ; D_JSD,β = (1−β)D_KL(p∥m_β) + βD_KL(q∥m_β), m_β = (1−β)p+βq | Divergences comparées |
| (22) | ∇_θ L_{t,j} ≈ −∇_θ log p^θ_{t,j}(x_j) · sg[log(p^tea_{t,j}(x_j)/p^θ_{t,j}(x_j))] | Estimateur KL inverse |
| (23)–(36) | C_fwd = 2N_s D, C_bwd = 4N_s D ; C_TOPD = (4+α)F_s ; C_RL-noV = 80F_s ; C_RL-V(k) = 16(6+3/k)F_s ; S_round = 16(6+3/k)/(4+α) ; ESPO : C = G(K_ESPO + 6µM)F_s ; C_TOPD-LLaDA = (K_TOPD + α + 3)F_s | Comptabilité de calcul |

[SOURCE] Variables : B = prompts, G = trajectoires par prompt ; α = N_t/N_s ; F_s = 2N_s D.

## 14. Données

| Jeu | Rôle | Taille | Classe |
|---|---|---|---|
| MATH, niveaux 3–5 (suivant Hu et al. 2026) | Entraînement (prompts) | 8 523 problèmes (« 8K ») | [SOURCE] §4.1 ; Tab. 5 |
| MATH500 | Évaluation | 500 | [SOURCE] |
| AIME2024 | Évaluation | 30 | [SOURCE] |
| GSM8K (test) | Évaluation | 1 319 | [SOURCE] |

[SOURCE] Réponses d'enseignant TraDo-8B-Instruct sur les mêmes sources de données pour la baseline SFT. [NON SPÉCIFIÉ] Séparation validation/test pour la sélection de checkpoint ; Prétraitement hors filtre de niveau.

## 15. Génération / préparation des données

[SOURCE] Rollouts étudiant : 64 prompts par round, 1 réponse par prompt, température 1,0, top-p 1,0, longueur max 2 000 tokens, décodage dynamique τ=0,9, blocs de 4 tokens, 4 pas/bloc. Lignes de trace construites après rollout. Enseignant : logits calculés une fois par état reconstruit. Prompt template SDAR : `<|im_start|>user {{problem}} Please reason step by step, and put your final answer within \boxed{}.<|im_end|><|im_start|>assistant` ; template LLaDA spécifique (App. C.2). [SOURCE] Même template pour rollout, supervision et évaluation. [SOURCE] Baseline SFT : réponses de TraDo-8B-Instruct générées sur les mêmes prompts.

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Étudiant (principal) | SDAR-4B-Chat (block-attention) | [SOURCE] |
| Enseignant (principal) | TraDo-8B-Instruct (gelé) | [SOURCE] |
| Étudiant (généralisation) | LLaDA-8B-Instruct (full-attention) | [SOURCE] §4.3 |
| Enseignant (généralisation) | LLaDA-8B-Instruct-ESPO (même taille) | [SOURCE] |
| Baseline RL | TraceRL / TraDo-4B-Instruct (version RL de SDAR-4B-Chat) ; ESPO | [SOURCE] |
| Modèle de récompense / valeur | Non utilisé par TOPD ; TraceRL+V utilise un modèle de valeur | [SOURCE] |

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] dLLM gelé, évalué sur le même (q, s_t, j) que l'étudiant.
- Étudiant : [SOURCE] politique d'échantillonnage et cible d'optimisation.
- Modèle de référence : Non applicable à cette publication.

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Rôle | Source |
|---|---|---|---|
| Divergence | KL inverse | Perte | [SOURCE] §3.4 |
| Supervision | Positions de la trace uniquement | Sélection | [SOURCE] §3.2 |
| Normalisation | 1/\|ã_{t,i}\| par pas, 1/(BG) par trajectoire | Perte | [SOURCE] Éq. (7) |
| Stop-gradient sur l'écart log-probabilité | Activé | Estimateur | [SOURCE] Tab. 5 |

**Paramètres expérimentaux — SDAR-4B (Tab. 5)** : prompts/round 64 ; réponses/prompt 1 ; T=1,0 ; max génération 2 000 ; bloc 4 / pas par bloc 4 ; remasquage dynamique τ=0,9 ; top-p 1,0 ; AdamW ; LR 2×10⁻⁷ constant ; weight decay 0 ; batch effectif 16 (4/GPU × 4 GPU × 1) ; clipping 1,0 ; 1 époque par rollout ; bf16, TF32 ; gradient checkpointing (couche entière) ; type de perte « Full-vocab Reverse-KL » ; données 8 523 problèmes MATH ; prompt max 784.
**LLaDA (Tab. 6)** : 32 prompts / 1 réponse par round ; 30 rounds, checkpoint retenu au round 8 ; évaluation tous les 2 rounds ; T=0,8 ; longueur max 512 / 512 pas de diffusion ; bloc 32 / horizon 128 ; remasquage statique low-confidence ; batch 4, accumulation 2 ; LR 2×10⁻⁷ ; perte « Reverse-KL with top-k=10 token KL » ; évaluation T=0,1 statique et dynamique.
**Infrastructure** : 4 GPU A100-80G ou A800 ; DeepSpeed ZeRO-2 avec offload CPU des paramètres ; ≈ 8–12 min par round (SDAR-4B) ; le run de 30 rounds ≈ 5 h.
**Baselines** : TraceRL (G=16, clipping ε=0,2, pénalité KL β=0,01, LR 1×10⁻⁶ cosine, batch 16, modèle de valeur) ; Semi-AR SFT (taux de masque U[0,1;0,9], LR 1×10⁻⁵, 1 époque, batch 16) ; ESPO (LoRA r=128, α=64, dropout 0,05 ; G=16 ; µ=8 ; M=2 ; LR 1×10⁻⁶ ; WD 0,1 ; clip 0,2 ; 256 pas de diffusion ; prompt max 400 ; completion max 256 ; T=1,0).

**Paramètres non spécifiés** : [NON SPÉCIFIÉ] β d'AdamW ; nombre de rounds pour les ablations au-delà de 60 (Tab. 4 mentionne un pic au pas 80) ; seeds des runs ; valeur de m (taille du minibatch) hors batch effectif.

## 19. Boucle d'entraînement

[SOURCE] Un « rollout step » = un round de rollout + sa mise à jour d'entraînement associée (une époque sur les lignes de trace), pas un pas d'optimiseur unique. Évaluation tous les 5 rounds. Comparaison principale au round 30 (TOPD) vs round 120 (TraceRL).

## 20. Inférence / génération

[SOURCE] Évaluation SDAR : statique (top-k=1, un nombre fixe de positions par pas) et dynamique (τ=0,9, top-k=0), blocs de 4 et 4 pas/bloc, T=1,0, top-p=1,0, 2 000 tokens max. LLaDA : décodage prefix-cache (Fast-dLLM) en blocs de 32 avec horizon 128. Extraction de réponse : \boxed{} avec normalisation SymPy ; GSM8K : réponse numérique finale.

## 21. Fonction de perte

[SOURCE] Éq. (7) avec KL inverse, ou son estimateur échantillonné (Éq. 8–9, 22). [SOURCE] Le texte (§3.4) indique que l'implémentation utilise l'estimateur échantillonné ; l'App. A.5 indique que « SDAR TOPD uses full-vocabulary KL » et que LLaDA utilise une KL top-k=10 ; Tab. 5 indique « Full-vocab Reverse-KL » : **ambiguïté non levée par la publication** pour les expériences SDAR.

## 22. Optimisation

[SOURCE] AdamW, LR 2e-7 constant, WD 0, clip 1,0, batch effectif 16 (SDAR) ; DeepSpeed ZeRO-2. [NON SPÉCIFIÉ] β, ε.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Filtre de trace (Éq. 4) ; filtre de niveau sur MATH (3–5). Ablations de la sélection : masque aléatoire avec M masques indépendants pour égaler le nombre de lignes de trace (App. A.4). [SOURCE] Aucune pondération de trajectoires.

## 24. Critères d'arrêt

[SOURCE] Nombre de rounds fixe (30 pour la comparaison principale ; > 60 dans certaines courbes). [SOURCE] « We did not observe training collapse »; surapprentissage statique léger après le pic. [NON SPÉCIFIÉ] Critère d'arrêt adaptatif.

## 25. Évaluation

[SOURCE] MATH500, AIME2024, GSM8K, statique et dynamique ; moyenne arithmétique des trois (« Avg. »). Étude d'ablations sur MATH500 uniquement.

## 26. Métriques

[SOURCE] avg@3 (MATH500, GSM8K), avg@20 (AIME2024) ; moyenne ± écart-type sur 3 runs (ablations) au pic de la moyenne ; longueur moyenne de réponse ; vitesse « per-round » et « to-accuracy » (FLOPs proxy) ; GPU-heures mesurées.

## 27. Protocole expérimental

[SOURCE] (a) Résultats principaux : « Main-table accuracies are taken from the best single run for each method, using the checkpoint selected on MATH500 under the corresponding decoding setting and then evaluated on the remaining benchmarks. » (b) Ablations multi-run : pic de la moyenne des courbes sur 3 runs. (c) Tab. 4 : run unique. (d) Comptabilité de calcul paramétrée par α (α=2 pour SDAR 4B avec enseignant 8B ; α=1 pour LLaDA).

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences.)

| Méthode | MATH500 st./dyn. | AIME2024 st./dyn. | GSM8K st./dyn. | Moy. st./dyn. | Trace |
|---|---|---|---|---|---|
| TraDo-8B-Instruct (enseignant) | 78,5 / 75,5 | 13,3 / 11,0 | 92,3 / 91,2 | 61,4 / 59,2 | Tab. 1 |
| SDAR-4B-Chat | 70,2 / 67,4 | 5,0 / 8,2 | 90,2 / 88,9 | 55,1 / 54,8 | Tab. 1 |
| + SFT | 74,3 / 68,0 | 9,0 / 7,3 | 91,4 / 89,7 | 58,2 / 55,0 | Tab. 1 |
| + TraceRL | 75,6 / 71,8 | 8,3 / 10,3 | 91,2 / 90,3 | 58,4 / 57,5 | Tab. 1 |
| + TOPD | 75,9 / 71,9 | 9,3 / 8,7 | 92,2 / 89,3 | 59,2 / 56,6 | Tab. 1 |
| LLaDA-8B-Instruct | 37,3 / 38,3 | 0,5 / 1,7 | 82,5 / 82,5 | 40,1 / 40,8 | Tab. 1 |
| + ESPO | 38,9 / 39,1 | 1,0 / 0,8 | 84,0 / 83,9 | 41,3 / 41,3 | Tab. 1 |
| + TOPD | 39,2 / 38,6 | 0,8 / 1,2 | 83,3 / 83,5 | 41,1 / 41,1 | Tab. 1 |

[SOURCE] Vitesse : « Per-Round » 24,0× et « To-Accuracy » 96,0× (SDAR-4B, vs TraceRL+V, modèle de calcul théorique) ; LLaDA : 10,9× et 511,6× (vs ESPO). [SOURCE] « TOPD reaches TraceRL-level MATH500 accuracy in roughly 4× fewer rollout rounds » (round 30 vs 120). [SOURCE] Mesuré (Tab. 12) : 0,53–0,80 GPU-h/round vs 1,9–2,0 (TraceRL+V) ; 16–24 GPU-h vs 228–240 GPU-h pour la précision comparable ; speedup mesuré de bout en bout « 10–15× ». [SOURCE] Longueur de réponse TOPD (Tab. 8) : MATH500 628,8 (statique) / 620,2 (dynamique) ; AIME2024 1 151,5 / 1 060,7 ; GSM8K 313,8 / 309,9. [SOURCE] Étude de cas (App. D) : 2 solutions correctes, 1 échec (glissement algébrique précoce).

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| État supervisé (peak mean ± σ, 3 runs) | Semi-AR SFT 73,5±0,1 / 69,6±0,4 ; Off-policy 74,5±0,6 / 71,0±1,3 ; On-policy (TOPD) 75,2±0,8 / 71,7±0,7 (statique / dynamique) | Tab. 2 |
| Trace vs masque aléatoire | Random-mask 74,3±1,0 / 71,3±1,0 ; Trace 75,2±0,8 / 71,7±0,7 | Tab. 3 |
| Divergence (run unique, pic ; statique) | Forward-KL 72,7 (pas 55) ; JSD β=0,5 75,3 (pas 35) ; Reverse-KL 75,9 (pas 30) ; Δbase +2,5 / +5,1 / +5,7 | Tab. 4 |
| Divergence (dynamique) | Forward-KL 68,9 ; JSD 71,7 ; Reverse-KL 72,6 (pas 80) ; Δbase +1,5 / +4,3 / +5,2 | Tab. 4 |
| Diagnostics | Écart statique–dynamique plus faible que le SFT semi-AR ; nuage précision/longueur | Fig. 8 |

## 30. Contraintes techniques

- [SOURCE] L'enseignant et l'étudiant doivent être évalués sur le même prompt, état et position.
- [SOURCE] Les logits de l'enseignant sont calculés une fois par état reconstruit et réutilisés pour toutes les positions retenues.
- [SOURCE] L'état reconstruit respecte les contraintes de planning (positions au-delà du point courant remises à [MASK]).
- [SOURCE] Même format de prompt à l'entraînement et à l'évaluation.
- [NON SPÉCIFIÉ] Contraintes de mémoire pour reconstruire les états de longs rollouts.

## 31. Dépendances

[SOURCE] SDAR-4B-Chat, TraDo-8B-Instruct, LLaDA-8B-Instruct, ESPO ; JetEngine ; Fast-dLLM ; DeepSpeed ZeRO-2 ; SymPy ; datasets MATH, GSM8K, AIME2024 ; GPU A100-80G/A800 ; TraceRL (baseline). [NON SPÉCIFIÉ] Versions.

## 32. Coûts / ressources

[SOURCE] Voir §18 et Tab. 12 : coût par round et GPU-heures ; formules théoriques (App. F). [SOURCE] Les auteurs précisent que les accélérations théoriques ne sont pas des temps muraux mesurés.

## 33. Limites identifiées par les auteurs

1. [SOURCE] Enseignant 8B / étudiant 4B seulement (peu de dLLM publics plus grands) ; en full-attention, enseignant de même taille (modèle RL de même famille).
2. [SOURCE] Évaluation limitée aux mathématiques.
3. [SOURCE] Dépendance à un enseignant gelé (capacité et alignement avec le domaine).
4. [SOURCE] Pas de vérificateur externe ni de correction symbolique ; erreurs dominantes : glissements locaux (signe, simplification, contrainte géométrique, résultat intermédiaire pris pour final).
5. [SOURCE] Surapprentissage statique léger après le pic.

## 34. Points non spécifiés

Voir table GAP.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Tab. 5, 6 ; Alg. 1 ; Éq. (4)–(9) ; templates de prompt (App. C.2) ; protocole d'évaluation (App. C.5) ; configurations des baselines (App. C.4) ; planning de décodage (App. A.3).

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Estimateur réellement utilisé pour SDAR (échantillonné vs plein vocabulaire) | Haute | §3.4 vs Tab. 5, App. A.5 |
| GAP-002 | Code source | Haute | — |
| GAP-003 | Définition de la taille de minibatch m et nombre de pas par round | Moyenne | Alg. 1 ; Tab. 5 |
| GAP-004 | β et ε d'AdamW | Basse | Tab. 5 |
| GAP-005 | Jeu de sélection de checkpoint distinct du jeu de rapport | Haute (interprétation) | App. C.5 |
| GAP-006 | Cas où une décision engagée peut être révisée (pertinence du filtre de trace) | Moyenne | §3.2 |
| GAP-007 | Écarts entre Tab. 1 et Tab. 7 pour TraceRL (75,6/71,8 vs 75,5/71,9) | Basse | Tab. 1 ; Tab. 7 |
| GAP-008 | Rounds d'entraînement des runs d'ablation (pic au pas 80 en Tab. 4) | Basse | Tab. 4 |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-4-001 | Trajectoire on-policy | [SOURCE] | §3.1, Éq. (3) |
| REQ-4-002 | Trace-aligned decision selection | [SOURCE] | §3.2, Éq. (4) |
| REQ-4-003 | Distributions appariées | [SOURCE] | §3.3, Éq. (5)–(6) |
| REQ-4-004 | Perte KL inverse | [SOURCE] | §3.4, Éq. (7) |
| REQ-4-005 | Estimateur échantillonné | [SOURCE] | Éq. (8)–(9), (22) |
| REQ-4-006 | Hyperparamètres SDAR | [SOURCE] | Tab. 5 |
| REQ-4-007 | Hyperparamètres LLaDA | [SOURCE] | Tab. 6 |
| REQ-4-008 | Règles de décodage | [SOURCE] | App. A.3, Éq. (15)–(17) |
| REQ-4-009 | Comptabilité de calcul | [SOURCE] | App. F |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Élevée** (algorithme et équations complets ; ambiguïté estimateur)
- Architecture : **Moyenne**
- Paramètres : **Élevée** (Tab. 5–6 et baselines détaillés)
- Reproductibilité : **Moyenne** (code non mentionné ; ambiguïté de la perte)

---

# STM 5 — Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models (TSD)

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | Temporal Self-Distillation: Faster Inference in Discrete Diffusion Language Models |
| Auteurs | Shijian Xu\*, Andrea Miele\* (University of Basel), Metod Jazbec (University of Amsterdam), Volker Roth (Basel), Eric Nalisnick (Johns Hopkins University), Ilija Bogunovic (Basel) — \* contribution égale |
| Date / version | arXiv:2609.15177v1 [cs.LG], 14 septembre 2026 ; « Preprint » |
| URL / code | [NON SPÉCIFIÉ] (aucune URL de code dans le PDF) |
| Méthode | TSD (Temporal Self-Distillation) |
| Objectif principal | Réduire le nombre de passes avant (NFE) nécessaires au décodage des dLLM masqués en rendant les prédictions précoces plus fiables |

## 2. Résumé technique

[SOURCE] TSD est une méthode on-policy d'auto-distillation à travers les pas de temps de débruitage. Pour chaque position masquée à un pas précoce t̂, la distribution prédite par le modèle est rapprochée (stop-gradient) de la distribution que le **même modèle** produit pour la même position au pas où elle est finalement engagée (t_ℓ). Une pondération par un vérificateur V(x⁰) et, pour certaines tâches, par la longueur, module la perte. Aucun modèle enseignant séparé ni jeu de données hors-ligne n'est requis.

## 3. Problème traité

- [SOURCE] Le décodage parallèle agressif (beaucoup de tokens engagés par pas) dégrade fortement la qualité des dLLM.
- [SOURCE] Les décodages adaptatifs restent contraints par les prédictions aux pas précoces ; les approches par distillation sont typiquement hors-ligne (génération d'un jeu de trajectoires enseignant puis entraînement séparé).

## 4. Objectif de la méthode

[SOURCE] Améliorer la qualité de génération sous budget d'inférence contraint (peu de NFE) ; viser le régime à faible calcul plutôt que le pic de précision à grand budget.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| La prédiction à l'instant d'engagement est conditionnée par un contexte plus complet et constitue une cible naturelle pour les prédictions précoces | [SOURCE] | §3 |
| Le décodage est à démasquage seul (« unmasking-only ») : chaque position a un temps d'engagement unique t_ℓ | [SOURCE] | §2 ; §7 |
| La KL, asymétrique et non bornée, peut produire des mises à jour instables si les probabilités de la cible sont faibles ; la JSD est bornée | [SOURCE] | §4.3 |
| La pondération par récompense nécessite un signal de correction programmatique | [SOURCE] | §7 |
| Les solutions correctes de code/maths varient en longueur : biaiser vers des solutions courtes favorise un décodage à moins de NFE | [SOURCE] | §3 ; §4.2 |
| Les solutions de Sudoku/Countdown ont une longueur fixe : pas de pondération de longueur | [SOURCE] | §4.3 |

## 6. Périmètre de la méthode

[SOURCE] dLLM masqués ; décodage Fast-dLLM par seuil de confiance ; checkpoints LLaDA-8B-Instruct (math, code) et GDSD (planification) ; applicable après un post-entraînement RL (GDSD). Non couvert [SOURCE §7] : sampleurs qui révisent les tokens engagés ; génération ouverte sans vérificateur ; amélioration de capacité.

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Prompt c ∼ S | Jeu d'entraînement par tâche | [SOURCE] Alg. 1 |
| Modèle p_θ | Checkpoint initial | [SOURCE] |
| Vérificateur V | Score de la génération finale | [SOURCE] Alg. 1 |
| Seuil de décodage d'entraînement λ | 0,9 | [SOURCE] Tab. 2 |
| Divergence D | JSD β=0,5 (principal) | [SOURCE] §3 |
| Longueur cible L_target, γ_len | Par tâche | [SOURCE] Tab. 1 |

## 8. Sorties

[SOURCE] Modèle (adaptateurs LoRA) dont les prédictions précoces anticipent mieux la distribution à l'engagement, utilisable avec des seuils de confiance plus bas. Par tâche : un modèle dédié, sauf les trois benchmarks de code qui partagent un entraînement (App. B.1).

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Rollout Fast-dLLM stop-gradient | Produire x^{0:T_c} et les temps {t_ℓ} | [SOURCE] Alg. 1 |
| Tirage de t̂ ∼ Unif{1..T_c} | Un pas par trajectoire | [SOURCE] Éq. (7) |
| Ensemble M_t̂ des positions masquées à t̂ | Positions supervisées | [SOURCE] Éq. (8) |
| Évaluation du modèle aux pas t̂ et t_ℓ | Prédiction précoce et cible | [SOURCE] |
| Vérificateurs par tâche | V(x⁰) | [SOURCE] App. A.3 |
| Module de pondération par longueur | V_len | [SOURCE] Éq. (9) |
| LoRA | Paramétrisation de la mise à jour | [SOURCE] §4.1 |

## 10. Architecture / organisation de la méthode

```text
Prompt ─► rollout Fast-dLLM (stop-grad, λ=0.9) ─► trajectoire x^{T_c}…x^0 + temps d'engagement t_ℓ
                                                   │            │
                                  t̂ ~ Unif{1..T_c}│            └─► V(x⁰) (+ pondération longueur)
                                                   ▼
  p_θ(·|c, x^t̂)_ℓ  (ℓ ∈ M_t̂, gradient)   vs   sg[p_θ(·|c, x^{t_ℓ})_ℓ]  ──► D (JSD) · V ──► mise à jour LoRA
```
[SOURCE] Fig. 2.

## 11. Flux d'exécution

[SOURCE] Alg. 1 : tant que non convergé : échantillonner c ; rollout stop-gradient avec temps d'engagement ; tirer t̂ ; calculer M_t̂ ; calculer L̂_TSD = V(x⁰)·Σ_{ℓ∈M_t̂} D(p_θ(·|c,x^t̂)_ℓ, sg[p_θ(·|c,x^{t_ℓ})_ℓ]) ; mise à jour de θ. [SOURCE] En pratique, G rollouts par prompt et moyenne de la perte sur le groupe.

## 12. Algorithme

[SOURCE] Algorithm 1 « TSD Training » (corps du texte), tel que ci-dessus. [NON SPÉCIFIÉ] Détails de l'implémentation de « inner update iterations » (4) : sont-elles des passes répétées sur les mêmes rollouts ? (hypothèse [DÉDUCTION] non confirmée par le texte).

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1) | q_s(x^s \| x) = Π_ℓ {1−s si x^s_ℓ = x_ℓ ; s si [MASK] ; 0 sinon} | Processus de masquage forward |
| (2) | L_MDM(θ) = −E[(1/s) Σ_ℓ 1[x^s_ℓ = [MASK]] log p_θ(x \| c, x^s)_ℓ] | Objectif MDM standard |
| (3)–(5) | Trajectoire x^{T_c},…,x^0 ; M_t = {ℓ : x^t_ℓ = [MASK]} ; U_t ⊆ M_t ; x^{t−1}_ℓ = x^0_ℓ si ℓ∈U_t, x^t_ℓ sinon ; x^0_ℓ ∼ p_θ(·\|c,x^t)_ℓ | Débruitage |
| (6) | L_TSD(θ) = E_{c, x^{0:T_c}∼sg[p_θ(·\|c)]}[V(x⁰)(1/T_c) Σ_{t=1}^{T_c} Σ_{ℓ∈M_t} D(p_θ(·\|c,x^t)_ℓ, sg[p_θ(·\|c,x^{t_ℓ})_ℓ])] | Objectif d'auto-distillation temporelle |
| (7) | t̂ ∼ Unif({1,…,T_c}) | Tirage d'un pas |
| (8) | L̂_TSD(θ) = V(x⁰) Σ_{ℓ∈M_t̂} D(p_θ(·\|c,x^t̂)_ℓ, sg[p_θ(·\|c,x^{t_ℓ})_ℓ]) | Estimateur Monte-Carlo à un pas (« unbiased » selon les auteurs) |
| D_KL | D_KL(p‖q) = Σ_v p(v) log(p(v)/q(v)) | KL « reverse (mode-seeking) » (p = prédiction précoce, q = cible stop-gradient) |
| D_JS,β | D_JS,β(p‖q) = β D_KL(q‖m) + (1−β) D_KL(p‖m), m = βq + (1−β)p ; borné par log 2 ; symétrique à β=½ | JSD |
| (9) | V_len(x⁰) = V(x⁰)(1 − γ_len (\|x⁰\| − L_target)₊/(L − L_target)) | Pondération par longueur |

[SOURCE] Variables : T_c = nombre de pas du rollout pour le prompt c ; t_ℓ = pas d'engagement de la position ℓ ; λ = seuil de confiance Fast-dLLM ; L = taille du canevas ; |x⁰| = longueur après troncature au premier EOS. [SOURCE] γ_len = 0 : poids brut ; γ_len = 1 : poids nul pour les rollouts atteignant la limite du canevas.

## 14. Données

| Tâche | Jeu d'entraînement | Évaluation | Source |
|---|---|---|---|
| GSM8K | split train de GSM8K | GSM8K (exact numérique) | [SOURCE] App. A.2 |
| MATH-500 | split train de MATH (≈ 7,5 K) | MATH-500 (équivalence symbolique normalisée) | [SOURCE] |
| Countdown | split train TinyZero (Countdown-Tasks-3to4) | 256 problèmes synthétiques (d1) | [SOURCE] |
| Sudoku | jeu d'entraînement 4×4 de Black-Phoenix (disponible dans d1) | 256 problèmes synthétiques (d1), précision par cellule | [SOURCE] |
| Code (MBPP, HumanEval, LiveCodeBench) | KodCode-Light-RL-10K (un seul run partagé) | MBPP (split sanitized) ; HumanEval (openai_humaneval test) ; LiveCodeBench v5 (516 problèmes, code generation lite) | [SOURCE] |

[NON SPÉCIFIÉ] Splits de validation, taille exacte des jeux Countdown/Sudoku d'entraînement.

## 15. Génération / préparation des données

[SOURCE] Rollouts sur le modèle courant : Fast-dLLM, longueur de bloc 32, seuil 0,9 ; G = 4 rollouts par prompt. Prompts d'entraînement mono-tour, zéro-shot, sans message système : échafaudage « Respond in the following format: <reasoning> … </reasoning> <answer> … </answer> » pour GSM8K/MATH/Countdown/Code ; MATH ajoute « You are a math expert… Wrap the final answer in a \boxed{} » ; Countdown : expression arithmétique utilisant chaque nombre une fois dans <answer> ; Code : « You are a coding expert… » ; Sudoku : bloc dédié (grille 4×4, chaîne de 16 caractères). Les prompts d'évaluation diffèrent légèrement (Countdown : \boxed{}). **Récompenses (App. A.3)** : GSM8K/MATH : binaire (exact/équivalent), format non pris en compte ; Countdown : 1,0 (équation correcte) / 0,1 (valide syntaxiquement mais numériquement fausse) / 0,0 ; Sudoku : fraction de cases vides correctement remplies ; Code : somme à poids égaux du taux de réussite des tests et de la correction du format. Ablation « no-reward » : V ≡ 1.

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Modèle initial (math, code) | LLaDA-8B-Instruct (8,02 Md paramètres) | [SOURCE] |
| Modèle initial (Countdown, Sudoku) | Checkpoints GDSD publics (post-entraînés RL sur LLaDA-8B-Instruct) | [SOURCE] |
| Enseignant | Aucun modèle séparé : le modèle lui-même aux pas ultérieurs | [SOURCE] |
| Vérificateur | Fonctions par tâche | [SOURCE] |
| Baselines | LLaDA (Fast-dLLM), GDSD (Fast-dLLM), dParallel (checkpoint externe distillé en confiance ; indisponible pour Countdown/Sudoku) | [SOURCE] |

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] distribution stop-gradient du modèle au pas d'engagement t_ℓ.
- Étudiant : [SOURCE] distribution du modèle au pas précoce t̂ (porteuse de gradient).
- Modèle de référence : Non applicable à cette publication.

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Source |
|---|---|---|
| Divergence principale | JSD β=0,5 | [SOURCE] §3 ; Tab. 2 |
| Nombre de pas par trajectoire utilisés | 1 (t̂ uniforme) | [SOURCE] Éq. (7)–(8) |
| Poids de trajectoire | V(x⁰) (≥ 0) | [SOURCE] §3 |
| Pondération de longueur | Tâches math et code ; non pour planification | [SOURCE] §4.1 |

**Paramètres expérimentaux (App. A.2, Tab. 1–2)** : canevas L/L_prompt : GSM8K 256/200 ; MATH-500 256/256 ; Countdown 128/200 ; Sudoku 256/128 ; code 256/400 ; longueur cible : 160 (GSM8K), 180 (MATH), 160 (code) ; γ_len : 1 (GSM8K, MATH, code), 0 (Countdown, Sudoku). LoRA : rang 64, α 32, dropout 0,05 ; modules q_proj, k_proj, v_proj, up_proj des 32 blocs ; 83,9 M paramètres entraînables (1,05 %). AdamW β=(0,9 ; 0,99), weight decay 0,1 ; clipping 0,2 ; LR 3×10⁻⁶ constant ; ratio de warmup 10⁻⁴ ; 4 rollouts/prompt ; 4 itérations internes ; batch effectif 32 (2 GPU × 4 par device × 4 accumulations) ; décodage d'entraînement : Fast-dLLM, bloc 32, seuil 0,9 ; ≈ 3 000 pas par run ; checkpoint rapporté : 2 000 (MATH-500 : 1 000) ; 2 GPU NVIDIA A100/H200.
**Évaluation** : décodage déterministe (température 0, seed 42), pass@1, balayage du seuil λ ∈ {0,5 ; 0,6 ; 0,7 ; 0,8 ; 0,9 ; 0,95 ; 0,99} (certains balayages commencent à 0,6), canevas L ∈ {128, 256, 512} ; bloc de 32 tokens (4, 8, 16 blocs).
**Paramètres non spécifiés** : [NON SPÉCIFIÉ] coefficients exacts de la somme « test + format » du vérificateur de code ; critère de choix de L_target hors règle empirique ; valeurs des temps limites d'exécution de code.

## 19. Boucle d'entraînement

[SOURCE] ≈ 3 000 pas ; chaque pas : prompts échantillonnés, 4 rollouts par prompt, pertes moyennées sur le groupe ; 4 itérations de mise à jour interne ; batch effectif 32. [NON SPÉCIFIÉ] Définition précise d'un « pas » (rollouts par pas, itérations internes).

## 20. Inférence / génération

[SOURCE] Décodage Fast-dLLM : engage toutes les positions dont la confiance du token prédit dépasse le seuil λ, avec repli sur la position masquée de plus haute confiance si aucune ne dépasse ; baisser λ augmente le parallélisme. Coût mesuré en NFE moyen par génération ; une courbe vitesse-qualité par balayage de λ. Extraction : réponses \boxed{} (repli <answer>) pour math ; premier bloc ```python pour code ; etc.

## 21. Fonction de perte

[SOURCE] Éq. (8) multipliée par la pondération de longueur (Éq. 9) pour math/code ; avec D = JSD β=0,5 (principal) ; KL testée comme alternative. [NON SPÉCIFIÉ] Normalisation du nombre de positions de M_t̂.

## 22. Optimisation

[SOURCE] AdamW (β₁=0,9, β₂=0,99, WD 0,1), clip 0,2, LR 3e-6 constant, warmup 10⁻⁴ ; LoRA r=64. [NON SPÉCIFIÉ] Précision numérique de l'entraînement.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Un pas t̂ par trajectoire ; pondération par V(x⁰) ; pas de rejet de trajectoire (les trajectoires échouées ont un poids nul ou réduit) [SOURCE : « penalize or zero out »]. [SOURCE] Pondération de longueur pour décourager les rollouts qui « remplissent » le canevas.

## 24. Critères d'arrêt

[SOURCE] ≈ 3 000 pas ; checkpoint 2 000 (1 000 pour MATH-500, la frontière se stabilisant à 1 000 pas et les checkpoints tardifs dégradant le pic). [SOURCE] Hypothèse des auteurs : un entraînement prolongé surapprend aux vérificateurs à correspondance exacte.

## 25. Évaluation

[SOURCE] 7 benchmarks : GSM8K, MATH-500 (maths) ; Sudoku, Countdown (planification) ; HumanEval, MBPP, LiveCodeBench (code). Évaluation à L ∈ {128, 256, 512} ; principal L=256. Évaluation croisée de canevas : modèles entraînés à L=256 (Countdown : L=128).

## 26. Métriques

[SOURCE] Précision (GSM8K : exact numérique ; MATH-500 : équivalence symbolique normalisée) ; taux de passage Countdown (vérificateur à base de règles ; tolérance 10⁻⁵) ; précision par cellule (Sudoku) ; pass-rate exécution (code ; pass@1 pour LiveCodeBench) ; NFE moyen.

## 27. Protocole expérimental

[SOURCE] Un modèle dédié par benchmark (sauf code partagé). Évaluation sur les mêmes problèmes pour toutes les méthodes. Balayage de seuils : chaque point de courbe = un réglage de λ. Baselines : modèles de base décodés avec Fast-dLLM ; dParallel (si disponible). Sélection de checkpoint unifiée (2 000, MATH : 1 000). [NON SPÉCIFIÉ] Plusieurs seeds d'entraînement, intervalles de confiance.

## 28. Résultats expérimentaux

(Valeurs approximatives lues dans le texte et les figures ; non exigences.)

| Domaine | Résultat rapporté | Trace |
|---|---|---|
| Countdown (L=128→256) | ≈ 80 % de précision à ≈ 20 NFE ; GDSD : ≈ 90 NFE (légende Fig. 1) ou ≈ 100 NFE (texte §4.2) pour son pic ; « 4,5× plus rapide » | Fig. 1, §4.2 |
| Sudoku | 86–89 % sous 25 NFE ; pic GDSD ≈ 85 % | §4.2 |
| MBPP | 40 % de pass-rate avec ≈ 2,3× moins de NFE que LLaDA-8B-Instruct ; ≈ 1,5× moins que dParallel ; ≈ 50 % à ≈ 45 NFE | Fig. 1 ; §1 ; §4.2 |
| HumanEval | ≈ 40 % à ≈ 45 NFE ; la base demande plus du double | §4.2 |
| LiveCodeBench | Gains plus progressifs ; la base garde un léger avantage au NFE le plus élevé | §4.2 |
| GSM8K | ≈ 80 % à 35–50 NFE ; la base ≈ 65 NFE ; à 30 NFE, base < 40 % vs TSD proche de 80 % | §4.2 |
| MATH-500 | TSD et dParallel : 30–33 % ; la base conserve le pic le plus élevé ; pas d'amélioration Pareto cohérente entre canevas (à L=512, base et dParallel conservent un pic plus élevé) | §4.2 |

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| Pondération longueur vs récompense brute (MBPP L=128/256/512 ; GSM8K) | La version consciente de la longueur domine à NFE faible ; la version brute renforce des sorties verbeuses à grand canevas | Fig. 5, 15 |
| JS vs KL (Countdown, code) | KL : effondrement de la récompense et saturation du NFE à sa limite haute ; JS : stable | Fig. 6, 16 |
| Pondération par récompense vs sans (Countdown, MBPP) | Sans récompense : vitesse améliorée possible, pas d'amélioration de précision | Fig. 12–14 ; App. B.2 |
| Entraînement à L=256, évaluation à L ∈ {128, 256, 512} (Countdown) | Baisse nette de performance à L=128 ; d'où l'entraînement à L=128 pour Countdown | Fig. 7 ; App. A.2 |
| Évaluation croisée de canevas | Avantage à bas NFE conservé sur GSM8K, planification, code ; MATH-500 exception | §4.2 |

## 30. Contraintes techniques

- [SOURCE] Décodage à démasquage seul (temps d'engagement unique).
- [SOURCE] Signal de correction programmatique disponible pour la pondération.
- [SOURCE] L_target à choisir par tâche et fixé avant évaluation.
- [SOURCE] La KL non bornée est instable lorsque la cible a de faibles probabilités.

## 31. Dépendances

[SOURCE] LLaDA-8B-Instruct ; checkpoints GDSD (HuggingFace) ; dParallel (baseline) ; Fast-dLLM ; LoRA ; jeux GSM8K, MATH, TinyZero Countdown, Sudoku (Black-Phoenix / d1), KodCode-Light-RL-10K, MBPP, HumanEval, LiveCodeBench v5 ; codebase d1 pour Countdown/Sudoku ; 2 GPU A100/H200. [NON SPÉCIFIÉ] Versions.

## 32. Coûts / ressources

[SOURCE] 2 GPU A100/H200 ; ≈ 3 000 pas ; batch effectif 32. [NON SPÉCIFIÉ] Temps mural, coût en NFE d'entraînement par pas, consommation mémoire.

## 33. Limites identifiées par les auteurs

1. [SOURCE] TSD accélère mais n'ajoute pas de capacité : il ne peut rendre disponible plus tôt que ce que la politique produirait finalement ; pas d'amélioration du pic à grand budget (MATH-500).
2. [SOURCE] Hypothèse de démasquage seul ; les samplers révisant des tokens exigent une autre définition de cible.
3. [SOURCE] La pondération par récompense restreint à des tâches à signal de correction programmatique ; la génération ouverte est hors périmètre actuel.
4. [SOURCE] L_target doit être choisi par tâche.

## 34. Points non spécifiés

Voir table GAP.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Checkpoints (LLaDA-8B-Instruct, GDSD) ; Alg. 1 ; Éq. (8)–(9) ; JSD β=0,5 ; Tab. 1–2 ; récompenses par tâche (App. A.3) ; prompts (App. A.4) ; protocole d'évaluation et sweeps de seuil.

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Dépôt de code | Haute | — |
| GAP-002 | Poids exacts du vérificateur de code | Moyenne | App. A.3 |
| GAP-003 | Détail des « inner update iterations » | Moyenne | Tab. 2 |
| GAP-004 | Normalisation de la somme sur M_t̂ | Moyenne | Éq. (8) |
| GAP-005 | Seeds d'entraînement, variance | Haute | §4 |
| GAP-006 | Valeurs numériques exactes des courbes (lues sur figures) | Moyenne | Fig. 1, 3, 4 |
| GAP-007 | Règle de choix de L_target sur de nouvelles tâches | Moyenne | §3 |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-5-001 | Objectif d'auto-distillation temporelle | [SOURCE] | §3, Éq. (6) |
| REQ-5-002 | Estimateur à un pas | [SOURCE] | Éq. (7)–(8), Alg. 1 |
| REQ-5-003 | JSD β=0,5 | [SOURCE] | §3 |
| REQ-5-004 | Pondération longueur | [SOURCE] | Éq. (9) |
| REQ-5-005 | Hyperparamètres | [SOURCE] | App. A.2, Tab. 1–2 |
| REQ-5-006 | Récompenses par tâche | [SOURCE] | App. A.3 |
| REQ-5-007 | Protocole d'évaluation | [SOURCE] | §4.1 ; App. A.4 |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Élevée**
- Architecture : **Moyenne**
- Paramètres : **Élevée** (tables détaillées en annexe)
- Reproductibilité : **Moyenne** (code non cité ; variance et détails de vérificateur de code absents)

---

# STM 6 — Hesitation-Aware On-Policy Distillation for Diffusion Language Models (HOPD)

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | Hesitation-Aware On-Policy Distillation for Diffusion Language Models |
| Auteurs | Jianguo Huang, Lipeng Wan, Yanchen Deng\*, Bo An (Nanyang Technological University ; Xi'an Jiaotong University) — \* auteur correspondant |
| Date / version | arXiv:2609.33301v1 [cs.CL], 27 septembre 2026 |
| URL / code | [NON SPÉCIFIÉ] |
| Méthode | HOPD (Hesitation-Aware On-Policy Distillation) |
| Objectif principal | Étendre la distillation on-policy des dLLM à toutes les positions masquées de chaque pas de débruitage, avec pondérations a posteriori |

## 2. Résumé technique

[SOURCE] À chaque pas de débruitage, un dLLM propose un token à chaque position masquée mais n'engage qu'un sous-ensemble. Les propositions non engagées sont appelées « hésitations ». Une étude pilote (SDAR-4B base, enseignant TraDo-8B-Instruct) montre que les hésitations constituent 24,2 % des paires état-position supervisables mais 66,1 % de la KL totale enseignant-étudiant. HOPD supervise **toutes** les positions masquées de chaque état pré-action visité par une KL inverse sur le vocabulaire complet, pondérée par des poids de position (w_j = 1 + e_j) et de bloc (w_b = max(1−ρ_b, ρ_min)) calculés a posteriori sur la trajectoire complète. Le décodeur est inchangé ; aucune passe avant supplémentaire n'est requise par rapport à la supervision de la trace seule.

## 3. Problème traité

[SOURCE] La supervision « trace-only » (positions engagées uniquement) écarte les propositions non engagées, qui portent une part disproportionnée du désaccord enseignant-étudiant. Étendre l'OPD aux dLLM exige de spécifier l'ensemble de positions cibles de chaque état visité.

## 4. Objectif de la méthode

[SOURCE] Superviser l'étudiant sur les hésitations, allouer la supervision selon l'historique de propositions, améliorer précision et efficacité de décodage.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| Commitment (engagement) et supervision sont deux décisions distinctes ; chaque paire état-position masquée est supervisable | [SOURCE] | §3 |
| Une proposition différée (égale au token final) devrait idéalement être engagée plus tôt ; une proposition rétractée est légitimement incertaine | [SOURCE] | §3 |
| Les statistiques de trajectoire (retraction count e_j, first-step retention ρ_b) sont des signaux **heuristiques** d'allocation | [SOURCE] | §3 |
| Supervision des hésitations → le décodage engage plus de tokens par pas | [SOURCE] (attente confirmée empiriquement) | §1 ; §5.2 |
| Décodage : positions engagées figées dans l'état | [SOURCE] | §2.1 ; §3 |

## 6. Périmètre de la méthode

[SOURCE] dLLM par blocs (SDAR-1.7B, SDAR-4B) avec enseignant TraDo-8B-Instruct ; extension à LLaDA-8B-Instruct (full-attention) avec enseignants ESPO ; domaines : mathématiques (MATH500, AIME2024, GSM8K) et code (LiveCodeBench-v2, LiveBench). Aucune section « limitations » n'est présente dans la publication [NON SPÉCIFIÉ].

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Étudiant π_θ | SDAR-4B-Chat ou SDAR-1.7B-Chat ; LLaDA-8B-Instruct | [SOURCE] §5.1 ; App. C.1 |
| Enseignant π_tea gelé | TraDo-8B-Instruct ; ESPO-Math / ESPO-Code (LLaDA) | [SOURCE] |
| Prompts | MATH niveaux 3–5 (8 523) ; PrimeIntellect (5 954 stdio) | [SOURCE] Tab. 7 |
| Décodage d'entraînement | Dynamique τ=0,9, T=1,0 | [SOURCE] |

## 8. Sorties

[SOURCE] Étudiant distillé ; checkpoints évalués tous les 5 rounds (SDAR) / 4 rounds (LLaDA). [DÉDUCTION] Lignes d'entraînement incluant des poids de position et de bloc.

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Moteur de rollout (JetEngine pour SDAR ; Fast-dLLM pour LLaDA) | Trajectoires on-policy synchronisées avec l'étudiant avant chaque round | [SOURCE] §5.1 ; App. C.1 |
| Enregistrement de l'historique de propositions | e_j, ρ_b | [DÉDUCTION] (poids calculés sur la trajectoire complète) |
| Calcul de poids hindsight | w_j, w_b | [SOURCE] Éq. (4) |
| Perte KL inverse plein vocabulaire | ℓ_{t,j} | [SOURCE] Éq. (6) |
| Enseignant gelé | Distribution à chaque position masquée de s_t | [SOURCE] |

## 10. Architecture / organisation de la méthode

```text
Prompt ─► rollout étudiant (dynamique τ=0.9) ─► états pré-action s_t, propositions, engagements
                         │ trajectoire complète
         poids : w_j = 1 + e_j ; w_b = max(1 − ρ_b, 0.1)
                         ▼
   pour chaque s_t, chaque j ∈ M(s_t) = A_t ∪ R_t :  ℓ_{t,j} = D_KL(π_θ(·|q,s_t,j) ‖ π_tea(·|q,s_t,j))
                         ▼
         L_HOPD = Σ_t w_{b(t)} (1/|M(s_t)|) Σ_j w_j ℓ_{t,j}  ─► mise à jour
```
[SOURCE] Fig. 2.

## 11. Flux d'exécution

[SOURCE] À chaque round : l'étudiant courant génère les trajectoires définissant les états pré-action ; ces trajectoires restent fixes pendant les mises à jour du round ; à chaque état, l'étudiant est apparié à l'enseignant gelé ; une époque d'entraînement sur les lignes de trace (« trace rows ») ; 30 rounds.

## 12. Algorithme

[SOURCE] La publication ne fournit pas de pseudo-code numéroté de HOPD ; elle décrit la méthode par le texte (§4) et les équations (4)–(6). [NON SPÉCIFIÉ] Le pseudo-code complet ne peut pas être reconstruit avec certitude (définition de la ligne d'entraînement, ordre des calculs de poids, normalisation sur plusieurs blocs). Le pipeline textuel de §11 est une reconstruction fidèle du texte.

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1) | L_OPD(θ) = E_{q, y∼π_θ} Σ_i D_KL(π_θ(·\|q,y_{<i}) ∥ π_tea(·\|q,y_{<i})) | OPD autoregressif (KL inverse) |
| (2) | ∇_θ L ≈ −∇_θ log π_θ(x\|q,y_{<i}) sg[log π_tea(x\|·) − log π_θ(x\|·)] | Estimateur à un échantillon |
| (3) | L_TOPD(θ) = E Σ_{t:\|A_t\|>0} (1/\|A_t\|) Σ_{j∈A_t} D_KL(π_θ(·\|q,s_t,j) ∥ π_tea(·\|q,s_t,j)) | TOPD (rappelé comme point de départ) |
| (4) | w_j = 1 + e_j ; w_b = max(1 − ρ_b, ρ_min) ; ρ_min = 0,1 | Poids a posteriori |
| (5) | L_HOPD(θ) = E_{q,ξ∼π_θ} Σ_{t:\|M(s_t)\|>0} w_{b(t)} (1/\|M(s_t)\|) Σ_{j∈M(s_t)} w_j ℓ_{t,j}(θ) | Objectif HOPD |
| (6) | ℓ_{t,j}(θ) = D_KL(π_θ(·\|q,s_t,j) ∥ π_tea(·\|q,s_t,j)), j ∈ M(s_t) | Perte par position (plein vocabulaire) |
| (7) | ℓ^FKL_{t,j} = D_KL(r ∥ p) | KL directe (ablation) |
| (8) | ℓ^JSD_{t,j} = (1−β)D_KL(p ∥ m_β) + βD_KL(r ∥ m_β), m_β = (1−β)p + βr | JSD (ablation ; β=0,5) |
| (9) | ℓ^ST_{t,j} = −log p(x̂_j) sg[log r(x̂_j) − log p(x̂_j)] | Substitut échantillonné (ablation) ; écart clippé à [−2,2] |

Variables : e_j = nombre de propositions échantillonnées à la position j différentes de son token final (« retraction count ») ; ρ_b = fraction de positions du bloc b dont la première proposition est égale au token final ; b(t) = bloc débruité au pas t ; p, r = distributions étudiant et enseignant. [SOURCE] Les poids sont « calculés à partir de la trajectoire complète et partagés entre les lignes d'entraînement » ; ils sont utilisés « à l'échelle brute » (non normalisés). [SOURCE] Définitions de catégories : committed (j ∈ A_t) ; deferred (j ∈ R_t, x̂_{t,j} = s_{T,j}) ; retracted (j ∈ R_t, x̂_{t,j} ≠ s_{T,j}).

## 14. Données

| Jeu | Rôle | Taille | Source |
|---|---|---|---|
| MATH (niveaux 3–5) | Entraînement math | 8 523 problèmes (« 8K ») | [SOURCE] §5.1 ; Tab. 7 |
| PrimeIntellect, vérifié DeepCoder (problèmes stdio) | Entraînement code | 5 954 (« 6K ») | [SOURCE] |
| MATH500, AIME2024, GSM8K | Évaluation math | — | [SOURCE] |
| LiveCodeBench-v2, LiveBench | Évaluation code | — | [SOURCE] |
| Étude pilote | 256 prompts MATH ; une trajectoire par prompt | — | [SOURCE] App. A |

[NON SPÉCIFIÉ] Tailles des jeux d'évaluation (hors mention « avg@3 / avg@20 »), séparation validation.

## 15. Génération / préparation des données

[SOURCE] Rollouts de l'étudiant : 64 prompts/round, 1 réponse/prompt, T=1,0, top-p=1,0, 2 000 tokens, décodage dynamique τ=0,9, graine 1234 ; moteur JetEngine synchronisé avec l'étudiant avant chaque round. Templates : math « …Please reason step by step, and put your final answer within \boxed{}. » ; code stdio « This is the problem: {problem} You should put your code in ```python ```. Use input()… » (App. B.3). L'enseignant reçoit le même prompt et la réponse partiellement débruitée de l'étudiant. Baselines : SFT (réponses de l'enseignant, masque de bloc indépendant de taux ∈ [0,1 ; 0,9]) ; Off-policy (trajectoires enseignant gelé, positions engagées, KL inverse plein vocabulaire).

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Étudiants | SDAR-4B-Chat, SDAR-1.7B-Chat ; LLaDA-8B-Instruct | [SOURCE] |
| Enseignants | TraDo-8B-Instruct ; ESPO-Math et ESPO-Code (LoRA fusionné pour math) | [SOURCE] |
| Baseline RL | TraceRL : TraDo-4B-Instruct publié (4B) ; entraîné avec l'implémentation publique pour 1,7B (objectif PPO avec pénalité KL, récompenses : correspondance binaire (math) / fraction de tests réussis (code), pas de modèle de valeur) | [SOURCE] App. B.2 |
| Reward model / modèle de valeur | Aucun pour HOPD | [DÉDUCTION] |

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] gelé ; distribution à chaque position masquée du même état.
- Étudiant : [SOURCE] génère les états ; porteur du gradient.
- Modèle de référence : Non applicable à cette publication.

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Rôle | Source |
|---|---|---|---|
| Positions supervisées | Toutes les positions masquées M(s_t) | Ensemble cible | [SOURCE] §4 |
| ρ_min | 0,1 | Plancher du poids de bloc | [SOURCE] Éq. (4) |
| Perte par position | KL inverse, plein vocabulaire | — | [SOURCE] Éq. (6) |
| Poids | w_j = 1 + e_j ; w_b = max(1−ρ_b, ρ_min) | Allocation | [SOURCE] |

**Paramètres expérimentaux (Tab. 7)** : SDAR : 64 prompts × 1 réponse/round ; bloc 4 ; 4 pas/bloc ; décodage dynamique (commit si c_j > 0,9, repli top-1) ; T=1,0, top-p=1,0, top-k off ; 2 000 tokens max ; AdamW (β₁=0,9, β₂=0,999, ε=10⁻⁸) ; LR 2×10⁻⁷ constant ; WD 0 ; batch effectif 16 lignes (4 par GPU × 4 GPU × 1 accumulation) ; clip 1,0 ; 30 rounds, 1 époque/round ; checkpoints tous les 5 rounds ; bf16 ; gradient checkpointing ; DeepSpeed ZeRO-2 sans offload ; 4 GPU (2 GPU avec accumulation pour l'exécution 1,7B code) ; prompt max 784. LLaDA (Tab. 11) : 32 prompts × 1/round ; décodage low-confidence statique ; bloc 32 / 32 pas ; horizon 128 ; T=0,8 ; max réponse 512 (math) ; LR 2×10⁻⁷ ; batch effectif 32 (8 × 4 GPU × 1) ; ZeRO-2 avec offload CPU de l'optimiseur ; perte « top-k reverse KL (k=10) » ; checkpoints tous les 4 rounds ; seed 1234 ; prompt max 400 (math) / 512 (code).
**Évaluation** : SDAR : bloc 4, 4 pas/bloc, T=1,0, top-p=1,0, 2 000 tokens max, top-k=1 (statique) ; dynamique τ=0,9 ; LLaDA : 512 (math) / 1 024 (code) tokens max ; T=0,1 ; bloc 32 ; horizon 128.
**Paramètres non spécifiés** : [NON SPÉCIFIÉ] graines d'entraînement hors rollout ; éventuelle normalisation des poids ; nombre d'étapes d'optimiseur par round.

## 19. Boucle d'entraînement

[SOURCE] « A round thus contains multiple optimizer steps rather than a single update » : une époque sur les lignes de trace du round (64 trajectoires), batch effectif de 16 lignes ; 30 rounds.

## 20. Inférence / génération

[SOURCE] Le décodeur n'est pas modifié. Décodage statique (B/K plus hautes confiances par pas ; 1 token/pas pour B=K=4) et dynamique (commit de tout c_{t,j} > τ ; repli sur la meilleure position). Confiance c_{t,j} = π_θ(x̂_{t,j} | q, s_t, j) (probabilité du token échantillonné). Chaque bloc fini entre dans le cache KV avant le suivant. [SOURCE] Mesure de vitesse : balayage τ ∈ {0,5 ; 0,7 ; 0,8 ; 0,9 ; 0,95 ; 0,99}, tokens engagés par pas moyen et débit mural.

## 21. Fonction de perte

[SOURCE] Éq. (5)–(6). Ablations de divergence et de poids (§29).

## 22. Optimisation

[SOURCE] AdamW, LR 2e-7 constant, WD 0, clip 1,0, batch effectif 16 (SDAR) ou 32 (LLaDA). [NON SPÉCIFIÉ] Scheduler (« constant »), warmup.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Sélection : toutes les positions masquées ; pondération a posteriori ; aucun filtrage de trajectoires. Étude pilote : bootstrap de 5 000 tirages sur 256 prompts (graine 1234).

## 24. Critères d'arrêt

[SOURCE] 30 rounds ; checkpoint choisi sur MATH500 (math) ou LiveCodeBench-v2 (code), pour chaque règle de décodage, évalué tous les 5 rounds (LLaDA : 4 rounds, MATH500 ou LiveBench).

## 25. Évaluation

[SOURCE] MATH500, AIME2024, GSM8K (math) ; LiveCodeBench-v2, LiveBench (code) ; décodage statique et dynamique (τ=0,9) selon l'évaluation de TraceRL ; moyenne sur 5 benchmarks. Évaluation via la bibliothèque dLLM-RL (reproduit TraDo-4B-Instruct à un point près : MATH500 76,2 vs 75,6 ; GSM8K 91,3 vs 91,2, statique) ; programmes de code corrects seulement s'ils passent tous les tests stdio cachés ; étude pilote notée avec OpenCompass.

## 26. Métriques

[SOURCE] avg@3 (MATH500, GSM8K, benchmarks de code), avg@20 (AIME2024) ; moyenne arithmétique non pondérée ; tokens par pas ; débit mural (tokens/s) ; KL, entropie prédictive par catégorie.

## 27. Protocole expérimental

[SOURCE] Baselines sous le même point de départ, prompts, template, limites de réponse, pipeline d'évaluation : SFT, Off-policy, TraceRL, TOPD (réimplémenté dans la même base de code que HOPD ; seule la fonction objectif diffère ; estimateur échantillonné ; écart log-prob clippé à [−2,2] ; lignes et positions non pondérées ; variante « full-vocabulary reverse KL » en App. B.6). La ligne « enseignant » de la Tab. 1 est reprise de la publication TOPD [SOURCE App. B.4].

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences.)

| Modèle | Méthode | MATH500 st./dyn. | AIME2024 | GSM8K | LCB-v2 | LiveBench | Moy. st./dyn. |
|---|---|---|---|---|---|---|---|
| — | TraDo-8B (enseignant) | 78,5 / 75,5 | 13,3 / 11,0 | 92,3 / 91,2 | 25,9 / 22,4 | 22,7 / 20,6 | 46,5 / 44,1 |
| SDAR-1.7B | Base | 61,7 / 54,2 | 4,0 / 4,7 | 80,9 / 78,4 | 7,2 / 4,2 | 5,2 / 4,2 | 31,8 / 29,1 |
| SDAR-1.7B | + SFT | 49,2 / 46,3 | 1,0 / 3,8 | 77,0 / 74,7 | 7,8 / 6,6 | 6,2 / 8,6 | 28,2 / 28,0 |
| SDAR-1.7B | + Off-policy | 63,3 / 56,6 | 6,8 / 3,2 | 81,6 / 75,1 | 11,3 / 9,1 | 11,7 / 7,8 | 34,9 / 30,4 |
| SDAR-1.7B | + TraceRL | 62,5 / 56,1 | 6,7 / 3,5 | 81,1 / 78,7 | 9,4 / 4,7 | 10,7 / 5,7 | 34,1 / 29,7 |
| SDAR-1.7B | + TOPD | 64,5 / 58,1 | 2,2 / 3,2 | 81,9 / 78,2 | 10,4 / 7,7 | 9,4 / 6,2 | 33,7 / 30,7 |
| SDAR-1.7B | + HOPD | 66,8 / 59,5 | 3,7 / 5,0 | 83,6 / 78,8 | 11,1 / 8,9 | 10,9 / 9,9 | 35,2 / 32,4 |
| SDAR-4B | Base | 70,5 / 66,1 | 7,5 / 7,8 | 90,5 / 88,9 | 19,3 / 11,7 | 19,0 / 9,6 | 41,4 / 36,8 |
| SDAR-4B | + SFT | 60,7 / 53,3 | 5,7 / 4,8 | 83,7 / 81,7 | 12,9 / 11,0 | 12,8 / 10,9 | 35,1 / 32,3 |
| SDAR-4B | + Off-policy | 75,6 / 71,3 | 7,8 / 8,0 | 91,3 / 90,1 | 21,5 / 19,3 | 19,3 / 17,2 | 43,1 / 41,2 |
| SDAR-4B | + TraceRL (TraDo-4B) | 76,2 / 71,0 | 7,7 / 8,8 | 91,3 / 89,7 | 18,0 / 13,9 | 14,8 / 9,9 | 41,6 / 38,7 |
| SDAR-4B | + TOPD | 74,9 / 71,3 | 11,2 / 7,2 | 92,0 / 90,0 | 21,9 / 18,6 | 19,8 / 17,4 | 43,9 / 40,9 |
| SDAR-4B | + HOPD | 74,9 / 72,5 | 15,0 / 9,7 | 91,1 / 90,7 | 20,9 / 19,8 | 22,1 / 19,8 | 44,8 / 42,5 |

[SOURCE] Vitesse (SDAR-4B, MATH500 dynamique, τ=0,9) : 72,5 % vs 71,3 % (TOPD), avec « 11 % de tokens en plus par pas » ; débit mural : +4–18 % (4B), +8–23 % (1,7B) selon le seuil (App. B.5). [SOURCE] Étude pilote : cf. §29. [SOURCE] LLaDA-8B (Tab. 3, st./dyn.) : base moyenne 26,5 / 26,8 ; ESPO (enseignant) 28,4 / 27,6 ; TOPD 27,9 / 27,3 ; HOPD 28,4 / 27,5. [SOURCE] Variante TOPD « full reverse KL » (Tab. 10) : SDAR-4B moy. 43,8 / 42,2 ; SDAR-1.7B 33,6 / 30,9.

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| Étude pilote : parts par catégorie | committed 75,8 % des paires / 33,9 % de la KL ; deferred 11,5 % / 28,5 % ; retracted 12,7 % / 37,6 % | Fig. 1 (gauche) |
| Pilote : KL moyenne selon le retraction count e_j | 0 : 0,0924 (141 254 positions) ; 1 : 0,5044 ; 2 : 0,6084 ; ≥ 3 : 0,7196 | Tab. 6 |
| Pilote : KL moyenne selon e_b^(0) = B(1−ρ_b) | 0 : 0,0672 ; 1 : 0,3824 ; 2 : 0,4530 ; 3 : 0,5021 | Tab. 6 |
| Pilote : taux d'hésitation | 27,6 % (réponses incorrectes) vs 22,8 % (correctes) ; IC bootstrap 95 % de l'écart [2,9 ; 6,7] points | App. A |
| Hésitations après entraînement (SDAR-4B math) | Committed : base 75,2 % / TOPD 72,6 % / HOPD 77,1 % ; deferred 11,9 / 12,0 / 10,5 % ; retracted 12,9 / 15,4 / 12,4 % ; entropie (nats) committed 0,141 / 0,177 / 0,124 ; deferred 0,962 / 1,173 / 1,017 ; retracted 1,433 / 1,984 / 1,615 | Tab. 2 |
| Divergence (SDAR-4B math ; statique : MATH500/AIME/GSM8K/moy.) | FKL plein 75,6/9,8/91,1/58,8 ; JSD(0,5) 75,9/8,2/91,5/58,5 ; KL inverse échantillonnée 75,3/7,8/91,7/58,3 ; KL inverse pleine 74,9/15,0/91,1/60,3 | Tab. 4 |
| Divergence (dynamique) | 71,1/8,8/90,3/56,7 ; 71,8/8,5/90,6/57,0 ; 71,9/8,3/91,0/57,1 ; 72,5/9,7/90,7/57,6 | Tab. 4 |
| Poids hindsight (SDAR-4B math ; statique / dynamique moy.) | Uniforme 59,3 / 57,5 ; w_j seul 58,4 / 56,9 ; w_b seul 58,4 / 57,6 ; w_j et w_b 60,3 / 57,6 | Tab. 5 |
| Poids hindsight (SDAR-1.7B math) | Uniforme 51,0 / 46,7 ; w_j 49,0 / 46,2 ; w_b 50,6 / 46,5 ; les deux 51,4 / 47,8 ; w_j seul : MATH500 statique 62,2 (round 20) → 47,3 (round 30) | Tab. 13, 14 |
| Poids hindsight (code, moy. LCB-v2 et LiveBench ; statique / dynamique) | SDAR-4B : uniforme 21,4/18,8 ; w_j 21,3/19,3 ; w_b 20,0/19,4 ; les deux 21,5/19,8 ; SDAR-1.7B : 9,9/8,9 ; 10,4/9,9 ; 9,3/8,3 ; 11,0/9,4 | Tab. 15 |

[SOURCE] Les auteurs notent que « the best weighting scheme can depend on the task and decoding rule » (App. C.3) et que la comparaison avec TOPD « full reverse KL » « does not isolate the effect of supervising hesitations, since HOPD also uses hindsight weights » (App. B.6).

## 30. Contraintes techniques

- [SOURCE] Enseignant et étudiant évalués sur le même état pré-action et les mêmes positions.
- [SOURCE] Les poids de hindsight exigent la trajectoire complète ; ils sont calculés après le rollout.
- [SOURCE] Aucune passe avant supplémentaire n'est requise par rapport à la supervision des positions engagées ; le coût supplémentaire est l'évaluation de la perte à plus de positions.
- [SOURCE] La perte plein vocabulaire est évaluée à chaque position de chaque état enregistré.

## 31. Dépendances

[SOURCE] SDAR, TraDo, LLaDA, ESPO ; JetEngine ; Fast-dLLM ; dLLM-RL (évaluation de TraceRL) ; OpenCompass (pilote) ; DeepSpeed ; MATH, PrimeIntellect (DeepCoder), LiveCodeBench-v2, LiveBench, AIME2024, GSM8K, MATH500 ; 4 GPU. [NON SPÉCIFIÉ] Modèle de GPU, versions.

## 32. Coûts / ressources

[SOURCE] 4 GPU ; DeepSpeed ZeRO-2 ; débit mural comparé (App. B.5). [NON SPÉCIFIÉ] Temps d'entraînement, mémoire, modèle de GPU.

## 33. Limites identifiées par les auteurs

[SOURCE] Il n'y a pas de section « Limitations ». Les passages suivants le sont de fait : (1) l'effet des poids dépend de la tâche et du mode de décodage (App. C.3) ; (2) la comparaison avec TOPD plein vocabulaire n'isole pas l'effet de la supervision des hésitations (App. B.6) ; (3) effondrement observé de la variante « w_j seul » sur SDAR-1.7B (Tab. 14) ; (4) certaines métriques sont hétérogènes (ex. variante w_j seul meilleure en code 1,7B dynamique, Tab. 15).

## 34. Points non spécifiés

Voir table GAP.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Tab. 7, 8, 9, 11 ; Éq. (4)–(6) ; templates (App. B.3) ; protocole d'évaluation (App. B.4) ; sweeps de seuil ; configurations de baselines (App. B.2).

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Pseudo-code complet de HOPD | Haute | §4 |
| GAP-002 | Code source | Haute | — |
| GAP-003 | Scheduler/warmup de l'optimiseur | Basse | Tab. 7 |
| GAP-004 | Modèle de GPU et temps d'entraînement | Moyenne | App. B.1 |
| GAP-005 | Traitement des états couvrant plusieurs blocs pour w_b | Moyenne | Éq. (5) |
| GAP-006 | Écarts entre valeurs TOPD de cette publication et celle du papier source (réimplémentation) | Moyenne | Tab. 1 ; App. B.2 |
| GAP-007 | Nombre de seeds ; variance | Haute | §5 |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-6-001 | Catégories committed/deferred/retracted | [SOURCE] | §3 |
| REQ-6-002 | Poids hindsight | [SOURCE] | §3–4, Éq. (4) |
| REQ-6-003 | Objectif HOPD | [SOURCE] | Éq. (5)–(6) |
| REQ-6-004 | Configuration d'entraînement | [SOURCE] | App. B.1, Tab. 7 |
| REQ-6-005 | Configuration des baselines | [SOURCE] | App. B.2, Tab. 8 |
| REQ-6-006 | Protocole d'évaluation | [SOURCE] | App. B.4, Tab. 9 |
| REQ-6-007 | Généralisation LLaDA | [SOURCE] | §6 ; App. C.1, Tab. 11 |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Moyenne** (équations complètes ; pseudo-code absent)
- Architecture : **Moyenne**
- Paramètres : **Élevée**
- Reproductibilité : **Moyenne** (hyperparamètres complets ; code non cité ; pseudo-code manquant)

---

# STM 7 — d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models

## 1. Identification de la publication

| Champ | Valeur |
|---|---|
| Titre | d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models |
| Auteurs | Ruitao Liu (Tsinghua University), Qinghao Hu (MIT), Song Han (MIT ; NVIDIA) |
| Date / version | arXiv:2609.35362v1 [cs.LG], 28 septembre 2026 |
| URL | Code : https://github.com/mit-han-lab/d-OPD [SOURCE] |
| Méthode | d-OPD (future-aware on-policy distillation) |
| Objectif principal | Corriger la cible enseignant AR lors de la conversion AR → dLLM par blocs, en tenant compte du contexte futur visible dans le bloc actif |

## 2. Résumé technique

[SOURCE] Dans la distillation on-policy d'un AR vers un dLLM par blocs, l'enseignant AR cause conditionne sur le préfixe causal, alors que l'étudiant voit tout le bloc partiellement débruité, y compris des tokens futurs visibles. d-OPD corrige la distribution de l'enseignant : pour chaque position masquée i, la cible est p̃_{T,i}(v) ∝ r_i(v)·exp(Δs_i(v)), où r_i est le prior causal de l'enseignant et Δs_i(v) = s_i(v) − s_i(x̂_i) mesure la compatibilité du suffixe complété avec le candidat v. La correction est restreinte à l'union des top-k de l'enseignant et de l'étudiant (k=16), et n'est appliquée que si le rollout est jugé correct par un vérificateur. La perte est la KL directe KL(p̃_T ‖ p_θ). Seule la cible change ; l'architecture étudiante, l'objectif de distillation et la procédure globale sont inchangés.

## 3. Problème traité

[SOURCE] Un désalignement de conditionnement de cible : l'étudiant block-dLLM prédit à partir du bloc partiellement masqué entier (tokens visibles à gauche **et** à droite), tandis que la cible AR standard ne dépend que du préfixe causal. L'OPD aligne les états d'entraînement avec les états de génération, mais pas les informations conditionnantes de la cible.

## 4. Objectif de la méthode

[SOURCE] Fournir une supervision enseignant qui reflète l'information visible de l'étudiant ; améliorer la moyenne sur 6 benchmarks et réduire le temps d'entraînement pour atteindre un niveau donné.

## 5. Hypothèses scientifiques et techniques

| Hypothèse | Classe | Trace |
|---|---|---|
| La cible idéale est le postérieur de l'enseignant AR conditionné sur tout l'état visible de l'étudiant | [SOURCE] | §3.2, Éq. (7)–(8) |
| Ce postérieur exact est intractable (marginalisation sur préfixes non débruités et futurs non observés) | [SOURCE] | §4.1 |
| Une trajectoire complétée on-policy fournit une réalisation compatible du préfixe incertain et du futur incertain | [SOURCE] | §4.1 |
| Normaliser les vraisemblances futures par un candidat de référence (x̂_i) évite de biaiser en faveur des tokens non scorés | [SOURCE] | §4.1, Éq. (16)–(18) |
| Les futurs de rollouts incorrects peuvent fournir des preuves bruitées ; d'où le portail de correctness | [SOURCE] | §4.2 |
| Les top-k de l'enseignant et de l'étudiant capturent l'essentiel de la masse de probabilité | [SOURCE] | §4.2 ; Tab. 3 |

## 6. Périmètre de la méthode

[SOURCE] Conversion AR → block-dLLM sur la famille Qwen3 (0,6B–8B), tailles de bloc N ∈ {4, 8, 16} (8B), tâches avec vérificateur pour le portail. Non traité [SOURCE App. F] : agrégation de plusieurs futurs ; autres familles de modèles et tailles de bloc ; combinaison SFT puis d-OPD (proposée comme extension).

## 7. Entrées

| Entrée | Description | Classe |
|---|---|---|
| Étudiant p_θ | Initialisé du checkpoint AR correspondant (bloc dLLM) | [SOURCE] §5 |
| Enseignant AR p_T | Même checkpoint, gelé, utilisé comme enseignant causal et comme scoreur de compatibilité future | [SOURCE] §5 |
| Mélange d'entraînement | 61 816 prompts (mélange OPDLM) | [SOURCE] §5 ; App. C.1 |
| Signal de correctness R(ŷ) | Verdict booléen par vérificateur de tâche | [SOURCE] Alg. 1 ; App. C.3 |
| Taille de l'ensemble candidat k | 16 | [SOURCE] §5 |

## 8. Sorties

[SOURCE] Étudiant block-dLLM entraîné (checkpoint avec la meilleure moyenne sur 6 benchmarks parmi ceux évalués). [SOURCE] Cible corrigée p̃_{T,i} par position masquée (Alg. 1).

## 9. Composants nécessaires

| Composant | Rôle | Classe |
|---|---|---|
| Moteur de rollout de l'étudiant avec enregistrement des états visités | Générer des réponses et reconstruire les états de débruitage (temps de premier démasquage de chaque token) | [SOURCE] App. C.2 |
| Vérificateur de tâche | R(ŷ) | [SOURCE] App. C.3 |
| Scoreur AR gelé avec cache | Calcul de r_i et des scores futurs s_i(v) | [SOURCE] App. B |
| Ensemble candidat (union top-k enseignant ∪ top-k étudiant) | Restreindre le scoring | [SOURCE] §4.2 |
| Module de perte KL | Éq. (19) | [SOURCE] |
| Banc d'évaluation | 6 benchmarks | [SOURCE] App. C.4 |

## 10. Architecture / organisation de la méthode

```text
Prompt ─► rollout étudiant (bloc dLLM) ─► réponse complétée ŷ, états visités, R(ŷ)
                       │ un état X_t par bloc (round de débruitage tiré uniformément)
                       ▼
  pour chaque position masquée i :  prior causal r_i(v) = p_T(v | q, z, Â_i)
      si R(ŷ)=1 et futur visible : candidats V_i = top-k(enseignant) ∪ top-k(étudiant)
            s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v) ; Δs_i(v) = s_i(v) − s_i(x̂_i)
            p̃_T,i ∝ r_i(v)·exp(Δs_i(v)) sur V_i, 1 ailleurs, puis normalisation
      sinon : p̃_T,i = r_i
                       ▼
        KL(p̃_T,i ‖ p_θ,i) moyennée sur les positions masquées ─► mise à jour
```
[SOURCE] Fig. 2.

## 11. Flux d'exécution

[SOURCE] Une itération on-policy : génération des réponses avec p_θ et enregistrement des états visités ; correctness de chaque réponse ; tirage d'un état par bloc ; pour chaque état de bloc actif : contexte causal z, bloc complété X̂ ; calcul de p_{θ,i} sur toutes les positions masquées ; construction des cibles ; moyenne des pertes token-level sur toutes les positions masquées de la réponse ; mise à jour par l'Éq. (19).

## 12. Algorithme

[SOURCE] Algorithm 1 « One on-policy training iteration of d-OPD » (App. A) : lignes 1–26 : générer et enregistrer les états ; obtenir ŷ et R(ŷ) ; échantillonner un état visité par bloc ; pour chaque X_t : z, X̂ ; exécuter p_θ sur X_t pour obtenir p_{θ,i} ; pour chaque position masquée i : Â_i, r_i ; **si R(ŷ)=1 et X_t contient un contexte futur visible pour i** : V_i^(k) = union top-k enseignant et étudiant ; Ĉ_i ; s_i(x̂_i) ; pour v ∈ V_i^(k) : s_i(v), Δs_i(v) ; repondérer par exp(Δs_i(v)) et normaliser → p̃_{T,i} ; **sinon** p̃_{T,i} = r_i ; moyenne des pertes token-level par rollout ; mise à jour de θ par Éq. (19).

## 13. Équations et fonctions objectif

| Éq. | Expression | Rôle |
|---|---|---|
| (1)–(4) | X = (x_1..x_N) ; X_t = (x_{t,1}..x_{t,N}) ; p_{θ,i}(·\|X_t,q,z) ; X̂ = (x̂_1..x̂_N) | Notations du bloc actif |
| (5) | r_i(v) = p_T(v \| q, z, x̂_{<i}) | Cible causale standard |
| (6) | p_T(X \| q, z) = Π_i p_T(x_i \| q, z, x_{<i}) | Joint de l'enseignant sur le bloc |
| (7) | π(X \| X_t, q, z) = p_T(X\|q,z)·1{X ⪰ X_t} / Σ_{X'} p_T(X'\|q,z)·1{X' ⪰ X_t} | Postérieur de bloc conditionné sur la cohérence avec les tokens visibles |
| (8) | π_i(v \| X_t, q, z) = Σ_{X : x_i = v} π(X \| X_t, q, z) | Cible exacte (marginale) |
| (9) | π_i(v\|·) = Σ_{A_i∈𝒜_i(X_t)} π(A_i\|·) π_i(v\|·, A_i) | Décomposition sur les préfixes compatibles |
| (10) | π_i(v\|·,A_i) = p_T(v\|q,z,A_i)·p_T(O_i\|q,z,A_i,v) / Σ_{v'} … | Bayes à préfixe fixé ; O_i = tokens visibles à droite de i |
| (11)–(12) | r_i(v) = p_T(v\|q,z,Â_i) ; π_i(v\|·,Â_i) ∝ r_i(v) p_T(O_i\|q,z,Â_i,v) | Résolution du préfixe par Â_i |
| (13)–(15) | Ĉ_i = (x̂_{i+1},…,x̂_N) ; s_i(v) = log p_T(Ĉ_i\|q,z,Â_i,v) ; p̃_{T,i}(v) = r_i(v)exp(s_i(v)) / Σ_{v'} r_i(v')exp(s_i(v')) | Cible « completed-future » |
| (16)–(18) | Δs_i(v) = s_i(v) − s_i(x̂_i) ; exp(Δs_i(v)) = p_T(Ĉ_i\|…,v)/p_T(Ĉ_i\|…,x̂_i) ; p̃_{T,i}(v) = r_i(v)exp(Δs_i(v)) / Σ_{v'} r_i(v')exp(Δs_i(v')) | Cible relative à la référence (identique sur le vocabulaire complet) |
| (19) | L_i(θ) = D_KL(p̃_{T,i} ‖ p_{θ,i}) | Perte (KL directe de la cible vers l'étudiant) |
| (20)–(22) | Construction du support énumérable (App. D) : ω(A) = p_T(A\|q)/Σ_{A'}p_T(A'\|q) ; p̄_{j+1}(v) = Σ_A ω(A)p_T(v\|q,A) ; p_T(X\|q) = Π_j p_T(x_j\|q,x_{<j}) | Diagnostic de cible exacte |
| (23)–(27) | Complétion échantillonnée X̂, Â_i, Ĉ_i ; r_i ; s_i ; Δs_i dans le diagnostic | Diagnostic |
| (28) | D_KL(π_i(·\|X_t,q) ‖ p̂_i(·\|X_t,q)) | Métrique de qualité de cible |
| (29)–(30) | (N,V) ∈ {(4,6),(4,12),(4,24),(4,48),(6,6),(6,8),(6,12),(6,16),(8,6),(8,8)} ; p_S^(σ)(X) ∝ exp(log p_T(X) + σ Z_X), Z_X ∼ N(0,1), σ ∈ {0 ; 0,5 ; 1,0 ; 1,5} | Configurations du diagnostic |

[SOURCE] Cas limites : si aucun token futur n'est visible, O_i = ∅, le terme de compatibilité est constant en v et la cible se réduit au prior causal ; si tout le préfixe gauche est visible, 𝒜_i(X_t) contient une seule complétion. [SOURCE] Remarque sur la notation : dans (7)–(8) la forme exacte marginalise préfixes et futurs, la forme pratique (13)–(18) utilise une seule trajectoire complétée.

## 14. Données

| Jeu | Rôle | Taille | Source |
|---|---|---|---|
| Mélange d'entraînement OPDLM | Prompts (+ informations de tâche nécessaires à la vérification) | 61 816 prompts, longueur de réponse max 4 000 | [SOURCE] §5 ; App. C.1 |
| MMLU | Évaluation | 14 042 exemples | [SOURCE] App. C.4 |
| MMLU-Pro | Évaluation | 12 032 | [SOURCE] |
| GPQA-Diamond | Évaluation | 198 | [SOURCE] |
| GSM8K | Évaluation | 1 319 | [SOURCE] |
| MATH500 | Évaluation | 500 | [SOURCE] |
| AIME25 | Évaluation | 30 | [SOURCE] |

[SOURCE] Même prétraitement et formatage que le mélange d'origine ; pas de déduplication ni de filtrage supplémentaires ; chaque prompt rendu avec le template de chat Qwen (`add_generation_prompt=True`, `enable_thinking=False`). [NON SPÉCIFIÉ] Composition par domaine du mélange (renvoyée à la publication source du mélange) ; séparation validation.

## 15. Génération / préparation des données

[SOURCE] À chaque itération on-policy : jusqu'à 128 prompts, 1 rollout par prompt, T=1,0, top-p=1,0, top-k=0, 4 pas de débruitage par bloc, longueur max 4 000 tokens ; limite globale de longueur M_s croissant de 100 à 4 000 tokens sur les 100 premières itérations selon un schedule cosinus. Pour l'état : enregistrement du premier instant de démasquage de chaque token ; reconstruction des états visités ; pour chaque bloc de réponse, tirage uniforme d'un round de débruitage visité où le bloc contient au moins une position masquée à prédire ; tirage indépendant entre blocs ; les états de blocs sélectionnés sont combinés en un exemple d'entraînement par trajectoire ; pas de packing de plusieurs exemples dans une séquence. Cibles : construites à partir de la trajectoire complétée.

## 16. Modèles

| Rôle | Modèle | Classe |
|---|---|---|
| Étudiants | Qwen3-0.6B, 1.7B, 4B, 8B convertis en block dLLM (tailles de bloc N=4 pour toutes les échelles ; N ∈ {8, 16} pour 8B) | [SOURCE] §5 |
| Enseignant / scoreur | Checkpoint AR Qwen3 correspondant, gelé ; chargé dans son dtype natif (bf16) | [SOURCE] §5 ; App. B |
| Vérificateur | « Task-specific verification procedure used in the OPDLM training setup » | [SOURCE] App. C.3 (détail : [NON SPÉCIFIÉ]) |
| Baselines | OPDLM (principal) ; SFT ; Blockwise SFT (B-SFT) ; BARD | [SOURCE] §5 |
| Modèle de récompense | Aucun | [DÉDUCTION] |

## 17. Teacher / Student / Reference Model

- Enseignant : [SOURCE] AR gelé ; fournit le prior causal r_i et sert de scoreur de compatibilité future p_T(Ĉ_i|·).
- Étudiant : [SOURCE] block-dLLM initialisé du checkpoint AR ; générateur de rollouts et cible d'optimisation.
- Modèle de référence : [SOURCE] « candidat de référence » x̂_i (token réalisé par le rollout) utilisé comme référence des scores relatifs ; ce n'est pas un modèle.

## 18. Paramètres et hyperparamètres

**Paramètres méthodologiques**

| Paramètre | Valeur | Rôle | Source |
|---|---|---|---|
| k (taille top-k) | 16 par défaut (essais 4, 8, 16, 32, 64) | Ensemble candidat | [SOURCE] §5 ; Tab. 3 |
| Portail de correctness | Activé par défaut (essai Off/On) | Restreint la correction | [SOURCE] §4.2 ; Tab. 3 |
| Divergence | KL directe KL(p̃_T ‖ p_θ) | Perte | [SOURCE] Éq. (19) |
| États | Un par bloc, tiré uniformément | Entraînement | [SOURCE] §4.3 |
| Branches par position | ≤ 33 (1 référence + ≤ 32 distinctes pour k=16) | Scoring | [SOURCE] App. B |

**Paramètres expérimentaux** : 483 itérations on-policy (une passe sur 61 816 prompts) sur 8 GPU NVIDIA H100 80 Go ; ≤ 128 prompts/itération ; T=1,0, top-p=1,0, top-k=0 ; 4 pas/bloc ; longueur max 4 000 ; AdamW LR 1×10⁻⁵, β₁=0,9, β₂=0,999, ε=10⁻⁸, weight decay 0 ; warmup linéaire sur 5 itérations puis cosinus jusqu'à 10 % du pic ; clipping 1,0 ; BF16, TF32, gradient checkpointing, DeepSpeed ZeRO-3 avec offload CPU des paramètres et états d'optimiseur ; garde-fou : le batch de branches est réduit si le cache de préfixe étendu dépasse 16 GiB ; calculs en float32 pour logits du scoreur, log-probabilités, sommes de log-vraisemblance, différences, facteurs de correction, normalisation. Évaluation : zéro-shot, template Qwen3, même longueur de bloc que l'entraînement, 4 pas/bloc, remasquage statique low-confidence, T=1,0, top-p=1,0, top-k=1, une complétion par problème, ≤ 4 096 nouveaux tokens ; checkpoints évalués aux itérations 50, 100, …, 450 et 483.
**Paramètres non spécifiés** : [NON SPÉCIFIÉ] graines ; nombre de rollouts par GPU ; composition des vérificateurs ; paramètres de BARD/B-SFT.

## 19. Boucle d'entraînement

[SOURCE] Une passe sur le mélange, 483 itérations on-policy, un exemple d'entraînement par rollout conservé (« we do not pack multiple trajectory-level examples »), mise à jour de l'étudiant par l'Éq. (19) ; recharge des poids dans le moteur de rollout à chaque itération (inclus dans la mesure de temps, App. E).

## 20. Inférence / génération

[SOURCE] Évaluation en décodage par blocs : N identique à l'entraînement, 4 pas/bloc, low-confidence static remasking, T=1,0, top-p=1,0, top-k=1 ; une complétion ; ≤ 4 096 tokens. [NON SPÉCIFIÉ] Mesures de tokens/pas ou de débit d'inférence (non rapportées).

## 21. Fonction de perte

[SOURCE] Éq. (19), par position masquée ; « The token-level losses are averaged over all masked positions collected across the sampled denoising states of a rollout » (§4.3).

## 22. Optimisation

[SOURCE] AdamW, LR 1e-5, warmup 5 itérations, cosinus jusqu'à 10 %, WD 0, clip 1,0, ZeRO-3 avec offload CPU, BF16/TF32.

## 23. Échantillonnage / sélection / filtrage

[SOURCE] Sélection d'états : un tour de débruitage visité par bloc, tirage uniforme parmi les rounds où le bloc contient au moins une position masquée. Filtrage : portail de correctness (le correctif future-aware n'est appliqué qu'aux rollouts vérifiés corrects ; la supervision causale standard reste appliquée à tous les rollouts ; les exemples sans vérificateur applicable reçoivent la supervision causale). Ensemble candidat : union top-k (k=16) enseignant/étudiant. Position éligible : un token à droite de la position masquée est visible dans l'état échantillonné.

## 24. Critères d'arrêt

[SOURCE] 483 itérations fixes ; sélection du checkpoint à meilleure moyenne sur 6 benchmarks parmi ceux évalués (aucune sélection par benchmark). 

## 25. Évaluation

[SOURCE] MMLU, MMLU-Pro, GPQA-Diamond, GSM8K, MATH500, AIME25 ; pipeline d'évaluation « repository-local OPDLM » de leur snapshot de code ; efficacité d'entraînement : temps mural pour atteindre la meilleure moyenne d'OPDLM ; surcoût du scoring futur ; diagnostic de cible exacte.

## 26. Métriques

[SOURCE] Précision par benchmark ; moyenne non pondérée sur 6 benchmarks ; temps mural jusqu'à la cible (« time to matched performance ») ; surcoût relatif de l'itération (% d'augmentation moyenne du temps d'itération vs OPDLM, moyenne non pondérée sur 4 échelles) ; KL moyenne à la cible exacte (49 948 marginales) ; couverture de masse top-k.

## 27. Protocole expérimental

[SOURCE] Dans chaque cadre de comparaison : mêmes prompts, configuration de rollout, sélection d'états, budget d'optimisation et seeds. Efficacité : temps mural de bout en bout par itération (rollout, vérificateur, scoring, optimisation, sérialisation, rechargement des poids) ; évaluation périodique hors zone chronométrée ; 8 GPU H100 80 Go ; même configuration distribuée pour OPDLM et d-OPD à chaque échelle. Diagnostic (App. D) : support énumérable construit à partir de Qwen3-8B ; (N,V) ∈ 10 configurations ; 400 blocs complets par configuration ; masquage indépendant avec probabilité de rétention 0,45 ; étudiant perturbé (σ ∈ {0 ; 0,5 ; 1,0 ; 1,5}) ; 49 948 marginales ; correction appliquée au vocabulaire restreint complet (pas d'approximation top-k).

## 28. Résultats expérimentaux

(Résultats rapportés, non exigences.)

| Échelle / taille de bloc | Méthode | MMLU | MMLU-P | GPQA-D | GSM8K | MATH500 | AIME25 | Moy. |
|---|---|---|---|---|---|---|---|---|
| Qwen3-8B (AR) | — | 76,1 | 60,5 | 44,4 | 91,7 | 83,2 | 16,7 | 62,1 |
| 8B, N=4 | SFT | 69,1 | 54,3 | 39,9 | 87,3 | 75,4 | 13,3 | 56,6 |
| 8B, N=4 | B-SFT | 70,9 | 52,6 | 39,4 | 88,5 | 72,6 | 16,7 | 56,8 |
| 8B, N=4 | BARD | 68,6 | 55,3 | 38,4 | 88,6 | 76,4 | 16,7 | 57,3 |
| 8B, N=4 | OPDLM | 69,2 | 54,0 | 38,4 | 87,2 | 76,8 | 10,0 | 55,9 |
| 8B, N=4 | d-OPD | 74,3 | 55,9 | 38,9 | 89,3 | 77,2 | 23,3 | 59,8 |
| 8B, N=8 | OPDLM / d-OPD | 65,0 / 67,7 | 45,3 / 48,0 | 24,2 / 27,8 | 77,3 / 83,6 | 59,0 / 61,2 | 6,7 / 13,3 | 46,3 / 50,3 |
| 8B, N=16 | OPDLM / d-OPD | 64,5 / 67,1 | 43,3 / 46,5 | 22,7 / 29,8 | 75,4 / 80,6 | 49,6 / 53,8 | 3,3 / 3,3 | 43,1 / 46,9 |
| Qwen3-4B (AR) | — | 72,1 | 56,2 | 39,9 | 90,4 | 81,8 | 16,7 | 59,5 |
| 4B, N=4 | SFT / B-SFT / BARD | — | — | — | — | — | — | 51,2 / 53,5 / 52,0 |
| 4B, N=4 | OPDLM / d-OPD | 66,2 / 67,4 | 47,8 / 50,5 | 30,3 / 32,3 | 82,9 / 88,3 | 75,0 / 75,4 | 6,7 / 16,7 | 51,5 / 55,1 |
| Qwen3-1.7B (AR) | — | 58,6 | 39,4 | 28,3 | 81,5 | 72,0 | 6,7 | 47,7 |
| 1,7B, N=4 | OPDLM / d-OPD | 42,9 / 55,2 | 29,9 / 30,6 | 22,2 / 27,8 | 73,9 / 74,3 | 60,2 / 59,4 | 6,7 / 6,7 | 39,3 / 42,3 |
| Qwen3-0.6B (AR) | — | 46,0 | 27,3 | 24,2 | 63,5 | 52,4 | 6,7 | 36,7 |
| 0,6B, N=4 | OPDLM / d-OPD | 41,1 / 46,7 | 18,8 / 19,9 | 19,7 / 21,7 | 40,6 / 44,6 | 31,0 / 33,6 | 0,0 / 0,0 | 25,2 / 27,8 |

[SOURCE] Gain de moyenne sur 6 benchmarks : « 2,6–3,9 points » de 0,6B à 8B (§5.1) ; « up to 4,0 points » (résumé, N=8). [SOURCE] Efficacité (Tab. 2 ; temps pour atteindre la meilleure moyenne OPDLM) : 0,6B : OPDLM 18,39 h vs d-OPD 13,15 h (1,40×) ; 1,7B : 19,92 vs 14,75 h (1,35×) ; 4B : 30,72 vs 22,34 h (1,38×) ; 8B : 28,26 vs 17,91 h (1,58×). [SOURCE] Surcoût du scoring futur : 2,63 % de temps d'itération moyen (k=16, portail et restriction activés). [SOURCE] Diagnostic (Tab. 4) : KL moyenne à la cible exacte 0,4444 (cible causale) vs 0,1249 (future-aware), réduction 71,9 % sur 49 948 marginales.

## 29. Ablations

| Ablation | Résultat | Trace |
|---|---|---|
| Taille k (masse couverte enseignant / étudiant ; surcoût) | k=4 : 97,87 / 95,43 % ; 2,44 % ; k=8 : 98,88 / 98,30 % ; 2,57 % ; k=16 : 99,45 / 99,44 % ; 2,63 % ; k=32 : 99,67 / 99,64 % ; 3,50 % ; k=64 : 99,79 / 99,77 % ; 6,02 % | Tab. 3 |
| Portail de correctness (8B) | OPDLM 55,9 ; d-OPD sans portail 58,8 ; avec portail 59,8 | Tab. 3 ; Tab. 5 |
| Détail du portail (8B ; MMLU / MMLU-P / GPQA-D / GSM8K / MATH500 / AIME25) | Off : 74,2 / 56,9 / 37,4 / 87,9 / 76,2 / 20,0 ; On : 74,3 / 55,9 / 38,9 / 89,3 / 77,2 / 23,3 | Tab. 5 |
| Tailles de bloc | Gains sur 8B : N=4 +3,9 ; N=8 +4,0 ; N=16 +3,8 | Tab. 1 |

## 30. Contraintes techniques

- [SOURCE] Le scoreur doit pouvoir évaluer p_T(Ĉ_i|q,z,Â_i,v) pour chaque candidat avec partage du préfixe (cache) ; chaque travail de scoring contient une branche de référence et une branche par candidat distinct.
- [SOURCE] La correction n'est validée (commit) que si le normaliseur plein vocabulaire est fini et strictement positif ; sinon, la cible causale est conservée pour la position.
- [SOURCE] Les entrées exactement nulles restent nulles après correction et sont représentées par −∞ dans les log-probabilités.
- [SOURCE] La correction se réduit exactement à la cible causale lorsque le portail est fermé ou qu'aucun futur n'est visible à droite.
- [SOURCE] Aucune écrêtage des différences de scores finies.
- [SOURCE] Candidate-branch batch réduit si le cache de préfixe étendu estimé dépasse 16 GiB.

## 31. Dépendances

[SOURCE] Qwen3 (0,6B–8B) ; mélange et pipeline d'évaluation d'OPDLM ; Hugging Face `DynamicCache` ; DeepSpeed ZeRO-3 ; 8× H100 80 Go ; benchmarks listés ; baselines SFT, B-SFT, BARD. [NON SPÉCIFIÉ] Versions logicielles.

## 32. Coûts / ressources

[SOURCE] 8× NVIDIA H100 80 Go ; temps mural jusqu'à la cible : 13,15 à 22,34 h (d-OPD) vs 18,39 à 30,72 h (OPDLM) ; 483 itérations ; surcoût 2,63 % par itération ; surcoût de couverture k=64 : 6,02 %.

## 33. Limites identifiées par les auteurs

1. [SOURCE] Une seule trajectoire complétée on-policy est utilisée pour construire la cible ; l'agrégation de plusieurs futurs compatibles pourrait réduire la variance (non réalisée).
2. [SOURCE] Complémentarité avec le SFT non étudiée (pipeline à deux étapes proposé comme travail futur).
3. [SOURCE] Évaluation sur d'autres familles de modèles et tailles de bloc : travail futur.
4. [SOURCE] Le portail de correctness suppose des verdicts de vérificateur ; sans vérificateur applicable, la supervision causale standard est utilisée.

## 34. Points non spécifiés

Voir table GAP.

## 35. Éléments nécessaires à une reproduction

[SOURCE] Mélange de 61 816 prompts ; Alg. 1 ; Éq. (15)–(19) ; k=16 ; portail ; paramètres d'entraînement (App. C.2) ; états (App. C.2) ; protocole d'évaluation (App. C.4) ; protocole de timing (App. E) ; protocole de diagnostic (App. D) ; code public.

## 36. Éléments impossibles à déterminer à partir de la publication

| ID | Information manquante | Importance pour reproduction | Source concernée |
|---|---|---|---|
| GAP-001 | Vérificateurs de tâche utilisés (renvoyés à la procédure d'OPDLM, non détaillée ici) | Haute | App. C.3 |
| GAP-002 | Composition du mélange par domaine | Moyenne | App. C.1 |
| GAP-003 | Seeds, variance (un seul checkpoint rapporté ; sélection sur la moyenne des 6 benchmarks de rapport) | Haute | Tab. 1 ; App. C.4 |
| GAP-004 | Configuration de BARD et B-SFT | Moyenne | §5 |
| GAP-005 | Résultats SFT/B-SFT/BARD pour 1,7B et 0,6B (non rapportés) | Basse | Tab. 1 |
| GAP-006 | Mesures de tokens/pas ou de débit d'inférence | Basse | — |
| GAP-007 | Détail de l'écrêtage des états dont aucun bloc n'a de position masquée | Basse | App. C.2 |

## 37. Traçabilité vers la publication

| REQ | Contenu | Classe | Localisation |
|---|---|---|---|
| REQ-7-001 | Désalignement de conditionnement | [SOURCE] | §3.1 |
| REQ-7-002 | Cible exacte | [SOURCE] | §3.2, Éq. (6)–(10) |
| REQ-7-003 | Cible completed-future relative à la référence | [SOURCE] | §4.1, Éq. (11)–(18) |
| REQ-7-004 | Ensemble candidat top-k union | [SOURCE] | §4.2 ; App. B |
| REQ-7-005 | Portail de correctness | [SOURCE] | §4.2 ; App. C.3 |
| REQ-7-006 | Perte KL directe | [SOURCE] | §4.3, Éq. (19) |
| REQ-7-007 | Algorithme d'itération | [SOURCE] | App. A, Alg. 1 |
| REQ-7-008 | Hyperparamètres d'entraînement | [SOURCE] | App. C.2 |
| REQ-7-009 | Protocole de timing | [SOURCE] | App. E |
| REQ-7-010 | Diagnostic de cible exacte | [SOURCE] | §3.3 ; App. D |

## 38. Niveau de confiance de l'extraction

- Compréhension générale : **Élevée**
- Algorithme : **Élevée** (pseudo-code et équations complets)
- Architecture : **Moyenne**
- Paramètres : **Élevée** (annexes C–E détaillées)
- Reproductibilité : **Moyenne** (code public ; vérificateurs et seeds absents du texte)

---

# Annexe — Comparaison descriptive des publications

> Cette annexe est **descriptive uniquement**. Elle ne modifie aucune STM individuelle, ne fusionne aucune méthode et ne formule aucune exigence commune.

| Publication | Type de modèle étudiant | Source de supervision | Mécanisme principal | Données d'entraînement | Taille(s) de modèles rapportées |
|---|---|---|---|---|---|
| STM 1 — NPD | Autoregressif | Enseignant AR (top-k logits sur séquences packées) | Génération asynchrone + filtre ΔIFD + refresh clairsemé | Jeu interne 1,2 M ; 50k « MetaQA » | 1B–7B (étudiants), 7B–72B (enseignants) |
| STM 2 — FLARE | dLLM par blocs (backbone hybride) | Données SFT externes (pas d'enseignant) | Objectif L_AR + L_diff à deux flux ; masque clean/noisy ; serving unifié | Mélanges Nemotron (Long-CoT, Math, IF) | 1,7B–9B |
| STM 3 — OPDLM | dLLM par blocs (converti d'un AR) | ARLM d'origine gelé (préfixe causal de la séquence terminale) | Self-OPD, KL directe, curriculum de longueur | ≈ 62K prompts publics | 0,6B–8B |
| STM 4 — TOPD | dLLM (bloc ou full-attention) | dLLM enseignant gelé (même état/position) | Supervision sur la trace ; KL inverse | MATH niveaux 3–5 (8 523) | 4B (étudiant), 8B (enseignant) ; LLaDA 8B |
| STM 5 — TSD | dLLM masqué | Le modèle lui-même (pas d'engagement) | Auto-distillation temporelle pondérée par vérificateur | Jeux par tâche (GSM8K, MATH, Countdown, Sudoku, KodCode) | LLaDA-8B (LoRA) |
| STM 6 — HOPD | dLLM (bloc ou full-attention) | dLLM enseignant gelé (toutes positions masquées) | Supervision des hésitations + poids a posteriori ; KL inverse pleine | MATH niveaux 3–5 ; PrimeIntellect | 1,7B et 4B (étudiants), 8B (enseignant) |
| STM 7 — d-OPD | dLLM par blocs (converti d'un AR) | AR gelé corrigé par le futur visible | Cible future-aware, portail de correctness, KL directe | Mélange de 61 816 prompts | 0,6B–8B |

**Différences de protocoles (descriptives)** : décodage d'évaluation glouton/statique (STM 3), température 1,0 avec moyenne sur plusieurs échantillons (STM 4, 6), température 0 avec balayage de seuil (STM 5), températures 1,0 + top-k=1 (STM 7), température 1 + top-p 0,95 + top-k 50 (STM 2) ; ensembles de benchmarks différents d'une publication à l'autre. Aucun chiffre ne doit être comparé d'une publication à l'autre sans vérifier ces protocoles.

---

# Annexe — Tableau de couverture du corpus

| Publication | STM correspondante | État de l'analyse | Informations manquantes principales |
|---|---|---|---|
| NPD (2605.05940v1) | STM 1 | Complète (PDF lu intégralement, annexes comprises) | Calcul exact d'IFD ; renormalisation top-k ; τ final ; jeu de données interne ; hyperparamètres GRPO ; code |
| FLARE (2606.01774v2) | STM 2 | Corps, ablations, annexes B/C/D lus ; annexe A (dérivations de kernels) lue partiellement | Loi du masque aléatoire ; formule du logit shift ; valeurs γ_s, S, N ; détails d'optimiseur secondaires |
| OPDLM (2606.06712v1) | STM 3 | Complète | Optimiseur, clipping ; normalisation de la perte ; détail du sampler ; incohérences de tableaux |
| TOPD (2607.16872v1) | STM 4 | Complète | Estimateur effectif pour SDAR (ambiguïté) ; code ; taille de minibatch |
| TSD (2609.15177v1) | STM 5 | Complète (valeurs de courbes lues sur figures ou texte) | Code ; poids du vérificateur de code ; seeds ; détail des itérations internes |
| HOPD (2609.33301v1) | STM 6 | Complète | Pseudo-code ; code ; scheduler ; seeds |
| d-OPD (2609.35362v1) | STM 7 | Complète | Vérificateurs ; composition du mélange ; seeds ; configuration des baselines |

**Audit final (résumé)** : 7 publications analysées ; 7 STM indépendantes ; aucune STM commune, aucune architecture générique, aucune résolution de contradiction ; chaque STM comporte les 38 rubriques, une table GAP et une évaluation de confiance ; les informations absentes sont marquées [NON SPÉCIFIÉ].
