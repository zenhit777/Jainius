# Audit J2 — parité entraînement ↔ serving et alignement `shifted` (2026-10-15)

Objectifs servis : G2, G4. Image : `jainius-flare-serve:j1` (extractions),
checkpoint FLARE-2B publié (mêmes SHA qu'en J1). Fork lu et exécuté tel quel
(D-04) : aucune modification. Sources papier : pages arXiv HTML (voir
`SOURCES.md` — même contenu que les PDF, à utiliser désormais).

## Résultat

| Question | Réponse (mesurée) |
|---|---|
| Mode GDN retenu | **2 (num_clean=1) en denoise, 1 en commit** = le natif serving |
| Tolérance de parité noyau (chemin de production packé) | **max_abs 3,0e-8**, cosinus 1,0 (modes 1 et 2) |
| Chemin propre vs forward HF (poids publiés) | argmax **25/25**, cosinus 0,9998 |
| Règle d'alignement `shifted` | **étudiant[i-1] et enseignant[i-1] pour le jeton i, graine exclue** |
| Écart KL bon vs mauvais (24 positions) | **2,40** contre 3,43 / 5,37 / 3,95 |
| Comportemental (notre forward vs serveur T=0) | 2/4 exacts, 2 bascules near-tie ≤ tolérance |

**La parité entraînement ↔ serving est établie sous la règle d'alignement
shifted « indice i-1 des deux côtés, graine exclue ».**

## 1. Noyaux GDN : le mode est tranché numériquement (E1)

- Chemin de production (noyau packé, batchs graphés) contre notre récurrence,
  conventions serving exactes : modes 1 et 2 à 3,0e-8
  (`logs/packed_parity.log`). Le mode 2 (mixte, num_clean=1) est celui du
  denoise natif (`_set_denoise_flags`), le mode 1 celui du commit.
- Fait établi (divergence du chemin non graphé uniquement) : le noyau
  non-packé de repli ignore un beta par tête (beta=1 effectif, preuve
  `logs/debug_beta*.log`) ; son mode 0 accumule sans réamorcer entre blocs.
  Le chemin de production (graphes CUDA, comme en J1) n'est pas concerné :
  son beta est calculé en noyau et respecté. Consigné ici, sans demande de
  correctif fork (D-04, aucun besoin pour nos objectifs).
- Conséquence : notre sémantique historique « tout depuis S̃_end » (mode 0)
  n'est PAS celle du serving. Le code gagne des modes explicites
  (`GDNRefLayer.forward_noisy_block(..., read_mode=0/1/2)`, défaut 0 = gelé).

## 2. Masques et layout (E2)

- Denoise serving : bloc [graine, MASK, MASK], ligne graine causale, autres
  lignes bidirectionnelles + passé propre complet, GDN mode 2
  (code serving lu : `_hybrid_diffusion_mask`, `dllm_gdn_num_clean=1`).
- Implémenté tel quel : `TwoStreamRunner.denoise_block_shifted` (+ test
  unitaire de layout et d'isolation documents). `JainiusModel` et le chemin
  `current` restent gelés (test de refus `shifted` intact).

## 3. Modèle complet sur poids publiés (E3)

- Propre : runner vs forward HF, 25/25 argmax identiques, cosinus 0,9998
  (max_abs 0,78 : bruit bf16/noyaux, sans effet comportemental).
- Réplique de l'algorithme serving avec notre forward sur 4 invites J1 :
  2/4 exacts ; cas « planets » : serveur « Mercury », nous « Earth » avec
  écart top-1/top-2 de 0,625 < 0,84 (tolérance) — bascule near-tie
  compatible avec la parité, pas une réfutation (`logs/replica.log`,
  `logs/margin.log`).

## 4. Alignement des cibles (E4)

- Enseignant : forward AR gelé (même contexte propre) ; cibles `shifted` =
  log-softmax NON décalé consommé à l'indice i-1
  (`compute_teacher_targets_shifted`, `current`+`roll(1)` inchangé).
  Cohérent avec OPDLM (cibles `pARLM(·|x0^{<b},x0^{b,<i})`, page HTML §4).
- KL sur checkpoint publié, 24 positions de texte réel : bon 2,40 ;
  mauvais étudiant 3,43 (+43 %) ; mauvais enseignant 5,37 (+124 %) ;
  sans décalage 3,95 (+65 %). Le bon alignement gagne partout.

## 5. Code, tests, limites

- Modifié (strict J2) : `src/jainius/model/gdn.py` (modes),
  `src/jainius/model/twostream.py` (denoise shifted),
  `src/jainius/distill/targets.py` (cibles shifted),
  `tests/numerical/test_j2_shifted.py` (5 tests, passent).
- Intégration `train_step` shifted reportée en J3 (nécessite les rollouts
  Diffusion-Trust qui produisent les blocs à graine) — le forward et les
  cibles, objets de J2, sont fonctionnels et validés.
- Non-régression : 37 passés, 2 ignorés (masques, two-stream, GDN,
  enseignants, perte, pas d'entraînement). Suite complète au commit.
- Bruts : `logs/` (parités noyaux, modèle, KL, réplique, check_env).
