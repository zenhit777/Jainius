# Registre des décisions — M0 (arbitrages documentaires)

Date : 2026-10-11. Statut : LIC-1 tranchée (usage privé non commercial, § ci-dessous) ; E1 reste ouverte.

## M0-D1 — Convention de logits : `current` = référence, `shifted` = variante isolée

- **Constat** : `AGENTS.md` §4.3 fixait `current` comme convention de référence (baseline
  OPDLM), `shifted` isolée en branche expérimentale. Le tableau D-06 de
  `docs/CAHIER_TECHNIQUE_JAINIUS.md` fixait à l'inverse `shifted` comme défaut provisoire
  en attente de E1. Contradiction confirmée dans les fichiers réels.
- **Arbitrage (M0)** : `current` est la convention de référence opérationnelle. `shifted`
  reste une variante expérimentale distincte, adoptée seulement si E1 la valide
  (H-01 résolue + composite dev ≥ `current` − 1 écart-type inter-seeds + chemin serving
  existant). Aucune bascule silencieuse, aucune convention présentée comme validée
  sans test numérique (alignement prédictions/cibles/positions supervisées).
- **Corrections appliquées** : `CAHIER_TECHNIQUE_JAINIUS.md` — ligne D-06 du tableau §2,
  défaut de `DiffusionSpec.logit_convention` (§9.1), phrase de conséquence §10.3.
  `Cahier_des_charges.md` et `AGENTS.md` inchangés (déjà cohérents avec cet arbitrage).
- **Ouvert** : E1 (comparaison `current` vs `shifted`, jalon M3) ; H-01 (cible KL sous
  `shifted` non établie pour OPDLM).

## M0-D2 — FLARE : oracle externe en lecture seule, aucune intégration en M0

- **Constat** : le tableau D-03 du cahier technique (« Révision si Gate LIC-1 validée en
  faveur du vendoring ») pouvait se lire comme une validation de copie. Vérification :
  D-03 statue que FLARE/OPDLM sont des **oracles, pas des dépendances**, et §20.1
  (gate LIC-1) exige une décision explicite avant tout vendoring — statut
  « Décision requise », donc **LIC-1 NON validée**. `AGENTS.md` §4.2/§13 du cahier
  technique : aucune copie sans vérification licence + compatibilité. Les mentions
  « réutilisation » du cahier des charges (§1.2, §3.2, §7.2) portent sur les mécanismes
  (idées, interfaces, gestion d'états) et s'entendent comme conditionnées à LIC-1
  et aux vérifications de compatibilité (architecture Qwen3.5-0.8B, SGLang, logiciel).
- **Règle opérationnelle M0 (inchangée, explicitée)** : FLARE = référence externe en
  lecture seule et oracle numérique éventuel ; **aucun code FLARE copié dans le projet** ;
  licence non déclarée validée ; compatibilités architecturale / SGLang / logicielle /
  juridique distinguées et toutes **À VÉRIFIER** (H-02, H-07, H-11).
- **Conditions d'une intégration ultérieure** : décision LIC-1 (usage non commercial
  établi ou licence obtenue) + démonstration de compatibilité (T-GDN-03, T-SG-*) +
  périmètre autorisé par le prompt de la phase concernée.
- **Aucune correction de fichier requise** : les documents convergent déjà vers cette
  lecture une fois la conditionnalité explicitée. Aucune licence inventée.

## Addendum d'audit OPDLM–FLARE (2026-10-11, sans commit)

Audit mené sans implémentation M1, sans entraînement, sans copie de code dans `src/`
(checkouts en lecture seule sous `_artefacts/oracles/`, non versionnés) :
FLARE `6ca547a` (2026-08-17, vérifié `git log`) et OPDLM `3bb8bf6` (2026-06-09, vérifié).

- **Licences vérifiées sur fichiers réels** : FLARE `LICENSE` = PolyForm Noncommercial
  1.0.0 (`Required Notice: Copyright 2026 Yuchen Zhu and HybridDiffusion contributors`,
  amont TorchTitan BSD-3-Clause) — usage, modification et distribution permis pour tout
  usage non commercial. OPDLM `LICENSE` = MIT (© 2026 DIVE Lab, Texas A&M).
  Ceci n'est pas un avis juridique : LIC-1 reste ouverte et porte désormais sur un seul
  point : le caractère non commercial de l'usage prévu de Jainius, à trancher par le
  responsable. Aucun vendoring avant cette décision.
- **M0-D2 est une précaution provisoire, pas un renoncement** : le cahier des charges
  impose l'intégration FLARE (référence d'exécution §1.2/§3.2, réutilisation §7.2,
  serving natif F-010) et `AGENTS.md` §4.2 la conditionne sans l'interdire. La règle
  « aucun code copié » vaut pour M0 ; le chemin conforme de réutilisation est :
  checkout lecture seule (fait) → décision LIC-1 → gates de compatibilité H-02
  (T-GDN-03), H-07/H-11 (T-SG-01) → vendoring sous PolyForm NC si usage non commercial
  confirmé, sinon réimplémentation propre. Aucun moteur parallèle ni kernel contournement
  (option C rejetée, ADR-001/ADR-008 maintenus).
- **Ancres de compatibilité établies (structurelles, exécution restant À VÉRIFIER)** :
  flavor FLARE `0.8B` (`torchtitan/models/qwen3_5/__init__.py:67-87`) identique au
  `config.json` source (24 couches, dim 1024, 8/2 têtes, GDN 16/16×128, conv 4,
  vocab 248320, tying, rope θ 1e7/partiel 0.25) ; auto-upgrade d'architecture du fork
  (`eval/sglang/srt/configs/model_config.py:156-167`) ; chargement tolérant
  (`eval/sglang/srt/models/qwen3_5.py:1081-1086`, ignore mtp/visual) ; sémantique
  denoise-sans-écriture / commit-causal (`low_confidence_shift_hybrid_diffusion.py:91-117`,
  `gdn_dllm_backend.py:764-876`) conforme à C-03 ; SGLang amont sans Qwen3.5 dLLM
  (docs publiques 2026-10-11 : LLaDA2.0, SDAR) → fork requis (D-11 confirmé).
- **Précisions reversées au cahier technique (sans réécriture)** : H-08 résolue —
  confiance = `sampled_token_prob` (`OPDLM/sample/bd3lm_rl_rollout.py:209-211`) ;
  OPDLM entièrement attention pure (KV-cache incrémental, masque additif 4D,
  `rl_sdar.py:318-357,513-557`) → D-04 EXPÉRIMENTAL confirmé, adaptation GDN requise ;
  curriculum yaml `start 100 / end 2000 / ramp 20 / cos`, `end 4000` via lanceurs
  (précise §10.6/§11.3, pas de contradiction) ; `one_state_per_block=True` dans tous
  les lanceurs ; `top_k_logits=16` réservé 4B/8B.

## M0-D3 — `.gitignore` : couverture secrets et artefacts

- **Correction appliquée** : `.gitignore` passé de (`.env`, `_artefacts/*`) à
  (`.env`, `.env.*`, `!.env.example`, `_artefacts/`).
- **Vérifié** : `.env` (vide, 0 variable) et `_artefacts/` non suivis par Git ;
  aucun fichier déjà versionné n'est concerné ; l'exclusion n'a donc aucun effet
  rétroactif à traiter.

## Décision LIC-1 (2026-10-11) — usage privé non commercial, TRANCHÉE

- **Déclaration du responsable** : Jainius est destiné à un usage personnel et privé,
  sans objectif commercial ni application commerciale anticipée. Consignée le
  2026-10-11 via le prompt de reprise ; aucune pièce commerciale n'existe dans le dépôt.
- **Termes vérifiés** (fichier réel `_artefacts/oracles/FLARE/LICENSE`, lu à l'audit) :
  PolyForm Noncommercial 1.0.0 — usage, modification et distribution permis pour tout
  usage non commercial ; l'usage personnel pour la recherche et l'expérimentation est
  un usage permis ; obligation de fournir les termes et la mention
  `Required Notice: Copyright 2026 Yuchen Zhu and HybridDiffusion contributors` à
  quiconque reçoit une copie ; composants amont TorchTitan sous BSD-3-Clause (conservée).
  Ceci n'est pas un avis juridique.
- **Autorisé par cette décision** : utilisation locale privée, modification et stockage
  local du code FLARE (kernels GDN two-stream, fork SGLang) et comparaison-oracle, dans
  la limite exacte ci-dessus. Toute copie dans le dépôt versionné doit conserver
  `LICENSE` + `NOTICE` dans le répertoire concerné et rester identifiée comme
  PolyForm NC (le dépôt Jainius n'a pas de fichier de licence propre à ce jour).
- **Non autorisé / non démontré** : aucun usage commercial ; la compatibilité technique
  n'est pas démontrée par cette décision (gates H-02/H-07/H-11 inchangées) ; OPDLM reste
  sous MIT (© 2026 DIVE Lab, Texas A&M) — conserver sa notice en cas de réutilisation.
- **Ambiguïté résiduelle documentée** : tout partage au-delà de l'usage privé
  (publication du dépôt avec code vendoré, diffusion d'un checkpoint dérivé) est une
  distribution au sens de la licence : exigerait termes + notices jointes et
  réexamen préalable. Réutilisation limitée en conséquence à l'usage privé local
  jusqu'à nouvel arbitrage explicite.
