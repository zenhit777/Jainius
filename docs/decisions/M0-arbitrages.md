# Registre des décisions — M0 (arbitrages documentaires)

Date : 2026-10-11. Statut : validé localement (M0), revue responsable requise pour LIC-1 et E1.

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

## M0-D3 — `.gitignore` : couverture secrets et artefacts

- **Correction appliquée** : `.gitignore` passé de (`.env`, `_artefacts/*`) à
  (`.env`, `.env.*`, `!.env.example`, `_artefacts/`).
- **Vérifié** : `.env` (vide, 0 variable) et `_artefacts/` non suivis par Git ;
  aucun fichier déjà versionné n'est concerné ; l'exclusion n'a donc aucun effet
  rétroactif à traiter.
