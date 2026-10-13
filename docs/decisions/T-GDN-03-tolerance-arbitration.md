# Décision — T-GDN-03 : arbitrage de la tolérance relative

**Statut** : **ACCEPTÉE** (option (a), arbitrage utilisateur du 2026-10-13).
**Date** : 2026-10-12 (proposition). **Acceptation** : 2026-10-13.

## Faits établis (audit 8e4d9e3)

- Récurrence exacte : écart 2e-8 (critère 1e-3) — OK.
- État final exact : écart 6e-8 (critère 1e-3) — OK.
- Les deux chemins de l'oracle concordent entre eux : 1.3e-4.
- Résidu final vs critère strict 1e-3 absolu / 1e-3 relatif :
  **absolu 1.33e-4 = OK ; relatif 2.23e-3 = KO** (rapport 2.2×
  le seuil).
- Le résidu est prouvé être du bruit interne du kernel :
  - une référence fp64 reproduit le résidu (donc ce n'est pas
    une erreur d'implémentation déterministe) ;
  - il provient du TF32/fast-math du kernel Triton ;
  - le flag host (déterminisme) n'a aucun effet.
- Le bruit numérique propre au modèle en bf16 est ≈ 0.25,
  soit ~100× le résidu incriminé (2.23e-3).

## Analyse

Le test compare une implémentation GDN à un oracle Triton. Le
résidu (2.23e-3 relatif) est inférieur à la divergence entre
les deux chemins de l'oracle lui-même (1.3e-4 est la
concorde, mais l'échelle du bruit de kernel est du même ordre
que le résidu) et ~100× inférieur au bruit bf16 intrinsèque du
modèle. Autrement dit : le seuil relatif de 1e-3 est plus
stricte que la précision effective de la chaîne de calcul
(Triton TF32 + bf16) — il exige une exactitude que le
matériel/le kernel ne garantit pas.

Le critère absolu (1e-3) est lui respecté (1.33e-4, marge
7.5×).

## Options

| # | Option | Conséquence |
|---|--------|-------------|
| a | **Arbitrer la tolérance relative à 5e-3** (absolu inchangé à 1e-3) | T-GDN-03 PASS (marge 2.2×), H-02 clos, M1 closable. Documentée comme arbitrage explicite, pas comme rétroactivité. |
| b | Garder le critère strict 1e-3/1e-3 | T-GDN-03 reste FAIL dans chaque audit futur ; M1/M2 formellement non clos ; aucun impact fonctionnel (le résidu est du bruit prouvé). |
| c | Écrire un kernel GDN custom (sans TF32) | Contredit ADR-001 (réutiliser les kernels existants) ; coût élevé, hors périmètre. |
| d | Marquer le test passé sans changer le critère | **Interdit** : green-washing (AGENTS.md §6, règle d'honnêteté des tests). |

## Recommandation

**Option (a)**, pour trois raisons :

1. Le résidu est prouvé être du bruit de kernel, pas un défaut
   d'implémentation (référence fp64, TF32, flag host sans
   effet).
2. La marge reste significative : 2.23e-3 mesuré vs 5e-3 (2.2×
   de marge), et l'absolu reste à 1e-3 (7.5× de marge) — le
   test garde un pouvoir de détection réel sur une
   régression structurelle (une vraie erreur de parité est
   typiquement O(1e-1) ou plus, pas O(1e-3)).
3. Les alternatives sont pires : (b) bloque formellement M1
   sans gain de rigueur, (c) contredit ADR-001, (d) est
   interdit.

**Ce n'est pas un ajustement de seuil après observation
d'un échec au sens du protocole expérimental** : c'est
l'arbitrage d'un critère de tolérance numérique dont on a
démontré qu'il était plus strict que la précision effective
de la chaîne de calcul. L'arbitrage est consigné ici, avec la
preuve, avant toute clôture de jalon. Le gel des seuils du
protocole P-G/1.0 (qualité) n'est pas concerné — c'est un
critère de parité numérique (M1), distinct du protocole de
mesure B0.

## Validation requise

L'arbitrage d'un critère d'acceptation de test de parité est
une décision de l'utilisateur (AGENTS.md §3 : l'agent ne
modifie pas seul une convention expérimentale validée).
~~À l'approbation, le test T-GDN-03 est mis à jour avec
tolérance rel=5e-3 (abs=1e-3), documentée dans le test et
dans ce document (statut → ACCEPTÉE).~~

**Approuvé le 2026-10-13 : test mis à jour (rel=5e-3, abs=1e-3),
re-exécuté PASS dans l'environnement dédié FLARE (voir audit
de clôture). H-02 est clos sur cette base.**
