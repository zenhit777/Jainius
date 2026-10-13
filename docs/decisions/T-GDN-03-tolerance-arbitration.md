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
- Amplitude des sorties de référence (`refmax`, audit-20261014
  `t_gdn_03_sens.log`) : ≈ 5.9e-2 en config nominale
  (4.4e-2 à 9.7e-2 sur les 6 configs fp32). Le résidu absolu
  1.33e-4 ne s'interprète que rapporté à cette échelle → 0.22 %.
- Le résidu est strictement multiplicatif : à `vscale=4.0`
  (amplitude ×8), l'écart absolu est ×8 (1.06e-3) mais l'écart
  relatif est inchangé (2.231e-3) — c'est un bruit relatif de
  la chaîne de calcul, et le relatif est la grandeur pertinente.
- Ordres de grandeur des résolutions relatives : fp16 2^-10
  ≈ 0.098 %, TF32 2^-8 ≈ 0.39 %, bf16 2^-7 ≈ 0.78 %. Le résidu
  mesuré (0.12–0.22 % selon la config) est du même ordre que la
  résolution de la chaîne (kernel Triton TF32/fast-math) : le
  seuil relatif de 1e-3 (0.1 %) exigeait une exactitude au
  niveau du quantum de représentation.
- Correction clôture A4 (2026-10-13) : la comparaison « ~100×
  inférieur au bruit bf16 » (0.25 absolu sur des activations vs
  2.23e-3 relatif sur des sorties) mettait en regard deux
  grandeurs inhomogènes ; elle était invalide et est retirée.
  À titre homogène (relatif vs relatif) : le bruit relatif
  propre mesuré en bf16 sur ce même test est 7.26e-3, soit ~3×
  le résidu fp32 — informatif, non nécessaire à l'arbitrage.

## Analyse

Le test compare une implémentation GDN à un oracle Triton. Le
résidu relatif (2.23e-3, soit 0.22 %) est du même ordre que la
résolution relative de la chaîne de calcul (TF32 ≈ 0.39 %,
voir Faits établis) et strictement multiplicatif (inchangé à
amplitude ×8) : le seuil relatif de 1e-3 (0.1 %) exigeait une
exactitude au niveau du quantum de représentation, que le
matériel/le kernel ne garantit pas. (Correction A4 : l'ancien
argument « ~100× sous le bruit bf16 » comparait un écart
absolu à un écart relatif ; il est retiré et remplacé par
l'argument de résolution ci-dessus.)

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
