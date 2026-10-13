> OBSOLÈTE en portée : test gelé (voir DECISIONS.md §C-1). Seuil retenu : 0,1.
# Décision — T-LOSS-05 : arbitrage de tolérance E0 (parité câblage)

**Statut** : **ACCEPTÉE** (arbitrage utilisateur du 2026-10-13).
**Contexte** : `scripts/t_loss_05_parity.py` (code officiel OPDLM
`forward_process` exécuté verbatim : SDAR-from-stock Qwen3-0.6B,
teacher Qwen3-0.6B, venv isolé sm_120), décision E0
(`docs/decisions/preM2-E0-convention-prompt.md`, option (b)).

## Faits établis (mesurés, 3 runs : E/F/G)

- **Teacher** : simple ≡ officiel clean_fill **bit-exact (0.0)** (mêmes
  conditions SDPA) — parité teacher PASS.
- **Masque xt-réponse** : contenu **bit-exact** vs officiel — PASS.
- **Masque x0-réponse** : 12 cellules d'écart (bloc-bidirectionnel
  officiel vs causal D-10) — delta architectural **par conception**,
  non un bug (E0).
- **Perte câblage** (masque natif + teacher officiel des deux côtés) :
  **Δ0.020–0.076**, vs bruit inter-process du code officiel lui-même
  (0.005–0.040, autotune FA) et plancher FA-vs-eager bit-exact (0.0).
  Le câblage (slice/roll/norm/KL) est démontré au plancher de bruit.
- **Perte architecture nominale** (causal D-10 vs natif) : Δ0.30 —
  effet E0 mesuré (borne E0 : 30.0).

## Arbitrage

Critère E0-b accepté : **|Δperte| ≤ 0.5** à masque natif + teacher
officiel partagés, avec deltas architecturaux documentés ci-dessus.
Le seuil strict 1e-3 (auto-défini, jamais une exigence cahier —
le cahier exige « même perte » sous « mêmes trajectoires, mêmes
masques », conditions impossibles entre ces architectures cf. E0)
reste rapporté sans modification rétroactive : il est FAIL (0.020),
et c'est l'arbitrage E0-b — pas un assouplissement silencieux —
qui fait foi pour M2.

## Conséquences

- T-LOSS-05 **PASS sous E0-b** ; E0 est clos.
- D-10 (`causal_x0=True`) inchangé ; aucune modification du code
  officiel ni de Jainius pour faire passer le test.
- La qualité du raisonnement reste à mesurer (B0) — cette parité
  ne prouve que le câblage de la perte, pas la qualité.
