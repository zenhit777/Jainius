# Décision M0 — B0-plain seul comme référence, B0-MTP reporté

Date : 2026-10-12. Statut : **ACCEPTÉE** (décision du responsable).

## Contexte

Le cahier technique §18 (jalon M0) exige comme livrable :
« B0-plain et B0-MTP à C = 1, 4, 8 sous SGLang amont ».

Spike `docs/audits/audit-spike-sglang-20261012/REPORT.md` :

- **B0-plain (AR) : PASS** — SGLang amont (`lmsysorg/sglang:dev-cu13`)
  charge nativement Qwen3.5-0.8B (535 tok/s bs=1, 1507 tok/s C=4,
  couches récurrentes reconnues, CUDA graphs).
- **B0-MTP (FROZEN_KV_MTP) : FAIL** — work-in-progress amont :
  crash d'init + timeout worker sur `dev-cu13` **et** `latest` ;
  cause documentée par les développeurs SGLang eux-mêmes
  (`spec_info.py` : « FIXME(kpham_sgl): Remove FROZEN_KV_MTP here
  once we have established support for it in the scheduler »).

## Décision

1. **B0-plain seul** (C = 1, 4, 8) constitue la référence B0 pour
   M2. B0 est mesurable dès maintenant via l'image Docker existante.
2. **B0-MTP est reporté** : à re-spiker lorsqu'une version de SGLang
   stabilisera FROZEN_KV_MTP pour les modèles à têtes MTP
   (Qwen3.5). Aucun fork Jainius, aucun contournement maison
   (conforme à ADR-001/ADR-002 : pas de kernel avant correction).
3. La déduction du gain MTP (H-04) est suspendue au re-spike ;
   la comparaison B0-plain vs B1 (débit) reste valide sans MTP.

## Conséquences

- M0 peut être clôturé une fois B0-plain mesuré (composite qualité
  7 benchmarks × 3 répétitions + B0-plain C=1/4/8 + P3 re-mesuré
  sur GPU vide).
- Le jalon M5b (serving performances) hérite du report MTP : les
  gains de débit MTP seront mesurés quand le support amont sera
  fonctionnel, ou arbitrés à nouveau.
- Risque résiduel : sans MTP, le plafond de débit de B0 est plus
  bas que celui anticipé (H-04 « ≥ 1,5× » devra être réévalué en
  conséquence — le gain diffusion reste mesurable vs B0-plain).

## Alternatives écartées

- Bloquer M0/M2 en attendant SGLang : coûte un jalon entier pour
  un gain non garanti (le MTP n'est pas requis pour la correction
  de la conversion OPDLM).
- Contourner le MTP par un fork Jainius : interdit par ADR-001
  (pas de kernel avant correction) et par la règle LIC-1
  (pas de code FLARE copié) ; et le bug est amont, pas dans Jainius.
