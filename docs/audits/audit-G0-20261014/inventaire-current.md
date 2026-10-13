# Inventaire — code et tests supposant la convention `current` (session G0, 2026-10-14)

Lecture seule du dépôt, sans modification de code. La convention de production est `shifted` (DECISIONS.md D-02) ; `current` est gelée (code conservé, non étendu). La règle d'alignement des cibles sous `shifted` sera établie et validée numériquement au jalon J2 (PRD §6).

## 1. `src/jainius/model/arch.py`

- `DiffusionSpec.logit_convention` (l.46) : défaut `"current"`. `unit_block` (l.52-54) encode déjà la formule `shifted` (B−1) — réutilisable tel quel.
- `JainiusModel.__init__` (l.78-81) : lève `ConventionMismatch` si ≠ `current` — garde à lever en J2, pas à supprimer (conserver le verrou pour `current`).
- `legacy_leaky_forward` (l.133-169) : flux concaténé `[x0 | xt]`, positions répétées, masque JAI-MASK, suppose logit i → jeton i. Chemin historique refusé par défaut en entraînement (P1) — caractérisation uniquement.
- `prefill` / `denoise` / `commit` (l.83-109) : forwards causaux, neutres vis-à-vis de la convention — réutilisables tels quels.
- `TwoStreamBatch`, `TwoStreamHidden`, `ConventionMismatch`, `StateLeakError`, `LegacyLeakyError`, `model_version` : neutres — réutilisables.
- À refaire sous `shifted` (J2) : défaut de convention, levée des gardes, variante du chemin concaténé (jeton graine, bloc logique 4 = 3 + graine, alignement cible).

## 2. `src/jainius/distill/targets.py`

- `compute_teacher_targets` (l.23-44) : `logits.roll(1)` + `log_softmax` — alignement `current` pur. À refaire sous `shifted` (nouvelle règle J2).
- Réutilisable : la signature `(x0, hidden_fn, lm_head)` et le `log_softmax` float32 comme forme de sortie.

## 3. `src/jainius/distill/teacher.py`

- `hidden_for_loss` (l.85-97) : `.roll(1, 1)` — `current`. `teacher_logits_for_loss` (l.100-106), `logprobs_simple` (l.109-115) : idem.
- `logprobs_extended` / `basic_block_attention` (l.24-57, 118-171) : port `clean_fill` OPDLM + `roll` — `current`, et gelé avec T-LOSS-05 (DECISIONS.md §C-1 : plus aucun travail dessus).
- Réutilisables tels quels : `Teacher.__init__` (gel), `is_frozen`, `logits`, `hidden`.

## 4. `src/jainius/train/loop.py`

- En-tête (l.6) : « Convention `current` uniquement ». `train_step` (l.84-174) : étudiant NON décalé, enseignant décalé via `compute_teacher_targets` (l.99-107, 148-159) — bloc d'alignement à refaire sous `shifted` (J2).
- `run_smoke` (l.187-308) : `response_len % block_size == 0` (l.230) — à revoir sous `shifted` (bloc logique 4, unité 3).
- Réutilisables : squelette curriculum → rollouts → KL → AdamW → checkpoint/reprise, asserts de finitude, clipping 1.0, empreinte et fraîcheur des trajectoires (`param_fingerprint`, `assert_traj_fresh`).

## 5. `src/jainius/decode/rollout.py`

- `rollout` (l.23-106) : bloc = `[MASK] × B`, logit i → jeton i, décodage causal par blocs. Documenté (l.8-10) : ce N'EST PAS du Diffusion-Trust. PRD §7 : la distillation on-policy réelle (J3) remplace ce chemin.
- `generate` (l.109-136) : enveloppe du même chemin — même statut.
- Réutilisables : `decode/sampler.py` et `decode/trajectory.py` (mécanique d'échantillonnage et de trajectoire, sans mention de convention).

## 6. Modules neutres — réutilisables tels quels

- `src/jainius/distill/loss.py` (`kl_teacher_to_student`, voie backward) : mathématique KL indépendante de la convention ; l'entrée `teacher_logits` change de contenu, pas l'interface.
- `src/jainius/model/masks.py` (`build_two_stream_mask`) : paramétré par `unit_block`, déjà alimenté par `spec.unit_block` (compatible `shifted`).
- `src/jainius/model/twostream.py` (`TwoStreamRunner.forward_train`, `final_norm`, `lm_head`) : aucune mention de convention ; côté étudiant à confirmer en J2 (parité serving).
- `src/jainius/model/gdn.py`, `state.py` : cœur validé (T-GDN-03) — à conserver (PRD §7).
- `src/jainius/distill/curriculum.py`, `train/master.py`, `train/checkpoint.py`, `eval/`, `model/__init__.py` : sans hypothèse de convention.

## 7. Tests supposant `current`

- `tests/numerical/test_target_alignment.py` : verrouille l'alignement `current` (roll enseignant, pas de roll étudiant). Conserver (non-régression du gel) ; nouveaux tests `shifted` à créer en J2.
- `tests/numerical/test_two_stream.py::test_two_stream_rejects_shifted` (l.48-50) : verrouille le refus `shifted` — à inverser/retirer quand `shifted` sera implémentée (J2/J3).
- `tests/unit/test_train_step.py` (l.53) : suppose `roll(1)` dans `compute_teacher_targets` — suit le sort de `targets.py`.
- `tests/numerical/test_teacher.py`, `test_teacher_extended_blocked.py`, `test_loss.py` : cibles/perte voie `current` — conserver (gel T-LOSS-05), compléter en J2.
- `tests/numerical/test_masks.py`, `test_gdn_*.py`, `test_p2_gdn_grad.py`, `test_p4_strict.py`, `test_twostream_runner.py`, `test_denoise_no_write.py` : neutres ou côté étudiant/état — réutilisables.
- `tests/integration/test_rollout.py`, `test_smoke.py`, `test_generate.py`, `tests/unit/test_p1_legacy_refusal.py` : supposent le décodage causal par blocs (ex. `n_forwards == 7`) — à faire évoluer en J3 (Diffusion-Trust).

## 8. Détail historique (ne pas rouvrir)

- `docs/decisions/preM2-B1-alignement.md`, `preM2-E0-convention-prompt.md` : traçabilité du choix `current` (bug B1, option E0-b) — histoire, la production est `shifted` (D-02).
