# ARCHITECTURE — Jainius après J3 (2026-10-15)

Description de l'existant, vérifiée sur le code. Hiérarchie : PRD >
DECISIONS > AGENTS > `decisions/` > ce fichier (PRD §8).

## 1. Principe (PRD §2, D-02/D-03)

Un seul mécanisme de diffusion (FLARE : double flux, convention `shifted`,
décodage Diffusion-Trust) et un seul mécanisme d'entraînement (distillation
on-policy OPDLM : rollouts étudiant, cibles enseignant AR gelé, KL).
Le fork SGLang de FLARE est utilisé tel quel (D-04) ; notre forward
d'entraînement s'aligne sur lui (parité J2).

## 2. Convention `shifted` (production)

- Bloc runtime Bb = `DiffusionSpec.block_size` (défaut 4) : réponse découpée
  en slices `[s, e)` de Bb jetons ; position `s` = graine (jamais
  supervisée), `[s+1, e)` supervisées. Règle J2 : la position `i` se prédit
  depuis l'indice `i-1`, étudiant comme enseignant.
- `DiffusionSpec.unit_block` = `block_size - 1` (masques par bloc).
- La convention `current` est gelée : `JainiusModel` (qui la refuse sinon),
  `forward_train`, `train_step`, `rollout`, `compute_teacher_targets`
  (`roll(1)`) restent en place mais ne sont plus étendus.

## 3. Modules (`src/jainius/`)

| Module | Rôle | Entrées / sorties |
|---|---|---|
| `model/twostream.py` — `TwoStreamRunner` | référence deux-flux autour du checkpoint HF (normes/MLP/têtes HF réutilisés, SDPA exact, GDN via `GDNRefLayer`) | `prefill`/`commit` (causal, écrivent), `denoise_block` (B0, mode 0), `denoise_block_shifted` (serving : mode 2, graine causale), `forward_train` (gelé `current`), `forward_train_shifted` → `(h_x0, h_noisy [B,nb,Bb,D], starts)` |
| `model/gdn.py` — `GDNRefLayer` | récurrence GDN de référence | `forward_clean`, `forward_noisy_block(..., read_mode=0/1/2, num_clean)` (0 = serving-denoise historique B0 — PAS le natif ; 2+`num_clean=1` = denoise natif ; 1 = commit natif) |
| `model/arch.py` — `JainiusModel` | enveloppe causale `current` (gelée) | `prefill`/`denoise`/`commit` sur cache HF, `forward_two_stream` refusé par défaut (`LegacyLeakyError`) |
| `model/state.py`, `model/masks.py` | empreintes d'état, masques `current` | — |
| `decode/rollout.py` — `rollout_shifted` | trajectoire Diffusion-Trust on-policy (graine prefill/commit, `logits[i-1]`, seuil/top-1, dernier pas tout le résidu) | `(runner, prompt_ids, response_len multiple de Bb, sampler)` → `Trajectory` (x0 Bb/bloc, `Round`s, `n_forwards = 1 + nb×(denoise + commit)`) |
| `decode/sampler.py`, `decode/trajectory.py` | échantillonnage (confiance = prob), `Trajectory`/`Round` | `SamplerSpec(static/dynamic, threshold, steps_per_block, T, top_k/p, seed)` |
| `distill/targets.py` | cibles enseignant uniques | `compute_teacher_targets` (`current`, `roll(1)`, gelé), `compute_teacher_targets_shifted` (non décalé, consommé en `i-1`) |
| `distill/loss.py` | KL enseignant→étudiant masquée, par morceaux | `kl_teacher_to_student(h, head, ..., sup_mask, teacher_logits=log-probs)` |
| `distill/teacher.py` — `Teacher` | enseignant gelé (`requires_grad=False`, `eval`, forwards `no_grad`) | instance DISTINCTE de l'étudiant (même checkpoint) |
| `distill/curriculum.py` | longueurs OPDLM (`rollout_length`, arrondi au bloc) | — |
| `train/loop.py` | `train_step_shifted` (KL compacte par bloc, clip 1.0, asserts finitude/gel/mise à jour), `run_distill_loop` (`DistillConfig`, curriculum → rollouts → pas → checkpoint) ; `run_smoke`/`train_step` gelés | — |
| `train/checkpoint.py`, `train/master.py` | sauvegarde atomique + reprise vérifiée (`code_sha`, RNG, maîtres), AdamW sur maîtres FP32 | `save_ckpt`/`load_ckpt`, `MasterOptimizer` |
| `eval/` | harnais + runner B0 (composite sans IFEval, B0-plain 0,3205 — acquis M0) | — |

## 4. Flux J3 (boucle)

`run_distill_loop` : prompts (synthétiques, J4 apportera de vrais prompts
seuls) → `rollout_length` → `rollout_shifted` (étudiant, `no_grad`,
empreinte poids) → `train_step_shifted` (xt masqué, `forward_train_shifted`
dans le graphe, KL compacte, clip, pas) → `save_ckpt` (pas 1 : modèle,
optimiseur, maîtres, RNG, budget). Enseignant : forwards `no_grad` sur x0.

## 5. États vérifiés

- J1 : fork servi en image `jainius-flare-serve:j1`, checkpoint
  `HybridDiffusion-2B`, 0,8B chargeable (`audit-J1-20261015`).
- J2 : parité noyau 3,0e-8 (modes 1/2), alignement `shifted` validé
  (`audit-J2-20261015`).
- J3 : boucle ci-dessus, 9 tests (`audit-J3-20261015`).
- Ne PAS supposer : débits (J5), rétention (J4), `temperature > 0` au-delà
  du câblage, blocs > 4 (paramètre, PRD §4).
