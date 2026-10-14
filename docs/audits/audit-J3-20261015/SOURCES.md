# Sources J3 (2026-10-15)

- `docs/PRD.md` §2, §6 (jalon J3 : rollouts par débruitage de l'étudiant,
  états par bloc, KL ; boucle complète : perte finie, enseignant gelé,
  reprise exacte, checkpoint exportable).
- `docs/DECISIONS.md` (D-01 à D-08, en particulier D-02 FLARE seul,
  D-03 distillation OPDLM seule, D-04 fork utilisé tel quel).
- `docs/audits/audit-J2-20261015/REPORT.md` (parité serving, règle
  d'alignement `shifted` i-1 des deux côtés, graine exclue ; modes GDN).
- Oracle FLARE lu tel quel (D-04, aucune modification) :
  `_artefacts/oracles/FLARE/eval/sglang/srt/dllm/algorithm/low_confidence_shift_hybrid_diffusion.py`
  (lignes 1-12 : layout `[graine, MASK, MASK]` ; 49-52 : masque
  graine-causale ; 57-89 : échantillonnage + confiance ; 91-104 : flags
  denoise `causal_mode=2, num_clean=1` ; 106-117 : flags commit `mode=1` ;
  204-237 : boucle de débruitage, `logits[i-1]`, seuil puis top-1,
  dernier pas tout le résidu ; 239-249 : commit + graine suivante ;
  266-315 : graines prefill/suivante).
- Config serving : `_artefacts/oracles/FLARE/eval/configs/hybrid_diffusion_shift_b3_g1.yaml`
  (`block_size: 3`, seuil 0,95).
- Code Jainius lu et modifié : `src/jainius/model/twostream.py`,
  `src/jainius/decode/rollout.py`, `src/jainius/train/loop.py`,
  `src/jainius/distill/targets.py` (J2, inchangé),
  `src/jainius/distill/loss.py` (inchangé, réutilisé tel quel).
