# B1 — Cohérence de B0 vs valeurs publiées Qwen3.5-0.8B (non-thinking)

**Date** : 2026-10-13 (clôture). **SHA mesuré** : `9ecab1a` (B0 mesuré en
`docs/audits/audit-m2-20261013/b0-results.json`, protocole P-G/1.0,
hash `8bd3da29…`, 3 répétitions, 0 échec de génération).
**Référence publiée** : fiche officielle `Qwen/Qwen3.5-0.8B` (carte HF,
consultée le 2026-10-13) : MMLU-Pro 29.7, MMLU-Redux 48.5, IFEval 52.1
(non-thinking). Aucun chiffre officiel pour GSM8K, MATH-500, MMLU
(standard), HumanEval, MBPP.

Protocole B0 (rappel) : SGLang `dev-cu13`, `enable_thinking=False`,
temperature 0.0 / top_p 1.0, budgets 2048 (gsm8k, math500), 1024
(ifeval), 512 (défaut). L'échantillonnage officiel non-thinking est
temperature 0.7 / top_p 0.8 / top_k 20 (fiche) : **aucune comparaison
n'est à protocole identique**.

## 1. Tableau de cohérence

| Benchmark | B0 (mean 3 reps) | Publié (non-thinking) | Écart | Explication |
|---|---|---|---|---|
| GSM8K | 0.5219 | — (non publié) | — | Pas de référence officielle. Extraction après `####`, consigne ad hoc, temp 0. |
| MATH-500 | 0.3313 | — (non publié) | — | Pas de référence officielle. Extraction `\boxed{}` (accolades imbriquées gérées, testé). Budget passé à 2048 après constat de 60 % de troncature à 512. |
| MMLU | 0.4476 | MMLU-Redux 48.5 (variante ≠) | −3.7 (indicatif) | Variante différente (Redux ≠ standard). Scoring par génération d'une lettre (système : « une seule lettre ») vs logprobs côté officiel ; réponses verbeuses comptées 0 (cf. `failures_sample`). Temp 0 vs sampling officiel. |
| MMLU-Pro | 0.1882 | 29.7 | −10.9 | 10 choix (A–J) : la génération libre d'une lettre est systématiquement défavorisée vs scoring logprobs ; même effet verbosité qu'en MMLU, amplifié par 10 options. Temp 0. |
| IFEval | 0.2181 | 52.1 | −30.3 | Voir §2 : métrique = prompt-level strict (la plus exigeante) ; 41/541 prompts (7.6 %) en échec automatique (borne sup 0.30 même si tous passaient) ; pas de variante loose ; sampling temp 0 ; wording ad hoc. |
| HumanEval | 0.2825 | — (non publié) | — | Pas de référence officielle. pass@1, temp 0, consigne « code seul ». |
| MBPP | 0.2540 | — (non publié) | — | Pas de référence officielle. pass@1, signature imposée (`mbpp_signature`). |
| **Composite** | **0.3205** | — | — | Moyenne équipondérée P-G/1.0, sans référence officielle. |

## 2. IFEval — définition exacte de la métrique (porte B1)

Source : Zhou et al., arXiv:2311.07911 §2 (IFEval). Quatre scores
officiels : **prompt-level strict-accuracy** (% de prompts dont TOUTES
les instructions vérifiables sont suivies), **inst-level
strict-accuracy** (% d'instructions suivies), et leurs deux variantes
**loose** (8 transformations anti-faux-négatifs : markdown, intro/outro…).

Métrique du harnais Jainius (`score_ifeval`, `src/jainius/eval/b0_runner.py:410`) :
**prompt-level strict** — un prompt réussit ssi toutes ses contraintes
passent, sans transformation loose, avec des vérificateurs stricts
réimplémentés. C'est la plus exigeante des quatre ; le chiffre officiel
52.1 de la fiche (sans précision strict/loose — standard industriel :
prompt-level strict) ne lui est donc pas directement comparable.

Part des prompts touchés par les deux contraintes non implémentées
(mesuré le 2026-10-13 sur `google/IFEval` en cache,
`logs/ifeval_unimpl.json`, `logs/ifeval_unimpl.log`) :

- `detectable_format:constrained_response` : **10/541 (1.85 %)** ;
- `language:response_language` : **31/541 (5.73 %)** ;
- **union : 41/541 (7.58 %)**, disjointes (0 prompt avec les deux) ;
- ces prompts sont en **échec automatique** (`NotImplementedError` →
  fail, honnêteté du score).

Borne supérieure : même si les 41 passaient, IFEval ≤ (119+41)/541 ≈
**0.30**, toujours −22 pts sous le publié. L'écart est donc dominé par
les autres facteurs (strict-only, sampling, wording, troncature 1024),
pas par les 2 contraintes manquantes. **Conséquence pour B2** : les
implémenter ne suffira pas à rejoindre 52.1 ; le choix
(implémenter vs exclure d'IFEval du composite en P-G/1.1) reste à
arbitrer, avec le constat chiffré ci-dessus.

## 3. Limites

- Les chiffres publiés sont auto-rapportés par l'équipe Qwen (sans
  protocole détaillé par benchmark) ; ce sont des références
  indicatives, pas des oracles.
- Aucune re-mesure B0 dans cette clôture (gel P-G/1.0 respecté ; B2/B3
  NOT RUN, voir rapport).
