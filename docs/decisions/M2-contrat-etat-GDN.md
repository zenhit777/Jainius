# Contrat d'état GDN pour M2 — planification des blocs bruités

Statut : CONTRACTUEL — implémenté et VÉRIFIÉ dans `twostream.py`
(chemin de référence) ; oracle FLARE T-GDN-03 BLOCKED (préflight documenté).
Références : FLARE Éq. (6) (planification), cahier technique §10.2 (H-02),
tests `test_two_stream_gdn_future_leak` (+ variantes),
`docs/decisions/preM2-*.md`, audit pré-pilote (base 6e4c82a).

## 1. Fait établi (caractérisé, pas supposé)

Sur Qwen3.5-0.8B, le forward concaténé `[x0|xt]` de référence pollue le
flux bruité par récurrence, malgré un masque d'attention exact
(T-MASK-01) et appliqué (sonde 4D fp32 : 2.2e-5 vs causal natif) :
- le futur de la réponse propre `x0` modifie les hidden `xt` ;
- le préfixe prompt de `xt` lui-même les modifie aussi ;
- en bf16 comme en fp32, à 1 comme à 2 blocs.
Le masque ne gouverne que les couches softmax ; les 18 couches GDN
accumulent tout le flux concaténé (états récurrents + contexte conv
noyau 4) sans ré-initialisation. Tout pilote qui calculerait la perte
sur ce forward entraînerait sur des cibles partiellement copiées :
**BLOCKED pour la distillation** (pas d'interprétation des pertes du
smoke antérieur comme apprentissage).

## 2. Contrat requis (M2) — sémantique détaillée (B0, pré-implémentation)

Sources FLARE lues (checkout `_artefacts/oracles/FLARE@6ca547a`, lecture
seule, rien copié dans `src/`) :
- `torchtitan/models/qwen3_5/model/block_gated_delta_rule/README.md`
  (récurrence Éq. 6 simplifiée l.45-56 ; two-stream l.150-184 ;
  packing `cu_seqlens` l.58-60) ;
- `.../chunk_fla_style_wy.py:51-55` (cross term ≡ récurrence bruitée
  avec `v_n = 0`, état initial = état courant du préfixe propre, avancé
  après chaque bloc) et `:109-114` (premier bloc bruité d'un document :
  PAS d'état propre initial — zéro ; blocs suivants : état courant) ;
- `.../fused_recurrent.py:365-479`
  (`fused_recurrent_block_causal_gated_delta_rule` avec `initial_state`
  / `output_final_state`, forme `[N, HV, K, V]`, l.283) ;
- `.../chunk_two_stream_wy.py:1-28` (Phase A : `S_j` depuis `S_0` par
  block-scan + replay bruité ; Phase B : gradients vers le propre).

### (a) Initialisation des blocs bruités depuis S_clean(s_b)

Chaque bloc bruité `b` démarre de l'état propre à sa frontière `s_b`
(snapshot immuable, jamais du flux bruité). Premier bloc d'un document :
état ZÉRO (pas même l'état propre initial — `chunk_fla_style_wy.py:109`).
Forwards de débruitage : lecture seule, aucune écriture.

### (b) Lecture des jetons bruités : état de fin de bloc S̃_end

Tous les jetons du bloc sont lus depuis l'état de FIN de bloc
(bidirectionnalité intra-bloc réalisée par la planification, pas par un
masque). Référence : README l.52-56 (« share the state at the appropriate
block boundary ») ; backward associé (`chunk_fla_style_wy.py:449-454`,
gradient bruité routé dans la transition propre — sémantique attendue
de T-GDN-05).

### (c) Contexte de la convolution courte : PROPRE (pas bruité)

Le flux bruité reçoit « the correct clean prefix for its block »
(README l.179-184, `twostream` ; `block_train_conv` appelé depuis
`model/dllm_model.py`, README l.71). Donc : historique conv[K-1] =
tokens propres à la frontière (pas l'historique bruité), mis à jour
ensuite causalement DANS le bloc bruité sur copie locale, sans
réécriture de l'historique propre. Statut : ÉTABLI au niveau README ;
la granularité exacte intra-bloc du noyau `twostream` n'a pas été relue
ligne à ligne (Triton) → **À VÉRIFIER** contre T-GDN-03 si l'oracle
devient exécutable.

### (d) Application aux chemins Jainius

- `denoise` : lecture depuis S̃_end du bloc (interdit : lecture causale
  intra-bloc — c'est le comportement actuel, conservé sous nom distinct
  jusqu'à validation) ; ne modifie ni l'état propre ni l'historique
  conv propre (garde `StateLeakError` étendue aux copies locales).
- `commit` : forward causal du bloc finalisé, SEUL écrivain (état +
  historique conv) — inchangé.
- `forward_two_stream` : flux propre causal (écrit ses états) ; chaque
  bloc bruité initialisé depuis le snapshot propre de sa frontière
  (zéro si début de document), lecture depuis S̃_end ; reset par ligne
  et par document (états + conv) ; blocs bruités indépendants
  conditionnellement aux snapshots (base de T-GDN-02).

### Points non tranchés : À VÉRIFIER

- Détail intra-bloc du noyau conv `twostream` (voir (c)).
- T-GDN-03 (oracle non exécutable ici — préflight §B4).
- `gen_block_size`/`maximum_unroll` côté serving : hors M1 (M5).

## 3. Non-contrats explicites

- Aucun changement de masque d'attention ne suffit (récurrence non
  gouvernée) — B2.5 de la mission respecté.
- Aucun reset/restauration inventé ici : snapshot/restore par copie
  profonde bit-exacte déjà validée (T-STATE-01) ; le reset par ligne
  reste à implémenter (zéro par ligne des 24 couches).

## 4. État d'implémentation (VÉRIFIÉ — audit pré-pilote, base 6e4c82a)

> Mise à jour post-audit. L'état antérieur (« PROPOSITION, aucune
> implémentation ») est remplacé par l'état réel mesuré. Les tests cités
> ont été exécutés ; voir `_artefacts/audit/` pour les logs bruts.

### Implémenté et VÉRIFIÉ (chemin de référence `twostream.py`)

| Composant | Statut | Preuve (test) |
|-----------|--------|---------------|
| `_clean_layers` (causal, GDN naïve) | ✅ VÉRIFIÉ | T-GDN-01 `test_gdn_01_naive_vs_hf_causal` PASS (parité float64 ≤ 1e-6) |
| `denoise_block` (lecture S̃_end, pas d'écriture, passé complet) | ✅ VÉRIFIÉ (corrigé) | `test_denoise_reads_end_state` PASS ; `test_denoise_no_write` PASS. **Correction** : le masque « passé » était causal-tronqué (position i ne voyait que i+1 clés du passé) — remplacé par le passé propre complet du même document (sémantique `_noisy_block_layers`), + garde `StateLeakError` par fingerprint `runner_state_fingerprint`. |
| `forward_train` (snapshots propres par frontière, blocs alignés `resp_start`) | ✅ VÉRIFIÉ | Propriété de fuite (mesurée, Qwen3.5-0.8B bf16, seed 0) : fuite futur-x0 = **0.0**, fuite préfixe-xt = **0.0** ; dépendances légitimes présentes (3.4 / 6.1 / 2.6). Zéro si début de document ; assertion bloc non-chevauchant-doc. |
| `_noisy_block_layers` (lecture S̃_end, pas d'écriture) | ✅ VÉRIFIÉ | Copies locales immuables ; même propriété de fuite que `forward_train`. |
| GDN two-stream float64 (`GDNRefLayer`, `GDNRef`) | ✅ VÉRIFIÉ | T-GDN-01 PASS ; T-GDN-05 (différences finies) — voir corr. |
| Batched == séquentiel (T-GDN-02) | ✅ VÉRIFIÉ (tolérance) | `test_gdn_02_batched_is_sequential` PASS. **Correction** : `torch.equal` (bit-exact) trop strict en bf16 (mesuré ~0.02 sur \|v\|~10) → allclose atol=0.25/rtol=0.05 + borne contamination inter-lignes < 0.5 (une vraie fuite donnerait O(1)). |
| Packing + reset inter-docs (T-GDN-04) | ✅ VÉRIFIÉ (tolérance) | `test_gdn_04_packing_reset` PASS. **Correction** : même motif bf16 (mesuré ~0.15) → allclose atol=0.5/rtol=0.05 + borne rupture d'isolation < 1.0. |

### Non implémenté / BLOCKED

| Composant | Statut | Blocage (vérifié) |
|-----------|--------|-------------------|
| T-GDN-03 (oracle FLARE) | ⛔ BLOCKED | Préflight `_artefacts/audit/preflight_t_gdn_03.json` : torch **2.11.0+cu128** ≠ exigé **2.10.0** ; **fla absent** ≠ exigé **0.5.1**. Noyaux présents dans le checkout. Règle : BLOCKED, ne pas substituer. |
| `reset_rows` (JAI-STATE M1-B) | ⛔ non implémenté | API HF `DynamicCache.reset()` globale uniquement (hors périmètre de cet audit). |
| Parité numérique vs oracle FLARE | ⛔ BLOCKED | Dépend de T-GDN-03. |

### Tests H-02 (fuite GDN deux-flux) — état vérifié

| Test | Statut | Commentaire |
|------|--------|-------------|
| `test_runner_no_leak_desired_property` | ✅ PASS | Propriété voulue SATISFAITE (fuites 0.0, blocs alignés `resp_start`) ; dépendances légitimes présentes. **Retiré** de `xfail(strict)` car résolu. |
| `test_runner_no_leak_characterization` | ✅ PASS | Caractérisation AVANT/APRÈS : le chemin ANCIEN (flux concaténé, `arch.py`) fuit encore (H-02, BLOCKED) ; le runner planifié ne fuit PAS. |
| `test_two_stream_gdn_future_leak` (chemin ancien) | ✅ PASS | Caractérise la fuite H-02 du chemin concaténé (préserver la mesure). |
| `test_gdn_05_gradient_noisy_to_clean` | ✅ PASS (corrigé) | T-GDN-05 : gradient float64 par différences finies bruité→propre. **Correction** : `forward_noisy_block` exige l'historique conv propre (3e arg, contrat B0 §(c)) — l'appel à 2 args levait TypeError. |
| `test_denoise_reads_end_state` | ✅ PASS (corrigé) | **Correction** : les jetons de substitution `(MASK+7)%248320` partagent la même ligne d'embedding que MASK (ids 248077…248319) — le test mesurait 0.0 par construction. Jetons valides non dégénérés (100/200/300). |

### Corrections appliquées durant cet audit (traçabilité)

1. `twostream.py` : bloc mort post-`return` dans `forward_train` (docstring
   orpheline, guillemet manquant) → **SyntaxError** bloquant l'import du
   paquet `jainius.model` — supprimé.
2. `train/loop.py` : `train_step` consommait `teacher.model(ids).hidden_states[-1]`
   **sans** `output_hidden_states=True` → `None[-1]` TypeError (smoke ERROR).
   Corrigé.
3. `train/checkpoint.py` : `code_sha` résout via racine du dépôt
   (`--show-toplevel` + `cwd`), indicateur `code_dirty` ajouté au manifeste,
   avertissements journalisés à la reprise si arbre sale.
4. `denoise_block` : masque « passé » causal-tronqué → passé propre complet
   du même document (aligné sur `_noisy_block_layers`), + garde non-écriture.
5. Tests : générateurs CUDA (`test_loss`), fakes `train_step` (contrat
   `output_hidden_states`), tolérances bf16 justifiées (T-GDN-02/04),
   jetons non dégénérés (`test_denoise_reads_end_state`), `conv_init` propre
   (T-GDN-05), fingerprint `runner_state_fingerprint` (T-STATE-02 runner).
6. Hygiène : 10 `.pyc` dé-suivis de l'index Git (`git rm --cached`),
   `*.pyc`/`*.pyo` ajoutés à `.gitignore`. **Aucun commit** (règle « aucun
   commit sans instruction séparée ») — l'index est prêt pour un commit
   d'hygiène distinct, à valider par l'architecte.

### Décisions A7 liées à GDN

- A1/B1 : la fonction unique de cibles (`compute_teacher_targets`) est
  indépendante du chemin GDN — pas d'impact sur H-02.
- A2 : la propriété de fuite est SATISFAITE (PASS) sur le runner planifié ;
  la caractérisation de la fuite du chemin ancien est conservée
  (`test_runner_no_leak_characterization` + `test_two_stream_gdn_future_leak`).
