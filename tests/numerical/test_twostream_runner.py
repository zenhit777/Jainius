"""Runner planifié : équivalence propre, fuite nulle, batchs, packing.

- `test_runner_clean_matches_hf` : le chemin propre (softmax exact +
  GDN naïve) reproduit le forward HF causal (bf16, tolérance large :
  ordre de kernels différent ; la parité stricte est T-GDN-01).
- `test_runner_no_leak_desired_property` : PROPRIÉTÉ VOULUE — avec blocs
  alignés (`resp_start`), le futur x0 et le préfixe xt NE MODIFIENT PAS
  les hidden xt (== 0.0 bit-exact) ; dépendances légitimes présentes.
  SATISFAITE par la planification B0 (mesurée : fuites 0.0).
- `test_runner_no_leak_characterization` : CARACTÉRISATION AVANT/APRÈS —
  le chemin ANCIEN (flux concaténé) fuit encore (H-02, BLOCKED) ; le
  chemin NOUVEAU (runner planifié) ne fuit PAS. Mesure conservée.
- `test_gdn_02_batched_is_sequential` (T-GDN-02) : batched == séquentiel
  (tolérance bf16 justifiée ; contamination inter-lignes bornée).
- `test_gdn_04_packing_reset` (T-GDN-04) : [docA, docB] packé avec reset
  == concaténation des forwards séparés (tolérance bf16 justifiée).
- `test_denoise_reads_end_state` : la prédiction de la position 0 d'un bloc
  dépend des positions 1 à 3 (lecture S̃_end, pas lecture causale).
  Jetons de substitution valides non dégénérés (ids 248077…248319 partagent
  la même embedding — non discriminant).
- `test_denoise_no_write` : denoise ne modifie JAMAIS l'état récurrent
  (fingerprint `runner_state_fingerprint`, garde `StateLeakError`).

A2 : la propriété de fuite est SATISFAITE (PASS) sur le runner planifié ;
la caractérisation de la fuite du chemin ancien est conservée.
"""

import copy
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.model.twostream import TwoStreamRunner

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MASK_ID = 248077


def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval()
    return tok, model


def _prompt(tok, text="Hi there"):
    s = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tok(s, return_tensors="pt")["input_ids"].to("cuda")


def test_runner_clean_matches_hf():
    # NOTE (transformers 5.17) : `hidden_states[-1]` HF est l'état NORMÉ
    # (vérifié bit-exact : == final_norm(sortie couche 23)), pas la sortie
    # brute — la comparaison se fait donc contre la sortie hookée exacte.
    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    cap = {}
    lay = model.model.layers[23]
    orig = lay.forward

    def hook(ms, *a, **k):
        out = orig(*a, **k)
        cap['l23'] = out.detach().clone()
        return out

    lay.forward = hook.__get__(lay, type(lay))
    with torch.no_grad():
        model(ids)  # forward HF complet : le hook capture la couche 23
        h, _ = runner.prefill(ids)
    lay.forward = orig.__get__(lay, type(lay))
    assert h.shape == cap['l23'].shape
    # bf16 : bruit d'ordre de kernels (mesuré ~0.05 cumulé) ; un bug de
    # câblage donne ~20+ (cf. pré-pilote). T-GDN-01 couvre le strict.
    assert torch.allclose(h.float(), cap['l23'].float(), atol=0.5,
                          rtol=0.05), "chemin propre divergent du forward HF"
    del model
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()


def test_runner_no_leak_desired_property():
    """PROPRIÉTÉ VOULUE (A2) — SATISFAITE par le runner planifié (B0) :
    avec blocs alignés sur la réponse (`resp_start`), le futur x0 et le
    préfixe xt NE MODIFIENT PAS les hidden xt (== 0.0 bit-exact), tandis
    que les dépendances LÉGITIMES (préfixe propre, tokens du bloc, propre
    des blocs précédents) sont présentes.

    Mesures (Qwen3.5-0.8B bf16, seed 0) : fuite futur-x0 = 0.0, fuite
    préfixe-xt = 0.0 ; dépendances légitimes 3.4 / 6.1 / 2.6 (>> 0).
    NOTE : sans `resp_start` (blocs couvrant prompt+réponse), un bloc à
    cheval sur la frontière utilise ses propres tokens prompt — ce n'est
    PAS une fuite (input du bloc), c'est un artefact de découpage.
    """
    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    l0 = ids.shape[1]
    xt = torch.cat([ids[:, :l0], torch.full((1, 8), MASK_ID, device="cuda")],
                   dim=1)
    x0 = torch.cat([ids[:, :l0], torch.full((1, 8), 100, device="cuda")],
                   dim=1)
    with torch.no_grad():
        _, h_xt = runner.forward_train(x0, xt, unit_block=4, resp_start=l0)
        # Fuites (doivent être 0) :
        x0b = x0.clone()
        x0b[0, -1] = (x0b[0, -1] + 1) % 248320
        _, h_xt_b = runner.forward_train(x0b, xt, unit_block=4,
                                         resp_start=l0)
        xtb = xt.clone()
        xtb[0, :l0] = (xtb[0, :l0] + 7) % 248320
        _, h_xt_c = runner.forward_train(x0, xtb, unit_block=4,
                                         resp_start=l0)
        # Dépendances légitimes (doivent être > 0) :
        x0c = x0.clone()
        x0c[0, 0] = (x0c[0, 0] + 7) % 248320
        _, h_xt_d = runner.forward_train(x0c, xt, unit_block=4,
                                         resp_start=l0)
        xtd = xt.clone()
        xtd[0, l0] = 100
        _, h_xt_e = runner.forward_train(x0, xtd, unit_block=4,
                                         resp_start=l0)
        x0f = x0.clone()
        x0f[0, l0] = 200
        _, h_xt_f = runner.forward_train(x0f, xt, unit_block=4,
                                         resp_start=l0)
    # Fuites nulles (bit-exact) :
    assert torch.equal(h_xt[:, l0:], h_xt_b[:, l0:]), (
        "fuite futur-x0 (H-02) dans le runner planifié")
    assert torch.equal(h_xt[:, l0:], h_xt_c[:, l0:]), (
        "fuite préfixe-xt (H-02) dans le runner planifié")
    # Dépendances légitimes présentes (test discriminant) :
    assert (h_xt[:, l0:] - h_xt_d[:, l0:]).abs().max() > 0, (
        "préfixe propre ignoré : le runner ne lit pas l'état propre")
    assert (h_xt[:, l0:] - h_xt_e[:, l0:]).abs().max() > 0, (
        "tokens du bloc ignorés : le runner ne lit pas ses inputs")
    assert (h_xt[:, l0:] - h_xt_f[:, l0:]).abs().max() > 0, (
        "propre des blocs précédents ignoré : snapshot de frontière absent")
    del model
    torch.cuda.empty_cache()


def test_runner_no_leak_characterization():
    """CARACTÉRISATION (A2) — état AVANT/APRÈS la planification (B0).

    Chemin ANCIEN (JainiusModel.forward_two_stream, flux concaténé) :
    la fuite GDN H-02 EST présente (documentée, BLOCKED pour la
    distillation) — conservée ici pour ne pas perdre la mesure.
    Chemin NOUVEAU (TwoStreamRunner.forward_train, blocs alignés) :
    la fuite N'EST PLUS présente (planification B0) — mesurée ici.
    Ne PAS modifier pour faire « passer » : ces mesures sont des faits.
    """
    from jainius.model.arch import DiffusionSpec, JainiusModel, TwoStreamBatch

    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    l0 = ids.shape[1]
    xt = torch.cat([ids[:, :l0], torch.full((1, 8), MASK_ID, device="cuda")],
                   dim=1)
    x0 = torch.cat([ids[:, :l0], torch.full((1, 8), 100, device="cuda")],
                   dim=1)
    with torch.no_grad():
        # Chemin ANCIEN (flux concaténé) — fuite attendue (H-02) :
        legacy = JainiusModel(model, DiffusionSpec())
        sup = torch.zeros_like(x0, dtype=torch.bool)
        sup[0, l0:] = True
        h_old = legacy.forward_two_stream(
            TwoStreamBatch(x0=x0, xt=xt, sup_mask=sup),
            allow_legacy_leaky=True).h_xt
        x0b = x0.clone()
        x0b[0, -1] = (x0b[0, -1] + 1) % 248320
        h_old_b = legacy.forward_two_stream(
            TwoStreamBatch(x0=x0b, xt=xt, sup_mask=sup),
            allow_legacy_leaky=True).h_xt
        leak_old = (h_old[0, l0:] - h_old_b[0, l0:]).abs().max().item()

        # Chemin NOUVEAU (runner planifié) — fuite ABSENTE :
        _, h_new = runner.forward_train(x0, xt, unit_block=4,
                                        resp_start=l0)
        _, h_new_b = runner.forward_train(x0b, xt, unit_block=4,
                                          resp_start=l0)
        leak_new = (h_new[0, l0:] - h_new_b[0, l0:]).abs().max().item()
    # Le chemin ancien fuit (fait établi H-02, BLOCKED pour distillation) :
    assert leak_old > 0, (
        f"fuite H-02 absente du chemin ancien (inattendu, {leak_old}) : "
        f"revérifier la caractérisation")
    # Le chemin nouveau ne fuit PAS (planification B0) :
    assert leak_new == 0.0, (
        f"le runner planifié fuit ({leak_new}) : la propriété B0 est rompue")
    del model
    torch.cuda.empty_cache()


def test_gdn_02_batched_is_sequential():
    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    with torch.no_grad():
        _, state = runner.prefill(ids)
        b1 = torch.full((1, 4), MASK_ID, device="cuda")
        b2 = torch.full((1, 4), 100, device="cuda")
        l_seq = [runner.denoise_block(state, b) for b in (b1, b2)]
        bb = torch.cat([b1, b2], dim=0)
        # Batch : même état répliqué (indépendance conditionnelle).
        st2 = copy.deepcopy(state)
        st2.attn_k = [torch.cat([k, k], dim=0) for k in st2.attn_k]
        st2.attn_v = [torch.cat([v, v], dim=0) for v in st2.attn_v]
        st2.docs = [torch.cat([d, d], dim=0) for d in st2.docs]
        st2.gdn = [type(s)(S=torch.cat([s.S, s.S], dim=0),
                            conv=torch.cat([s.conv, s.conv], dim=0))
                   for s in st2.gdn]
        l_bat = runner.denoise_block(st2, bb)
    # CORRECTION (tolérance justifiée) : `torch.equal` (bit-exact) était
    # trop strict en bf16 — le matmul/softmax batch-2 vs batch-1
    # accumule l'arrondi différemment (mesuré : max ~0.02 sur |v|~10).
    # Une CONTAMINATION inter-lignes (vraie fuite) donnerait un écart
    # O(1) — borné ci-dessous. Tolérance allclose = bruit bf16.
    for i, (a, b) in enumerate(((l_seq[0], l_bat[0:1]), (l_seq[1], l_bat[1:2]))):
        diff = (a - b).abs().max().item()
        assert diff < 0.5, f"T-GDN-02 bloc {i} : contamination inter-lignes (diff {diff})"
        assert torch.allclose(a.float(), b.float(), atol=0.25, rtol=0.05), (
            f"T-GDN-02 bloc {i} : batched ≠ séquentiel (diff {diff})")
    del model
    torch.cuda.empty_cache()


def test_gdn_04_packing_reset():
    tok, model = _load()
    runner = TwoStreamRunner(model)
    a = _prompt(tok, "Hello")
    b = _prompt(tok, "Hi there friend")
    packed = torch.cat([a, b], dim=1)
    doc = torch.cat([torch.zeros_like(a), torch.ones_like(b)], dim=1)
    with torch.no_grad():
        h_pack, _ = runner.prefill(packed, doc_ids=doc)
        h_a, _ = runner.prefill(a)
        h_b, _ = runner.prefill(b)
    ref = torch.cat([h_a, h_b], dim=1)
    # CORRECTION (tolérance justifiée) : `torch.equal` (bit-exact) était
    # trop strict en bf16 — le SDPA packé (masque doc, K plus large) et
    # le SDPA séparé accumulent l'arrondi différemment (mesuré : max
    # ~0.15). Une RUPTURE d'isolation/reset (vraie fuite inter-docs)
    # donnerait un écart O(1) — borné ci-dessous.
    diff = (h_pack - ref).abs().max().item()
    assert diff < 1.0, (
        f"T-GDN-04 : isolation/reset inter-documents en défaut (diff {diff})")
    assert torch.allclose(h_pack.float(), ref.float(), atol=0.5, rtol=0.05), (
        f"T-GDN-04 : packing ≠ séparés (diff {diff})")
    del model
    torch.cuda.empty_cache()


def test_denoise_reads_end_state():
    """B4 : la prédiction de la position 0 d'un bloc dépend des positions
    1 à 3 (lecture S̃_end, pas lecture causale).

    Si le décodage était causal intra-bloc, modifier les positions 1-3
    ne changerait PAS la prédiction de la position 0. Avec la sémantique
    B0 (lecture depuis S̃_end), elle CHANGE.

    CORRECTION (discriminabilité) : l'ancienne version utilisait
    `(MASK+7)%248320` etc. comme jetons de substitution — mais dans
    Qwen3.5-0.8B, les ids 248077…248319 partagent la MÊME ligne
    d'embedding (vérifié : embedding identiques) — le test mesurait donc
    0.0 par construction, indépendamment de l'implémentation. On utilise
    des jetons valides non dégénérés (100/200/300).
    """
    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    with torch.no_grad():
        _, state = runner.prefill(ids)
        # Bloc 4 jetons : [MASK, MASK, MASK, MASK].
        b_full = torch.full((1, 4), MASK_ID, device="cuda")
        logits_full = runner.denoise_block(state, b_full)

        # Modifier les positions 1-3 (mais pas la position 0) avec des
        # jetons dont l'embedding DIFFÈRE (ids valides non dégénérés).
        b_modified = b_full.clone()
        b_modified[0, 1] = 100
        b_modified[0, 2] = 200
        b_modified[0, 3] = 300
        logits_modified = runner.denoise_block(state, b_modified)

    # Avec lecture S̃_end : la position 0 dépend des positions 1-3.
    diff_0 = (logits_full[0, 0] - logits_modified[0, 0]).abs().max().item()
    assert diff_0 > 1e-2, (
        f"position 0 indépendante de 1-3 : lecture causale (pas S̃_end) ? "
        f"diff={diff_0}")
    del model
    torch.cuda.empty_cache()


def test_denoise_no_write():
    """B4 : denoise ne modifie JAMAIS l'état récurrent.

    Empreinte de l'état du runner avant et après denoise doit être
    identique. Vérifié par fingerprint SHA-256 (bit-exact).

    CORRECTION (API) : l'ancienne version appelait `cache_fingerprint`
    (conçu pour le cache HF `DynamicCache`), qui lève AttributeError
    sur `RunnerState`. On utilise `runner_state_fingerprint` (ajouté
    dans `twostream.py`, même sémantique bit-exacte).
    """
    from jainius.model.twostream import runner_state_fingerprint

    tok, model = _load()
    runner = TwoStreamRunner(model)
    ids = _prompt(tok)
    with torch.no_grad():
        _, state = runner.prefill(ids)
        fp_before = runner_state_fingerprint(state)
        b = torch.full((1, 4), MASK_ID, device="cuda")
        # denoise doit utiliser une copie de l'état (pas d'écriture).
        _ = runner.denoise_block(state, b)
        fp_after = runner_state_fingerprint(state)
        assert fp_before == fp_after, "denoise a écrit l'état récurrent"
    # La garde `StateLeakError` doit être présente (défensive) :
    # on ne peut pas la déclencher sans muter l'état, mais on vérifie
    # qu'elle est importable et que denoise_block la référence.
    import inspect
    assert "StateLeakError" in inspect.getsource(
        TwoStreamRunner.denoise_block), (
        "la garde StateLeakError est absente de denoise_block")
    del model
    torch.cuda.empty_cache()
