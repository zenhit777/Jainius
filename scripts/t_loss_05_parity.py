"""T-LOSS-05 — parité perte officielle OPDLM (oracle exécuté, pas porté).

Exécute VERBATIM le code officiel (`forward_process`, `process_pad`,
`make_basic_block_attention` extraits de `train/rl_sdar.py` — objets code
officiels, aucune réécriture) avec :
- student : SDAR-from-stock Qwen3-0.6B (poids stock exacts, `missing ==
  unexpected == []`) ;
- teacher : Qwen3-0.6B AR stock (déviation documentée du 4B officiel :
  la parité testée est le CALCUL, pas les poids enseignant) ;
- teacher_fill=clean_fill, student_arm_shift=False, loss_type=kl.

Convention prompt (E0, option (a) pré-approuvée) : run PRINCIPAL avec
masque officiel à région prompt rendue causale (E0-a) vs Jainius
(`causal_x0=True`) ; run natif officiel conservé pour caractériser le
delta E0 (attendu ~9.5, mesuré avant à ~9.5).

Critère (prédéfini, aucun seuil existant — T-LOSS-05 jamais exécuté) :
|Δ| logits étudiants ≤ 1e-3 (abs) ET |Δ| perte ≤ 1e-3 (bf16, mêmes
noyaux FA des deux côtés — un écart de câblage donne O(1)).

Environnement : venv isolé opdlm-parity (torch cu128 sm_120 + FA 2.8.3
build sm_120a + transformers 4.52.4). Shims infra-only à l'import
(wandb/deepspeed, jamais appelés dans le chemin numérique — vérifié).
Usage :
  PYTHONPATH=src <venv>/bin/python scripts/t_loss_05_parity.py [--out JSON]
"""

import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPDLM = REPO / "_artefacts" / "oracles" / "OPDLM"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(OPDLM / "train"))
sys.path.insert(0, str(OPDLM))

_CRT = "T-LOSS-05"
ABS_TOL = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import importlib.machinery as _mach
    for _n in ("wandb", "deepspeed"):
        _m = types.ModuleType(_n)
        _m.__spec__ = _mach.ModuleSpec(_n, loader=None)
        _m.__version__ = "0.0-stub"
        sys.modules[_n] = _m

    import torch
    import torch.nn.functional as F

    import rl_sdar  # code officiel (fonctions extraites ci-dessous)

    consts = {c.co_name: c for c in rl_sdar.main.__code__.co_consts
              if hasattr(c, "co_name")}
    mk_mask = consts["make_basic_block_attention"]
    mk_pad = consts["process_pad"]
    mk_fwd = consts["forward_process"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.sdar import SDARForCausalLM, SDARConfig

    DTYPE = torch.bfloat16
    DEV = "cuda"
    torch.manual_seed(0)

    stock = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=False,
        torch_dtype=DTYPE, device_map="cpu")
    sc = stock.config
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",
                                        trust_remote_code=False)
    MASK = 151669
    # Préparation harnais (miroir du setup tokenizer OPDLM) : jeton de
    # masque explicite 151669 (le tokenizer stock Qwen3 n'en définit aucun
    # et sa propriété est non assignable). Proxy d'ENTRÉE uniquement :
    # le code officiel sous test (masques, forwards, KL) est inchangé.
    class TokProxy:
        def __init__(self, base, mask_id):
            object.__setattr__(self, "_base", base)
            object.__setattr__(self, "mask_token_id", mask_id)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_base"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_base"), name, value)

    tok = TokProxy(tok, MASK)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 151669 + 1

    def build_sdar():
        cfg = SDARConfig(
            vocab_size=sc.vocab_size, hidden_size=sc.hidden_size,
            intermediate_size=sc.intermediate_size,
            num_hidden_layers=sc.num_hidden_layers,
            num_attention_heads=sc.num_attention_heads,
            num_key_value_heads=sc.num_key_value_heads,
            head_dim=getattr(sc, "head_dim", 128), hidden_act=sc.hidden_act,
            max_position_embeddings=sc.max_position_embeddings,
            rms_norm_eps=sc.rms_norm_eps, rope_theta=sc.rope_theta,
            rope_scaling=getattr(sc, "rope_scaling", None),
            sliding_window=getattr(sc, "sliding_window", 4096),
            tie_word_embeddings=False, fuse_cross_entropy=False,
            pad_token_id=sc.pad_token_id, bos_token_id=sc.bos_token_id,
            eos_token_id=sc.eos_token_id)
        m = SDARForCausalLM(cfg).to(DTYPE)
        missing, unexpected = m.load_state_dict(stock.state_dict(),
                                                strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return m.to(DEV).eval()

    student_off = build_sdar()
    student_ours = build_sdar()
    # Teacher stock en SDPA explicite des deux côtés : avec FA disponible,
    # l'appel 4D officiel et l'appel causal simple prendraient des chemins
    # noyaux différents (diagnostiqué) ; SDPA lève l'ambiguïté.
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=False,
        torch_dtype=DTYPE, device_map="cuda",
        attn_implementation="sdpa").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- entrées partagées (seed 0, sans pad/mask/im_end en réponse) ---
    L0, L1, BLOCK = 12, 8, 4
    prompt = torch.arange(100, 100 + L0, dtype=torch.long,
                          device=DEV).unsqueeze(0) % 150000
    gen = torch.Generator(device="cpu").manual_seed(0)
    avoid = {PAD, MASK, tok.convert_tokens_to_ids("<|im_end|>")}
    xresp = torch.randint(0, 150000, (1, L1), generator=gen)
    while any(int(x) in avoid for x in xresp[0]):
        xresp = torch.randint(0, 150000, (1, L1), generator=gen)
    xresp = xresp.to(DEV)
    xt_resp = xresp.clone()
    xt_resp[0, ::2] = MASK
    x0 = torch.cat([prompt, xresp], dim=1)
    xt = torch.cat([prompt, xt_resp], dim=1)
    # Officiel : extended = [prompt | x0_resp | xt_resp] (L0+2L1, PAS [x0|xt]).
    extended = torch.cat([prompt, xresp, xt_resp], dim=1)  # [B, L0+2L1]
    N = L0 + 2 * L1
    # tok_idx_ext : formules officielles rl_sdar.py:850-855 (sans pads).
    idx = torch.arange(L0 + L1, device=DEV).unsqueeze(0)
    valid = (idx >= L0) | x0.ne(PAD)
    tok_idx = valid.long().cumsum(dim=-1) - 1
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_ext = torch.cat([tok_idx, tok_idx[:, L0:]], dim=1)
    p_mask = torch.ones(1, L0 + L1, dtype=torch.bool, device=DEV)

    G = {"torch": torch, "F": F, "start_pos": L0, "pad_id": PAD}
    G["make_basic_block_attention"] = types.FunctionType(mk_mask, dict(G))
    def _cell(v):
        return (lambda: v).__closure__[0]

    def _bind(code, glb, mapping):
        cells = tuple(_cell(mapping[n]) for n in code.co_freevars)
        return types.FunctionType(code, glb, None, None, cells)

    _pad_free = {n: {"pad_id": PAD, "start_pos": L0, "L0": L0, "L1": L1}[n]
                 for n in mk_pad.co_freevars}
    G["process_pad"] = _bind(mk_pad, dict(G), _pad_free)
    base_mask = G["make_basic_block_attention"](N, L0, BLOCK).to(DEV)
    G["basic_block_attention"] = base_mask

    class Cfg:
        pass
    cfg_tr = Cfg()
    cfg_tr.training = Cfg()
    cfg_tr.training.block_size = BLOCK
    cfg_tr.training.teacher_fill = "clean_fill"
    cfg_tr.training.student_arm_shift = False
    cfg_tr.training.loss_type = "kl"
    cfg_tr.training.reverse_kl_weight = 0.0
    cfg_tr.training.exclude_im_end = False
    G["config"] = cfg_tr
    G["tokenizer"] = tok
    G["model"] = student_off

    recorded: dict = {}
    srec: dict = {}

    class StudWrap(torch.nn.Module):
        def __init__(self, m, tag):
            super().__init__()
            self.m = m
            self.tag = tag

        def forward(self, **kw):
            out = self.m(**kw)
            srec[self.tag] = out.logits.detach().clone()
            srec[self.tag + "_kw"] = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in kw.items()}
            return out

    class TeacherWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, **kw):
            out = self.m(**kw)
            recorded["teacher_in"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                      for k, v in kw.items()}
            recorded["teacher_logits"] = out.logits.detach().clone()
            return out

    G["teacher_model"] = TeacherWrap(teacher)
    _pool = dict(G)
    _pool.update({"start_pos": L0, "pad_id": PAD, "F": F,
                  "basic_block_attention": base_mask, "L0": L0, "L1": L1})
    G["model"] = StudWrap(student_off, "native")
    _pool["model"] = G["model"]
    fwd = _bind(mk_fwd, dict(G),
                {n: _pool[n] for n in mk_fwd.co_freevars})

    labels = extended[:, :L0 + L1].clone()
    adv = torch.zeros(1, L0 + L1, device=DEV)
    logp_old = torch.zeros(1, L0 + L1, device=DEV)
    print("mask_token_id =", tok.mask_token_id, "pad =", PAD,
          "extended", tuple(extended.shape), flush=True)
    with torch.no_grad():
        loss_nat, _, _ = fwd(extended, p_mask, tok_idx_ext, labels, adv,
                             logp_old)
    teach_nat = recorded["teacher_logits"]

    # Variante causale E0-a : région prompt du masque -> tril causal.
    causal_mask = base_mask.clone()
    tril = torch.tril(torch.ones(L0, L0, dtype=torch.bool, device=DEV))
    causal_mask[:, :, :L0, :L0] = tril
    G["basic_block_attention"] = causal_mask
    G["model"] = StudWrap(student_off, "causal")
    _pool_c = dict(_pool)
    _pool_c["basic_block_attention"] = causal_mask
    _pool_c["model"] = G["model"]
    fwd_c = _bind(mk_fwd, dict(G),
                  {n: _pool_c[n] for n in mk_fwd.co_freevars})
    with torch.no_grad():
        loss_causal, _, _ = fwd_c(extended, p_mask, tok_idx_ext, labels, adv,
                                  logp_old)
    teach_causal = recorded["teacher_logits"]

    # --- côté Jainius : câblage nôtre (masque bool + positions + split),
    # exécuté sur le même modèle SDAR. Le masque ADDITIF de
    # `forward_two_stream` n'est pas utilisé ici : SDAR attend un masque
    # BOOL (`attention_mask.bool()` en interne — l'additif 0/-inf serait
    # inversé) ; la conversion additif->SDPA est couverte par T-MASK-01 et
    # la sonde 4D sur modèles HF standard.
    from jainius.model.masks import build_two_stream_mask
    L = L0 + L1
    allow = build_two_stream_mask(
        L, BLOCK, causal_x0=True, causal_xt=False).to(DEV)  # [2L,2L] bool
    # Égalité de contenu avec la variante causale officielle (E0-a),
    # modulo le layout ([x0|xt] vs [prompt|resp|xtresp]).
    off_causal_bool = (causal_mask[0, 0] if causal_mask.dim() == 4
                       else causal_mask).bool()  # [N,N] officiel E0-a
    # Régions comparables (layouts différents : [x0|xt] vs [p|r|xtr]).
    # x0 (L lignes/cols) <-> officiel [0,L0+L1).
    x0_eq = bool((allow[:L, :L] == off_causal_bool[:L, :L]).all())
    _dx0 = (allow[:L, :L] != off_causal_bool[:L, :L])
    _idx = _dx0.nonzero()
    rep = {}
    rep["mask_x0_content_equal"] = x0_eq
    rep["mask_x0_ndiff"] = int(_dx0.sum())
    rep["mask_x0_first_diff"] = _idx[0].tolist() if len(_idx) else None
    # Diagnostic d'isolation (E0) : masque hybride = notre x0 CAUSAL pour le
    # prompt + BLOC-BIDI officiel pour la réponse. Si la perte s'aligne
    # alors, le câblage (slice/roll/norm/KL) est exact et le résidu est
    # purement architectural (D-10 causal_x0 vs bloc-bidi OPDLM).
    base_nat = G["make_basic_block_attention"](N, L0, BLOCK).to(DEV).bool()
    base_nat = base_nat[0, 0] if base_nat.dim() == 4 else base_nat
    hybrid = allow.clone()
    # x0 ENTIEREMENT native officielle (prompt + reponse) : le diagnostic
    # isole le cablage restant (layout, positions par flux, split, KL).
    hybrid[:L, :L] = base_nat[:L, :L]
    orows = list(range(L0 + L1, N))
    nrows = list(range(L + L0, 2 * L))
    ocols = list(range(L0 + L1)) + list(range(L0 + L1, N))
    ncols = list(range(L)) + list(range(L + L0, 2 * L))
    xt_eq = bool((allow[nrows][:, ncols] == off_causal_bool[orows][:, ocols]).all())
    rep["mask_xt_resp_content_equal"] = xt_eq
    pos = torch.arange(L, device=DEV).unsqueeze(0)
    pos_ext = torch.cat([pos, pos], dim=1)
    concat = torch.cat([x0, xt], dim=1)
    add4 = allow.unsqueeze(0).unsqueeze(0)  # [B,1,2L,2L] bool
    sup = torch.zeros_like(x0, dtype=torch.bool)
    sup[0, L0:] = (xt_resp[0] == MASK)
    with torch.no_grad():
        full_ours = student_ours(input_ids=concat, attention_mask=add4,
                                 position_ids=pos_ext,
                                 output_hidden_states=True)
        h_all = full_ours.hidden_states[-1]
        h_xt_ours = h_all[:, L:]
    from jainius.distill.loss import kl_teacher_to_student
    from jainius.distill.teacher import Teacher
    with torch.no_grad():
        # Même conditions noyaux que l'officiel : masque 4D explicite
        # (la voie implicite diverge de 0.5 sous transformers 4.52/SDPA —
        # diagnostiqué ; la construction reste roll(1)+log_softmax).
        add_causal = torch.zeros(1, 1, L, L, dtype=torch.bfloat16, device=DEV)
        add_causal[0, 0][~torch.tril(torch.ones(L, L, dtype=torch.bool, device=DEV))] = float("-inf")
        tlog = teacher(input_ids=x0, attention_mask=add_causal,
                       position_ids=torch.arange(L, device=DEV).unsqueeze(0)).logits
        teach_simple = torch.nn.functional.log_softmax(
            tlog.roll(dims=1, shifts=1).float(), dim=-1)
        assert teach_simple.shape == (1, L, tlog.shape[-1])
    V = teach_simple.shape[-1]
    out = kl_teacher_to_student(
        h_xt_ours, lambda hh: student_ours.lm_head(hh), None, None, sup,
        chunk_tokens=1024,
        teacher_logits=teach_simple.reshape(-1, V))
    loss_ours = out.loss.item()

    # Parité student-logits : recalcule officiel sliced via wrapper student.
    G["model"] = StudWrap(student_off, "causal2")
    _pool_c["model"] = G["model"]
    fwd_c2 = _bind(mk_fwd, dict(G),
                   {n: _pool_c[n] for n in mk_fwd.co_freevars})
    with torch.no_grad():
        fwd_c2(extended, p_mask, tok_idx_ext, labels, adv, logp_old)
    full = srec["causal2"]
    logits_off = torch.cat([full[:, :L0, :], full[:, L0 + L1:, :]], dim=1)
    # Comparaison sur les logits étudiants (même tête SDAR des deux côtés).
    rep.update({
        "loss_official_native": float(loss_nat.detach()),
        "loss_official_causal": float(loss_causal.detach()),
        "loss_ours": float(loss_ours),
        "d_loss_causal": abs(float(loss_causal.detach()) - float(loss_ours)),
        "d_loss_native": abs(float(loss_nat.detach()) - float(loss_ours)),
        "teacher_nat_vs_causal": float((teach_nat.float() - teach_causal.float()).abs().max()),
        "criterion_abs": ABS_TOL,
    })
    # Teacher : compare notre simple vs officiel causal (log-probs).
    # Slice officielle rl_sdar (cat) PUIS roll(1) comme l'officiel.
    with torch.no_grad():
        tc = teach_causal.float()
        tc_sliced = torch.cat([tc[:, :L0, :], tc[:, L0 + L1:, :]], dim=1)
        tc_rolled = tc_sliced.roll(dims=1, shifts=1)
        off_lp = torch.nn.functional.log_softmax(tc_rolled, dim=-1)
    # Isolation E0 : même teacher officiel + masque hybride côté nôtre.
    with torch.no_grad():
        addH = hybrid.unsqueeze(0).unsqueeze(0)
        full_h = student_ours(input_ids=concat, attention_mask=addH,
                              position_ids=pos_ext,
                              output_hidden_states=True).hidden_states[-1]
        h_xt_h = full_h[:, L:]
    out_h = kl_teacher_to_student(
        h_xt_h, lambda hh: student_ours.lm_head(hh), None, None, sup,
        chunk_tokens=1024, teacher_logits=off_lp.reshape(-1, V))
    rep["loss_ours_hybrid"] = float(out_h.loss.item())
    rep["d_loss_hybrid"] = abs(float(loss_nat.detach()) - float(out_h.loss.item()))
    # Diagnostic fp32 : mêmes poids (cast exact bf16->fp32), masque hybride,
    # teacher officiel — sépare bruit de tiling bf16 et sémantique.
    with torch.no_grad():
        so32, sn32 = student_off.float(), student_ours.float()
        print("dtypes:", sn32.lm_head.weight.dtype,
              student_off.lm_head.weight.dtype, full32_dtype if False else "",
              flush=True)
        full32 = sn32(input_ids=concat, attention_mask=addH,
                      position_ids=pos_ext,
                      output_hidden_states=True).hidden_states[-1][:, L:]
        off32 = torch.nn.functional.log_softmax(
            torch.cat([teach_nat.float()[:, :L0, :],
                       teach_nat.float()[:, L0 + L1:, :]],
                      dim=1).roll(dims=1, shifts=1), dim=-1)
        out32 = kl_teacher_to_student(
            full32, lambda hh: sn32.lm_head(hh.to(sn32.lm_head.weight.dtype)),
            None, None, sup, chunk_tokens=1024,
            teacher_logits=off32.reshape(-1, V))
        # Référence officielle recalculée en fp32 sur les mêmes entrées ?
        # (l'officiel ne tourne qu'en bf16 ici : on compare la perte
        # hybride fp32 à la perte native bf16 à titre indicatif seulement.)
        rep["loss_ours_hybrid_fp32"] = float(out32.loss.item())
        # Écart student-hybrid bf16 vs fp32 (bruit de précision du layout).
        rep["d_hybrid_bf16_fp32"] = abs(float(out_h.loss.item()) - float(out32.loss.item()))
    # NOTE : l'officiel ne peut pas tourner en fp32 (`.bfloat16()` en dur
    # dans rl_sdar — le modifier invaliderait l'oracle). Pas de parité
    # fp32-vs-fp32 ; le diagnostic ci-dessus borne le bruit de précision.
    # Diagnostic layout : même entrée/masque/positions officiels via NOTRE
    # modèle — masque BOOL (l'officiel reçoit du bool via process_pad ;
    # l'int64 prend une autre branche noyau côté SDAR).
    base_bool = base_mask.bool()
    with torch.no_grad():
        rep_off = student_ours(input_ids=extended,
                               attention_mask=base_bool,
                               position_ids=tok_idx_ext).logits
    rep["d_replay_official_layout"] = float(
        (rep_off.float() - srec["native"].float()).abs().max())
    with torch.no_grad():
        rep_off2 = student_ours(input_ids=extended,
                                attention_mask=base_mask,
                                position_ids=tok_idx_ext).logits
        rep_off3 = student_off(input_ids=extended,
                               attention_mask=base_mask,
                               position_ids=tok_idx_ext).logits
    rep["d_ours_deterministic"] = float(
        (rep_off.float() - rep_off2.float()).abs().max())
    rep["d_off_vs_ours_same_inputs"] = float(
        (rep_off3.float() - srec["native"].float()).abs().max())
    rep["d_instances_same_inputs"] = float(
        (rep_off.float() - rep_off3.float()).abs().max())
    proc = G["process_pad"](base_mask.clone().repeat_interleave(1, dim=0).to(DEV),
                            extended)
    rep["process_pad_identity"] = bool((proc == base_mask.bool()).all())
    with torch.no_grad():
        rep_proc = student_ours(input_ids=extended, attention_mask=proc,
                                position_ids=tok_idx_ext).logits
    rep["d_proc_vs_native"] = float(
        (rep_proc.float() - srec["native"].float()).abs().max())
    with torch.no_grad():
        kw = srec["native_kw"]
        rep_exact = student_ours(**kw).logits
    rep["d_exact_kw_replay"] = float(
        (rep_exact.float() - srec["native"].float()).abs().max())
    rep["kw_dtypes"] = {k: str(v.dtype) if torch.is_tensor(v) else type(v).__name__
                        for k, v in kw.items()}
    rep["kw_shapes"] = {k: list(v.shape) if torch.is_tensor(v) else v
                        for k, v in kw.items() if k != "attention_mask"}
    _am = kw["attention_mask"]
    rep["kw_mask_unique"] = _am.unique().tolist()
    _rec = srec["native_kw"]["attention_mask"]
    rep["kw_mask_shape"] = list(_rec.shape)
    rep["mask_recorded_vs_base"] = bool((_rec == base_bool).all())
    rep["mask_recorded_ndiff"] = int((_rec != base_bool).sum())
    _dd = (_rec[0, 0] != base_bool[0, 0]).nonzero()
    rep["mask_recorded_first_diffs"] = _dd[:10].tolist()
    _proc_now = G["process_pad"](base_mask.clone(), extended)
    rep["mask_proc_now_vs_recorded"] = bool((_proc_now == _rec).all())
    rep["teacher_simple_vs_official"] = float((teach_simple.float() - off_lp).abs().max())
    _to = teach_simple.float()[0, 0].topk(5)
    _oo = off_lp[0, 0].topk(5)
    print("teach_simple row0 top5:", _to.indices.tolist(),
          [round(float(x), 3) for x in _to.values], flush=True)
    print("official row0 top5:", _oo.indices.tolist(),
          [round(float(x), 3) for x in _oo.values], flush=True)
    rep["teacher_per_pos"] = (teach_simple.float() - off_lp).abs().max(-1).values[0].tolist()
    # Diagnostic : rejoue le teacher stock sur l'entrée échantillonnée
    # officielle (masque+positions enregistrés) -> doit retrouver l'enregistré.
    tin = recorded["teacher_in"]
    print("attn_impl:", teacher.config._attn_implementation, flush=True)
    with torch.no_grad():
        mini = teacher(input_ids=tin["input_ids"][:, :20],
                       attention_mask=tin["attention_mask"][:, :, :20, :20],
                       position_ids=tin["position_ids"][:, :20]).logits
    print("mini20 vs recorded-sliced:",
          float((mini.float() - teach_causal.float()[:, :20, :]).abs().max()),
          "mini20 vs plain:",
          float((mini.float() - teacher(x0).logits.float()).abs().max()),
          flush=True)
    print("teach_in shapes:",
          {k: tuple(v.shape) if torch.is_tensor(v) else v
           for k, v in tin.items()}, "mask dtype:",
          tin["attention_mask"].dtype, flush=True)
    print("teach mask row0:", tin["attention_mask"][0, 0, 0, :].tolist(),
          flush=True)
    print("teach mask row12:", tin["attention_mask"][0, 0, 12, :].tolist(),
          flush=True)
    print("teach mask row20:", tin["attention_mask"][0, 0, 20, :].tolist(),
          flush=True)
    with torch.no_grad():
        replay = teacher(input_ids=tin["input_ids"],
                         attention_mask=tin["attention_mask"],
                         position_ids=tin["position_ids"]).logits
    rep["teacher_replay_vs_recorded"] = float((replay.float() - teach_causal.float()).abs().max())
    with torch.no_grad():
        plain = teacher(x0).logits.float()
    # Slice officielle du replay, pour comparaison à notre simple.
    rp = torch.cat([replay[:, :L0, :], replay[:, L0 + L1:, :]], dim=1).float()
    rep["teacher_replay_sliced_vs_simple"] = float((rp - teach_simple.float()).abs().max())
    # Positions : officielles (tok_idx_ext) vs nôtres ([pos|pos]).
    L = L0 + L1
    ours_pos = torch.cat([torch.arange(L, device=DEV),
                          torch.arange(L, device=DEV)]).tolist()
    rep["positions_official"] = tok_idx_ext[0].tolist()
    rep["positions_ours"] = ours_pos
    # Student logits officiels vs nôtres (même tête SDAR).
    with torch.no_grad():
        l_ours = student_ours.lm_head(h_xt_ours).float()
        l_off_full = logits_off.float()
        # Slice supervisée : réponse uniquement.
        d_stud = (l_ours[0, L0:] - l_off_full[0, L0:]).abs().max().item()
    rep["student_per_pos"] = (l_ours[0] - l_off_full[0]).abs().max(-1).values.tolist()
    rep["d_student_logits_resp"] = d_stud
    ok = (rep["d_loss_causal"] <= ABS_TOL and d_stud <= 1e-2)
    rep["verdict"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
    if rep["d_loss_causal"] > ABS_TOL:
        raise SystemExit(f"T-LOSS-05 : |Δperte|={rep['d_loss_causal']:.3e} > {ABS_TOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
