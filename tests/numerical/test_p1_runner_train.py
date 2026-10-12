"""P1/P2/P3 — tests modèle réel (CUDA + poids Qwen3.5-0.8B).

- `test_p1_train_step_uses_runner_and_grads` (P1.1/1.2) : `train_step`
  avec `TwoStreamRunner` — perte finie, gradient jusqu'aux paramètres
  étudiants, refus legacy sans runner.
- `test_p1_runner_no_leak_loss_invariant` (P1.4/1.5) : perturbation du
  futur x0 -> invariance bit-exacte des hidden xt ET de la perte KL sur
  positions supervisées ; dépendances légitimes non nulles (discriminant).
  Perte déterministe, même état initial, tokens distincts non dégénérés.
- `test_p2_forward_train_all_gdn_layers_receive_grads` (P2 e2e) : vraie
  perte KL + backward — les 18 couches GDN reçoivent des gradients finis
  non nuls (fp32).
- `test_p3_quadratic_vs_optimized_equivalence` (P3) : `forward_train`
  optimisé == variante quadratique (tolérance bf16 documentée).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace

from jainius.distill.loss import kl_teacher_to_student
from jainius.distill.targets import compute_teacher_targets
from jainius.distill.teacher import Teacher
from jainius.model.arch import DiffusionSpec, JainiusModel, LegacyLeakyError
from jainius.model.twostream import TwoStreamRunner
from jainius.train.loop import train_step

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MASK_ID = 248077


def _load(dtype):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    s = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="cuda",
        trust_remote_code=False).eval()
    t = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="cuda",
        trust_remote_code=False).eval()
    return tok, s, t


def _prompt_ids(tok, text="Hi there"):
    s = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tok(s, return_tensors="pt")["input_ids"][0].tolist()


def test_p1_train_step_uses_runner_and_grads():
    torch.manual_seed(0)
    tok, sm, tm = _load(torch.bfloat16)
    try:
        student = JainiusModel(sm, DiffusionSpec())
        teacher = Teacher(tm)
        runner = TwoStreamRunner(sm)
        pids = _prompt_ids(tok)[:16]
        traj = SimpleNamespace(x0=[100, 200, 300, 400])
        optim = torch.optim.SGD(sm.parameters(), lr=1e-3)
        # Sans runner : refus explicite.
        try:
            train_step(student, teacher, optim, [pids], [traj])
            raise AssertionError("legacy accepté en entraînement (P1 violé)")
        except LegacyLeakyError:
            pass
        stats: dict = {}
        loss = train_step(student, teacher, optim, [pids], [traj],
                          stats=stats, runner=runner)
        assert loss == loss and abs(loss) != float("inf")
        assert "grad_norm" in stats and stats["grad_norm"] == stats["grad_norm"]
        grads = [p.grad for p in sm.parameters() if p.grad is not None]
        assert grads, "aucun gradient étudiant (graphe coupé)"
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        assert any(bool((g != 0).any()) for g in grads), "gradients tous nuls"
    finally:
        del sm, tm
        torch.cuda.empty_cache()


def test_p1_runner_no_leak_loss_invariant():
    torch.manual_seed(0)
    tok, sm, tm = _load(torch.bfloat16)
    try:
        runner = TwoStreamRunner(sm)
        pids = _prompt_ids(tok)[:16]
        l0 = len(pids)
        # Tokens distincts non dégénérés (100/200/300 : embeddings distincts,
        # hors plage 248077..248319 à embedding partagé).
        x0 = torch.tensor([pids + [100] * 8], device="cuda")
        xt = torch.tensor([pids + [MASK_ID] * 8], device="cuda")

        def loss_of(a, b):
            V = tm.lm_head.weight.shape[0]
            with torch.no_grad():
                _, h_xt = runner.forward_train(a, b, unit_block=4,
                                               resp_start=l0)
                hn = runner.final_norm(h_xt)
                tlog = compute_teacher_targets(
                    a,
                    lambda ids: tm(ids,
                                   output_hidden_states=True).hidden_states[-1],
                    tm.lm_head).reshape(-1, V)
            sup = torch.zeros_like(a, dtype=torch.bool)
            sup[0, l0:] = True
            out = kl_teacher_to_student(hn, lambda h: sm.lm_head(h),
                                        None, None, sup, chunk_tokens=1024,
                                        teacher_logits=tlog)
            return h_xt, out.loss

        h_ref, loss_ref = loss_of(x0, xt)
        # Futur x0 perturbé (dernier token 100 -> 101, réellement distinct).
        x0b = x0.clone()
        x0b[0, -1] = 101
        h_fut, loss_fut = loss_of(x0b, xt)
        assert torch.equal(h_ref[:, l0:], h_fut[:, l0:]), "fuite futur-x0 (P1)"
        assert float((loss_ref.detach() - loss_fut.detach()).abs()) == 0.0, (
            f"perte non invariante : {loss_ref} vs {loss_fut}")
        # Préfixe xt perturbé (prompt 0 -> autre id valide).
        xtb = xt.clone()
        xtb[0, 0] = (int(xtb[0, 0]) + 7) % 248320
        h_pre, _ = loss_of(x0, xtb)
        assert torch.equal(h_ref[:, l0:], h_pre[:, l0:]), "fuite préfixe-xt"
        # Dépendances légitimes NON nulles (discriminant).
        x0c = x0.clone()
        x0c[0, 0] = (int(x0c[0, 0]) + 7) % 248320
        h_leg1, _ = loss_of(x0c, xt)
        assert (h_ref[:, l0:] - h_leg1[:, l0:]).abs().max() > 0
        xtd = xt.clone()
        xtd[0, l0] = 200  # token du bloc actif modifié
        h_leg2, _ = loss_of(x0, xtd)
        assert (h_ref[:, l0:] - h_leg2[:, l0:]).abs().max() > 0
        x0f = x0.clone()
        x0f[0, l0] = 200  # propre du bloc précédent modifié
        h_leg3, _ = loss_of(x0f, xt)
        assert (h_ref[:, l0:] - h_leg3[:, l0:]).abs().max() > 0
    finally:
        del sm, tm
        torch.cuda.empty_cache()


def test_p2_forward_train_all_gdn_layers_receive_grads():
    torch.manual_seed(0)
    tok, sm, tm = _load(torch.float32)
    try:
        sm.train()
        runner = TwoStreamRunner(sm)
        pids = _prompt_ids(tok)[:12]
        l0 = len(pids)
        x0 = torch.tensor([pids + [100] * 8], device="cuda")
        xt = torch.tensor([pids + [MASK_ID] * 8], device="cuda")
        sm.zero_grad()
        _, h_xt_raw = runner.forward_train(x0, xt, unit_block=4,
                                           resp_start=l0)
        h_xt = runner.final_norm(h_xt_raw)
        V = tm.lm_head.weight.shape[0]
        with torch.no_grad():
            tlog = compute_teacher_targets(
                x0,
                lambda ids: tm(ids,
                               output_hidden_states=True).hidden_states[-1],
                tm.lm_head).reshape(-1, V)
        sup = torch.zeros_like(x0, dtype=torch.bool)
        sup[0, l0:] = True
        out = kl_teacher_to_student(h_xt, lambda h: sm.lm_head(h),
                                    None, None, sup, chunk_tokens=1024,
                                    teacher_logits=tlog)
        out.loss.backward()
        cfg = sm.config.get_text_config()
        layer_types = list(cfg.layer_types)
        gdn_idx = [i for i, t in enumerate(layer_types)
                   if t == "linear_attention"]
        assert len(gdn_idx) == 18, f"attendu 18 GDN, vu {len(gdn_idx)}"
        missing, nonfinite, allzero = [], [], []
        for i in gdn_idx:
            layer = sm.model.layers[i].linear_attn
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b",
                         "in_proj_a", "conv1d", "out_proj", "norm"):
                mod = getattr(layer, name, None)
                w = mod.weight if mod is not None else None
                if w is None or w.grad is None:
                    missing.append(f"L{i}.{name}")
                elif not bool(torch.isfinite(w.grad).all()):
                    nonfinite.append(f"L{i}.{name}")
                elif not bool((w.grad != 0).any()):
                    allzero.append(f"L{i}.{name}")
            for name in ("A_log", "dt_bias"):
                w = getattr(layer, name, None)
                if w is None or w.grad is None:
                    missing.append(f"L{i}.{name}")
                elif not bool(torch.isfinite(w.grad).all()):
                    nonfinite.append(f"L{i}.{name}")
                elif not bool((w.grad != 0).any()):
                    allzero.append(f"L{i}.{name}")
        assert not missing, f"sans gradient : {missing[:10]}"
        assert not nonfinite, f"gradient non fini : {nonfinite[:10]}"
        assert not allzero, f"gradient nul : {allzero[:10]}"
    finally:
        del sm, tm
        torch.cuda.empty_cache()


def test_p3_quadratic_vs_optimized_equivalence():
    torch.manual_seed(0)
    tok, sm, _ = _load(torch.bfloat16)
    try:
        runner = TwoStreamRunner(sm)
        pids = _prompt_ids(tok)[:16]
        l0 = len(pids)
        x0 = torch.tensor([pids + [100] * 8], device="cuda")
        xt = torch.tensor([pids + [MASK_ID] * 8], device="cuda")
        with torch.no_grad():
            h0_new, hn_new = runner.forward_train(x0, xt, unit_block=4,
                                                  resp_start=l0)
            h0_old, hn_old = runner._forward_train_quadratic(
                x0, xt, unit_block=4, resp_start=l0)
        for tag, a, b in (("h_x0", h0_new, h0_old),
                          ("h_xt", hn_new, hn_old)):
            diff = (a.float() - b.float()).abs().max().item()
            assert diff < 0.5, f"P3 {tag} : divergence {diff} (borne fuite O(1))"
            assert torch.allclose(a.float(), b.float(), atol=0.5,
                                  rtol=0.05), f"P3 {tag} : {diff}"
    finally:
        del sm
        torch.cuda.empty_cache()
