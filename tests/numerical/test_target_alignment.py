"""Alignement cibles enseignant/étudiant, convention `current` (pré-M2).

Dossier formel (OPDLM `rl_sdar.py`, coordonnées slicées (B, L0+L1)) :
- étudiant : slice `[.., :L0] + [.., L0+L1:]`, SANS roll
  (`student_arm_shift: False`, `rl_bd3lm.yaml:108`, code `:1269`) ;
- enseignant : même slice, AVEC `roll(1)` (`:1344-1348`) ;
- supervisées : réponse ∧ pad ∧ pré-EOS ∧ masquée-en-xt (`:1425-1428`) ;
- normalisation : moyenne par ligne puis batch (`:1428`, `:1449`).
Le décalage requis (roll enseignant, pas de roll étudiant) est présent
dans `train_step` : l'omission est INFIRMÉE. Différence assumée et
documentée (pas une omission) : le contexte prompt/ligne de l'étudiant
est causal (`causal_x0=True` exigé : D-10, AR-Trust, cohérence FLARE),
tandis qu'OPDLM utilise un prompt bloc-bidirectionnel — test (3) le
quantifie. Les positions xt répétées (§10.2, `tok_idx_resp`) sont
verrouillées par le test (1).

Modèle : Qwen3-0.6B auxiliaire fp32 (famille Qwen3, attention pure comme
le code OPDLM). Déterministe (seed 0).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.distill.loss import kl_teacher_to_student
from jainius.distill.teacher import Teacher, basic_block_attention
from jainius.model.arch import DiffusionSpec, JainiusModel, TwoStreamBatch
from jainius.model.masks import build_two_stream_mask

AUX_ID = "Qwen/Qwen3-0.6B"
MASK_ID = 151669
L1 = 8
BLOCK = 4


def _setup():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(AUX_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        AUX_ID, dtype=torch.float32, device_map="cuda", trust_remote_code=False
    ).eval()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Hello world! How are you today?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = tok(text, return_tensors="pt")["input_ids"].to("cuda")
    l0 = prompt.shape[1]
    gen = torch.Generator(device="cuda").manual_seed(0)
    xresp = torch.randint(0, 1000, (1, L1), generator=gen, device="cuda")
    xt_resp = xresp.clone()
    xt_resp[0, ::2] = MASK_ID
    return model, prompt, l0, xresp, xt_resp


def _manual_two_stream(model, x0, xt, length):
    """Réimplémentation indépendante du câblage (masque testé par T-MASK,
    positions et split explicites ici)."""
    concat = torch.cat([x0, xt], dim=1)
    mask = build_two_stream_mask(length, BLOCK)
    additive = torch.zeros(1, 1, 2 * length, 2 * length, dtype=torch.float32,
                           device="cuda")
    additive[:, :, ~mask] = torch.finfo(torch.float32).min
    pos = torch.arange(length, device="cuda").unsqueeze(0)
    pos_ext = torch.cat([pos, pos], dim=1)
    with torch.no_grad():
        hidden = model(concat, attention_mask=additive, position_ids=pos_ext,
                       output_hidden_states=True).hidden_states[-1]
    return hidden[:, :length], hidden[:, length:]


def test_alignment_wiring_vs_manual():
    """Le câblage `forward_two_stream` (masque, positions répétées, split)
    est bit-exact vs réimplémentation indépendante. ÉCHOUE sur l'ancienne
    implémentation (positions continues implicites)."""
    model, prompt, l0, xresp, xt_resp = _setup()
    student = JainiusModel(model, DiffusionSpec())
    x0 = torch.cat([prompt, xresp], dim=1)
    xt = torch.cat([prompt, xt_resp], dim=1)
    sup = torch.zeros_like(x0, dtype=torch.bool)
    with torch.no_grad():
        got = student.forward_two_stream(TwoStreamBatch(x0=x0, xt=xt,
                                                         sup_mask=sup))
        ref_x0, ref_xt = _manual_two_stream(model, x0, xt, x0.shape[1])
    assert torch.equal(got.h_x0, ref_x0) and torch.equal(got.h_xt, ref_xt)
    del model
    torch.cuda.empty_cache()


def test_alignment_shift_sensitivity():
    """Sans roll étudiant + roll enseignant : toute translation ±1 du
    student dégrade strictement la KL (verrouille le (non-)décalage)."""
    model, prompt, l0, xresp, xt_resp = _setup()
    teacher = Teacher(model)
    student = JainiusModel(model, DiffusionSpec())
    x0 = torch.cat([prompt, xresp], dim=1)
    xt = torch.cat([prompt, xt_resp], dim=1)
    sup = torch.zeros_like(x0, dtype=torch.bool)
    sup[0, l0:] = (xt_resp[0] == MASK_ID)
    with torch.no_grad():
        h_s = student.forward_two_stream(
            TwoStreamBatch(x0=x0, xt=xt, sup_mask=sup)).h_xt
        h_t = teacher.hidden(x0)
    head = lambda h: model.lm_head(h)
    loss_ref = kl_teacher_to_student(h_s, head, h_t, head, sup).loss.item()
    loss_p1 = kl_teacher_to_student(h_s.roll(1, 1), head, h_t, head, sup).loss.item()
    loss_m1 = kl_teacher_to_student(h_s.roll(-1, 1), head, h_t, head, sup).loss.item()
    assert loss_p1 > loss_ref * 1.1 and loss_m1 > loss_ref * 1.1, (
        f"alignement suspect : ref={loss_ref:.3f} +1={loss_p1:.3f} -1={loss_m1:.3f}"
    )
    del model
    torch.cuda.empty_cache()


def test_prompt_context_delta_characterized():
    """Différence assumée (non-omission) : prompt causal (Jainius,
    `causal_x0=True` exigé) vs prompt bloc-bidirectionnel (OPDLM). Mesuré
    sur ce montage : KL moyenne ~9.5 par position supervisée — même ordre
    que la perte elle-même. Conséquence : la parité numérique exacte avec
    le forward étudiant OPDLM (T-LOSS-05/E0) est impossible sous
    `causal_x0=True` ; E0 devra en tenir compte (tolérance ou convention
    commune pour le run de parité)."""
    model, prompt, l0, xresp, xt_resp = _setup()
    base = basic_block_attention(l0, L1, BLOCK, device="cuda")
    causal = base.clone()
    n = l0 + 2 * L1
    tril = torch.tril(torch.ones(1, 1, n, n, dtype=torch.bool, device="cuda"))
    causal[:, :, :l0, :l0] &= tril[:, :, :l0, :l0]

    def run(mask):
        add = torch.full(mask.shape, torch.finfo(torch.float32).min,
                         dtype=torch.float32, device="cuda")
        add[mask] = 0.0
        pos = torch.arange(l0 + L1, device="cuda").unsqueeze(0)
        pose = torch.cat([pos, pos[:, l0:]], dim=1)
        ext = torch.cat([prompt, xresp, xt_resp], dim=1)
        with torch.no_grad():
            full = model(input_ids=ext, attention_mask=add,
                         position_ids=pose).logits
        return torch.cat([full[:, :l0, :], full[:, l0 + L1 :, :]], dim=1)

    sup = torch.zeros(1, l0 + L1, dtype=torch.bool, device="cuda")
    sup[0, l0:] = (xt_resp[0] == MASK_ID)
    r_full = run(base)
    r_causal = run(causal)
    lp = F.log_softmax(r_full.float(), -1)
    lq = F.log_softmax(r_causal.float(), -1)
    delta = (lp.exp() * (lp - lq)).sum(-1)[sup].mean().item()
    assert 1.0 < delta < 30.0, f"effet prompt-contexte hors borne : {delta}"
    del model
    torch.cuda.empty_cache()
