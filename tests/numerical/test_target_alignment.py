"""Alignement cibles enseignant/étudiant, convention `current` (pré-M2).

Dossier formel (OPDLM `rl_sdar.py`, coordonnées slicées (B, L0+L1)) :
- étudiant : slice `[.., :L0] + [.., L0+L1:]`, SANS roll
  (`student_arm_shift: False`, `rl_bd3lm.yaml:108`, code `:1269`) ;
- enseignant : même slice, AVEC `roll(1)` (`:1344-1348`) ;
- supervisées : réponse ∧ pad ∧ pré-EOS ∧ masquée-en-xt (`:1425-1428`) ;
- normalisation : moyenne par ligne puis batch (`:1428`, `:1449`).
Le décalage requis (roll enseignant, pas de roll étudiant) DOIT être
présent dans `train_step` via `Teacher.hidden_for_loss` (contre-audit :
la version précédente consommait `teacher.hidden` sans roll — BUG B1,
cibles « jeton suivant », incohérent avec l'inférence). Différence
assumée et documentée (pas une omission) : le contexte prompt/ligne de
l'étudiant est causal (`causal_x0=True` exigé : D-10, AR-Trust,
cohérence FLARE), tandis qu'OPDLM utilise un prompt
bloc-bidirectionnel — test (4) le quantifie. Les positions xt répétées
(§10.2, `tok_idx_resp`) sont verrouillées par le test (1).

Modèle : Qwen3-0.6B auxiliaire fp32 (famille Qwen3, attention pure comme
le code OPDLM). Déterministe (seed 0).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.distill.loss import kl_teacher_to_student
from jainius.distill.targets import compute_teacher_targets
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


def test_train_step_targets_match_reference():
    """RÉGRESSION B1 : les cibles réellement consommées par `train_step`
    — via L'UNIQUE fonction `Teacher.teacher_logits_for_loss` (appelée
    par `train_step` lui-même) — coïncident avec la référence
    `logprobs_simple` aux positions supervisées. ÉCHOUE sur l'ancien
    chemin (`teacher.hidden` sans roll : écart max ~25 mesuré) car
    l'ancienne expression est comparée explicitement ci-dessous."""
    model, prompt, l0, xresp, xt_resp = _setup()
    teacher = Teacher(model)
    x0 = torch.cat([prompt, xresp], dim=1)
    sup = torch.zeros_like(x0, dtype=torch.bool)
    sup[0, l0:] = (xt_resp[0] == MASK_ID)
    with torch.no_grad():
        # A1/B1 non miroir : la fonction UNIQUE de cibles (aussi appelée par
        # train_step). Si train_step redevient non décalé, ce test échoue.
        consumed = F.log_softmax(
            compute_teacher_targets(
                x0,
                lambda ids: model(ids,
                                  output_hidden_states=True).hidden_states[-1],
                model.lm_head).float(), dim=-1)
        reference = teacher.logprobs_simple(x0)
        old_buggy = F.log_softmax(
            model.lm_head(teacher.hidden(x0)).float(), dim=-1)
    assert torch.allclose(
        consumed[sup], reference[sup], atol=1e-5), "cibles ≠ référence"
    old_diff = (old_buggy[sup] - reference[sup]).abs().max().item()
    assert old_diff > 1.0, (
        f"le test ne discriminerait plus l'ancien chemin (écart {old_diff})")
    del model
    torch.cuda.empty_cache()


def test_alignment_shift_sensitivity():
    """Avec l'enseignant CORRECT (décalé), la perte verrouille l'oracle ET
    reste sensible aux décalages : toute translation ±1 de l'étudiant OU
    tout décalage erroné injecté dégrade strictement la KL. Un test qui
    conclurait avec une cible non décalée est invalide (voir le piège)."""
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
        h_t = teacher.hidden_for_loss(x0)
    head = lambda h: model.lm_head(h)
    loss_ref = kl_teacher_to_student(h_s, head, h_t, head, sup).loss.item()
    # Verrou oracle : même calcul via logprobs_simple (fonction distincte).
    with torch.no_grad():
        oracle = teacher.logprobs_simple(x0)
        lp_s = F.log_softmax(head(h_s).float(), dim=-1)
        kl_oracle = torch.where(
            torch.isfinite(oracle), oracle.exp() * (oracle - lp_s),
            torch.zeros_like(oracle)).sum(-1)
        expected = (kl_oracle * sup).sum(1) / sup.sum(1).clamp_min(1.0)
    out = kl_teacher_to_student(h_s, head, h_t, head, sup)
    assert torch.allclose(out.per_row.cpu(),
                          expected.cpu(), atol=1e-5), "perte ≠ oracle"
    # Sensibilité : un enseignant non décalé ou un étudiant translaté
    # +1 changent la perte substantiellement (mesuré Qwen3-0.6B fp32 :
    # ref=12.53, raw=8.58, +1=3.41). Le sens de la variation n'est PAS
    # asserté (à poids AR initiaux, +1 « gagne » par auto-similarité —
    # voir le test piège) : seule compte la sensibilité + le verrou oracle.
    loss_p1 = kl_teacher_to_student(h_s.roll(1, 1), head, h_t, head, sup).loss.item()
    with torch.no_grad():
        h_t_raw = teacher.hidden(x0)
    loss_raw = kl_teacher_to_student(h_s, head, h_t_raw, head, sup).loss.item()
    assert abs(loss_p1 - loss_ref) > 0.2 * loss_ref
    assert abs(loss_raw - loss_ref) > 0.2 * loss_ref, (
        f"test aveugle au décalage enseignant : ref={loss_ref:.3f} "
        f"raw={loss_raw:.3f}")
    del model
    torch.cuda.empty_cache()


def test_alignment_shift_trap_documented():
    """PIÈGE CARACTÉRISÉ (ne pas « corriger ») : avec l'enseignant NON
    décalé (ancien bug B1), la KL est minimale SANS translation étudiant —
    à poids AR initiaux, tout est auto-similaire (mesuré : raw=8.58 <
    +1=15.53). C'est exactement ce que l'ancien test de sensibilité
    vérifiait, d'où sa fausse validation. Leçon verrouillée : seule la
    comparaison à l'oracle décalé (`test_train_step_targets_match_reference`
    + verrou oracle ci-dessus) prouve l'alignement, jamais la minimalité
    relative."""
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
        h_t_raw = teacher.hidden(x0)  # ANCIEN chemin (sans roll)
    head = lambda h: model.lm_head(h)
    loss_raw = kl_teacher_to_student(h_s, head, h_t_raw, head, sup).loss.item()
    loss_p1 = kl_teacher_to_student(h_s.roll(1, 1), head, h_t_raw, head,
                                    sup).loss.item()
    loss_m1 = kl_teacher_to_student(h_s.roll(-1, 1), head, h_t_raw, head,
                                    sup).loss.item()
    assert loss_p1 > loss_raw and loss_m1 > loss_raw, (
        "l'ancien test aurait détecté quelque chose : revérifier B1")
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
