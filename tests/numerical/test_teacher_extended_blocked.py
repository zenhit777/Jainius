"""Diagnostic de compatibilité ENSEIGNANT sur hybride (BLOCKED, séparé).

Ce test n'affirme que ce qu'il teste : la construction enseignante
étendue `clean_fill` (valide sur attention pure, T-TEACH-02) n'est PAS
transposable telle quelle au backbone récurrent Qwen3.5 — la récurrence
GDN consomme tout x0 avant xt (état pollué), là où le masque ne gouverne
que l'attention (constat M1-D0 : divergence réponse ~6-11 en bf16/fp32).
Il ne prouve rien sur FLARE et ne revendique aucune parité FLARE.

Marqué `xfail(strict=True)` : visible comme blocage connu dans la suite
de référence (verte), et PASSERA automatiquement le jour où la
planification d'états GDN le résorbera (signal de déblocage, pas de
dissimulation).
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.distill.teacher import Teacher

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MASK_ID = 248077
TOL = 5e-2


@pytest.mark.xfail(
    strict=True,
    reason="BLOCKED (D-04) : clean_fill étendu invalide sur backbone GDN "
    "sans planification d'états (M1-D0).",
)
def test_teacher_extended_cleanfill_blocked_on_hybrid():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    ).eval()
    teacher = Teacher(model)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = tok(text, return_tensors="pt")["input_ids"].to("cuda")
    xresp = torch.randint(0, 1000, (1, 8), device="cuda")
    xt = xresp.clone()
    xt[0, ::2] = MASK_ID
    x0 = torch.cat([prompt, xresp], dim=1)
    with torch.no_grad():
        simple = teacher.logprobs_simple(x0)
        extended = teacher.logprobs_extended(prompt, xresp, xt, MASK_ID)
    diff = (simple - extended).abs()[0, prompt.shape[1]:].max().item()
    assert diff <= TOL, f"clean_fill diverge sur hybride : {diff} > {TOL}"
    del model
    torch.cuda.empty_cache()
