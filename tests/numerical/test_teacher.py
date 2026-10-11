"""T-TEACH-01/02 — enseignant gelé, parité `clean_fill` (H-03).

T-TEACH-01 : gel (requires_grad, eval, sans-gradient, empreinte stable).
T-TEACH-02 : voie simple AR ≡ construction étendue `clean_fill` portée de
OPDLM (`rl_sdar.py:1297-1348`, positions répétées `:850-855`), sur
Qwen3-0.6B auxiliaire (famille Qwen3, attention pure comme le code OPDLM —
usage justifié : le test porte la construction, pas le backbone cible).
Modèle fp32, tolérance 1e-4 (règle float32 §13.5).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.distill.teacher import Teacher

AUX_ID = "Qwen/Qwen3-0.6B"
MASK_ID = 151669  # Qwen3 (OPDLM) ; le 0.6B partage le vocab Qwen3


def _aux():
    tok = AutoTokenizer.from_pretrained(AUX_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        AUX_ID, dtype=torch.float32, device_map="cuda", trust_remote_code=False
    )
    return tok, model


def test_teach_01_frozen():
    torch.manual_seed(0)
    _, model = _aux()
    teacher = Teacher(model)
    assert teacher.is_frozen()
    assert not teacher.model.training
    before = [p.detach().clone() for p in teacher.model.parameters()]
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    out = teacher.logits(ids)
    assert not out.requires_grad
    for a, b in zip(before, teacher.model.parameters()):
        assert torch.equal(a, b), "poids enseignant mutés"
    del model
    torch.cuda.empty_cache()


def test_teach_02_clean_fill_parity():
    torch.manual_seed(0)
    tok, model = _aux()
    teacher = Teacher(model)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 6*7?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = tok(text, return_tensors="pt")["input_ids"].to("cuda")
    l0 = prompt.shape[1]
    l1 = 8
    x0_resp = torch.randint(0, 1000, (1, l1), device="cuda")
    xt = x0_resp.clone()
    xt[0, ::2] = MASK_ID  # masques alternés
    x0_full = torch.cat([prompt, x0_resp], dim=1)
    with torch.no_grad():
        simple = teacher.logprobs_simple(x0_full)
        extended = teacher.logprobs_extended(prompt, x0_resp, xt, MASK_ID)
    assert simple.shape == extended.shape == (1, l0 + l1, 151936)
    diff = (simple - extended).abs()
    # Positions supervisées (réponse) : identité stricte attendue.
    resp_diff = diff[0, l0:].max().item()
    assert resp_diff <= 1e-4, f"parité clean_fill violée : {resp_diff}"
    del model
    torch.cuda.empty_cache()
