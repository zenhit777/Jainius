"""T-TOK-01/02 — tokenizer et jeton de masque (support M1-A, sortie M0).

T-TOK-01 : vocabulaire, ids spéciaux, id de masque 248077 < 248320 et absent
du tokenizer, embedding non dégénéré (F-002).
T-TOK-02 : rendu `apply_chat_template(enable_thinking=False)` stable et
identique entre appels (identité entraînement/rollout/évaluation/serving).
"""

import torch


def test_tok_01_vocab_and_mask_id(tokenizer, model_bf16):
    assert tokenizer.vocab_size == 248044
    assert model_bf16.config.get_text_config().vocab_size == 248320
    # Valeurs vérifiées le 2026-10-11 (M0 + M1-A) : le tokenizer utilise
    # eos=248046 (<|im_end|>) et pad=248044 ; le config text porte
    # eos_token_id=248044. Écart consigné, non bloquant (F-002 = préservation,
    # pas égalité config/tokenizer).
    assert tokenizer.eos_token_id == 248046
    assert tokenizer.pad_token_id == 248044
    mask_id = 248077
    assert mask_id < 248320
    # Le tokenizer ne doit produire cet id pour aucun texte courant.
    for probe in ["Hello", "MASK", "[MASK]", "<mask>", "reasoning"]:
        assert mask_id not in tokenizer.encode(probe), f"collision : {probe!r}"
    row = model_bf16.get_input_embeddings().weight[mask_id]
    assert torch.isfinite(row).all() and row.norm().item() > 0


def test_tok_02_chat_template_stable(tokenizer):
    messages = [{"role": "user", "content": "Hello"}]
    kwargs = dict(
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    a = tokenizer.apply_chat_template(messages, **kwargs)
    b = tokenizer.apply_chat_template(messages, **kwargs)
    assert a == b and len(a) > 0
