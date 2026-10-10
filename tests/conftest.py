"""Fixtures M1 : modèle et tokenizer Qwen3.5-0.8B (cache HF local)."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MASK_TOKEN_ID = 248077
SEED = 0


def chat_ids(tokenizer, content: str):
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt")


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)


@pytest.fixture(scope="module")
def model_bf16():
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False
    ).eval()
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def model_fp32():
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cuda", trust_remote_code=False
    ).eval()
    yield model
    del model
    torch.cuda.empty_cache()
