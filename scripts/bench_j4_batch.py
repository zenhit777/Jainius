"""Débit du pas batché J4 à l'échelle pilote (B=8, prompts OPDLM tronqués)."""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jainius.decode.rollout import rollout_shifted  # noqa: E402
from jainius.decode.sampler import SamplerSpec  # noqa: E402
from jainius.distill.teacher import Teacher  # noqa: E402
from jainius.model.twostream import TwoStreamRunner  # noqa: E402
from jainius.train.loop import train_step_shifted_batched  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
BB = 4
BATCH = 2
PTRUNC = 192


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    data = json.load(open("_artefacts/data/opdlm_train_data/opdlm_train.json",
                          encoding="utf-8"))

    def pids(q):
        txt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tok(txt, return_tensors="pt")["input_ids"][0].tolist()
        return ids[-PTRUNC:]

    buckets: dict[int, list] = {}
    for x in data[:6000]:
        ids = pids(x["question"])
        buckets.setdefault(len(ids), []).append(ids)
    full = [(k, v[:BATCH]) for k, v in buckets.items()
            if len(v) >= BATCH and k <= PTRUNC]
    full.sort(key=lambda kv: -len(buckets[kv[0]]))
    P, batch = full[0][0], full[0][1]
    print(f"P={P} (bucket le plus fourni)", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).train()
    teacher = Teacher(AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval())
    runner = TwoStreamRunner(student)
    mask_id = 248077
    R = 64
    spec = SamplerSpec(mode="static", temperature=1.0, steps_per_block=1,
                       seed=0)
    t0 = time.time()
    with torch.no_grad():
        trajs = [rollout_shifted(runner, p, R, spec, block_size=BB,
                                 mask_id=mask_id) for p in batch]
    dt_r = time.time() - t0
    print(f"rollouts {BATCH}x R={R} static/T=1/s=1: {dt_r:.1f}s -> "
          f"{BATCH * R / dt_r:.1f} tok/s (reponse)", flush=True)
    for p, t in zip(batch, trajs):
        t.weight_version = "bench"
    optim = torch.optim.AdamW(student.parameters(), lr=1e-5)
    t0 = time.time()
    stats: dict = {}
    loss = train_step_shifted_batched(
        student, teacher, optim, batch, trajs, stats=stats, runner=runner,
        block_size=BB, mask_id=mask_id)
    dt_t = time.time() - t0
    L = P + R
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"train_step B={BATCH} L={L} n_sup={stats['n_supervised']}: "
          f"{dt_t:.1f}s loss={loss:.4f} -> {BATCH * L / dt_t:.1f} tok/s "
          f"(seq), pic_mem={mem:.1f} Go", flush=True)
    tot = dt_r + dt_t
    print(f"pas complet: {tot:.1f}s pour {BATCH * R} tok reponse "
          f"({BATCH * L} seq) -> {BATCH * R / tot:.1f} tok-rep/s, "
          f"{BATCH * L / tot:.1f} tok-seq/s", flush=True)


if __name__ == "__main__":
    main()
