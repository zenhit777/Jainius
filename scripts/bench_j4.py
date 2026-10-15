"""Micro-benchmark J4 : débit rollout + pas shifted (décision de taille pilote)."""

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
from jainius.train.loop import train_step_shifted  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"
BB = 4


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).train()
    teacher_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval()
    teacher = Teacher(teacher_hf)
    runner = TwoStreamRunner(student)
    mask_id = 248077
    messages = [{"role": "user", "content": "Solve: a train travels 120 km in 2 hours. What is its speed?"}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    pids = tok(text, return_tensors="pt")["input_ids"][0].tolist()
    print(f"prompt_len={len(pids)}", flush=True)
    for resp_len, steps, mode in [(128, 1, "static"), (128, 4, "dynamic")]:
        spec = SamplerSpec(mode=mode, threshold=0.9, temperature=1.0,
                           steps_per_block=steps, seed=0)
        t0 = time.time()
        with torch.no_grad():
            traj = rollout_shifted(runner, pids, resp_len, spec,
                                   block_size=BB, mask_id=mask_id)
        dt = time.time() - t0
        L = len(pids) + len(traj.x0)
        print(f"rollout R={resp_len} {mode}/T=1.0 steps={steps}: "
              f"{dt:.1f}s, forwards={traj.n_forwards}, L={L}, "
              f"tok/s={(len(traj.x0)) / dt:.0f}", flush=True)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                       seed=0)
    with torch.no_grad():
        traj = rollout_shifted(runner, pids, 128, spec, block_size=BB,
                               mask_id=mask_id)
    optim = torch.optim.AdamW(student.parameters(), lr=1e-5)
    t0 = time.time()
    stats: dict = {}
    loss = train_step_shifted(student, teacher, optim, [pids], [traj],
                              stats=stats, runner=runner, block_size=BB,
                              mask_id=mask_id)
    dt = time.time() - t0
    L = len(pids) + len(traj.x0)
    print(f"train_step L={L} n_sup={stats['n_supervised']}: {dt:.1f}s, "
          f"loss={loss:.4f} -> tok/s={L / dt:.0f} (seq complete)",
          flush=True)


if __name__ == "__main__":
    main()
