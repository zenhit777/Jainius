"""Localise l'écart batché vs unitaire (J4) : forward étudiant, enseignant ou perte."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jainius.decode.rollout import rollout_shifted  # noqa: E402
from jainius.decode.sampler import SamplerSpec  # noqa: E402
from jainius.distill.targets import compute_teacher_targets_shifted  # noqa: E402
from jainius.model.twostream import TwoStreamRunner, shifted_supervision_mask  # noqa: E402



def _chat_ids(tok, content):
    messages = [{"role": "user", "content": content}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    return tok(text, return_tensors="pt")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MASK_ID = 248077
BB = 4


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval()
    runner = TwoStreamRunner(student)
    dev = "cuda"
    p1 = _chat_ids(tok, "Count to three.")["input_ids"][0].tolist()
    cands = ["Name a planet.", "Solve x+3=7.", "Count to five.",
             "What is 9*8?", "What is H2O?", "Say hello.",
             "Define entropy.", "Who wrote Hamlet?"]
    p2 = None
    for c in cands:
        q = _chat_ids(tok, c)["input_ids"][0].tolist()
        if len(q) == len(p1):
            p2 = q
            break
    assert p2 is not None, "pas de paire de même longueur"
    print("P:", len(p1), len(p2), flush=True)
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                       seed=0)
    with torch.no_grad():
        t1 = rollout_shifted(runner, p1, 8, spec, block_size=BB,
                             mask_id=MASK_ID)
        t2 = rollout_shifted(runner, p2, 8, spec, block_size=BB,
                             mask_id=MASK_ID)
        P, R, L = len(p1), 8, len(p1) + 8
        sup = shifted_supervision_mask(L, P, BB).to(dev)
        rows = []
        for pids, tr in ((p1, t1), (p2, t2)):
            x0 = torch.tensor([pids + tr.x0], dtype=torch.long, device=dev)
            xt = x0.clone()
            xt[0][sup] = MASK_ID
            _, hn, _ = runner.forward_train_shifted(
                x0, xt, block_size=BB, resp_start=P)
            rows.append(hn)
        h_unit = torch.cat(rows, dim=0)
        x0b = torch.tensor([p1 + t1.x0, p2 + t2.x0], dtype=torch.long,
                           device=dev)
        xtb = x0b.clone()
        xtb[:, sup] = MASK_ID
        _, h_batch, _ = runner.forward_train_shifted(
            x0b, xtb, block_size=BB, resp_start=P)
        d = (h_unit.float() - h_batch.float()).abs()
        print(f"student h_noisy: max_abs={d.max():.3e} "
              f"mean_abs={d.mean():.3e} ref_abs_mean={h_unit.float().abs().mean():.3e}",
              flush=True)
        f1 = student(x0b[:1], output_hidden_states=True).hidden_states[-1]
        f2 = student(x0b, output_hidden_states=True).hidden_states[-1][:1]
        d2 = (f1.float() - f2.float()).abs()
        print(f"teacher HF B=1 vs B=2: max_abs={d2.max():.3e} "
              f"mean_abs={d2.mean():.3e}", flush=True)
        g1 = compute_teacher_targets_shifted(
            x0b[:1], lambda ids: student(
                ids, output_hidden_states=True).hidden_states[-1],
            student.lm_head)
        g2 = compute_teacher_targets_shifted(
            x0b, lambda ids: student(
                ids, output_hidden_states=True).hidden_states[-1],
            student.lm_head)[:1]
        d3 = (g1.float() - g2.float()).abs()
        print(f"teacher targets B=1 vs B=2: max_abs={d3.max():.3e} "
              f"mean_abs={d3.mean():.3e}", flush=True)


if __name__ == "__main__":
    main()
