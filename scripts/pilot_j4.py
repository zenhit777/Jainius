"""Pilote J4 — conversion Qwen3.5-0.8B sur prompts OPDLM (réels, prompts seuls).

Protocole (écarts OPDLM documentés) :
- prompts : `divelab/opdlm_train_data` (licence MIT, voir manifeste),
  mélangés (seed), chat-template Qwen3.5, tronqués à gauche à 192 tokens ;
- batching : paires de longueur exacte (pas de padding, voie J3 inchangée) ;
- curriculum réduit : R=32 (pas 0-99) puis R=64 (pas 100+) — L≤256
  (pas de checkpointing d'activations, cf. DECISIONS C.2) ;
- rollouts : statique, T=1.0, 1 pas/bloc (diversité par température,
  pas de raffinement itératif — écart OPDLM : 4 pas/bloc dynamiques,
  motivé par le débit mesuré ~6,6 tok/s en pile de référence) ;
- optimisation : MasterOptimizer, AdamW, LR cosine 1e-5→1e-6 (warmup
  linéaire 5 pas, cf. OPDLM), batch effectif 2 (écart OPDLM : 8), clip 1.0 ;
- tokens comptés : réponse (comparable OPDLM Éq.7), supervisés, séquence.

Usage :
  python scripts/pilot_j4.py --out _artefacts/j4 --steps 700 [--resume DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jainius.decode.rollout import rollout_shifted  # noqa: E402
from jainius.decode.sampler import SamplerSpec  # noqa: E402
from jainius.distill.teacher import Teacher  # noqa: E402
from jainius.model.twostream import TwoStreamRunner  # noqa: E402
from jainius.train.checkpoint import load_ckpt, save_ckpt  # noqa: E402
from jainius.train.loop import (  # noqa: E402
    param_fingerprint,
    train_step_shifted_batched,
)
from jainius.train.master import MasterOptimizer  # noqa: E402
from jainius.train.remediation import (  # noqa: E402
    merge_budgets,
    should_final_save,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATA = "_artefacts/data/opdlm_train_data/opdlm_train.json"
BB = 4
MASK_ID = 248077
BATCH = 2
PTRUNC = 192
LR_PEAK, LR_END, WARMUP = 1e-5, 1e-6, 5


def lr_schedule(step: int, total: int) -> float:
    if step < WARMUP:
        return LR_PEAK * (step + 1) / WARMUP
    p = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return LR_END + 0.5 * (LR_PEAK - LR_END) * (1 + math.cos(math.pi * p))


def load_pairs(shuffle_seed: int, need: int):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    data = json.load(open(DATA, encoding="utf-8"))

    def pids(q):
        txt = tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        return tok(txt, return_tensors="pt")["input_ids"][0].tolist()[-PTRUNC:]

    rng = torch.Generator().manual_seed(shuffle_seed)
    order = torch.randperm(len(data), generator=rng).tolist()
    buckets: dict[int, list] = {}
    domains: dict[int, list] = {}
    for i in order:
        x = data[i]
        ids = pids(x["question"])
        buckets.setdefault(len(ids), []).append(ids)
        domains.setdefault(len(ids), []).append(x.get("domain"))
        if i % 500 == 0:
            tot = sum(len(v) // 2 for v in buckets.values())
            if tot >= need:
                break
    pairs, pair_domains = [], []
    for ln in sorted(buckets):
        ids_list, dom_list = buckets[ln], domains[ln]
        for j in range(0, len(ids_list) - 1, 2):
            if len(pairs) >= need:
                break
            pairs.append((ids_list[j], ids_list[j + 1]))
            pair_domains.append((dom_list[j], dom_list[j + 1]))
        if len(pairs) >= need:
            break
    return tok, pairs, pair_domains


def main() -> int:
    ap = argparse.ArgumentParser(description="Pilote J4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-every", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pilot.log.jsonl"
    logf = open(log_path, "a", encoding="utf-8")

    def log(**kw):
        kw["t"] = round(time.time(), 1)
        logf.write(json.dumps(kw) + "\n")
        logf.flush()
        print(json.dumps(kw), flush=True)

    start_step = 0
    resume_cfg = None
    resume_budget: dict = {}
    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).train()
    teacher = Teacher(AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval())
    assert teacher.is_frozen()
    optim = MasterOptimizer(student, lr=LR_PEAK)
    runner = TwoStreamRunner(student)
    tok, pairs, pair_domains = load_pairs(args.seed, args.steps)
    assert len(pairs) >= args.steps, f"paires insuffisantes : {len(pairs)}"
    config = {"pilot": "J4", "shifted": True, "seed": args.seed,
              "lr_peak": LR_PEAK, "lr_end": LR_END, "warmup": WARMUP,
              "batch": BATCH, "block_size": BB, "ptrunc": PTRUNC,
              "data": "divelab/opdlm_train_data (licence MIT, prompts seuls)",
              "steps": args.steps}
    if args.resume:
        d = Path(args.resume)
        manifest = load_ckpt(d, student, optim.optim, config,
                             masters=optim.masters)
        start_step = manifest["step"] + 1
        resume_budget = dict(manifest.get("budget", {}))
        pairs = pairs[start_step:]
        pair_domains = pair_domains[start_step:]
        log(event="resume", from_step=manifest["step"],
            tokens_response=tom)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log(event="start", start_step=start_step, pairs_available=len(pairs),
        fp=param_fingerprint(student)[:12])

    def save(step, tokens_response, tokens_sup, tokens_seq, updates):
        rng_cuda = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        d = save_ckpt(out, student, optim.optim, torch.get_rng_state(),
                      step, config, masters=optim.masters, rng_cuda=rng_cuda,
                      extra={"curriculum_step": step,
                             "tokens_response": tokens_response,
                             "tokens_supervised": tokens_sup,
                             "tokens_seq": tokens_seq, "updates": updates})
        log(event="ckpt", dir=str(d))
        return d

    _c = merge_budgets(resume_budget)  # cumulatifs (corrige remise à zéro J4)
    tom = _c["tokens_response"]
    tokens_sup = _c["tokens_supervised"]
    tokens_seq = _c["tokens_seq"]
    updates = _c["updates"]
    step = start_step - 1
    last_saved: int | None = (resume_budget.get("curriculum_step")
                              if args.resume else None)
    try:
        for k in range(start_step, args.steps):
            step = k
            R = 32 if step < 100 else 64
            lr = lr_schedule(step, args.steps)
            optim.optim.param_groups[0]["lr"] = lr
            p1, p2 = pairs[k - start_step]
            spec = SamplerSpec(mode="static", temperature=1.0,
                               steps_per_block=1, seed=args.seed + step)
            t0 = time.time()
            with torch.no_grad():
                trajs = [rollout_shifted(runner, p, R, spec, block_size=BB,
                                         mask_id=MASK_ID) for p in (p1, p2)]
            dt_r = time.time() - t0
            for t in trajs:
                t.weight_version = param_fingerprint(student)
            stats: dict = {}
            t1 = time.time()
            loss = train_step_shifted_batched(
                student, teacher, optim, [p1, p2], trajs, stats=stats,
                runner=runner, block_size=BB, mask_id=MASK_ID)
            dt_t = time.time() - t1
            updates += 1
            tom += 2 * R
            tokens_sup += stats["n_supervised"]
            tokens_seq += 2 * (len(p1) + R)
            log(event="step", step=step, R=R, lr=f"{lr:.2e}",
                loss=round(loss, 4), grad=round(stats["grad_norm"], 4),
                n_sup=stats["n_supervised"], rollout_s=round(dt_r, 1),
                train_s=round(dt_t, 1), domains=pair_domains[k - start_step],
                tokens_response=tom)
            if (step + 1) % args.ckpt_every == 0:
                save(step, tom, tokens_sup, tokens_seq, updates)
                last_saved = step
        # Finale conditionnelle (corrige FileExistsError J4 si steps % every == 0).
        if should_final_save(last_saved, args.steps - 1):
            last = save(args.steps - 1, tom, tokens_sup, tokens_seq, updates)
        else:
            last = out / f"step_{args.steps - 1:06d}"
        log(event="done", ckpt=str(last), tokens_response=tom,
            tokens_supervised=tokens_sup, tokens_seq=tokens_seq,
            fp=param_fingerprint(student)[:12])
    except Exception:
        traceback.print_exc()
        try:
            save(step, tom, tokens_sup, tokens_seq, updates)
            log(event="emergency_ckpt", step=step)
        except Exception:
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
