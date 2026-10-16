"""Pilote J4-bis — phase 2 depuis le checkpoint J4 avec prompts formatés.

Différences vs `scripts/pilot_j4.py` (J4, gelé pour reproductibilité) :
- reprise des POIDS + OPTIMISEUR + MAÎTRES depuis un checkpoint J4
  (config parente lue dans son `config.json` pour satisfaire le hash ;
  `force=True` pour code_sha, tracé au log) ;
- prompts seuls ENRICHIS (`remediation.build_training_text`, version
  `FORMAT_VERSION`) — `ground_truth_answer` / `tests_json` jamais lus ;
- curriculum : R constant (phase 1 a fait R=32 puis R=64) ;
- LR cosine redémarrée (warm restart 1e-5 -> 1e-6) ;
- compteurs cumulatifs via budget parent (`merge_budgets`) ;
- sauvegarde finale conditionnelle (`should_final_save`) ;
- gouvernance mémoire §9 (`apply_memory_governance`, log par pas,
  `empty_cache()` après chaque `save_ckpt`).

Usage :
  python scripts/pilot_j4b.py --out _artefacts/j4b/phase2 \\
      --from-ckpt _artefacts/j4/step_000699 --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import os

# Gouvernance mémoire : l'env doit précéder l'init CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    FORMAT_VERSION,
    apply_memory_governance,
    build_training_text,
    mem_snapshot,
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


def load_format_pairs(shuffle_seed: int, need: int):
    """Paires de longueur exacte, prompts seuls + consignes de format.

    Ne lit que `question` et `domain` (jamais les réponses de référence).
    Troncature à gauche : la consigne (en fin de prompt) est préservée.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    data = json.load(open(DATA, encoding="utf-8"))

    def pids(q, domain):
        txt = tok.apply_chat_template(
            [{"role": "user",
              "content": build_training_text(q, domain)}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return tok(txt, return_tensors="pt")["input_ids"][0].tolist()[-PTRUNC:]

    rng = torch.Generator().manual_seed(shuffle_seed)
    order = torch.randperm(len(data), generator=rng).tolist()
    buckets: dict[int, list] = {}
    for i in order:
        x = data[i]
        ids = pids(x["question"], x.get("domain"))
        buckets.setdefault(len(ids), []).append(ids)
        if i % 500 == 0:
            if sum(len(v) // 2 for v in buckets.values()) >= need:
                break
    pairs = []
    for ln in sorted(buckets):
        ids_list = buckets[ln]
        for j in range(0, len(ids_list) - 1, 2):
            if len(pairs) >= need:
                break
            pairs.append((ids_list[j], ids_list[j + 1]))
        if len(pairs) >= need:
            break
    return tok, pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Pilote J4-bis (phase 2 format)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-ckpt", required=True,
                    help="checkpoint J4 parent (ex. _artefacts/j4/step_000699)")
    ap.add_argument("--steps", type=int, default=300,
                    help="pas de phase 2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--response-len", type=int, default=64)
    ap.add_argument("--pair-offset", type=int, default=0,
                    help="décale l'index des paires (reprise : ne pas rejouer "
                         "les paires déjà vues, ex. 200 pour globals 900+)")
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

    mem_gov = apply_memory_governance()
    parent_dir = Path(args.from_ckpt)
    parent_config = json.loads((parent_dir.parent / "config.json").read_text())

    student = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).train()
    teacher = Teacher(AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).eval())
    assert teacher.is_frozen()
    optim = MasterOptimizer(student, lr=LR_PEAK)
    # Reprise poids + optim + maîtres + RNG (force: nouveau commit, tracé).
    manifest = load_ckpt(parent_dir, student, optim.optim, parent_config,
                         masters=optim.masters, force=True)
    parent_budget = manifest.get("budget", {})
    counters = merge_budgets(parent_budget)
    global_offset = manifest["step"] + 1
    log(event="phase2_start", parent=str(parent_dir),
        parent_step=manifest["step"],
        parent_code_sha=str(manifest.get("code_sha", ""))[:12],
        force_code_sha=True, counters_init=counters,
        mem_gov=mem_gov, fp=param_fingerprint(student)[:12])

    runner = TwoStreamRunner(student)
    _, pairs = load_format_pairs(args.seed, args.pair_offset + args.steps)
    assert len(pairs) >= args.pair_offset + args.steps, (
        f"paires insuffisantes : {len(pairs)}")
    pairs = pairs[args.pair_offset:]
    config = {"pilot": "J4-bis", "shifted": True, "seed": args.seed,
              "lr_peak": LR_PEAK, "lr_end": LR_END, "warmup": WARMUP,
              "batch": BATCH, "block_size": BB, "ptrunc": PTRUNC,
              "data": "divelab/opdlm_train_data (licence MIT, prompts seuls)",
              "prompt_format": FORMAT_VERSION,
              "parent_ckpt": str(parent_dir),
              "parent_config_hash": manifest.get("config_hash", ""),
              "response_len": args.response_len,
              "steps": args.steps}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log(event="start", global_offset=global_offset,
        pairs_available=len(pairs))

    last_saved = manifest["step"]

    def save(gstep):
        # J4-bis : vider le cache AVANT (défragmente avant le transitoire
        # de sauvegarde) comme après (incident OOM 16/10).
        torch.cuda.empty_cache()
        rng_cuda = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
        d = save_ckpt(out, student, optim.optim, torch.get_rng_state(),
                      gstep, config, masters=optim.masters, rng_cuda=rng_cuda,
                      extra={"curriculum_step": gstep, **counters,
                             "phase": "J4-bis"})
        torch.cuda.empty_cache()
        log(event="ckpt", dir=str(d))
        return d

    gstep = global_offset - 1
    try:
        for k in range(args.steps):
            gstep = global_offset + k
            R = args.response_len
            lr = lr_schedule(k, args.steps)
            optim.optim.param_groups[0]["lr"] = lr
            p1, p2 = pairs[k]
            spec = SamplerSpec(mode="static", temperature=1.0,
                               steps_per_block=1, seed=args.seed + gstep)
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
            counters["updates"] += 1
            counters["tokens_response"] += 2 * R
            counters["tokens_supervised"] += stats["n_supervised"]
            counters["tokens_seq"] += 2 * (len(p1) + R)
            # J4-bis : purge périodique (transitoire backward fragmenté ;
            # incident OOM 16/10). Coût négligeable devant ~16 s/pas.
            if k % 10 == 0:
                torch.cuda.empty_cache()
            log(event="step", step=gstep, phase_step=k, R=R,
                lr=f"{lr:.2e}", loss=round(loss, 4),
                grad=round(stats["grad_norm"], 4),
                n_sup=stats["n_supervised"], rollout_s=round(dt_r, 1),
                train_s=round(dt_t, 1), counters=dict(counters),
                **mem_snapshot())
            if (gstep + 1) % args.ckpt_every == 0:
                save(gstep)
                last_saved = gstep
        if should_final_save(last_saved, args.steps - 1 + global_offset):
            save(args.steps - 1 + global_offset)
            last_saved = args.steps - 1 + global_offset
        log(event="done", ckpt_step=last_saved, counters=counters,
            fp=param_fingerprint(student)[:12])
    except Exception:
        traceback.print_exc()
        try:
            if gstep >= global_offset and gstep != last_saved:
                save(gstep)
                log(event="emergency_ckpt", step=gstep)
        except Exception:
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
