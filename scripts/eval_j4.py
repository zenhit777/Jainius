"""Évaluation J4 — checkpoint Jainius via runner local (même protocole P-G/1.0 que B0).

Usage :
  python scripts/eval_j4.py --ckpt _artefacts/j4/step_000699 --out _artefacts/j4/eval_ckpt.json
  python scripts/eval_j4.py --ckpt ... --out ... --limit 50 --datasets gsm8k mmlu   # comparabilité
  python scripts/eval_j4.py --model Qwen/Qwen3.5-0.8B --out ... --limit 50 ...       # source locale

Le checkpoint ne contient qu'un state_dict (model.pt) : on le charge dans
un AutoModelForCausalLM du même MODEL_ID, puis on évalue avec
`local_runner.evaluate` (prompts/scorers/plafonds/stops B0 à l'identique,
moteur glouton local documenté).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jainius.eval import local_runner  # noqa: E402
from jainius.eval.b0_runner import (  # noqa: E402
    COMPOSITE,
    DATASETS,
    GENERATION,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPTS,
    STOP_TOKENS,
    load_all,
)
from jainius.eval.harness import protocol_hash  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def main() -> int:
    ap = argparse.ArgumentParser(description="Évaluation J4 (runner local)")
    ap.add_argument("--ckpt", default=None,
                    help="répertoire checkpoint (model.pt)")
    ap.add_argument("--model", default=None,
                    help="modèle HF (source locale, comparabilité)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    if bool(args.ckpt) == bool(args.model):
        raise SystemExit("donner exactement un de --ckpt / --model")

    t0 = time.time()
    if args.ckpt:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=False)
        sd = torch.load(Path(args.ckpt) / "model.pt", weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        label = args.ckpt
    else:
        model = args.model
        label = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    datasets = load_all(args.limit)
    names = args.datasets or list(DATASETS)
    res = local_runner.evaluate(model, tokenizer, datasets, names,
                                batch_over=args.batch)
    result = {
        "protocol": {"version": "P-G/1.0", "hash": protocol_hash(),
                      "generation": {**GENERATION, "n_repetitions": 1,
                                     "note": "T=0 déterministe local : "
                                             "1 répétition (B0 : 3)"},
                      "composite": COMPOSITE,
                      "system_prompts": SYSTEM_PROMPTS,
                      "stop_tokens": STOP_TOKENS,
                      "deviation_max_new_tokens": "idem B0 (2048 gsm8k/math500, 1024 ifeval)"},
        "model": label,
        "engine": "local-HF generate() glouton batché (J4, sans Docker)",
        "environment": {"limit": args.limit, "batch": args.batch,
                        "eval_s": round(time.time() - t0, 1)},
        "datasets": res["per_dataset"],
        "composite": res["composite"],
        "composite_no_ifeval": res["composite_no_ifeval"],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"écrit : {args.out}")
    print(f"composite = {res['composite']:.4f} "
          f"(sans IFEval : {res['composite_no_ifeval']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
