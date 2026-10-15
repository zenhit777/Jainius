"""Runner local J4 — même protocole P-G/1.0 que B0, moteur HF local.

Réutilise `b0_runner` (prompts, scorers, plafonds, stops) à l'identique ;
seul le moteur change (SGLang HTTP indisponible hors WSL/Docker :
génération gloutonne batchée locale, T=0 déterministe → 1 répétition,
écart documenté vs 3 répétitions B0). Comparabilité moteur vérifiée par
sous-ensemble source (local vs B0-SGLang) avant toute conclusion.
Sortie JSON au même schéma que B0 (`engine` adapté).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jainius.eval.b0_runner import (
    COMPOSITE,
    DATASETS,
    GENERATION,
    MAX_NEW_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    SCORERS,
    STOP_TOKENS,
    SYSTEM_PROMPTS,
    build_prompt,
    load_all,
)
from jainius.eval.harness import protocol_hash

MODEL_ID = "Qwen/Qwen3.5-0.8B"
BATCH_BY_DATASET = {"gsm8k": 8, "math500": 8, "ifeval": 8, "humaneval": 16,
                    "mbpp": 16, "mmlu": 64, "mmlu_pro": 64}


def _truncate_stops(text: str) -> str:
    cut = len(text)
    for s in STOP_TOKENS:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]


def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int, batch_size: int) -> list[str]:
    """Génération gloutonne batchée (padding à gauche, masque exact)."""
    out: list[str] = []
    pad_id = tokenizer.eos_token_id
    for b in range(0, len(prompts), batch_size):
        chunk = prompts[b: b + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        padding_side="left")
        input_ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        plen = input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                input_ids, attention_mask=mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
        new = gen[:, plen:]
        texts = tokenizer.batch_decode(new, skip_special_tokens=False)
        out.extend(_truncate_stops(t) for t in texts)
    return out


def run_dataset_local(name: str, ds, model, tokenizer,
                      batch_size: int) -> dict[str, Any]:
    scorer = SCORERS[name]
    max_new = MAX_NEW_TOKENS.get(name, DEFAULT_MAX_NEW_TOKENS)
    prompts = [build_prompt(tokenizer, name, row) for row in ds]
    responses = generate_batch(model, tokenizer, prompts, max_new,
                               batch_size)
    scores, unimplemented, failures_sample = [], set(), []
    failures = 0
    for idx, (row, resp) in enumerate(zip(ds, responses)):
        try:
            r = scorer(row, resp)
            ok, unimpl = (r if name == "ifeval" else (bool(r), []))
            scores.append(1.0 if ok else 0.0)
            unimplemented.update(unimpl)
            if not ok and len(failures_sample) < 5:
                failures_sample.append({"index": idx,
                                        "response_head": resp[:200]})
        except Exception as e:  # noqa: BLE001 — scorer isolé (comme B0)
            scores.append(0.0)
            failures += 1
            if len(failures_sample) < 5:
                failures_sample.append(
                    {"index": idx, "error": f"{type(e).__name__}: {e}"})
    n = len(scores)
    return {"rows": n, "score": sum(scores) / n if n else 0.0,
            "unimplemented_constraints": sorted(unimplemented),
            "generation_failures": failures,
            "failures_sample": failures_sample}


def evaluate(model_id_or_model, tokenizer=None, datasets=None,
             names=None, batch_over=None,
             dtype=torch.bfloat16) -> dict[str, Any]:
    """Évalue (modèle HF ou chemin). Retourne le dict résultat complet."""
    if isinstance(model_id_or_model, str):
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_model, dtype=dtype, device_map="cuda",
            trust_remote_code=False).eval()
    else:
        model = model_id_or_model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_model if isinstance(model_id_or_model, str)
            else MODEL_ID)
    ds_all = load_all(None) if datasets is None else datasets
    names = names or list(DATASETS)
    per_dataset: dict[str, Any] = {}
    table = dict(BATCH_BY_DATASET)
    if isinstance(batch_over, int):
        table = {n: batch_over for n in names}
    for name in names:
        t0 = time.time()
        r = run_dataset_local(name, ds_all[name], model, tokenizer,
                              table.get(name, 16))
        per_dataset.setdefault(name, {})["rep0"] = r
        mean = r["score"]
        per_dataset[name].update({
            "mean": mean, "std": 0.0, "repetitions": [mean],
            "unimplemented_constraints": r.pop("unimplemented_constraints"),
            "max_new_tokens": MAX_NEW_TOKENS.get(
                name, DEFAULT_MAX_NEW_TOKENS)})
        print(f"[{name}] score={mean:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    weights = COMPOSITE["weights"]
    composite = sum(per_dataset[n]["mean"] * weights[n] for n in names)
    w6 = {k: v for k, v in weights.items() if k != "ifeval"}
    s6 = sum(w6.values())
    composite_no_ifeval = (
        sum(per_dataset[n]["mean"] * w6[n] / s6 for n in names if n != "ifeval")
        if "ifeval" in names else None)
    return {"per_dataset": per_dataset, "composite": composite,
            "composite_no_ifeval": composite_no_ifeval}


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner local P-G/1.0 (J4)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = load_all(args.limit)
    names = args.datasets or list(DATASETS)
    res = evaluate(args.model, tokenizer, datasets, names,
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
        "model": args.model,
        "engine": "local-HF generate() glouton batché (J4, sans Docker)",
        "environment": {"limit": args.limit, "batch": args.batch},
        "datasets": res["per_dataset"],
        "composite": res["composite"],
        "composite_no_ifeval": res["composite_no_ifeval"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"écrit : {args.out}")
    print(f"composite = {res['composite']:.4f} "
          f"(sans IFEval : {res['composite_no_ifeval']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
