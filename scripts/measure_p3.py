"""P3 — mesure temps/pic mémoire forward_train + backward (avant/après).

Configuration fixe (reproductible) :
- modèle Qwen3.5-0.8B bf16, batch 1, unit_block 4, resp_start 0 ;
- ids synthétiques déterministes (seed 0), réponse = séquence entière ;
- warmup 1 (sauf L=4096 : 0 si OOM/timeout), 2 répétitions mesurées ;
- GPU : synchronize + reset_peak_memory_stats avant chaque mesure ;
- sortie JSON : résultats bruts + résumé + environnement.

Usage :
  python scripts/measure_p3.py --out _artefacts/audit-p3.json [--lengths 128 512]
"""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def env_info():
    import transformers
    info = {
        "python": None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16",
        "batch": 1,
        "unit_block": 4,
        "resp_start": 0,
        "seed": 0,
    }
    import sys
    info["python"] = sys.version.split()[0]
    return info


def measure_one(runner, L, variant, warmup=1, reps=2):
    from jainius.model.twostream import TwoStreamRunner
    assert variant in ("optimized", "quadratic")
    fn = (runner.forward_train if variant == "optimized"
          else runner._forward_train_quadratic)
    torch.manual_seed(0)
    x0 = torch.randint(0, 1000, (1, L), device="cuda")
    xt = x0.clone()
    xt[0, L // 2:] = 248077
    for _ in range(warmup):
        h0, h1 = fn(x0, xt, unit_block=4, resp_start=0)
        loss = h1.square().mean()
        loss.backward()
        runner.model.zero_grad()
    times, peaks = [], []
    fwd_times, bwd_times = [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats("cuda")
        t0 = time.perf_counter()
        h0, h1 = fn(x0, xt, unit_block=4, resp_start=0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = h1.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3)
        times.append(t2 - t0)
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
        peaks.append(peak)
        runner.model.zero_grad()
    return {"times": times, "peaks_gb": peaks,
            "fwd_times": fwd_times, "bwd_times": bwd_times,
            "loss": float(loss.detach().item())}


def measure_train_step(student, teacher, optim, L, warmup=1, reps=2):
    """Coût complet d'un train step réel (runner + teacher + KL + optim)."""
    from types import SimpleNamespace
    from jainius.train.loop import train_step
    torch.manual_seed(0)
    pids = list(range(100, 116))
    traj = SimpleNamespace(x0=[100 + (j % 900) for j in range(L - len(pids))])
    for _ in range(warmup):
        train_step(student, teacher, optim, [pids], [traj],
                   runner=student.runner)
    times, peaks = [], []
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats("cuda")
        t0 = time.perf_counter()
        loss = train_step(student, teacher, optim, [pids], [traj],
                          runner=student.runner)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3)
        times.append(t1 - t0)
        peaks.append(peak)
    return {"times": times, "peaks_gb": peaks, "loss": loss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[128, 512, 1024, 4096])
    ap.add_argument("--variants", nargs="+",
                    default=["optimized", "quadratic"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--train-step", action="store_true",
                    help="mesure aussi un train_step réel (runner+teacher+KL)")
    args = ap.parse_args()
    from jainius.model.twostream import TwoStreamRunner
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=False).train()
    runner = TwoStreamRunner(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {"env": env_info(), "n_trainable_params": n_params,
           "n_layers": len(model.model.layers), "results": {}}
    for L in args.lengths:
        for variant in args.variants:
            key = f"L{L}/{variant}"
            try:
                # Garde : le quadratique à L>=1024 est prohibitif par
                # construction (1024+ recomputes du préfixe) — mesuré
                # uniquement sur demande explicite avec warmup 0.
                w = 0 if (L >= 1024 and variant == "quadratic") else args.warmup
                r = measure_one(runner, L, variant, warmup=w, reps=args.reps)
                out["results"][key] = {"status": "OK", **r}
                print(f"{key}: times={r['times']} peaks={r['peaks_gb']}",
                      flush=True)
            except Exception as e:  # OOM / timeout -> preuve brute
                out["results"][key] = {"status": "FAIL",
                                       "error": f"{type(e).__name__}: {e}"}
                print(f"{key}: FAIL {e}", flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    if args.train_step:
        import copy
        from jainius.distill.teacher import Teacher
        from jainius.model.arch import DiffusionSpec, JainiusModel
        # Enseignant = instance séparée (Teacher gèle ses paramètres ;
        # partager l'objet modèle gèlerait aussi l'étudiant).
        teacher = Teacher(copy.deepcopy(model).eval())
        student = JainiusModel(model, DiffusionSpec())
        student.runner = runner
        optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for L in args.lengths:
            key = f"L{L}/train_step"
            try:
                r = measure_train_step(student, teacher, optim, L,
                                       warmup=args.warmup, reps=args.reps)
                out["results"][key] = {"status": "OK", **r}
                print(f"{key}: times={r['times']} peaks={r['peaks_gb']}",
                      flush=True)
            except Exception as e:
                out["results"][key] = {"status": "FAIL",
                                       "error": f"{type(e).__name__}: {e}"}
                print(f"{key}: FAIL {e}", flush=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
