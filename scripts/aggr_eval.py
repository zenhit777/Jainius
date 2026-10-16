"""Agrège les évals J4-bis par dataset en un composite (même formule que
local_runner.evaluate : poids COMPOSITE uniformes, sans-IFEval renormalisé).

Usage :
  python scripts/aggr_eval.py --ins _artefacts/j4b/eval_j4b_gsm8k.json ... \\
      --out _artefacts/j4b/eval_j4b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jainius.eval.harness import COMPOSITE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Agrégation composite J4-bis")
    ap.add_argument("--ins", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    per: dict = {}
    proto = None
    for f in args.ins:
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        proto = proto or r.get("protocol")
        for name, d in r["datasets"].items():
            per[name] = d
    weights = COMPOSITE["weights"]
    names = [n for n in weights if n in per]
    composite = sum(per[n]["mean"] * weights[n] for n in names)
    w6 = {k: v for k, v in weights.items() if k != "ifeval"}
    s6 = sum(w6.values())
    no_ifeval = sum(per[n]["mean"] * w6[n] / s6 for n in names if n != "ifeval")
    out = {"protocol": proto, "engine": "local-HF (J4-bis, par dataset)",
           "datasets": per, "composite": composite,
           "composite_no_ifeval": no_ifeval,
           "reference_B0_no_ifeval": 0.3376,
           "palier1": 0.2363, "palier2": 0.3038}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"composite = {composite:.4f} (sans IFEval : {no_ifeval:.4f})")
    for n in names:
        print(f"  {n}: {per[n]['mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
