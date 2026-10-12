"""Construit le manifeste JAI-LOAD du checkpoint source réel (opération
explicite et traçable, P5.1) et le vérifie immédiatement.

Usage :
  python scripts/build_source_manifest.py --out docs/audits/<audit-id>/source_manifest.json
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from jainius.train.source_manifest import (
    build_manifest,
    verify_manifest,
    write_manifest,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=False).eval()
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    manifest = build_manifest(sd)
    manifest["model_id"] = MODEL_ID
    manifest["torch"] = torch.__version__
    write_manifest(manifest, args.out)
    ok = verify_manifest(manifest, sd)
    print(f"tenseurs : {manifest['n_tensors']}, vérifiés : {ok['checked']}")
    print(f"manifeste : {args.out}")


if __name__ == "__main__":
    main()
