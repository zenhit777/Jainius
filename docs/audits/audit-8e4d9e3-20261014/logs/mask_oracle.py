import sys, types
sys.path.insert(0, "/mnt/d/Intelligence_Artificielle/Projects/Jainius/_artefacts/oracles/OPDLM/train")
sys.path.insert(0, "/mnt/d/Intelligence_Artificielle/Projects/Jainius/src")
# Shims infra-only (jamais utilises par make_basic_block_attention : pur torch).
import importlib.machinery as _mach
for name in ("wandb", "deepspeed"):
    _m = types.ModuleType(name)
    _m.__spec__ = _mach.ModuleSpec(name, loader=None)
    _m.__version__ = "0.0-stub"
    sys.modules[name] = _m
import torch
import rl_sdar
main_code = rl_sdar.main.__code__
names = [c.co_name for c in main_code.co_consts if hasattr(c, "co_name")]
print("nested:", [n for n in names if "block" in n or "pad" in n or "process" in n])
mk = next(c for c in main_code.co_consts
          if getattr(c, "co_name", "") == "make_basic_block_attention")
ns = {"torch": torch}
official = types.FunctionType(mk, ns)
from jainius.distill.teacher import basic_block_attention as ours
# (L0, L1, block) : plusieurs formes, dont bords non multiples.
for L0, L1, b in [(10, 8, 4), (7, 8, 4), (16, 12, 4), (5, 4, 4), (24, 16, 8)]:
    N = L0 + 2 * L1
    got = official(N, L0, b)
    exp = ours(L0, L1, b, device="cpu")
    assert got.shape == exp.shape == (1, 1, N, N), (got.shape, exp.shape)
    same = bool((got == exp).all())
    diff = int((got != exp).sum())
    print(f"L0={L0} L1={L1} b={b}: {'BIT-EXACT' if same else f'DIFF {diff}'}")
    assert same, "port du masque non fidele"
print("MASK-ORACLE-PASS")
