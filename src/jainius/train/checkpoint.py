"""Sauvegarde / reprise atomique, contrat §9.8 (JAI-TRAIN).

État d'entraînement (répertoire séparé) : paramètres, optimiseur, maîtres
FP32 (si `MasterOptimizer`), RNG CPU (+ CUDA si fourni), pas de
curriculum / curseur de données / compteurs de budget (`extra`),
empreintes (config, code). Écriture atomique (tmp + rename).
`ResumeMismatchError` (sous-classe de `ValueError` par compatibilité)
distingue l'incompatibilité de checkpoint de toute erreur générique.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch


class ResumeMismatchError(ValueError):
    """Checkpoint incompatible avec la reprise demandée (config, état,
    code ou verrous différents). Distincte des erreurs génériques."""


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def code_sha() -> str:
    """SHA du commit courant (best-effort, `unknown` hors dépôt)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_ckpt(directory: str | Path, model, optim, rng_state: torch.Tensor,
              step: int, config: dict, *, masters=None,
              rng_cuda: torch.Tensor | None = None,
              extra: dict | None = None) -> Path:
    """Sauvegarde atomique : modèle + optimiseur (+ maîtres) + RNG CPU/CUDA
    + budget/curseurs (`extra`) + manifeste (hashes, versions)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": step,
        "config_hash": config_hash(config),
        "code_sha": code_sha(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "has_masters": masters is not None,
        "has_rng_cuda": rng_cuda is not None,
    }
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        tmp = Path(tmp)
        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(optim.state_dict(), tmp / "optim.pt")
        if masters is not None:
            torch.save([m.detach().clone() for m in masters], tmp / "master.pt")
        torch.save(rng_state, tmp / "rng_cpu.pt")
        if rng_cuda is not None:
            torch.save(rng_cuda, tmp / "rng_cuda.pt")
        (tmp / "budget.json").write_text(json.dumps(extra or {}, indent=2))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        target = directory / f"step_{step:06d}"
        if target.exists():
            raise FileExistsError(f"checkpoint existant : {target}")
        os.replace(tmp, target)
    return target


def load_ckpt(directory: str | Path, model, optim, config: dict, *,
              masters: list | None = None) -> dict:
    """Recharge et vérifie (lève `ResumeMismatchError` sinon). Restaure
    modèle, optimiseur, maîtres (si demandés ET présents), RNG CPU/CUDA."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["config_hash"] != config_hash(config):
        raise ResumeMismatchError(
            "configuration différente "
            f"({manifest['config_hash'][:8]} vs {config_hash(config)[:8]})"
        )
    model.load_state_dict(
        torch.load(directory / "model.pt", weights_only=True))
    optim.load_state_dict(
        torch.load(directory / "optim.pt", weights_only=True))
    if masters is not None:
        if not manifest.get("has_masters", False):
            raise ResumeMismatchError("maîtres FP32 demandés mais absents")
        saved = torch.load(directory / "master.pt", weights_only=True)
        if len(saved) != len(masters):
            raise ResumeMismatchError("maîtres incompatibles avec le modèle")
        for dst, src in zip(masters, saved):
            dst.detach().copy_(src)
    rng_state = torch.load(directory / "rng_cpu.pt", weights_only=True)
    torch.set_rng_state(rng_state)
    if manifest.get("has_rng_cuda", False) and torch.cuda.is_available():
        cuda_states = torch.load(directory / "rng_cuda.pt", weights_only=True)
        torch.cuda.set_rng_state_all(cuda_states)
    return manifest
