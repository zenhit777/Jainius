"""Sauvegarde / reprise atomique (JAI-TRAIN, M1-C micro-test).

Mécanisme seul (pas de boucle d'entraînement) : écriture atomique
(répertoire temporaire + renommage), manifeste avec hash de configuration,
rechargement avec vérification. T-CKPT-01 en vérifie la fidélité :
reprise ⇒ pas d'optimisation identique (bit-exact, CPU déterministe).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def save_ckpt(directory: str | Path, model, optim, rng_state: torch.Tensor,
              step: int, config: dict) -> Path:
    """Sauvegarde atomique : état modèle + optimiseur + RNG + manifeste."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": step,
        "config_hash": config_hash(config),
        "torch_version": torch.__version__,
    }
    with tempfile.TemporaryDirectory(dir=directory.parent) as tmp:
        tmp = Path(tmp)
        torch.save(model.state_dict(), tmp / "model.pt")
        torch.save(optim.state_dict(), tmp / "optim.pt")
        torch.save(rng_state, tmp / "rng.pt")
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        target = directory / f"step_{step:06d}"
        if target.exists():
            raise FileExistsError(f"checkpoint existant : {target}")
        os.replace(tmp, target)
    return target


def load_ckpt(directory: str | Path, model, optim, config: dict) -> dict:
    """Recharge et vérifie le hash de configuration (refus sinon)."""
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest["config_hash"] != config_hash(config):
        raise ValueError(
            "ResumeMismatch : configuration différente "
            f"({manifest['config_hash'][:8]} vs {config_hash(config)[:8]})"
        )
    model.load_state_dict(torch.load(Path(directory) / "model.pt",
                                     weights_only=True))
    optim.load_state_dict(torch.load(Path(directory) / "optim.pt",
                                     weights_only=True))
    rng_state = torch.load(Path(directory) / "rng.pt", weights_only=True)
    torch.set_rng_state(rng_state)
    return manifest
