"""Sauvegarde / reprise atomique, contrat §9.8 (JAI-TRAIN).

État d'entraînement (répertoire séparé) : paramètres, optimiseur, maîtres
FP32 (si `MasterOptimizer`), RNG CPU (+ CUDA si fourni), curriculum,
curseurs de données, compteurs de budget (`extra`). Empreintes (config,
code). Écriture atomique (tmp + rename).
`ResumeMismatchError` (sous-classe de `ValueError` par compatibilité)
distingue l'incompatibilité de checkpoint de toute erreur générique.

A5 : load_ckpt vérifie code_sha (refus journalisé sauf --force), relit
et expose budget.json, restaure RNG CUDA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ResumeMismatchError(ValueError):
    """Checkpoint incompatible avec la reprise demandée (config, état,
    code ou verrous différents). Distincte des erreurs génériques."""


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_root() -> str | None:
    """Racine du dépôt git courant (`--show-toplevel`), ou None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def code_sha() -> str:
    """SHA du commit courant (best-effort, `unknown` hors dépôt).
    A5 : résout d'abord la RACINE du dépôt (`--show-toplevel`) puis
    exécute `rev-parse HEAD` avec `cwd` = racine (et non le CWD du
    processus, qui peut être un sous-dossier ou un autre répertoire)."""
    root = _git_root()
    if root is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, cwd=root)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def code_dirty() -> bool:
    """A5 : indicateur « dirty » — l'arbre de travail contient des
    modifications non commitées (`git status --porcelain` non vide).
    `True` = le code exécuté diffère du commit référencé par `code_sha`.
    `False` hors dépôt ou en cas d'erreur (ne bloque jamais)."""
    root = _git_root()
    if root is None:
        return False
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, timeout=10, cwd=root)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


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
        "code_dirty": code_dirty(),  # A5 : indicateur « dirty »
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
            # J4-bis : SANS clone — torch.save sérialise vers CPU de toute
            # façon ; le clone (+)3,2 Go CUDA faisait OOM la sauvegarde
            # sous plafond 90 % (incident 16/10, step ~977). detach suffit
            # (données identiques, roundtrip vérifié par test_ckpt_*).
            torch.save([m.detach() for m in masters], tmp / "master.pt")
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
              masters: list | None = None, force: bool = False) -> dict:
    """Recharge et vérifie (lève `ResumeMismatchError` sinon). Restaure
    modèle, optimiseur, maîtres (si demandés ET présents), RNG CPU/CUDA.

    A5 : vérifie code_sha (refus journalisé sauf ``force=True``), relit
    et expose budget.json (curriculum, curseurs, compteurs de budget).

    Args:
        directory : répertoire du checkpoint.
        model : modèle à charger.
        optim : optimiseur à charger.
        config : configuration courante (comparée au hash sauvegardé).
        masters : liste des copies maîtres FP32 (si `MasterOptimizer`).
        force : si True, ignore l'incompatibilité de code_sha et journalise.

    Returns:
        dict avec clés : step, config_hash, code_sha, budget, torch_version,
        cuda_available, has_masters, has_rng_cuda.

    Raises:
        ResumeMismatchError : configuration ou code incompatible (sauf force).
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())

    # A5 : vérifier code_sha (refus journalisé sauf --force).
    current_sha = code_sha()
    if manifest["code_sha"] != current_sha:
        msg = (f"code_sha incompatible : checkpoint={manifest['code_sha'][:8]} "
               f"vs courant={current_sha[:8]}")
        if force:
            logger.warning("REPRISE FORCEE : %s", msg)
        else:
            raise ResumeMismatchError(msg)

    # A5 : indicateur « dirty » (information, pas un refus — le refus
    # porte sur code_sha ci-dessus). Un checkpoint OU l'arbre courant
    # enregistré depuis un arbre sale est journalisé explicitement.
    if manifest.get("code_dirty", False):
        logger.warning(
            "checkpoint enregistré depuis un arbre de travail SALE "
            "(code_sha=%s) : le code exécuté incluait des modifications "
            "non commitées", str(manifest["code_sha"])[:8])
    if code_dirty():
        logger.warning(
            "reprise avec un arbre de travail SALE : le code exécuté peut "
            "différer du commit courant (code_sha=%s)",
            str(current_sha)[:8])

    # Vérifier config.
    if manifest["config_hash"] != config_hash(config):
        raise ResumeMismatchError(
            "configuration différente "
            f"({manifest['config_hash'][:8]} vs {config_hash(config)[:8]})"
        )

    model.load_state_dict(
        torch.load(directory / "model.pt", weights_only=True))
    optim.load_state_dict(
        torch.load(directory / "optim.pt", weights_only=True))

    # A5 : restaurer maîtres FP32.
    if masters is not None:
        if not manifest.get("has_masters", False):
            raise ResumeMismatchError("maîtres FP32 demandés mais absents")
        saved = torch.load(directory / "master.pt", weights_only=True)
        if len(saved) != len(masters):
            raise ResumeMismatchError("maîtres incompatibles avec le modèle")
        for dst, src in zip(masters, saved):
            dst.detach().copy_(src)

    # A5 : restaurer RNG CPU.
    rng_state = torch.load(directory / "rng_cpu.pt", weights_only=True)
    torch.set_rng_state(rng_state)

    # A5 : restaurer RNG CUDA (si présent et disponible).
    if manifest.get("has_rng_cuda", False) and torch.cuda.is_available():
        cuda_states = torch.load(directory / "rng_cuda.pt", weights_only=True)
        torch.cuda.set_rng_state_all(cuda_states)

    # A5 : relire et exposer budget.json (curriculum, curseurs, compteurs).
    budget_path = directory / "budget.json"
    manifest["budget"] = (json.loads(budget_path.read_text())
                          if budget_path.exists() else {})

    return manifest
