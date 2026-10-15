"""Remédiation J4-bis : prompts formatés (prompts seuls), compteurs cumulatifs,
sauvegarde finale conditionnelle, gouvernance mémoire.

Contexte (audit J4) : le checkpoint J4 ne produit jamais le marqueur final
``####`` (0/20) car la distillation sur prompts seuls ne supervise jamais la
convention de clôture, alors que l'évaluation P-G/1.0 l'exige (gsm8k, math500).
Remédiation acceptée : enrichir les prompts seuls avec des consignes de
format explicites, TOUJOURS sans réponses de référence.

Garantie « prompts seuls » : `build_training_text` ne prend que
(question, domain). `ground_truth_answer` / `tests_json` ne sont jamais lus
(aucun paramètre ne les reçoit).
"""

from __future__ import annotations

import os

# Version de la remédiation (tracée dans config.json du pilote).
FORMAT_VERSION = "v1-domain"

# Consignes miroirs des SYSTEM_PROMPTS d'évaluation (b0_runner), par domaine
# OPDLM. Formulation volontairement proche du protocole P-G/1.0 pour un
# transfert train -> éval maximal.
FORMAT_SUFFIX = {
    "math": (
        "\n\nShow your reasoning step by step. "
        "Conclude with the final number after #### "
        "and put the final answer in \\boxed{}."
    ),
    "code": (
        "\n\nComplete the task with code only. "
        "Output only the code, no explanations."
    ),
    "science": (
        "\n\nReason step by step. "
        "If the question is multiple-choice, respond with a single letter "
        "and nothing else. "
        "Otherwise conclude with the final result after ####."
    ),
    "chat": (
        "\n\nFollow every instruction in the prompt exactly "
        "and answer concisely."
    ),
}

DEFAULT_SUFFIX = FORMAT_SUFFIX["chat"]

BUDGET_KEYS = ("tokens_response", "tokens_supervised", "tokens_seq", "updates")


def build_training_text(question: str, domain: str | None) -> str:
    """Texte d'entraînement = question + consigne de format du domaine.

    Ne reçoit jamais la réponse de référence (signature restreinte).
    Domaine inconnu / absent -> consigne générique (chat).
    """
    suffix = FORMAT_SUFFIX.get(domain or "", DEFAULT_SUFFIX)
    return f"{question}{suffix}"


def merge_budgets(resumed: dict, fresh: dict | None = None) -> dict:
    """Compteurs cumulatifs : repris + session courante (défaut 0).

    Corrige le défaut J4 (tokens_sup/tokens_seq/updates remis à zéro à
    chaque reprise ; seul tokens_response était cumulé).
    """
    out = {}
    for k in BUDGET_KEYS:
        out[k] = int(resumed.get(k, 0)) + int((fresh or {}).get(k, 0))
    return out


def should_final_save(last_saved_step: int | None, last_step: int) -> bool:
    """Sauvegarde finale conditionnelle : oui sauf si le dernier pas est
    déjà sauvegardé par la boucle (corrige le FileExistsError J4 quand
    steps % ckpt_every == 0 ; sortie code 1, checkpoint intact).
    """
    return last_saved_step != last_step


def apply_memory_governance() -> dict:
    """Gouvernance mémoire (docs/environnement.md §9).

    - expandable_segments doit être actif (via env avant init CUDA) ;
    - plafonnement à ~90 % : l'échec devient une OOM franche au lieu d'une
      pagination WDDM silencieuse ;
    - retourne les réglages appliqués (tracés au log).
    """
    import torch

    applied = {
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "fraction": None,
    }
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)
        applied["fraction"] = 0.90
    return applied


def mem_snapshot() -> dict:
    """Compteurs allocateur CUDA (alloué vs réservé) pour le log par pas."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "mem_alloc_MiB": round(torch.cuda.memory_allocated() / 2**20, 1),
        "mem_reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1),
        "mem_max_alloc_MiB": round(
            torch.cuda.max_memory_allocated() / 2**20, 1),
    }
