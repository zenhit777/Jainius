"""Harnais d'évaluation minimal (P5.2, protocole P-G).

Couvre : GSM8K, MATH-500, MMLU, MMLU-Pro, IFEval, HumanEval, MBPP — par des
DÉFINITIONS versionnées (ids, versions, splits, prompts, politique de
génération, budgets, scoring, seeds, répétitions, environnement) et un
protocole P-G figé + hash. AUCUN benchmark complet n'est exécuté ici
(interdiction de pilote, règle absolue n°1) : le scoring est testé sur
fixtures synthétiques ; la mesure B0 est classée NOT RUN.

Le composite (définition, pondérations, normalisation, règle des scores
manquants, jeu de développement) est figé AVANT toute mesure — les
pondérations ci-dessous sont ARBITRAIRES et documentées comme telles
(choisies sans observation de résultats : aucune mesure n'existe).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

PROTOCOL_VERSION = "P-G/1.0"

# Identifiants et versions exactes (figés ; datasets NON téléchargés ici).
DATASETS = {
    "gsm8k": {"hf_id": "openai/gsm8k", "config": "main",
              "version": "1.0", "split": "test"},
    "math500": {"hf_id": "HuggingFaceH4/MATH-500", "version": "1.0",
                "split": "test"},
    "mmlu": {"hf_id": "cais/mmlu", "version": "1.0", "split": "test"},
    "mmlu_pro": {"hf_id": "TIGER-Lab/MMLU-Pro", "version": "1.0",
                 "split": "test"},
    "ifeval": {"hf_id": "google/IFEval", "version": "1.0",
               "split": "train"},
    "humaneval": {"hf_id": "openai/openai_humaneval", "version": "1.0",
                  "split": "test"},
    "mbpp": {"hf_id": "google-research-datasets/mbpp", "config": "full",
             "version": "1.0", "split": "test"},
}

# Politique de génération (figée).
GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512,
              "stop": ["<|im_end|>"], "n_repetitions": 3,
              "seeds": [0, 1, 2]}

# Composite (figé avant mesure ; pondérations égales documentées comme
# arbitraires-sans-observation, normalisation z_params absents -> moyenne
# simple ; scores manquants = 0.0 + drapeau `missing`).
COMPOSITE = {"weights": {k: 1.0 / len(DATASETS) for k in DATASETS},
             "normalization": "none (scores natifs 0..1)",
             "missing_rule": "score 0.0 + missing=True",
             "dev_set": "aucun (pas de sélection de checkpoint avant M3)"}


@dataclass(frozen=True)
class Protocol:
    version: str = PROTOCOL_VERSION
    chat_template: str = "qwen3.5-instruct (enable_thinking=False)"
    invalid_policy: str = "réponse invalide/erreur = score 0 + comptée"
    environment: str = "cf. requirements-lock.txt + source_manifest"


def protocol_hash() -> str:
    payload = json.dumps({"datasets": DATASETS, "generation": GENERATION,
                          "composite": COMPOSITE,
                          "protocol": asdict(Protocol())},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def score_exact_match(prediction: str, reference: str) -> float:
    """Scoring exact-match normalisé (GSM8K/MATH : dernière valeur)."""
    norm = lambda s: " ".join(str(s).strip().split())
    return 1.0 if norm(prediction) == norm(reference) else 0.0


def score_pass_at_k(passed: list[bool], k: int = 1) -> float:
    """HumanEval/MBPP : 1.0 si au moins une variante passe."""
    return 1.0 if any(passed[:k]) else 0.0


def composite(scores: dict[str, float | None]) -> dict:
    """Agrégat figé (NE PAS ajuster après observation)."""
    total, missing = 0.0, []
    for name, w in COMPOSITE["weights"].items():
        s = scores.get(name)
        if s is None:
            missing.append(name)
            s = 0.0
        total += w * s
    return {"composite": total, "missing": missing,
            "protocol_hash": protocol_hash()}
