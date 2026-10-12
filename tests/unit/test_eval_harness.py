"""Scoring du harnais sur fixtures synthétiques (P5.2, CPU).

Ne télécharge AUCUN dataset et n'exécute AUCUN benchmark (interdiction de
pilote) : vérifie les règles de scoring, le composite figé et le hash de
protocole. B0 = NOT RUN (cf. rapport).
"""

from jainius.eval.harness import (
    COMPOSITE,
    DATASETS,
    composite,
    protocol_hash,
    score_exact_match,
    score_pass_at_k,
)


def test_harness_covers_seven_benchmarks():
    assert set(DATASETS) == {"gsm8k", "math500", "mmlu", "mmlu_pro",
                             "ifeval", "humaneval", "mbpp"}


def test_exact_match_fixtures():
    assert score_exact_match(" 42 ", "42") == 1.0
    assert score_exact_match("43", "42") == 0.0


def test_pass_at_k_fixtures():
    assert score_pass_at_k([False, True], k=2) == 1.0
    assert score_pass_at_k([False, False], k=2) == 0.0


def test_composite_frozen_weights_and_missing():
    assert abs(sum(COMPOSITE["weights"].values()) - 1.0) < 1e-9
    full = {k: 1.0 for k in DATASETS}
    assert abs(composite(full)["composite"] - 1.0) < 1e-9
    assert composite(full)["missing"] == []
    partial = dict(full)
    partial["mmlu"] = None
    got = composite(partial)
    assert got["missing"] == ["mmlu"]
    assert got["composite"] < 1.0


def test_protocol_hash_stable():
    h1, h2 = protocol_hash(), protocol_hash()
    assert h1 == h2 and len(h1) == 64
