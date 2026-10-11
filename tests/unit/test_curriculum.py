"""T-CURR-01 — curriculum de longueur (F-007, sans entraînement).

Valeurs exactes aux pas 0, S_warm/2, S_warm, au-delà ; plafonds par domaine
par min() côté données (ici : vérifiés comme politique d'appel).
"""

import math

import pytest

from jainius.distill.curriculum import rollout_length


def test_curr_01_schedule_points():
    assert rollout_length(0) == 100
    # 100 + 3900*0.5 = 2050 ; round(512.5) = 512 (demi-pair) → 2048.
    assert rollout_length(50) == 2048
    assert rollout_length(100) == 4000
    assert rollout_length(101) == 4000
    assert rollout_length(10_000) == 4000


def test_curr_01_shapes():
    # sin et cos encadrent le linéaire à mi-rampe (concavité documentée).
    lin = rollout_length(50, shape="linear")
    assert rollout_length(50, shape="sin") > lin
    assert rollout_length(50, shape="cos") < lin
    assert rollout_length(0, shape="sin") == 100
    assert rollout_length(100, shape="cos") == 4000
    # Cohérence avec les formes OPDLM (rl.py) : valeur puis arrondi au bloc
    # (l'implémentation arrondit value/bloc, sans re-ajouter l_min).
    assert rollout_length(50) == round(2050 / 4) * 4 == 2048
    assert rollout_length(50, shape="sin") == round(
        (100 + 3900 * math.sin(math.pi / 4)) / 4
    ) * 4


def test_curr_01_caps_and_errors():
    caps = {"math": 4096, "code": 2048, "chat": 2048, "science": 1024}
    assert min(rollout_length(100), caps["code"]) == 2048
    assert min(rollout_length(100), caps["math"]) == 4000
    with pytest.raises(ValueError):
        rollout_length(-1)
    with pytest.raises(ValueError):
        rollout_length(0, warmup_rollouts=0)
    with pytest.raises(ValueError):
        rollout_length(0, shape="exp")
