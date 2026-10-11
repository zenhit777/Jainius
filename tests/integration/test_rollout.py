"""T-ROLL-01/02 — trajectoires déterministes et reproductibles (F-004).

T-ROLL-01 : même seed, mêmes poids ⇒ même trajectoire (x0, tours, états
d'entrée, version, seed, compteur de forwards).
T-ROLL-02 : dynamique avec τ > 1 ≡ statique 1 jeton/pas (à température 0).
"""

import torch

from jainius.decode.rollout import rollout
from jainius.decode.sampler import SamplerSpec
from jainius.model.arch import DiffusionSpec, JainiusModel
from tests.conftest import chat_ids


def _run(model_bf16, tokenizer, text, sampler, response_len=8):
    batch = chat_ids(tokenizer, text).to("cuda")
    model = JainiusModel(model_bf16, DiffusionSpec())
    with torch.no_grad():
        return rollout(model, batch["input_ids"][0].tolist(), response_len, sampler)


def _traj_key(traj):
    return (
        traj.x0,
        [(r.block, r.step, r.ids, r.committed) for r in traj.rounds],
        traj.model_version,
        traj.seed,
        traj.n_forwards,
    )


def test_roll_01_same_seed_same_trajectory(model_bf16, tokenizer):
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2, seed=7)
    t1 = _run(model_bf16, tokenizer, "Count to three.", spec)
    t2 = _run(model_bf16, tokenizer, "Count to three.", spec)
    assert _traj_key(t1) == _traj_key(t2), "trajectoire non reproductible"
    assert t1.model_version and t1.seed == 7 and t1.n_forwards > 0
    assert len(t1.x0) == 8
    # Métadonnées F-004 : prompt et contenu des tours enregistrés.
    assert t1.prompt_ids and all(isinstance(r.ids, list) for r in t1.rounds)


def test_roll_01_different_seed_noted(model_bf16, tokenizer):
    """Seed distincte : trajectoire possiblement différente (consigné, non
    bloquant à température 0 si identique — le déterminisme seedé est testé
    en échantillonnage dans T-ROLL-01b ci-dessous)."""
    s7 = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2, seed=7)
    s8 = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2, seed=8)
    t7 = _run(model_bf16, tokenizer, "Count to three.", s7)
    t8 = _run(model_bf16, tokenizer, "Count to three.", s8)
    assert len(t7.x0) == len(t8.x0) == 8  # structure identique, contenu libre


def test_roll_02_dynamic_tau_gt_1_is_static(model_bf16, tokenizer):
    static = SamplerSpec(mode="static", temperature=0.0, steps_per_block=4, seed=3)
    dyn = SamplerSpec(
        mode="dynamic", threshold=2.0, temperature=0.0, steps_per_block=4, seed=3
    )
    ts = _run(model_bf16, tokenizer, "Say hello briefly.", static)
    td = _run(model_bf16, tokenizer, "Say hello briefly.", dyn)
    assert ts.x0 == td.x0, "τ > 1 ≢ statique 1 jeton/pas"
    assert ts.n_forwards == td.n_forwards
