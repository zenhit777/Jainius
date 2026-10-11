"""T-GEN-01 — génération complète de référence (F-009).

Critères structurels uniquement (aucune exigence de qualité : le modèle
source n'a jamais vu le jeton MASK) : séquence complète, arrêt EOS ou
plafond `max_new_tokens`, texte décodable valide, statistiques présentes.
"""

import torch

from jainius.decode.rollout import generate
from jainius.decode.sampler import SamplerSpec
from jainius.model.arch import DiffusionSpec, JainiusModel


def test_gen_01_complete_generation(model_bf16, tokenizer):
    model = JainiusModel(model_bf16, DiffusionSpec())
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=2, seed=0)
    with torch.no_grad():
        text, traj, stats = generate(model, tokenizer, "Say hello.", spec,
                                     max_new_tokens=8)
    assert isinstance(text, str)
    # Arrêt EOS ou plafond : longueur ≤ max, pas de MASK résiduel.
    assert 0 < len(traj.x0) <= 8
    assert model.spec.mask_token_id not in traj.x0
    if traj.eos_stop:
        assert traj.x0[-1] == tokenizer.eos_token_id
    else:
        assert len(traj.x0) == 8
    assert stats["out_len"] == len(traj.x0) and stats["n_forwards"] > 0
    assert stats["n_rounds"] == len(traj.rounds)
    assert isinstance(stats["eos_stop"], bool)
    # Les ids générés sont dans le vocabulaire modèle.
    assert all(0 <= t < 248320 for t in traj.x0)
