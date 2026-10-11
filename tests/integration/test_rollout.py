"""T-ROLL-01/02 — trajectoires déterministes et reproductibles (F-004).

Sémantique démontrée (B3) : décodage CAUSAL par blocs (référence
d'avancement, pas Diffusion-Trust — le bidirectionnel intra-bloc exige
le contrat d'état GDN, M2).

T-ROLL-01 : même seed, mêmes poids ⇒ même trajectoire (x0, tours, états
d'entrée, version, seed, compteur de forwards).
T-ROLL-02 : dynamique avec τ > 1 ≡ statique 1 jeton/pas (à température 0).
"""

import torch

from jainius.decode.rollout import rollout
from jainius.decode.sampler import SamplerSpec
from jainius.model.arch import DiffusionSpec, JainiusModel
from jainius.train.loop import assert_traj_fresh, param_fingerprint
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


def test_rollout_causal_intra_block(model_bf16, tokenizer):
    """B3 : `denoise` est causal intra-bloc (pas Diffusion-Trust).

    Modifier le futur d'un bloc ([MASK, X, X, X] vs [MASK]×4) ne change
    PAS les logits de la position 0 — bit-exact. Spécifie le comportement
    réellement implémenté ; le bidirectionnel intra-bloc est interdit
    avant le contrat d'état GDN (M2, H-02 : la pile HF est causale par
    construction et le GDN ne suit aucun masque)."""
    batch = chat_ids(tokenizer, "Hi").to("cuda")
    model = JainiusModel(model_bf16, DiffusionSpec())
    mask_id = model.spec.mask_token_id
    with torch.no_grad():
        _, cache = model.prefill(batch["input_ids"])
        base = torch.full((1, 4), mask_id, device="cuda")
        future = base.clone()
        future[0, 1:] = 100
        l_base = model.denoise(base, cache)
        l_future = model.denoise(future, cache)
    assert torch.equal(l_base[0, 0], l_future[0, 0]), (
        "comportement non causal inattendu : revérifier B3")


def test_traj_weight_version_detects_stale(model_bf16, tokenizer):
    """B4.1 : modifier un paramètre invalide l'empreinte poids ; sans
    empreinte, la fraîcheur échoue aussi. Restauration bit-exacte après."""
    import pytest

    batch = chat_ids(tokenizer, "Hi").to("cuda")
    model = JainiusModel(model_bf16, DiffusionSpec())
    spec = SamplerSpec(mode="static", temperature=0.0, steps_per_block=1,
                       seed=0)
    with torch.no_grad():
        traj = rollout(model, batch["input_ids"][0].tolist(), 4, spec)
    assert traj.weight_version == "", "rollout() seul ne doit pas dater"
    with pytest.raises(ValueError, match="sans empreinte"):
        assert_traj_fresh(traj, model_bf16)
    traj.weight_version = param_fingerprint(model_bf16)
    assert_traj_fresh(traj, model_bf16)  # fraîche : ne lève pas
    param = next(model_bf16.parameters())
    saved = param.detach().clone()
    with torch.no_grad():
        param.add_(1.0)
    with pytest.raises(ValueError, match="obsolète"):
        assert_traj_fresh(traj, model_bf16)
    with torch.no_grad():
        param.copy_(saved)
    assert_traj_fresh(traj, model_bf16)  # restaurée : fraîche à nouveau
