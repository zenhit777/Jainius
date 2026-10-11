"""T-CKPT-01 — fidélité sauvegarde/reprise (F-008, mécanisme seul).

Protocole : jouet linéaire CPU déterministe, 1 pas SGD → save A ;
1 pas → save B ; recharge A dans un modèle+optimiseur frais → 1 pas ⇒
paramètres bit-exacts de B. Pas de boucle d'entraînement (M1-C micro).
"""

import torch

from jainius.train.checkpoint import load_ckpt, save_ckpt

CONFIG = {"lr": 0.01, "arch": "toy-linear-8x4", "seed": 0}


def _toy():
    torch.manual_seed(CONFIG["seed"])
    model = torch.nn.Linear(8, 4)
    optim = torch.optim.SGD(model.parameters(), lr=CONFIG["lr"])
    return model, optim


def _step(model, optim):
    torch.manual_seed(1234)  # batch synthétique fixe
    x = torch.randn(16, 8)
    loss = model(x).square().mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def test_ckpt_01_resume_bitexact(tmp_path):
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()
    _step(model, optim)
    dir_a = save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 1, CONFIG)
    _step(model, optim)
    save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 2, CONFIG)
    params_b = _params(model)

    torch.manual_seed(CONFIG["seed"])
    model2, optim2 = _toy()
    manifest = load_ckpt(dir_a, model2, optim2, CONFIG)
    assert manifest["step"] == 1
    _step(model2, optim2)
    for a, b in zip(_params(model2), params_b):
        assert torch.equal(a, b), "reprise non fidèle (F-008 violé)"


def test_ckpt_01_resume_mismatch_refused(tmp_path):
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()
    dir_a = save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 1, CONFIG)
    import pytest

    with pytest.raises(ValueError, match="ResumeMismatch"):
        load_ckpt(dir_a, model, optim, {**CONFIG, "lr": 0.02})
