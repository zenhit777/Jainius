"""T-CKPT-01 — fidélité sauvegarde/reprise (F-008, mécanisme seul).

Protocole : jouet linéaire CPU déterministe, 1 pas SGD → save A ;
1 pas → save B ; recharge A dans un modèle+optimiseur frais → 1 pas ⇒
paramètres bit-exacts de B. Pas de boucle d'entraînement (M1-C micro).
"""

import pytest
import torch

from jainius.train.checkpoint import (
    ResumeMismatchError,
    load_ckpt,
    save_ckpt,
)
from jainius.train.master import MasterOptimizer

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
    with pytest.raises(ResumeMismatchError, match="configuration"):
        load_ckpt(dir_a, model, optim, {**CONFIG, "lr": 0.02})


def test_ckpt_master_roundtrip_and_fields(tmp_path):
    """§9.8 : maîtres + budget/curseurs + code_sha ; type dédié ;
    reprise bit-exacte avec maîtres."""
    torch.manual_seed(CONFIG["seed"])
    model = torch.nn.Linear(8, 4)
    opt = MasterOptimizer(model, lr=CONFIG["lr"])
    extra = {"curriculum_step": 3, "length": 128, "updates": 3,
             "budget": {"prompts": 48}}
    opt.zero_grad()
    _step(model, opt)
    dir_a = save_ckpt(tmp_path, model, opt.optim, torch.get_rng_state(), 3,
                      CONFIG, masters=opt.masters, extra=extra)
    manifest = (dir_a / "manifest.json").read_text()
    assert '"has_masters": true' in manifest and '"code_sha"' in manifest
    assert (dir_a / "budget.json").read_text().find('"curriculum_step": 3') >= 0
    params_a = _params(model)

    torch.manual_seed(CONFIG["seed"])
    model2 = torch.nn.Linear(8, 4)
    opt2 = MasterOptimizer(model2, lr=CONFIG["lr"])
    manifest = load_ckpt(dir_a, model2, opt2.optim, CONFIG,
                         masters=opt2.masters)
    assert manifest["step"] == 3
    for a, b in zip(_params(model2), params_a):
        assert torch.equal(a, b)
    for a, b in zip(opt2.masters, opt.masters):
        assert torch.equal(a, b), "maîtres non restaurés"
    _step(model2, opt2)
    _step(model, opt)
    for a, b in zip(_params(model2), _params(model)):
        assert torch.equal(a, b), "reprise maîtres non fidèle"


def test_ckpt_masters_missing_rejected(tmp_path):
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()  # SGD, sans maîtres
    dir_a = save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 1, CONFIG)
    model2 = torch.nn.Linear(8, 4)
    opt2 = MasterOptimizer(model2, lr=CONFIG["lr"])
    with pytest.raises(ResumeMismatchError, match="maîtres"):
        load_ckpt(dir_a, model2, opt2.optim, CONFIG, masters=opt2.masters)
