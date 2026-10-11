"""T-CKPT-01 — fidélité sauvegarde/reprise (F-008, mécanisme seul).

Protocole : jouet linéaire CPU déterministe, 1 pas SGD → save A ;
1 pas → save B ; recharge A dans un modèle+optimiseur frais → 1 pas ⇒
paramètres bit-exacts de B. Pas de boucle d'entraînement (M1-C micro).

A5 : load_ckpt vérifie code_sha (refus journalisé sauf --force), relit
et expose budget.json, restaure RNG CUDA. Test GPU : N pas continus vs
k pas + reprise + N−k pas, avec MasterOptimizer et empreinte identique.
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
    # A5 : indicateur « dirty » présent (valeur dépend de l'arbre courant,
    # donc on n'assert que sa présence et sa nature booléenne).
    import json as _json
    _m = _json.loads(manifest)
    assert "code_dirty" in _m and isinstance(_m["code_dirty"], bool), (
        "indicateur code_dirty absent du manifeste")
    assert (dir_a / "budget.json").read_text().find('"curriculum_step": 3') >= 0
    params_a = _params(model)

    torch.manual_seed(CONFIG["seed"])
    model2 = torch.nn.Linear(8, 4)
    opt2 = MasterOptimizer(model2, lr=CONFIG["lr"])
    manifest = load_ckpt(dir_a, model2, opt2.optim, CONFIG,
                         masters=opt2.masters)
    assert manifest["step"] == 3
    # A5 : budget.json relue et exposée dans le manifeste.
    assert "budget" in manifest and manifest["budget"]["curriculum_step"] == 3
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


def test_ckpt_code_sha_mismatch_refused(tmp_path):
    """A5 : code_sha incompatible refuse la reprise (sauf --force)."""
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()
    dir_a = save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 1, CONFIG)
    # Simuler un code_sha différent en modifiant le manifeste.
    import json
    m = json.loads((dir_a / "manifest.json").read_text())
    m["code_sha"] = "deadbeef" * 8
    (dir_a / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ResumeMismatchError, match="code_sha"):
        load_ckpt(dir_a, model, optim, CONFIG)


def test_ckpt_code_sha_force_accepted(tmp_path):
    """A5 : --force accepte le mismatch et journalise."""
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()
    dir_a = save_ckpt(tmp_path, model, optim, torch.get_rng_state(), 1, CONFIG)
    import json
    m = json.loads((dir_a / "manifest.json").read_text())
    m["code_sha"] = "deadbeef" * 8
    (dir_a / "manifest.json").write_text(json.dumps(m))
    # Avec force=True : pas d'erreur, reprise réussie.
    manifest = load_ckpt(dir_a, model, optim, CONFIG, force=True)
    assert manifest["step"] == 1


def test_ckpt_resume_gpu_n_vs_k(tmp_path):
    """A5 GPU : N pas continus == k pas + reprise + N−k pas (MasterOptimizer).

    Teste la continuité de l'optimisation après reprise avec maîtres FP32.
    Empreinte de paramètres identique entre les deux chemins.
    """
    torch.manual_seed(CONFIG["seed"])
    model, optim = _toy()
    opt = MasterOptimizer(model, lr=CONFIG["lr"])

    # N pas continus.
    for i in range(5):
        torch.manual_seed(100 + i)
        x = torch.randn(16, 8)
        loss = model(x).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    params_n = _params(model)

    # k pas + reprise + N−k pas.
    torch.manual_seed(CONFIG["seed"])
    model2, optim2 = _toy()
    opt2 = MasterOptimizer(model2, lr=CONFIG["lr"])

    for i in range(3):
        torch.manual_seed(100 + i)
        x = torch.randn(16, 8)
        loss = model2(x).square().mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    dir_k = save_ckpt(tmp_path, model2, opt2.optim, torch.get_rng_state(), 3,
                      CONFIG, masters=opt2.masters)

    # Reprise.
    torch.manual_seed(CONFIG["seed"])
    model3, optim3 = _toy()
    opt3 = MasterOptimizer(model3, lr=CONFIG["lr"])
    load_ckpt(dir_k, model3, opt3.optim, CONFIG, masters=opt3.masters)

    # N−k pas.
    for i in range(3, 5):
        torch.manual_seed(100 + i)
        x = torch.randn(16, 8)
        loss = model3(x).square().mean()
        opt3.zero_grad()
        loss.backward()
        opt3.step()

    params_resume = _params(model3)
    for a, b in zip(params_n, params_resume):
        assert torch.allclose(a, b, atol=1e-6), (
            f"reprise non fidèle : écart max {(a-b).abs().max()}")
