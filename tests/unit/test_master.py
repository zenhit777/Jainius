"""B4.3 — poids maîtres FP32 : mises à jour réelles, reproductibles (CPU).

Le cas discriminant : modèle bf16 + lr=1e-5, où AdamW direct ne modifie
rien (ULP ≫ update). Avec maîtres, l'accumulation FP32 fonctionne même
quand la recopie bf16 arrondit — et le test l'exige explicitement au
lieu de se contenter de la finitude de la perte.
"""

import torch

from jainius.train.master import MasterOptimizer


def _toy(dtype=torch.bfloat16):
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4, dtype=dtype)
    torch.manual_seed(1234)
    x = torch.randn(16, 8, dtype=dtype)
    return model, x


def _grad_norm(model):
    return torch.stack(
        [p.grad.float().norm() for p in model.parameters() if p.grad is not None]
    ).sum().item()


def test_master_grads_finite_nonzero():
    model, x = _toy()
    opt = MasterOptimizer(model, lr=1e-5)
    opt.zero_grad()
    loss = model(x).square().mean()
    loss.backward()
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    assert _grad_norm(model) > 0, "gradients nuls : cas non discriminant"


def test_master_effective_update_and_reproducible():
    def run():
        model, x = _toy()
        opt = MasterOptimizer(model, lr=1e-2)
        before = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
        return model, before

    model, before = run()
    changed = [not torch.equal(a, b)
               for a, b in zip(before, model.parameters())]
    assert any(changed), "aucun paramètre modifié à lr=1e-2"
    model2, before2 = run()
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b), "pas d'optimisation non reproductible"


def test_master_accumulates_when_bf16_rounds():
    """À lr=1e-5, la recopie bf16 peut arrondir à zéro : les maîtres FP32
    doivent quand même accumuler (sinon l'entraînement est un no-op
    silencieux — le cas de l'audit)."""
    model, x = _toy()
    opt = MasterOptimizer(model, lr=1e-5)
    masters_before = [m.detach().clone() for m in opt.masters]
    opt.zero_grad()
    model(x).square().mean().backward()
    assert _grad_norm(model) > 0
    opt.step()
    assert any(not torch.equal(a, b)
               for a, b in zip(masters_before, opt.masters)), (
        "maîtres inchangés : accumulation FP32 défaillante")