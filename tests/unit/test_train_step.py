"""Clipping de norme 1.0 dans `train_step` (OPDLM `max_grad_norm`, CPU).

Fakes minimaux (pas de modèle) : le test verrouille le PLUMBING —
déclenchement (norme pré-clip > 1 journalisée), effet (norme post-clip
≈ min(brute, 1.0)), finitude. Direction préservée par construction
(`clip_grad_norm_` ne fait que renormaliser — asserté via le facteur
d'échelle exact).
"""

import math
from types import SimpleNamespace

import torch

from jainius.train.loop import train_step


class _FakeHead(torch.nn.Module):
    def __init__(self, dim=8, vocab=16, scale=1.0):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self.proj.weight.mul_(scale)

    def forward(self, h):
        return self.proj(h)


class _FakeStudent:
    def __init__(self, dim=8, gain=1.0):
        torch.manual_seed(1)
        self.spec = SimpleNamespace(block_size=4, mask_token_id=0)
        self.model = SimpleNamespace(
            lm_head=_FakeHead(dim, scale=100.0 * gain),
            parameters=lambda: list(self._params()),
        )
        self._params_list = list(self.model.lm_head.parameters()) + [
            torch.nn.Parameter(torch.randn(1, 12, dim) * 0.5 * gain)]
        self.model.parameters = lambda: iter(self._params_list)

    def forward_two_stream(self, batch):
        from jainius.model.arch import TwoStreamHidden
        h = self._params_list[-1]
        return TwoStreamHidden(h_x0=h.detach(), h_xt=h)


class _FixedLogitsHead(torch.nn.Module):
    """Logits déterministes (seed 3) — même conception que l'ancien fake
    (`teacher_logits_for_loss` → `randn(seed 3)`), adaptée au contrat
    `train_step` : `lm_head(hidden) → logits` (la cible est ensuite
    `roll(1)` + `log_softmax` dans `compute_teacher_targets`)."""

    def __init__(self, vocab=16):
        super().__init__()
        self.vocab = vocab

    def forward(self, h):
        torch.manual_seed(3)
        return torch.randn(h.shape[0], h.shape[1], self.vocab,
                           device=h.device)


class _FakeTeacherModel:
    """Contrat `train_step` (A1) : `teacher.model(ids,
    output_hidden_states=True).hidden_states[-1]` + `teacher.model.lm_head`."""

    def __init__(self, dim=8, vocab=16):
        self.dim = dim
        self.lm_head = _FixedLogitsHead(vocab)

    def __call__(self, ids, output_hidden_states=False):
        B, L = ids.shape
        return SimpleNamespace(
            hidden_states=(torch.zeros(B, L, self.dim, device=ids.device),))


class _FakeTeacher:
    def __init__(self, dim=8, vocab=16):
        self.model = _FakeTeacherModel(dim, vocab)


def _run_case(gain):
    torch.manual_seed(7)
    student = _FakeStudent(gain=gain)
    teacher = _FakeTeacher()
    optim = torch.optim.SGD(student.model.parameters(), lr=1e-3)
    traj = SimpleNamespace(x0=[1] * 8)
    stats: dict = {}
    loss = train_step(student, teacher, optim, [[2] * 4], [traj],
                      stats=stats)
    assert loss == loss and abs(loss) != float("inf")
    grads = torch.cat([p.grad.reshape(-1).float()
                       for p in student.model.parameters()
                       if p.grad is not None])
    return stats["grad_norm"], float(grads.norm().item())


def test_clip_triggers_and_scales():
    raw, post = _run_case(1.0)
    assert raw > 1.0, f"cas non discriminant (norme {raw})"
    assert abs(post - 1.0) < 1e-6, f"clip ineffectif : {post}"
    # Facteur d'échelle exact ⇒ direction préservée par construction.
    assert abs(post - min(raw, 1.0)) < 1e-6


def test_no_clip_below_threshold():
    raw, post = _run_case(1e-4)
    assert raw < 1.0, f"cas non discriminant (norme {raw})"
    assert abs(post - raw) / max(raw, 1e-12) < 1e-6, "clip abusif sous le seuil"
