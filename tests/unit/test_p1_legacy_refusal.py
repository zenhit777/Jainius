"""P1 — refus explicite du chemin `legacy_leaky` (CPU, sans poids).

Verrouille :
- `JainiusModel.forward_two_stream` lève `LegacyLeakyError` par défaut ;
- `legacy_leaky_forward` / `allow_legacy_leaky=True` restent accessibles
  (voie de caractérisation isolée) ;
- `train_step` sans runner lève `LegacyLeakyError` sauf
  `allow_legacy_leaky=True` (plumbing unitaire).
"""

from types import SimpleNamespace

import pytest
import torch

from jainius.model.arch import (
    DiffusionSpec,
    JainiusModel,
    LegacyLeakyError,
    TwoStreamBatch,
)
from jainius.train.loop import train_step


class _FakeHF(torch.nn.Module):
    def __init__(self, dim=8, layers=2):
        super().__init__()
        self.dtype = torch.float32
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, ids, attention_mask=None, position_ids=None,
                output_hidden_states=False):
        B, L = ids.shape
        h = torch.zeros(B, L, 8)
        ns = SimpleNamespace(hidden_states=(h,))
        return ns


def _batch():
    x0 = torch.tensor([[1, 2, 3, 4]])
    xt = torch.tensor([[1, 2, 9, 9]])
    sup = torch.tensor([[False, False, True, True]])
    return TwoStreamBatch(x0=x0, xt=xt, sup_mask=sup)


def test_forward_two_stream_refused_by_default():
    student = JainiusModel(_FakeHF(), DiffusionSpec())
    with pytest.raises(LegacyLeakyError):
        student.forward_two_stream(_batch())


def test_legacy_leaky_explicit_characterization():
    student = JainiusModel(_FakeHF(), DiffusionSpec())
    out = student.forward_two_stream(_batch(), allow_legacy_leaky=True)
    assert out.h_x0.shape == (1, 4, 8)
    out2 = student.legacy_leaky_forward(_batch())
    assert torch.equal(out.h_xt, out2.h_xt)


def test_train_step_refuses_legacy_without_runner():
    from tests.unit.test_train_step import (
        _FakeStudent,
        _FakeTeacher,
    )
    from types import SimpleNamespace as SN

    torch.manual_seed(0)
    student = _FakeStudent()
    teacher = _FakeTeacher()
    optim = torch.optim.SGD(student.model.parameters(), lr=1e-3)
    traj = SN(x0=[1] * 8)
    with pytest.raises(LegacyLeakyError):
        train_step(student, teacher, optim, [[2] * 4], [traj])
    # Voie explicite (plumbing) : OK.
    stats: dict = {}
    loss = train_step(student, teacher, optim, [[2] * 4], [traj],
                      stats=stats, allow_legacy_leaky=True)
    assert loss == loss and "grad_norm" in stats
