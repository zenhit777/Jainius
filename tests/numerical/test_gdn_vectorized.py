"""Étape D — parité de la vectorisation GDN (CPU, fp64 strict).

- `forward_clean` : nouveau `_core` vs `_core_legacy` (bit-exact attendu :
  même ordre d'opérations élément par élément), sorties + états + grads.
- `forward_noisy_block` : nouveau vs `_forward_noisy_block_legacy`
  (bit-exact attendu, einsum S̃_end inclus), sorties + états + grads.
- `_doc_positions` / `_doc_segments` : vs références Python brute-force
  (multi-documents, `past_doc`, cas limites), égalité exacte.
- `GDNRef` (indépendante, non vectorisée) : inchangée, fait toujours foi
  pour les équations (cf. `test_p2_gdn_grad.py`).

Critère : `torch.equal` (bit-exact). Tout écart, même d'1 ulp, est un
FAIL à investiguer — pas de `allclose` de confort ici.
"""

import torch

from jainius.model.gdn import GDNRefLayer
from jainius.model.twostream import TwoStreamRunner
from tests.numerical.test_p2_gdn_grad import _FakeGDN


def _inputs(B=2, L=6, D=32):
    torch.manual_seed(0)
    h = torch.randn(B, L, D, dtype=torch.float64) * 0.5
    return h + torch.arange(L, dtype=torch.float64).view(1, L, 1) * 0.1


def test_vectorized_core_bitexact_and_grads():
    ref = GDNRefLayer(_FakeGDN(), dtype=torch.float64)
    h = _inputs()
    mixed, z, beta, g = ref._project(h)
    B, L, _ = h.shape
    # Rejoue la préparation exacte de forward_clean (sans conv) :
    mixed_t = mixed.transpose(1, 2)
    from jainius.model.gdn import causal_short_conv_hist
    conv = causal_short_conv_hist(
        mixed_t, ref._conv_weight(mixed_t.dtype), hist=None)
    from jainius.model.gdn import silu
    conv = silu(conv).transpose(1, 2)
    q, k, v = torch.split(conv, [ref.kdim, ref.kdim, ref.vdim], -1)
    q = q.reshape(B, L, -1, ref.d_k)
    k = k.reshape(B, L, -1, ref.d_k)
    v = v.reshape(B, L, -1, ref.d_v)

    o_new, s_new = ref._core(q, k, v, g, beta)
    o_old, s_old = ref._core_legacy(q, k, v, g, beta)
    assert torch.equal(o_new, o_old), (
        f"_core : écart max {(o_new - o_old).abs().max().item():.3e}")
    assert torch.equal(s_new, s_old), "états _core divergents"

    # Gradients : mêmes d(q)/d(S0) des deux côtés.
    q2 = q.detach().clone().requires_grad_(True)
    s0 = torch.zeros(B, ref.n_v, ref.d_k, ref.d_v, dtype=torch.float64,
                     requires_grad=True)
    on, _ = ref._core(q2, k, v, g, beta, S0=s0)
    on.square().mean().backward()
    gq_new, gs_new = q2.grad.clone(), s0.grad.clone()
    q3 = q.detach().clone().requires_grad_(True)
    s0b = torch.zeros(B, ref.n_v, ref.d_k, ref.d_v, dtype=torch.float64,
                      requires_grad=True)
    # Entrées fraîches (pas de graphe partagé avec le premier backward).
    k3, v3, g3, b3 = (t.detach().clone() for t in (k, v, g, beta))
    oo, _ = ref._core_legacy(q3, k3, v3, g3, b3, S0=s0b)
    oo.square().mean().backward()
    assert torch.equal(gq_new, q3.grad), "grad q divergent"
    assert torch.equal(gs_new, s0b.grad), "grad S0 divergent"


def test_vectorized_noisy_block_bitexact_and_grads():
    ref = GDNRefLayer(_FakeGDN(), dtype=torch.float64)
    h = _inputs(B=2, L=8)
    with torch.no_grad():
        _, S_clean, conv_clean = ref.forward_clean(h)
    hb = _inputs(B=2, L=4) + 1.0
    S_init = S_clean.clone()
    o_new, s_new = ref.forward_noisy_block(hb, S_init, conv_clean)
    o_old, s_old = ref._forward_noisy_block_legacy(
        hb, S_init, conv_clean)
    assert torch.equal(o_new, o_old), (
        f"noisy : écart max {(o_new - o_old).abs().max().item():.3e}")
    assert torch.equal(s_new, s_old), "états noisy divergents"


def _brute_positions(doc, past_doc=None):
    B, L = doc.shape
    pos = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        if past_doc is not None and past_doc.shape[1] > 0:
            prev = past_doc[b, -1].item()
            run = 0
            for t in range(past_doc.shape[1] - 1, -1, -1):
                if past_doc[b, t].item() == doc[b, 0].item():
                    run += 1
                else:
                    break
            cur = run
        else:
            prev, cur = None, 0
        for t in range(L):
            d = doc[b, t].item()
            if prev is not None and d != prev:
                cur = 0
            pos[b, t] = cur
            cur += 1
            prev = d
    return pos


def _brute_segments(doc):
    B, L = doc.shape
    segs = []
    for b in range(B):
        d = doc[b].tolist()
        s = 0
        for e in range(1, L + 1):
            if e == L or d[e] != d[s]:
                segs.append((b, s, e))
                s = e
    return segs


def test_doc_positions_match_brute_force():
    runner = TwoStreamRunner.__new__(TwoStreamRunner)
    torch.manual_seed(0)
    cases = [
        (torch.zeros(2, 8, dtype=torch.long), None),
        (torch.tensor([[0, 0, 1, 1, 1, 0, 2, 2]]), None),
        (torch.tensor([[0, 0, 1, 1, 1, 0, 2, 2],
                        [3, 3, 3, 4, 4, 4, 4, 5]]), None),
        (torch.tensor([[1, 1, 2, 0]]),
         torch.tensor([[0, 0, 1, 1, 1, 1]])),  # prolonge run passé
        (torch.tensor([[2, 2, 2, 0]]),
         torch.tensor([[0, 0, 1, 1, 1, 1]])),  # nouveau doc
        (torch.tensor([[5]]), torch.tensor([[5, 5, 5]])),  # L=1 prolongé
        (torch.tensor([[6]]), torch.tensor([[5, 5, 5]])),  # L=1 nouveau
        (torch.zeros(1, 16, dtype=torch.long),
         torch.zeros(1, 16, dtype=torch.long)),  # passé uniforme
    ]
    for doc, past in cases:
        got = runner._doc_positions(doc, past)
        exp = _brute_positions(doc, past)
        assert torch.equal(got, exp), f"positions : {doc.tolist()} + {past}"


def test_doc_segments_match_brute_force():
    runner = TwoStreamRunner.__new__(TwoStreamRunner)
    torch.manual_seed(1)
    for shape in [(1, 1), (2, 8), (3, 16)]:
        doc = torch.randint(0, 3, shape)
        assert runner._doc_segments(doc) == _brute_segments(doc), shape
    assert runner._doc_segments(torch.zeros(2, 0, dtype=torch.long)) == []
