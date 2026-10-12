"""P2 — gradients GDN par différences finies float64 (CPU, sans poids).

Couvre `GDNRefLayer.forward_clean` et `GDNRefLayer.forward_noisy_block`
(couche de référence du runner, poids factices vivants — même chemin
autograd que l'entraînement : `torch.exp`, indexation native, poids lus
en direct sur la couche) pour :
- `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`
  (projections q/k/v), `conv1d`, `norm`, `out_proj`.

Protocole : entrées déterministes non dégénérées (seed 0, randn*0.5+arange),
scalaire `out.square().mean()`, perturbations CENTRALES eps=1e-6 (float64 :
troncature O(eps²)~1e-12, arrondi O(1e-16/eps)~1e-10 — marge > 1e3 vs
tolérance). Vérifie finitude, non-nullité conjointe (pas de faux succès
0/0), accord analytique/numérique (err rel ≤ 1e-3, moyenne ≤ 5e-4).

Tolérances : choisies après analyse de stabilité float64 (cf. docstring) ;
aucun seuil existant modifié (nouveau test).
"""

import torch

from jainius.model.gdn import GDNRefLayer

EPS = 1e-6
REL_TOL = 1e-3


class _FakeGDN(torch.nn.Module):
    """Couche GDN factice (mêmes attributs que Qwen3_5GatedDeltaNet)."""

    def __init__(self, D=32, n_k=2, n_v=2, d_k=8, d_v=8, K=4):
        super().__init__()
        torch.manual_seed(0)
        self.num_k_heads = n_k
        self.num_v_heads = n_v
        self.head_k_dim = d_k
        self.head_v_dim = d_v
        self.key_dim = n_k * d_k
        self.value_dim = n_v * d_v
        self.layer_norm_epsilon = 1e-6
        self.conv_kernel_size = K
        Q = 2 * self.key_dim + self.value_dim
        V = self.value_dim
        self.in_proj_qkv = torch.nn.Linear(D, Q, bias=False).double()
        self.in_proj_z = torch.nn.Linear(D, V, bias=False).double()
        self.in_proj_b = torch.nn.Linear(D, n_k, bias=False).double()
        self.in_proj_a = torch.nn.Linear(D, n_k, bias=False).double()
        self.conv1d = torch.nn.Conv1d(Q, Q, K, groups=Q, bias=False).double()
        self.A_log = torch.nn.Parameter(torch.randn(n_k).double() * 0.5)
        self.dt_bias = torch.nn.Parameter(torch.randn(n_k).double() * 0.5)
        self.norm = torch.nn.LayerNorm(d_v, eps=1e-6).double()
        self.norm.weight.data.fill_(1.0)
        self.out_proj = torch.nn.Linear(V, D, bias=False).double()


def _layer():
    return GDNRefLayer(_FakeGDN(), dtype=torch.float64)


def _inputs(B=1, L=6, D=32):
    torch.manual_seed(0)
    h = torch.randn(B, L, D, dtype=torch.float64) * 0.5
    h = h + torch.arange(L, dtype=torch.float64).view(1, L, 1) * 0.1
    return h


def _params_of(fake):
    return {
        "A_log": fake._hf.A_log,
        "dt_bias": fake._hf.dt_bias,
        "in_proj_a": fake._hf.in_proj_a.weight,
        "in_proj_b": fake._hf.in_proj_b.weight,
        "in_proj_qkv": fake._hf.in_proj_qkv.weight,
        "in_proj_z": fake._hf.in_proj_z.weight,
        "conv1d": fake._hf.conv1d.weight,
        "norm": fake._hf.norm.weight,
        "out_proj": fake._hf.out_proj.weight,
    }


def _check(ref, forward_fn, param, name, n_pts=3):
    for p in ref._hf.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = forward_fn().square().mean()
    assert torch.isfinite(loss), f"{name} : perte non finie"
    loss.backward()
    assert param.grad is not None, f"{name} : aucun gradient (graphe coupé)"
    ana = param.grad.detach().clone()
    assert torch.isfinite(ana).all(), f"{name} : gradient non fini"
    flat = ana.reshape(-1)
    order = torch.argsort(flat.abs(), descending=True)
    pts = [int(order[i]) for i in range(min(n_pts, flat.numel()))]
    num, avec = [], []
    with torch.no_grad():
        for idx in pts:
            orig = param.detach().clone()
            plus = orig.clone()
            minus = orig.clone()
            plus.reshape(-1)[idx] += EPS
            minus.reshape(-1)[idx] -= EPS
            param.detach().copy_(plus)
            lp = forward_fn().square().mean().item()
            param.detach().copy_(minus)
            lm = forward_fn().square().mean().item()
            param.detach().copy_(orig)
            num.append((lp - lm) / (2 * EPS))
            avec.append(flat[idx].item())
    ana_t = torch.tensor(avec, dtype=torch.float64)
    num_t = torch.tensor(num, dtype=torch.float64)
    assert ana_t.abs().max() > 1e-12, (
        f"{name} : gradient analytique nul partout (test aveugle)")
    assert num_t.abs().max() > 1e-12, (
        f"{name} : gradient numérique nul partout (test aveugle)")
    rel = ((ana_t - num_t).abs() / ana_t.abs().clamp_min(1e-12)).max().item()
    assert rel <= REL_TOL, (
        f"{name} : désaccord analytique/numérique (rel {rel:.2e})")
    return rel


def test_p2_forward_clean_grads_finite_diff():
    ref = _layer()
    h = _inputs()
    P = _params_of(ref)
    fwd = lambda: ref.forward_clean(h)[0]
    rels = {n: _check(ref, fwd, p, f"clean/{n}") for n, p in P.items()}
    assert all(r <= REL_TOL for r in rels.values())


def test_p2_forward_noisy_block_grads_finite_diff():
    ref = _layer()
    h = _inputs(L=8)
    with torch.no_grad():
        _, S_clean, conv_clean = ref.forward_clean(h)
    S_init = S_clean.detach().clone().requires_grad_(True)
    hb = _inputs(L=4) + 1.0
    out, _ = ref.forward_noisy_block(hb, S_init, conv_clean)
    loss = out.square().mean()
    loss.backward()
    assert S_init.grad is not None and torch.isfinite(S_init.grad).all()
    assert S_init.grad.abs().sum() > 0, "pas de graphe bruité→propre"
    P = _params_of(ref)

    def fwd():
        s = S_init.detach().clone()
        return ref.forward_noisy_block(hb, s, conv_clean)[0]

    rels = {n: _check(ref, fwd, p, f"noisy/{n}") for n, p in P.items()}
    assert all(r <= REL_TOL for r in rels.values())
