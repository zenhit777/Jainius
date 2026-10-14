"""J2 — convention `shifted` : layout serving, modes GDN, cibles.

Règle serving (décodage Diffusion-Trust à graine, validée J2 contre le
fork natif) : bloc [graine, MASK×(B-1)] ; le jeton en position i≥1
s'échantillonne depuis logits[i-1] ; la graine (position 0) n'est jamais
supervisée ; GDN mode 2 (graine causale, reste lu depuis S̃_end).
"""

import torch
import torch.nn.functional as F

from jainius.distill.targets import (
    compute_teacher_targets,
    compute_teacher_targets_shifted,
)
from jainius.model.gdn import GDNRefLayer
from jainius.model.twostream import TwoStreamRunner


def test_serving_mask_seed_row_causal():
    """Ligne graine causale (elle seule), autres lignes bidirectionnelles."""
    Bb, B = 3, 2
    doc = torch.zeros(B, Bb, dtype=torch.long)
    allow = TwoStreamRunner._serving_denoise_block_mask(Bb, doc, "cpu")
    assert allow.shape == (B, Bb, Bb)
    assert allow[0, 0].tolist() == [True, False, False]
    assert allow[0, 1].tolist() == [True, True, True]
    assert allow[0, 2].tolist() == [True, True, True]


def test_serving_mask_doc_isolation():
    """Deux documents : aucune visibilité inter-documents dans le bloc."""
    Bb = 3
    doc = torch.tensor([[0, 0, 0], [0, 0, 1]])
    allow = TwoStreamRunner._serving_denoise_block_mask(Bb, doc, "cpu")
    assert allow[1, 2].tolist() == [False, False, True]
    assert allow[1, 0].tolist() == [True, False, False]


def test_gdn_read_modes_match_naive():
    """Modes 0/1/2 de `forward_noisy_block` sur fausse couche float64 :
    mode2(nc=0) ≡ mode0, mode2(nc=T) ≡ mode1, mode1 ≡ déroulé causal."""
    from types import SimpleNamespace

    torch.manual_seed(0)
    B, T, D, H, K, Vd, C = 1, 4, 16, 2, 4, 6, 3
    mix_dim = H * (K + K + Vd)
    lin = lambda o, i: torch.nn.Linear(i, o, dtype=torch.float64)
    fake = SimpleNamespace(
        num_k_heads=H, num_v_heads=H, head_k_dim=K, head_v_dim=Vd,
        key_dim=H * K, value_dim=H * Vd, layer_norm_epsilon=1e-6,
        in_proj_qkv=lin(mix_dim, D), in_proj_z=lin(H * Vd, D),
        in_proj_b=lin(H, D), in_proj_a=lin(H, D),
        conv1d=torch.nn.Conv1d(mix_dim, mix_dim, C, groups=mix_dim,
                               dtype=torch.float64),
        conv_kernel_size=C, A_log=torch.randn(H, dtype=torch.float64),
        dt_bias=torch.randn(H, dtype=torch.float64),
        norm=torch.nn.LayerNorm(Vd, dtype=torch.float64),
        out_proj=lin(D, H * Vd),
    )
    cell = GDNRefLayer(fake, dtype=torch.float64)
    h = torch.randn(B, T, D, dtype=torch.float64)
    S0 = torch.zeros(B, H, K, Vd, dtype=torch.float64)
    C0 = torch.zeros(B, mix_dim, C - 1, dtype=torch.float64)
    o0, _ = cell.forward_noisy_block(h, S0, C0)
    o1, _ = cell.forward_noisy_block(h, S0, C0, read_mode=1)
    o2a, _ = cell.forward_noisy_block(h, S0, C0, read_mode=2, num_clean=0)
    o2b, _ = cell.forward_noisy_block(h, S0, C0, read_mode=2, num_clean=T)
    assert torch.equal(o2a, o0)  # nc=0 : tout depuis S_end
    assert torch.equal(o2b, o1)  # nc=T : tout causal
    assert not torch.equal(o0, o1)
    try:
        cell.forward_noisy_block(h, S0, C0, read_mode=7)
        raise AssertionError("read_mode=7 aurait dû lever")
    except ValueError:
        pass


def test_shifted_targets_unshifted():
    """Cibles shifted = log-softmax NON décalé (l'indice i-1 est pris
    des deux côtés) ; `current` garde son `roll(1)` (gelé, inchangé)."""
    torch.manual_seed(0)
    B, L, V = 1, 6, 32
    hidden = torch.randn(B, L, 8)
    W = torch.randn(V, 8)
    hf = lambda h: hidden
    lh = lambda h: h @ W.T
    t_sh = compute_teacher_targets_shifted(None, hf, lh)
    assert t_sh.shape == (B, L, V)
    assert torch.allclose(t_sh, F.log_softmax(hidden @ W.T, dim=-1))
    x0 = torch.zeros(B, L, dtype=torch.long)
    t_cur = compute_teacher_targets(x0, hf, lh)
    assert torch.allclose(
        t_cur, F.log_softmax((hidden @ W.T).roll(1, 1), dim=-1))


def test_denoise_shifted_no_write(model_bf16):
    """Bloc [graine, M, M] : logits [1, 3, V], état non muté (CUDA)."""
    from jainius.model.twostream import runner_state_fingerprint

    runner = TwoStreamRunner(model_bf16)
    dev = next(model_bf16.parameters()).device
    gstates, _ = runner.fresh_states(1, dev)
    from jainius.model.twostream import RunnerState
    state = RunnerState(gdn=gstates, attn_k=[], attn_v=[], docs=[])
    fp = runner_state_fingerprint(state)
    z = runner.denoise_block_shifted(
        state, torch.tensor([[10, 248077, 248077]], device=dev))
    assert z.shape[-1] > 1000 and z.shape[1] == 3
    assert runner_state_fingerprint(state) == fp
    assert torch.isfinite(z.float()).all()
