"""T-LOSS-01…05 — KL enseignant→étudiant sur cas synthétiques (M1-C).

T-LOSS-01 : vs implémentation naïve float64 ; KL ≥ 0 ; = 0 si égales.
T-LOSS-02 : découpage (chunks 1/7/1024) ≡ plein ; top_k=V ≡ None.
T-LOSS-03 : normalisation par ligne/batch, masquage, `-inf` sans NaN.
T-LOSS-04 : différences finies du gradient sur tête miniature.
T-LOSS-05 : parité code officiel OPDLM — SKIP (requiert l'env OPDLM).
Tout déterministe (seeds fixées), CPU ou GPU indifférent.
"""

import pytest
import torch
import torch.nn.functional as F

from jainius.distill.loss import (
    kl_teacher_to_student,
    kl_teacher_to_student_backward,
)

torch.manual_seed(11)


def _heads(dim=16, vocab=64):
    ws = torch.randn(vocab, dim) * 0.5
    wt = torch.randn(vocab, dim) * 0.5
    return (lambda h: h @ ws.T), (lambda h: h @ wt.T)


def _case(bsz=2, length=9, dim=16, vocab=64, density=0.5):
    hs = torch.randn(bsz, length, dim, requires_grad=True)
    with torch.no_grad():
        ht = torch.randn(bsz, length, dim)
    gen = torch.Generator().manual_seed(3)
    mask = torch.rand(bsz, length, generator=gen) < density
    mask[:, 0] = True  # au moins une position par ligne ici
    return hs, ht, mask


def _naive_kl(logits_t, logits_s):
    """Référence indépendante float64 (chemin distinct, sans chunks)."""
    pt = F.softmax(logits_t.double(), dim=-1)
    ls = F.log_softmax(logits_s.double(), dim=-1)
    lt = torch.log(pt)
    terms = torch.where(
        torch.isfinite(lt), pt * (lt - ls), torch.zeros_like(pt)
    ).sum(-1)
    return terms


def test_loss_01_vs_naive_and_properties():
    lm_s, lm_t = _heads()
    hs, ht, mask = _case()
    out = kl_teacher_to_student(hs, lm_s, ht, lm_t, mask, chunk_tokens=1024)
    ref = _naive_kl(lm_t(ht).detach(), lm_s(hs).detach()).reshape(2, -1)
    expected = torch.stack([ref[r][mask[r]].mean() for r in range(2)]).float()
    assert torch.allclose(out.per_row, expected, atol=1e-5), (
        "écart à la référence naïve"
    )
    assert (out.per_row >= 0).all() and out.loss.item() >= 0
    assert out.n_supervised == int(mask.sum())
    # Distributions égales (mêmes cachés, même tête) ⇒ KL = 0.
    out0 = kl_teacher_to_student(hs, lm_s, hs.detach(), lm_s, mask)
    assert out0.loss.item() == 0.0


def test_loss_02_chunk_and_topk_coherence():
    lm_s, lm_t = _heads()
    hs, ht, mask = _case()
    ref = kl_teacher_to_student(hs, lm_s, ht, lm_t, mask, chunk_tokens=1024)
    for chunk in (1, 7):
        got = kl_teacher_to_student(hs, lm_s, ht, lm_t, mask, chunk_tokens=chunk)
        # Tolérance float32 (et non égalité de bits) : le découpage change
        # les formes GEMM (mv vs mm) donc l'ordre d'arrondi — mesuré ~2e-7.
        assert torch.allclose(got.per_row, ref.per_row, atol=1e-6), (
            f"chunk {chunk} divergent"
        )
        assert abs(got.loss.item() - ref.loss.item()) <= 1e-6
    full_k = kl_teacher_to_student(hs, lm_s, ht, lm_t, mask, top_k=64)
    assert torch.allclose(full_k.per_row, ref.per_row, atol=1e-6), (
        "top_k=V ≢ vocabulaire complet"
    )
    sparse = kl_teacher_to_student(hs, lm_s, ht, lm_t, mask, top_k=16)
    assert torch.isfinite(sparse.loss) and sparse.loss.item() >= 0


def test_loss_03_masking_normalization_nonfinite():
    torch.manual_seed(5)
    dim, vocab = 8, 32
    ws = torch.randn(vocab, dim)
    wt = torch.randn(vocab, dim)
    # Enseignant avec -inf explicites (tokens impossibles).
    wt[0, :] = float("-inf")
    wt[1, :] = float("-inf")
    hs = torch.randn(2, 6, dim)
    ht = torch.randn(2, 6, dim)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, [1, 2, 4]] = True  # 3 positions ligne 0
    mask[1, [5]] = True  # 1 position ligne 1
    out = kl_teacher_to_student(
        hs, lambda h: h @ ws.T, ht, lambda h: h @ wt.T, mask
    )
    assert out.n_supervised == 4
    assert torch.isfinite(out.loss) and torch.isfinite(out.per_row).all()
    # Normalisation : moyenne par ligne (3 et 1 positions), puis batch.
    assert out.loss.item() == pytest.approx(out.per_row.mean().item())
    # Cas vide : perte 0, pas de crash.
    empty = torch.zeros(2, 6, dtype=torch.bool)
    out_e = kl_teacher_to_student(
        hs, lambda h: h @ ws.T, ht, lambda h: h @ wt.T, empty
    )
    assert out_e.n_supervised == 0 and out_e.loss.item() == 0.0


def test_loss_04_finite_difference_gradient():
    torch.manual_seed(9)
    dim, vocab = 6, 12
    ws = torch.randn(vocab, dim, requires_grad=True)
    wt = torch.randn(vocab, dim)
    hs = torch.randn(1, 4, dim, dtype=torch.float64, requires_grad=True)
    ht = torch.randn(1, 4, dim, dtype=torch.float64)
    ws64 = ws.detach().to(torch.float64).requires_grad_(True)
    mask = torch.tensor([[True, False, True, False]])
    out = kl_teacher_to_student(
        hs, lambda h: h @ ws64.T, ht, lambda h: h @ wt.double().T, mask
    )
    out.loss.backward()
    analytic = ws64.grad.clone()
    eps = 1e-5
    numeric = torch.zeros_like(ws64)
    with torch.no_grad():
        for i in range(vocab):
            for j in range(dim):
                plus, minus = ws64.clone(), ws64.clone()
                plus[i, j] += eps
                minus[i, j] -= eps
                lp = kl_teacher_to_student(
                    hs, lambda h, w=plus: h @ w.T,
                    ht, lambda h: h @ wt.double().T, mask,
                ).loss.item()
                lm = kl_teacher_to_student(
                    hs, lambda h, w=minus: h @ w.T,
                    ht, lambda h: h @ wt.double().T, mask,
                ).loss.item()
                numeric[i, j] = (lp - lm) / (2 * eps)
    denom = analytic.abs().max().item()
    rel = ((analytic - numeric).abs().max() / denom).item()
    assert rel <= 1e-4, f"gradient incohérent : err rel {rel}"


def test_loss_chunked_backward_equals_simple():
    """Égalité voie bornée ≡ voie simple : valeurs (allclose serré) ET
    gradients (allclose) sur cas jouet avec graphes. L'ordre d'accumulation
    diffère entre morceaux : égalité `allclose`, pas bit-exacte (documenté
    dans `loss.py`)."""
    torch.manual_seed(21)
    dim, vocab = 16, 64
    ws = torch.randn(vocab, dim) * 0.5
    wt = torch.randn(vocab, dim) * 0.5
    hs = torch.randn(2, 9, dim, requires_grad=True)
    ht = torch.randn(2, 9, dim)
    gen = torch.Generator().manual_seed(3)
    mask = torch.rand(2, 9, generator=gen) < 0.5
    mask[:, 0] = True
    ref = kl_teacher_to_student(hs, lambda h: h @ ws.T, ht,
                                lambda h: h @ wt.T, mask, chunk_tokens=1024)
    ref.loss.backward()
    grad_ref = hs.grad.clone()
    hs2 = hs.detach().clone().requires_grad_(True)
    got = kl_teacher_to_student_backward(
        hs2, lambda h: h @ ws.T, ht, lambda h: h @ wt.T, mask,
        chunk_tokens=3)
    assert not got.loss.requires_grad, "la perte bornée doit être détachée"
    assert torch.allclose(got.per_row, ref.per_row, atol=1e-6)
    assert abs(got.loss.item() - ref.loss.item()) <= 1e-6
    assert torch.allclose(hs2.grad, grad_ref, atol=1e-6), (
        "gradients voie bornée ≠ voie simple")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A4 mesure la mémoire GPU (device=cuda) — sauté sans CUDA (CI CPU)",
)
def test_loss_backward_memory_bounded_large():
    """A4 : kl_teacher_to_student_backward ne fait pas OOM à grande échelle.

    L=4096, 8 lignes, vocabulaire 248320 (Qwen3.5-0.8B). La voie simple
    ferait ~340 Go (mesuré : ~10,4 Mo/token supervisé × 4096 × 8 ≈ 340 Go).
    La voie bornée doit tenir dans la mémoire GPU disponible.

    Égalité allclose avec la voie simple sur un cas réduit (L=256) pour
    vérifier que le mécanisme de chunking est correct.
    Garde `skipif` CUDA : la CI CPU (`.github/workflows/cpu.yml`) exécute ce
    fichier — sans CUDA, le test est proprement sauté (pas un échec).
    """
    torch.manual_seed(42)
    dim, vocab = 1024, 248320  # Qwen3.5-0.8B dimensions
    ws = torch.randn(vocab, dim).to("cuda") * 0.01
    wt = torch.randn(vocab, dim).to("cuda") * 0.01

    # Cas réduit pour égalité (vocabulaire complet à 248320 serait trop lent).
    hs_small = torch.randn(8, 256, dim, requires_grad=True, device="cuda")
    ht_small = torch.randn(8, 256, dim, device="cuda")
    # CORRECTION A4 : `torch.rand(..., device="cuda")` exige un générateur
    # CUDA (l'ancien `torch.Generator()` CPU lève RuntimeError) — bug du
    # test, non de la perte.
    gen = torch.Generator(device="cuda").manual_seed(3)
    mask_small = torch.rand(8, 256, generator=gen, device="cuda") < 0.5
    mask_small[:, 0] = True

    ref = kl_teacher_to_student(hs_small, lambda h: h @ ws.T, ht_small,
                                lambda h: h @ wt.T, mask_small,
                                chunk_tokens=1024)
    ref.loss.backward()
    grad_ref = hs_small.grad.clone()
    hs_small2 = hs_small.detach().clone().requires_grad_(True)

    got = kl_teacher_to_student_backward(
        hs_small2, lambda h: h @ ws.T, ht_small,
        lambda h: h @ wt.T, mask_small, chunk_tokens=512)

    assert not got.loss.requires_grad, "la perte bornée doit être détachée"
    assert torch.allclose(got.per_row, ref.per_row, atol=1e-4), (
        f"écart per_row : max {(got.per_row - ref.per_row).abs().max()}")
    assert abs(got.loss.item() - ref.loss.item()) <= 1e-3
    assert torch.allclose(hs_small2.grad, grad_ref, atol=1e-4), (
        "gradients voie bornée ≠ voie simple à grande échelle")

    # Cas grand : L=4096, 8 lignes — vérifie qu'aucun OOM ne se produit.
    hs_large = torch.randn(8, 4096, dim, requires_grad=True, device="cuda")
    ht_large = torch.randn(8, 4096, dim, device="cuda")
    mask_large = torch.rand(8, 4096, generator=gen, device="cuda") < 0.5
    mask_large[:, 0] = True

    # Mesure A4 : pic mémoire de la voie BORNÉE (chunk_tokens=512).
    # La voie simple ferait ~340 Go (mesuré, cf. docstring) ; la voie
    # bornée doit rester bornée (<< 340 Go). On borne à 16 Go : largement
    # au-dessus du nécessaire, très en-dessous de la voie simple.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats("cuda")
    # NB : le backward est appliqué INTERNE à `kl_teacher_to_student_backward`
    # par morceau (loss.py:205) et la perte retournée est DÉTACHÉE — la borne
    # mémoire se joue donc pendant cet appel (pas à un `.backward()` ultérieur).
    got_large = kl_teacher_to_student_backward(
        hs_large, lambda h: h @ ws.T, ht_large,
        lambda h: h @ wt.T, mask_large, chunk_tokens=512)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated("cuda") / (1024 ** 3)
    assert torch.isfinite(got_large.loss), "perte non finie à grande échelle"
    assert peak < 16.0, (
        f"voie bornée non bornée : pic {peak:.2f} Go (attendu << 340 Go)")
    print(f"\n[A4] pic mémoire voie bornée (L=4096, 8 lignes) = {peak:.2f} Go")


@pytest.mark.skip(
    reason="BLOCKED : T-LOSS-05 requiert l'environnement OPDLM officiel "
    "(torch 2.6 + code `rl_sdar`, GPU) pour la parité mêmes-trajectoires."
)
def test_loss_05_official_parity():
    raise AssertionError("ne doit pas s'exécuter")
