"""B4.2 — échantillonnage device-aware (CPU + CUDA).

Couvre le crash constaté (`multinomial` + générateur CPU sur logits CUDA)
sans charger de modèle : logits synthétiques. Reproductibilité seedée,
différence température nulle/positive, validité des sorties.
"""

import pytest
import torch

from jainius.decode.sampler import SamplerSpec, sample_token


def _logits(device, peak=True):
    """Ligne 1D [V] comme `logits[i]` transmis par `rollout`."""
    base = torch.zeros(8, device=device)
    if peak:
        base[3] = 5.0
    return base


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sampler_seeded_reproducible(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA absent")
    spec = SamplerSpec(temperature=1.0, seed=0)
    a = sample_token(_logits(device),
                     spec, torch.Generator(device=device).manual_seed(0))
    b = sample_token(_logits(device),
                     spec, torch.Generator(device=device).manual_seed(0))
    assert a == b, "échantillonnage non reproductible à seed fixée"
    token, conf = a
    assert 0 <= token < 8 and 0.0 <= conf <= 1.0


def test_sampler_temp0_vs_temp_positive():
    spec0 = SamplerSpec(temperature=0.0, seed=0)
    spec1 = SamplerSpec(temperature=1.0, seed=0)
    rng = lambda: torch.Generator(device="cpu").manual_seed(0)
    t0, c0 = sample_token(_logits("cpu"), spec0, rng())
    assert t0 == 3
    assert c0 == pytest.approx(float(torch.softmax(
        _logits("cpu"), -1)[3].item()))
    seen = {sample_token(_logits("cpu", peak=False), spec1,
                         torch.Generator(device="cpu").manual_seed(s))[0]
            for s in range(20)}
    assert len(seen) > 1, "temp>0 n'explore pas (uniforme devrait varier)"


def test_sampler_wrong_device_generator_rejected():
    """Le contrat device est explicite : générateur CPU + logits CUDA
    lève (comportement torch documenté ici, pas masqué)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA absent")
    with pytest.raises(RuntimeError):
        sample_token(_logits("cuda"), SamplerSpec(temperature=1.0, seed=0),
                     torch.Generator(device="cpu").manual_seed(0))
