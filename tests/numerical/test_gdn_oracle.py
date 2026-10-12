"""T-GDN-03 — parité référence deux-flux vs noyau FLARE officiel (oracle).

Protocole (cahier §13 T-GDN-03) : référence deux-flux (récurrence naïve
`gdn_run` + lecture de fin de bloc, EXACTEMENT
`torch_recurrent_gated_delta_rule`) vs noyau officiel FLARE sur entrées
aléatoires :
- oracle PRINCIPAL : `ChunkBlockCausalGatedDeltaRuleFunction` (chemin
  d'entraînement WY, `causal_mode=0` = lecture fin de bloc = sémantique
  S̃_end B0) ;
- garde SECONDAIRE : `fused_recurrent_block_causal_gated_delta_rule`
  `causal_mode=1` (token-causal) vs récurrence causale naïve.

Environnement dédié requis : torch 2.10.0+cu128 + fla 0.5.1 + triton
(voir `setup_envs_models_qwen35.sh` de FLARE). Le checkout FLARE est
importé EN LECTURE SEULE depuis `_artefacts/oracles/FLARE` (oracle
externe, M0-D2 — aucun code copié dans `src/`, aucun vendoring).
SKIP propre si le checkout est absent.

Critère (défini avant exécution, aucun seuil préexistant modifié —
T-GDN-03 n'a jamais été exécuté) : err relative max ≤ 1e-3 ET err
absolue max ≤ 1e-3 (fp32 ; un écart de sémantique donne O(1)).

Constat documenté (2026-10-13, RTX 5090) : le helper d'inférence
`fused_recurrent_block_causal_gated_delta_rule` en `causal_mode=0` NE
reproduit PAS la lecture de fin de bloc (écart O(1) ~0.2-0.3) alors que
son état final est exact (6e-8) et que son `causal_mode=1` est exact
(2e-8) — anomalie côté oracle d'inférence, SANS EFFET sur le chemin
d'entraînement (chunk WY) qui fait foi ici.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FLARE = REPO / "_artefacts" / "oracles" / "FLARE"

flare_available = FLARE.is_dir()
if flare_available:
    sys.path.insert(0, str(FLARE))

requires_flare = pytest.mark.skipif(
    not flare_available, reason="BLOCKED : checkout FLARE absent")


@requires_flare
def test_gdn_03_oracle_kernel_parity():
    from torchtitan.models.qwen3_5.model.block_gated_delta_rule.chunk import (
        ChunkBlockCausalGatedDeltaRuleFunction as Oracle,
    )
    from torchtitan.models.qwen3_5.model.block_gated_delta_rule import (
        fused_recurrent_block_causal_gated_delta_rule as fused,
    )

    from jainius.model.gdn import gdn_run, l2norm

    torch.manual_seed(0)
    B, T, H, K, V, BS = 2, 8, 2, 32, 32, 4
    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.float32) * 0.5
    g = torch.nn.functional.logsigmoid(
        torch.rand(B, T, H, device="cuda", dtype=torch.float32))
    beta = torch.rand(B, T, H, device="cuda", dtype=torch.float32).sigmoid()
    # Conventions des équations : q,k l2norm (eps 1e-6), q /= sqrt(K).
    qn = l2norm(q) / (K ** 0.5)
    kn = l2norm(k)
    S0 = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")

    out, final = Oracle.apply(qn, kn, v, g, beta, 1.0, BS, S0, True,
                              0, None, False, 64)

    # Référence naïve : récurrence causale par bloc, lecture depuis S_end.
    # (o[t,v] = Σ_k S_end[k,v]·q[t,k] — somme sur K, PAS sur V.)
    ref = torch.zeros_like(out)
    S = S0.clone()
    for s in range(0, T, BS):
        e = s + BS
        for b in range(B):
            for hh in range(H):
                _, send = gdn_run(qn[b, s:e, hh], kn[b, s:e, hh],
                                  v[b, s:e, hh], g[b, s:e, hh],
                                  beta[b, s:e, hh], S0=S[b, hh])
                S[b, hh] = send
                ref[b, s:e, hh] = (
                    send.unsqueeze(0)
                    * qn[b, s:e, hh].unsqueeze(-1)).sum(1)
    assert out.shape == ref.shape == (B, T, H, V)
    assert torch.isfinite(out).all(), "oracle : sorties non finies"
    amax = (out - ref).abs().max().item()
    rmax = amax / max(ref.abs().max().item(), 1e-12)
    assert amax <= 1e-3 and rmax <= 1e-3, (
        f"T-GDN-03 : écart oracle (abs {amax:.2e}, rel {rmax:.2e})")
    fmax = (final - S).abs().max().item()
    assert fmax <= 1e-3, f"T-GDN-03 : état final divergent ({fmax:.2e})"

    # Garde secondaire : token-causal exact (récurrence seule).
    out1, _ = fused(qn, kn, v, g=g, beta=beta, scale=1.0,
                    block_size=BS, causal_mode=1,
                    output_final_state=False)
    outs, _ = gdn_run(qn[0, :, 0], kn[0, :, 0], v[0, :, 0],
                      g[0, :, 0], beta[0, :, 0])
    assert (out1[0, :, 0] - outs).abs().max().item() <= 1e-5, (
        "oracle token-causal divergent de la récurrence naïve")
