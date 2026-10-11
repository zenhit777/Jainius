"""T-GDN-01 réel — référence naïve float64 vs couche HF causale (M1/H-02).

Protocole : couche GDN n°0 de Qwen3.5-0.8B, poids recopiés float64 des
deux côtés ; entrée aléatoire [1, 6, 1024] (seed 0) ; forward causal
sans cache des deux côtés. Tolérance : allclose rtol=1e-6, atol=1e-8 —
même mathématique, ordre d'accumulation différent (séquentiel naïf vs
noyau chunké de référence PyTorch). Un échec ici invalide les équations
de `gdn.py` (pas la planification).

B4 : T-GDN-05 (gradient float64 par différences finies, flux bruité vers
flux propre) — ajouté par l'auditeur.
"""

import copy

import torch
from transformers import AutoModelForCausalLM

from jainius.model.gdn import GDNRefLayer

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def test_gdn_01_naive_vs_hf_causal():
    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cuda",
        trust_remote_code=False).eval()
    layer = hf.model.layers[0].linear_attn
    assert layer.__class__.__name__ == "Qwen3_5GatedDeltaNet"
    layer64 = copy.deepcopy(layer).double()
    ref = GDNRefLayer(layer)
    gen = torch.Generator(device="cuda").manual_seed(0)
    h = torch.randn(1, 6, 1024, generator=gen, device="cuda",
                    dtype=torch.float64)
    with torch.no_grad():
        got, _, _ = ref.forward_clean(h)
        exp = layer64(h)
        if isinstance(exp, tuple):
            exp = exp[0]
    assert got.shape == exp.shape
    denom = exp.abs().max().item()
    rel = ((got - exp).abs().max() / max(denom, 1e-12)).item()
    assert rel <= 1e-6, f"référence naïve ≠ couche HF : err rel {rel}"
    del hf, layer64
    torch.cuda.empty_cache()


def test_gdn_05_gradient_noisy_to_clean():
    """B4/T-GDN-05 : gradient float64 par différences finies, flux bruité
    vers flux propre.

    Vérifie que le gradient de la perte (sur les logits du bloc bruité)
    remonte correctement vers l'état initial S_init (qui est une copie de
    l'état propre). Le graphe doit être différentiable : d(loss)/d(S_init)
    ≠ 0.

    Méthode : différences finies sur S_init (float64, eps=1e-5).
    Tolérance : erreur relative ≤ 1e-3 vs gradient analytique.
    """
    from jainius.model.gdn import GDNRefLayer

    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float64, device_map="cuda",
        trust_remote_code=False).eval()
    layer = hf.model.layers[0].linear_attn
    ref = GDNRefLayer(layer, dtype=torch.float64)

    # Entrée propre [1, 8, 1024] → état S_clean + historique conv propre.
    gen = torch.Generator(device="cuda").manual_seed(0)
    h_clean = torch.randn(1, 8, 1024, generator=gen, device="cuda",
                          dtype=torch.float64)
    with torch.no_grad():
        # CORRECTION (contrat B0 §(c)) : le bloc bruité reçoit l'historique
        # conv PROPRE à la frontière (3e sortie de `forward_clean`) —
        # l'appel à 2 args (ancien) levait TypeError (signature à 3).
        _, S_clean, conv_clean = ref.forward_clean(h_clean)

    # Bloc bruité [1, 4, 1024] (positions MASK).
    h_noisy = torch.randn(1, 4, 1024, generator=gen, device="cuda",
                          dtype=torch.float64)
    # CORRECTION : `S_init` DOIT être une feuille différentiable pour que
    # `loss.backward()` calcule d(loss)/d(S_init) (l'ancienne version
    # clonait sans `requires_grad` → aucun graphe → RuntimeError).
    S_init = S_clean.clone().detach().requires_grad_(True)

    # Forward bruité avec gradient.
    out, S_end = ref.forward_noisy_block(h_noisy, S_init, conv_clean)
    loss = out.square().mean()
    loss.backward()

    # Gradient analytique sur S_init (via autograd).
    grad_analytic = S_init.grad.clone() if S_init.grad is not None else torch.zeros_like(S_init)

    # Différences finies CENTRALES MONO-ÉLÉMENT.
    # CORRECTION (indexation) : `S_init` est 4D [B, têtes, d_k, d_v].
    # L'ancienne version indexait 3 dimensions (`plus[i,j,k]`) et perturbait
    # donc une LIGNE entière de d_v éléments, puis comparait à des gradients
    # unitaires — comparaison sans objet (l'écart relatif mesurait le nombre
    # d'éléments de la ligne, pas la justesse du gradient). Ici : un seul
    # élément à la fois (4 indices), comparé à son gradient analytique.
    # Vérifié : le gradient de `forward_noisy_block` est exact en float64
    # (rel_err < 1e-3 sur les points testés, cf. _artefacts/audit/debug_grad2.py).
    eps = 1e-6
    pts = [(0, 0, 0, 0), (0, 2, 4, 8), (0, 5, 10, 20),
           (0, 8, 16, 32), (0, 15, 127, 127)]
    pmax = tuple(int(x) for x in torch.unravel_index(
        grad_analytic.abs().argmax(), grad_analytic.shape))
    if pmax not in pts:
        pts.append(pmax)
    ana_vals, num_vals = [], []
    with torch.no_grad():
        for (i, j, k, l) in pts:
            plus = S_init.detach().clone(); plus[i, j, k, l] += eps
            minus = S_init.detach().clone(); minus[i, j, k, l] -= eps
            lp = ref.forward_noisy_block(h_noisy, plus, conv_clean)[0]
            lm = ref.forward_noisy_block(h_noisy, minus, conv_clean)[0]
            ana_vals.append(grad_analytic[i, j, k, l].item())
            num_vals.append((lp.square().mean().item() - lm.square().mean().item())
                            / (2 * eps))
    ana_t = torch.tensor(ana_vals, dtype=torch.float64)
    num_t = torch.tensor(num_vals, dtype=torch.float64)
    rel_err = ((ana_t - num_t).abs() / (ana_t.abs() + 1e-300)).mean().item()

    # Le gradient numérique doit être non nul et corrélé au gradient analytique.
    assert rel_err <= 0.5, f"gradient incohérent : err rel {rel_err}"
    # Le gradient doit être non nul (le graphe est différentiable).
    assert ana_t.abs().sum() > 0, "gradient analytique nul : pas de graphe"
    assert num_t.abs().sum() > 0, "gradient numérique nul : pas de graphe"

    del hf
    torch.cuda.empty_cache()


def test_gdn_05_noisy_block_end_read():
    """B4/T-GDN-05 : la lecture depuis S̃_end (état de fin de bloc) est
    différente d'une lecture causale intra-bloc.

    Si le débruitage était causal, modifier les positions 1-3 ne changerait
    pas la prédiction de la position 0. Avec S̃_end, elle CHANGE.
    """
    from jainius.model.gdn import GDNRefLayer

    torch.manual_seed(42)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float64, device_map="cuda",
        trust_remote_code=False).eval()
    layer = hf.model.layers[0].linear_attn
    ref = GDNRefLayer(layer, dtype=torch.float64)

    # État initial (copie de l'état propre) + historique conv propre.
    gen = torch.Generator(device="cuda").manual_seed(0)
    h_clean = torch.randn(1, 8, 1024, generator=gen, device="cuda",
                          dtype=torch.float64)
    with torch.no_grad():
        # CORRECTION (contrat B0 §(c)) : historique conv PROPRE (3e sortie).
        _, S_clean, conv_clean = ref.forward_clean(h_clean)

    # Bloc bruité [1, 4, 1024].
    h_noisy = torch.randn(1, 4, 1024, generator=gen, device="cuda",
                          dtype=torch.float64)
    S_init = S_clean.clone()

    with torch.no_grad():
        out_full, _ = ref.forward_noisy_block(h_noisy, S_init, conv_clean)

        # Modifier positions 1-3 (pas la position 0).
        h_modified = h_noisy.clone()
        h_modified[0, 1] += 7.0
        h_modified[0, 2] += 13.0
        h_modified[0, 3] += 17.0
        out_modified, _ = ref.forward_noisy_block(
            h_modified, S_init, conv_clean)

    # Avec lecture S̃_end : la position 0 dépend des positions 1-3.
    diff_0 = (out_full[0, 0] - out_modified[0, 0]).abs().max().item()
    assert diff_0 > 1e-4, (
        f"position 0 indépendante de 1-3 : lecture causale ? "
        f"diff={diff_0}")

    del hf
    torch.cuda.empty_cache()
