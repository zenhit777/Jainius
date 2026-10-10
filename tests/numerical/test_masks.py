"""T-MASK-01…03 — masque deux-flux ``[x0 | xt]`` (cahier technique §10.2).

Référence indépendante : implémentation brute-force par boucles explicites
sur les règles de visibilité (chemin de code distinct de ``masks.py``).
Défaut Jainius : ``causal_x0=True``, ``causal_xt=False``.
"""

import pytest
import torch

from jainius.model.masks import build_two_stream_mask


def brute_force_mask(L, B, causal_x0=True, causal_xt=False, doc_ids=None):
    """Référence indépendante : boucles explicites q/kv, 0=x0, 1=xt."""
    mask = torch.zeros(2 * L, 2 * L, dtype=torch.bool)
    for q in range(2 * L):
        q_x0 = q < L
        qp = q if q_x0 else q - L
        qb = qp // B
        for k in range(2 * L):
            k_x0 = k < L
            kp = k if k_x0 else k - L
            kb = kp // B
            if q_x0 and k_x0:
                ok = qp >= kp if causal_x0 else qb >= kb
            elif not q_x0 and not k_x0:
                ok = qb == kb and (qp >= kp if causal_xt else True)
            elif not q_x0 and k_x0:
                ok = qb > kb
            else:  # q∈x0, kv∈xt : jamais visible
                ok = False
            if ok and doc_ids is not None and doc_ids[q] != doc_ids[k]:
                ok = False
            mask[q, k] = ok
    return mask


@pytest.mark.parametrize("L,B", [(8, 2), (8, 3), (12, 4)])
def test_mask_01_truth_table(L, B):
    """T-MASK-01 : table de vérité exacte vs référence indépendante (F-003)."""
    got = build_two_stream_mask(L, B)
    assert got.shape == (2 * L, 2 * L)
    assert torch.equal(got, brute_force_mask(L, B))


def test_mask_02_doc_isolation():
    """T-MASK-02 (niveau masque) : aucun flux cross-document.

    Niveaux modèle (logits inchangés sous perturbation cross-doc) : requiert
    JAI-ARCH (M1-B) — ici : le masque ne contient aucune entrée cross-doc.
    """
    L, B = 8, 2
    doc_ids = torch.tensor([0] * L + [0] * L)
    doc_ids[L + 4 :] = 1  # second document sur la moitié de xt
    doc_ids[4:L] = 1  # et la moitié correspondante de x0
    mask = build_two_stream_mask(L, B, doc_ids=doc_ids)
    cross = doc_ids[:, None] != doc_ids[None, :]
    assert not mask[cross].any(), "fuite cross-document dans le masque"
    # Sans doc_ids, les mêmes paires seraient visibles : l'isolation vient
    # bien des doc_ids.
    assert build_two_stream_mask(L, B)[cross].any()


def test_mask_03_visibility_rules():
    """T-MASK-03 : xt ne voit ni x0 du même bloc ni le futur ; voit le passé."""
    L, B = 8, 2
    mask = build_two_stream_mask(L, B)
    x0 = mask[:L, :L]
    xt = mask[L:, L:]
    cross = mask[L:, :L]
    rev = mask[:L, L:]
    assert not rev.any(), "x0 ne doit jamais voir xt"
    for b in range(L // B):
        rows = slice(L + b * B, L + (b + 1) * B)
        # Pas de visibilité du bloc x0 courant (strictement antérieur requis).
        assert not cross[rows, b * B : (b + 1) * B].any()
        # Visibilité intra-bloc xt complète et bidirectionnelle.
        block = xt[b * B : (b + 1) * B, b * B : (b + 1) * B]
        assert block.all(), "bloc xt diagonal non bidirectionnel"
        # xt ne voit aucun bloc futur.
        assert not xt[b * B : (b + 1) * B, (b + 1) * B :].any()
    # Premier bloc xt : rien du passé à voir sauf rien (cohérence).
    assert x0.tril().equal(x0), "x0 causal violé"
