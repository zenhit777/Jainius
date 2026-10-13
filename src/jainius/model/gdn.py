"""Référence GDN naïve float64, jeton par jeton (B, H-02).

Équations : EXACTEMENT `torch_recurrent_gated_delta_rule`
(`transformers/integrations/hub_kernels.py`, repli de référence utilisé
ici en l'absence de `fla`) —
  q,k l2norm (eps 1e-6, fp auteur) ; q /= sqrt(d) ;
  S = S·exp(g) ; δ = β(v − Σ_k S·k) ; S += k⊗δ ; o = Σ_k S·q.
États fonctionnels (pas d'in-place) pour un autograd propre (T-GDN-05).
`step_block` réalise la sémantique planifiée FLARE (note B0) : init depuis
l'état propre de frontière (zéro si début de document), lecture de TOUS
les jetons depuis l'état de FIN de bloc, sans réécriture du propre.

B2 — Référence indépendante : `GDNRef` utilise des poids aléatoires
(aucune dépendance HF) pour vérifier les équations GDN elles-mêmes
(indépendamment de la parité avec la couche HF).
"""

from __future__ import annotations

import math

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def gdn_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor,
             v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor):
    """Un pas : S [K, V], q/k [K], v [V], g/beta scalaires (tenseurs 0-d).

    P1/P2 : opérations PyTorch différentiables uniquement — `torch.exp`
    et indexation native. Aucun `float()`, `math.exp` ou extraction
    scalaire Python sur le chemin du gradient (le graphe autograd vers
    `A_log`, `dt_bias`, `in_proj_a/b`, q/k/v doit rester intact).

    Étape D : `g`/`beta` acceptent aussi la forme batchée [N] (une valeur
    par séquence×tête) — l'extension de dimensions préserve exactement
    l'ordre des opérations élément par élément."""
    decay = torch.exp(g)
    while decay.dim() < S.dim():
        decay = decay.unsqueeze(-1)
    S = S * decay
    kv = (S * k.unsqueeze(-1)).sum(dim=-2)
    beta_b = beta
    while beta_b.dim() < (v - kv).dim():
        beta_b = beta_b.unsqueeze(-1)
    delta = (v - kv) * beta_b
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    o = (S * q.unsqueeze(-1)).sum(dim=-2)
    return o, S


def gdn_run(q, k, v, g, beta, S0=None):
    """Déroulé causal : q/k [T, K], v [T, V], g/beta [T]. -> (outs, S_end).
    dtype = celui des entrées (float64 référence, dtype modèle en runner).

    P1/P2 : `g[t]` / `beta[t]` restent des tenseurs 0-d (indexation native),
    jamais convertis en `float` Python."""
    K, V = k.shape[-1], v.shape[-1]
    S = (torch.zeros(K, V, dtype=q.dtype, device=q.device)
         if S0 is None else S0.clone())
    outs = []
    for t in range(q.shape[0]):
        o, S = gdn_step(S, q[t], k[t], v[t], g[t], beta[t])
        outs.append(o)
    return torch.stack(outs), S


def gdn_run_batched(q, k, v, g, beta, S0=None):
    """Déroulé causal vectorisé sur N séquences×têtes indépendantes.

    q/k [N, T, K], v [N, T, V], g/beta [N, T] (scalaires par tête),
    S0 [N, K, V] ou None. -> (outs [N, T, V], S_end [N, K, V]).

    Étape D (vectorisation logicielle) : même ordre mathématique que
    `gdn_run` élément par élément (la boucle sur les jetons porte une
    dépendance séquentielle réelle S_t(S_{t-1}) et est conservée) ;
    seules les dimensions indépendantes (batch, têtes) sont fusionnées.
    Autograd préservé (aucune extraction scalaire).
    """
    N, T, K = q.shape
    V = v.shape[-1]
    S = (torch.zeros(N, K, V, dtype=q.dtype, device=q.device)
         if S0 is None else S0.clone())
    outs = []
    for t in range(T):
        o, S = gdn_step(S, q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t])
        outs.append(o)
    return torch.stack(outs, dim=1), S


def gdn_block_end_read(q, k, v, g, beta, S_init):
    """Bloc bruité : recurrence depuis S_init, lecture de tous les jetons
    depuis S_end (sémantique B0 §(b)). -> (outs_end, S_end)."""
    _, S_end = gdn_run(q, k, v, g, beta, S0=S_init)
    outs = []
    for t in range(q.shape[0]):
        outs.append((S_end * q[t].unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outs), S_end


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))


def causal_short_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Conv causale depthwise : x [B, C, L], weight [C, K] -> [B, C, L],
    avec historique gauche `hist` [B, C, K-1] optionnel (B0 §(c) : propre).

    Équivaut à `causal_conv1d_fn` (padding=K-1, trim) pour hist=zeros.
    """
    return causal_short_conv_hist(x, weight, hist=None)


def causal_short_conv_hist(x, weight, hist=None):
    B, C, L = x.shape
    K = weight.shape[-1]
    if hist is None:
        hist = torch.zeros(B, C, K - 1, dtype=x.dtype, device=x.device)
    full = torch.cat([hist.to(dtype=x.dtype), x], dim=-1)
    out = torch.nn.functional.conv1d(
        full, weight.unsqueeze(1), bias=None, padding=0, groups=C)
    return out[:, :, :L]


class GDNRefLayer:
    """Couche GDN de référence extraite d'une couche HF (poids FIGÉS,
    recopiés float64). `forward_clean` reproduit le causal HF (T-GDN-01).
    `forward_noisy_block` applique la sémantique planifiée B0.

    P1/P2 — différentiabilité : les copies `W_*`/`A_log`/`dt_bias` ci-dessous
    servent aux formes et aux tests de référence SANS gradient. Le chemin
    d'entraînement (`TwoStreamRunner.forward_train` + `train_step`) lit les
    poids VIVANTS de la couche HF (`self._hf`, cast différentiable vers le
    dtype de calcul) afin que le graphe autograd atteigne les paramètres
    `in_proj_qkv/z/b/a`, `A_log`, `dt_bias`, `conv1d`, `norm`, `out_proj` et
    les projections q/k/v. Aucune conversion Python sur ce chemin."""

    def __init__(self, hf_layer, dtype: torch.dtype = torch.float64):
        g = hf_layer
        self._hf = g  # référence vivante (chemin gradient, P1/P2)
        self.dtype = dtype
        self.n_k = g.num_k_heads
        self.n_v = g.num_v_heads
        self.d_k = g.head_k_dim
        self.d_v = g.head_v_dim
        self.kdim = g.key_dim
        self.vdim = g.value_dim
        self.eps = float(g.layer_norm_epsilon)

        def conv(t):
            return t.detach().to(dtype).clone()

        self.W_qkv = conv(g.in_proj_qkv.weight)
        self.W_z = conv(g.in_proj_z.weight)
        self.W_b = conv(g.in_proj_b.weight)
        self.W_a = conv(g.in_proj_a.weight)
        self.conv_w = conv(g.conv1d.weight.squeeze(1))
        self.conv_k = int(g.conv_kernel_size)
        self.A_log = conv(g.A_log)
        self.dt_bias = conv(g.dt_bias)
        self.norm_w = conv(g.norm.weight)
        self.W_out = conv(g.out_proj.weight)
        self.repeat = self.n_v // self.n_k

    def _live(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Cast différentiable vers le dtype de calcul (le gradient traverse
        le cast vers le paramètre vivant)."""
        return t.to(dtype) if t.dtype != dtype else t

    def _project(self, h):
        cdtype = h.dtype
        hf = self._hf
        W_qkv = self._live(hf.in_proj_qkv.weight, cdtype)
        W_z = self._live(hf.in_proj_z.weight, cdtype)
        W_b = self._live(hf.in_proj_b.weight, cdtype)
        W_a = self._live(hf.in_proj_a.weight, cdtype)
        A_log = self._live(hf.A_log, cdtype)
        dt_bias = self._live(hf.dt_bias, cdtype)
        mixed = h @ W_qkv.T
        z = (h @ W_z.T).reshape(h.shape[0], h.shape[1], -1, self.d_v)
        b = torch.sigmoid(h @ W_b.T)
        a = h @ W_a.T
        g = -torch.exp(A_log) * torch.nn.functional.softplus(
            a + dt_bias)
        return mixed, z, b, g

    def _core(self, q, k, v, g, beta, S0=None):
        """q/k [B, T, H, K], v [B, T, H, V] -> (outs [B, T, H, V], S_end).

        Étape D : boucle unique sur les jetons, dimensions batch×têtes
        fusionnées ([B*H, T, ...]) via `gdn_run_batched`. Même ordre
        mathématique que `_core_legacy` (référence conservée ci-dessous,
        vérifiée par `test_gdn_vectorized_parity`)."""
        B, T, H, K = q.shape
        Vd = v.shape[-1]
        if g.shape[-1] != H or beta.shape[-1] != H:
            raise ValueError(
                f"_core : têtes g/beta {g.shape[-1]}/{beta.shape[-1]} ≠ "
                f"têtes q/k/v {H} (répétition GQA non supportée, comme le "
                f"chemin historique).")
        scale = math.sqrt(K)
        qn = l2norm(q) / scale
        kn = l2norm(k)
        qf = qn.permute(0, 2, 1, 3).reshape(B * H, T, K)
        kf = kn.permute(0, 2, 1, 3).reshape(B * H, T, K)
        vf = v.permute(0, 2, 1, 3).reshape(B * H, T, Vd)
        gf = g.reshape(B, T, H).permute(0, 2, 1).reshape(B * H, T)
        bf = beta.reshape(B, T, H).permute(0, 2, 1).reshape(B * H, T)
        s0 = None if S0 is None else S0.reshape(B * H, K, Vd)
        outs, send = gdn_run_batched(qf, kf, vf, gf, bf, S0=s0)
        return (outs.reshape(B, H, T, Vd).permute(0, 2, 1, 3),
                send.reshape(B, H, K, Vd))

    def _core_legacy(self, q, k, v, g, beta, S0=None):
        """Référence historique (boucles Python B×H) — conservée pour les
        tests de parité de la vectorisation. Ne pas utiliser en entraînement."""
        B, T, H, _ = q.shape
        outs, sends = [], []
        for b in range(B):
            ob, sb = [], []
            for hh in range(H):
                qq = l2norm(q[b, :, hh])
                kk = l2norm(k[b, :, hh])
                qq = qq / math.sqrt(qq.shape[-1])
                o, send = gdn_run(qq, kk, v[b, :, hh], g[b, :, hh],
                                  beta[b, :, hh],
                                  S0=None if S0 is None else S0[b, hh])
                ob.append(o)
                sb.append(send)
            outs.append(torch.stack(ob))
            sends.append(torch.stack(sb))
        # [B, H, T, V] -> [B, T, H, V] (ordre HF).
        return torch.stack(outs).permute(0, 2, 1, 3), torch.stack(sends)

    def _finish(self, core, z):
        cdtype = core.dtype
        hf = self._hf
        norm_w = self._live(hf.norm.weight, cdtype)
        n = core.reshape(-1, self.d_v)
        var = (n.double().pow(2).mean(-1, keepdim=True))
        n = n * torch.rsqrt(var + self.eps)
        n = n * norm_w.to(core.dtype)
        n = n * silu(z.reshape(-1, self.d_v).to(torch.float64))
        return n.to(cdtype)

    def _conv_weight(self, dtype: torch.dtype) -> torch.Tensor:
        hf = self._hf
        return self._live(hf.conv1d.weight.squeeze(1), dtype)

    def forward_clean(self, h, S0=None, conv_hist=None):
        """Causal exact : (out [B, L, Vdim], S_end [B, H, K, V], conv_end
        [B, C, K-1] = derniers inputs mixtes, hist inclus)."""
        B, L, _ = h.shape
        mixed, z, beta, g = self._project(h)
        mixed_t = mixed.transpose(1, 2)
        conv = causal_short_conv_hist(mixed_t, self._conv_weight(mixed_t.dtype),
                                      hist=conv_hist)
        conv = silu(conv).transpose(1, 2)
        q, k, v = torch.split(conv, [self.kdim, self.kdim, self.vdim], -1)
        q = q.reshape(B, L, -1, self.d_k)
        k = k.reshape(B, L, -1, self.d_k)
        v = v.reshape(B, L, -1, self.d_v)
        if self.repeat > 1:
            q = q.repeat_interleave(self.repeat, dim=2)
            k = k.repeat_interleave(self.repeat, dim=2)
        core, send = self._core(q, k, v, g, beta, S0=S0)
        hf = self._hf
        W_out = self._live(hf.out_proj.weight, core.dtype)
        out = self._finish(core, z).reshape(B, L, -1) @ W_out.T
        if conv_hist is None:
            full_in = mixed_t
        else:
            full_in = torch.cat(
                [conv_hist.to(dtype=mixed_t.dtype), mixed_t], dim=-1)
        # P1/P2 : `clone()` différentiable (pas de `detach`) — le gradient
        # vers le préfixe propre via l'historique conv est préservé.
        conv_end = full_in[:, :, -(self.conv_k - 1):].clone()
        return out, send, conv_end

    def _block_qkvz(self, h_block, conv_init):
        """Projections + conv (historique = copie locale) d'un bloc."""
        B, L, _ = h_block.shape
        mixed, z, beta, g = self._project(h_block)
        mixed_t = mixed.transpose(1, 2)
        conv = causal_short_conv_hist(mixed_t, self._conv_weight(mixed_t.dtype), hist=conv_init)
        conv = silu(conv).transpose(1, 2)
        q, k, v = torch.split(conv, [self.kdim, self.kdim, self.vdim], -1)
        q = q.reshape(B, L, -1, self.d_k)
        k = k.reshape(B, L, -1, self.d_k)
        v = v.reshape(B, L, -1, self.d_v)
        if self.repeat > 1:
            q = q.repeat_interleave(self.repeat, dim=2)
            k = k.repeat_interleave(self.repeat, dim=2)
        return q, k, v, z, g, beta

    def forward_noisy_block(self, h_block, S_init, conv_init):
        """Sémantique planifiée B0 : récurrence depuis S_init (copie, pas
        d'écriture du propre), lecture de TOUS les jetons depuis S_end.
        -> (out [B, L, Vdim], S_end).

        Étape D : batch×têtes fusionnées + lecture S̃_end en un einsum
        (référence historique conservée : `_forward_noisy_block_legacy`)."""
        B, L, _ = h_block.shape
        q, k, v, z, g, beta = self._block_qkvz(h_block, conv_init)
        H, K = q.shape[2], self.d_k
        Vd = v.shape[-1]
        if g.shape[-1] != H or beta.shape[-1] != H:
            raise ValueError(
                f"noisy_block : têtes g/beta {g.shape[-1]}/{beta.shape[-1]} "
                f"≠ têtes q/k/v {H} (comme le chemin historique).")
        qn = l2norm(q) / math.sqrt(self.d_k)
        kn = l2norm(k)
        N = B * H
        qf = qn.permute(0, 2, 1, 3).reshape(N, L, K)
        kf = kn.permute(0, 2, 1, 3).reshape(N, L, K)
        vf = v.permute(0, 2, 1, 3).reshape(N, L, Vd)
        gf = g.reshape(B, L, H).permute(0, 2, 1).reshape(N, L)
        bf = beta.reshape(B, L, H).permute(0, 2, 1).reshape(N, L)
        s0 = S_init.reshape(N, K, Vd)
        _, send = gdn_run_batched(qf, kf, vf, gf, bf, S0=s0)
        # Lecture S̃_end pour tous les jetons en une opération :
        # o[b,t,h,v] = Σ_k S_end[b,h,k,v] · q[b,t,h,k].
        core = torch.einsum("nhkv,nhtk->nhtv",
                            send.reshape(B, H, K, Vd),
                            qn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        hf = self._hf
        W_out = self._live(hf.out_proj.weight, core.dtype)
        out = self._finish(core, z).reshape(B, L, -1) @ W_out.T
        return out, send.reshape(B, H, K, Vd)

    def _forward_noisy_block_legacy(self, h_block, S_init, conv_init):
        """Référence historique (boucles Python B×H×T) — conservée pour
        les tests de parité de la vectorisation."""
        B, L, _ = h_block.shape
        q, k, v, z, g, beta = self._block_qkvz(h_block, conv_init)
        H = q.shape[2]
        outs, sends = [], []
        for b in range(B):
            ob = []
            sb = []
            for hh in range(H):
                qq = l2norm(q[b, :, hh]) / math.sqrt(self.d_k)
                kk = l2norm(k[b, :, hh])
                _, send = gdn_run(qq, kk, v[b, :, hh], g[b, :, hh],
                                  beta[b, :, hh], S0=S_init[b, hh])
                o_end = (send * qq.unsqueeze(-1)).sum(dim=-2)
                ob.append(o_end)
                sb.append(send)
            outs.append(torch.stack(ob))
            sends.append(torch.stack(sb))
        core = torch.stack(outs).permute(0, 2, 1, 3)  # [B, T, H, V]
        hf = self._hf
        W_out = self._live(hf.out_proj.weight, core.dtype)
        out = self._finish(core, z).reshape(B, L, -1) @ W_out.T
        return out, torch.stack(sends)


# =============================================================================
# B2 — Référence indépendante (poids aléatoires, aucune dépendance HF)
# =============================================================================

class GDNRef:
    """Référence indépendante float64 jeton par jeton (B2).

    Utilise des poids aléatoires (aucune dépendance à une couche HF).
    Permet de vérifier les équations GDN elles-mêmes, indépendamment
    de la parité avec la couche HF (T-GDN-01).

    Équation : EXACTEMENT `torch_recurrent_gated_delta_rule`
    (`transformers/integrations/hub_kernels.py`) —
      q,k l2norm (eps 1e-6) ; q /= sqrt(d_k) ;
      S = S·exp(g) ; δ = β(v − Σ_k S·k) ; S += k⊗δ ; o = Σ_k S·q.

    `forward_clean` : causal standard (T-GDN-01).
    `forward_noisy_block` : sémantique B0 (lecture depuis S̃_end, pas d'écriture).
    """

    def __init__(self, n_heads: int = 4, d_k: int = 32, d_v: int = 64,
                 conv_kernel: int = 4, seed: int = 0):
        torch.manual_seed(seed)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.conv_kernel = conv_kernel
        self.eps = 1e-6

        # Projections (poids aléatoires, float64).
        self.W_qkv = torch.nn.Parameter((torch.randn(n_heads * 2 * d_k + n_heads * d_v,
                                                     n_heads * d_k) * 0.1).double())
        self.W_z = torch.nn.Parameter((torch.randn(n_heads * d_v,
                                                   n_heads * d_k) * 0.1).double())
        self.W_b = torch.nn.Parameter((torch.randn(n_heads * d_v,
                                                   n_heads * d_k) * 0.1).double())
        self.W_a = torch.nn.Parameter((torch.randn(n_heads * d_k,
                                                   n_heads * d_k) * 0.1).double())
        self.conv_w = torch.nn.Parameter(
            (torch.randn(n_heads * (2 * d_k + d_v), conv_kernel) * 0.1).double())
        self.A_log = torch.nn.Parameter(torch.randn(n_heads * d_k).double())
        self.dt_bias = torch.nn.Parameter(torch.randn(n_heads * d_k).double())
        self.norm_w = torch.nn.Parameter(torch.ones(n_heads * d_v).double())
        self.W_out = torch.nn.Parameter(
            (torch.randn(n_heads * d_v, n_heads * d_v) * 0.1).double())

    def _project(self, h):
        """Projections : (mixed [B,L,H*2Dk+H*Dv], z, beta, g)."""
        B, L, H = h.shape[0], h.shape[1], self.n_heads
        Dk = self.d_k
        mixed = h @ self.W_qkv.T  # [B, L, H*(2*Dk + Dv)]
        z = (h @ self.W_z.T).reshape(B, L, H, self.d_v)
        b = torch.sigmoid(h @ self.W_b.T).reshape(B, L, H, self.d_v)
        a = h @ self.W_a.T  # [B, L, H*Dk]
        g = -torch.exp(self.A_log) * torch.nn.functional.softplus(
            a + self.dt_bias)  # [H, Dk] → broadcast
        return mixed, z, b, g

    def _split_qkvz(self, mixed):
        """Split projections en q, k, v (après conv)."""
        B, L = mixed.shape[:2]
        H = self.n_heads
        Dk = self.d_k
        # mixed : [B, L, H*(2*Dk + Dv)] → split en 2*H*Dk + H*Dv
        parts = torch.split(mixed, [H * Dk, H * Dk, H * self.d_v], dim=-1)
        q_raw, k_raw, v_raw = parts[0], parts[1], parts[2]
        return (q_raw.reshape(B, L, H, Dk),
                k_raw.reshape(B, L, H, Dk),
                v_raw.reshape(B, L, H, self.d_v))

    def forward_clean(self, h):
        """Causal standard : out [B, L, H*Dv], S_end [B, H, Dk, Dv]."""
        B, L = h.shape[:2]
        mixed, z, beta, g = self._project(h)
        # Conv causale sur mixed.
        conv = causal_short_conv(mixed.transpose(1, 2), self.conv_w).transpose(1, 2)
        conv = silu(conv)
        q, k, v = self._split_qkvz(conv)
        # L2norm + scaling.
        q = l2norm(q) / math.sqrt(self.d_k)
        k = l2norm(k)
        # Récurrence causale.
        outs, sends = [], []
        S = torch.zeros(B, self.n_heads, self.d_k, self.d_v, dtype=h.dtype)
        for t in range(L):
            ob, sb = [], []
            for hh in range(self.n_heads):
                qq = q[:, t, hh]  # [Dk]
                kk = k[:, t, hh]  # [Dk]
                vv = v[:, t, hh]  # [Dv]
                gg = g[hh]        # [Dk]
                bb = beta[:, t, hh]  # [Dv]
                # P1/P2 : tenseurs conservés (pas de `float()` — graphe intact).
                o, S_new = gdn_step(S[:, hh], qq, kk, vv, gg, bb)
                ob.append(o)
                sb.append(S_new)
            outs.append(torch.stack(ob, dim=1))  # [B, H, Dv]
            sends.append(torch.stack(sb, dim=1))  # [B, H, Dk, Dv]
        core = torch.cat(outs, dim=1)  # [B, L, H, Dv]
        out = self._finish(core, z)
        return out, sends[-1]

    def _finish(self, core, z):
        """Fin de couche : norm + silu(z) + projection sortie."""
        B, L, H, V = core.shape
        n = core.reshape(B * L, -1)  # [B*L, H*Dv]
        var = (n.double().pow(2).mean(-1, keepdim=True))
        n = n * torch.rsqrt(var + self.eps)
        n = n * self.norm_w.to(core.dtype)
        n = n * silu(z.reshape(B * L, -1).to(torch.float64))
        return (n @ self.W_out.T).reshape(B, L, H * self.d_v)

    def forward_noisy_block(self, h_block, S_init):
        """Sémantique B0 : récurrence depuis S_init (copie), lecture de TOUS
        les jetons depuis S_end. -> (out [B, T, H*Dv], S_end)."""
        B, T = h_block.shape[:2]
        mixed, z, beta, g = self._project(h_block)
        conv = causal_short_conv(mixed.transpose(1, 2), self.conv_w).transpose(1, 2)
        conv = silu(conv)
        q, k, v = self._split_qkvz(conv)
        q = l2norm(q) / math.sqrt(self.d_k)
        k = l2norm(k)
        # Récurrence depuis S_init (copie locale).
        outs, sends = [], []
        S = S_init.clone()  # [B, H, Dk, Dv] — copie immuable de l'entrée
        for t in range(T):
            ob, sb = [], []
            for hh in range(self.n_heads):
                qq = q[:, t, hh]
                kk = k[:, t, hh]
                vv = v[:, t, hh]
                gg = g[hh]
                bb = beta[:, t, hh]
                # P1/P2 : tenseurs conservés (pas de `float()` — graphe intact).
                o, S_new = gdn_step(S[:, hh], qq, kk, vv, gg, bb)
                ob.append(o)
                sb.append(S_new)
            outs.append(torch.stack(ob, dim=1))
            sends.append(torch.stack(sb, dim=1))
        core = torch.cat(outs, dim=1)  # [B, T, H, Dv]
        out = self._finish(core, z)
        return out, sends[-1]  # S_end = état de FIN de bloc (bidirectionnel)
