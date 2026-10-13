"""Runner deux-flux de référence (B, H-02) — PAS un moteur de serving.

Sémantique planifiée (note B0, `docs/decisions/M2-contrat-etat-GDN.md`) :
- flux propre : causal exact (softmax + GDN), écrit ses états ;
- chaque bloc bruité démarre du snapshot propre de sa frontière (zéro si
  début de document), lecture de tous les jetons depuis S̃_end, sans
  réécriture ; historique conv = préfixe propre (copie locale) ;
- reset par ligne et par document (états GDN + conv + KV attention).

Implémentation : normes/MLP/projections/qk-norm/RoPE réutilisés (modules
HF en place, dtype modèle) ; SDPA exact avec masque explicite ;
récurrence GDN par `gdn.GDNRefLayer` (validée T-GDN-01).
`prefill`/`denoise_block`/`commit` sont des chemins d'inférence (appelés
sous `torch.no_grad` par les tests) ; `forward_train` RESTE dans le graphe
autograd (aucun `detach`/`no_grad` sur le chemin étudiant — P1/P2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from jainius.model.gdn import GDNRefLayer


@dataclass
class RunnerGDNState:
    S: torch.Tensor  # [H, K, V]
    conv: torch.Tensor  # [C, K-1]


@dataclass
class RunnerState:
    gdn: list = field(default_factory=list)  # par couche GDN
    attn_k: list = field(default_factory=list)  # [B, T, Hk, D] post-RoPE
    attn_v: list = field(default_factory=list)  # [B, T, Hv, D]
    docs: list = field(default_factory=list)  # doc ids passés par couche


def runner_state_fingerprint(state: RunnerState) -> str:
    """Empreinte SHA-256 bit-exacte d'un `RunnerState` (garde
    « denoise n'écrit jamais l'état », T-STATE-02 du runner).
    Relecture bit-level via `view(torch.uint8)` (fonctionne sur bf16,
    fp16, int64… — contrairement à `.numpy()` qui rejette le bfloat16)."""
    import hashlib

    digest = hashlib.sha256()

    def add(t: torch.Tensor) -> None:
        c = t.detach().to("cpu").contiguous()
        digest.update(c.view(torch.uint8).numpy().tobytes())

    for s in state.gdn:
        add(s.S)
        add(s.conv)
    for t in (*state.attn_k, *state.attn_v, *state.docs):
        add(t)
    return digest.hexdigest()


class TwoStreamRunner:
    """Chemin de référence planifié autour d'un checkpoint HF chargé."""

    def __init__(self, hf_model):
        self.model = hf_model
        text = hf_model.model
        self.embed = text.embed_tokens
        self.final_norm = text.norm
        self.lm_head = hf_model.lm_head
        cfg = hf_model.config.get_text_config()
        self.n_layers = int(cfg.num_hidden_layers)
        self.layer_types = list(cfg.layer_types)
        self.head_dim = int(cfg.head_dim)
        self.n_q = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.rep = self.n_q // self.n_kv
        self.scaling = self.head_dim ** -0.5
        rotary = getattr(text, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("module rotary introuvable (.model.rotary_emb)")
        self.rotary = rotary
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_rotary_pos_emb as rope_fn)
        self._rope_fn = rope_fn
        self.layers = []
        self.dtype = next(hf_model.parameters()).dtype
        for i, layer in enumerate(text.layers):
            if self.layer_types[i] == "linear_attention":
                cell = GDNRefLayer(layer.linear_attn, dtype=self.dtype)
                self.layers.append({"kind": "gdn", "mod": layer, "cell": cell})
            else:
                self.layers.append({"kind": "attn", "mod": layer, "cell": None})

    # ---------- briques ----------
    def _rope(self, pos1d):
        B = pos1d.shape[0]
        ar = pos1d
        pos3 = torch.stack([ar, ar, ar], dim=0)
        return self.rotary(torch.empty(B, pos1d.shape[1], 64,
                                       device=pos1d.device), pos3)

    def _attn_qkv(self, attn_mod, h):
        B, T, _ = h.shape
        qg = attn_mod.q_proj(h).view(B, T, -1, self.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)  # [B, T, Hq, D] chacun
        q = attn_mod.q_norm(q).transpose(1, 2)
        k = attn_mod.k_norm(
            attn_mod.k_proj(h).view(B, T, -1, self.head_dim)).transpose(1, 2)
        v = attn_mod.v_proj(h).view(B, T, -1, self.head_dim).transpose(1, 2)
        return q, gate.reshape(B, T, -1), k, v

    def _sdpa(self, q, k, v, allow):
        neg = torch.finfo(torch.float32).min
        additive = torch.zeros(allow.shape, dtype=torch.float32,
                               device=q.device)
        additive[~allow] = neg
        additive = additive[:, None, :, :]
        w = (q.float() @ k.float().transpose(-1, -2)) * self.scaling + additive
        w = torch.softmax(w, dim=-1)
        # RoPE peut promouvoir q/k en fp32 : calcul interne fp32, sortie
        # au dtype d'entrée (comme `eager_attention_forward` HF).
        return (w @ v.float()).to(v.dtype)

    def _attn_block(self, attn_mod, h, pos1d, k_past, v_past, allow):
        """SDPA exact sur [passé | bloc]. `k_past/v_past` [B, T, H, D] (post-
        RoPE, positions absolues figées) ou None. Retourne (out, k_new,
        v_new) avec k/v post-RoPE du bloc courant."""
        B, T, _ = h.shape
        q, gate, k, v = self._attn_qkv(attn_mod, h)
        cos, sin = self._rope(pos1d)
        q, k = self._rope_fn(q, k, cos, sin)
        knew = k.permute(0, 2, 1, 3)  # [B, T, Hkv, D]
        vnew = v.permute(0, 2, 1, 3)
        if k_past is not None:
            k = torch.cat([k_past.permute(0, 2, 1, 3), k], dim=2)
            v = torch.cat([v_past.permute(0, 2, 1, 3), v], dim=2)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        o = self._sdpa(q, k, v, allow).transpose(1, 2).reshape(B, T, -1)
        o = o * torch.sigmoid(gate)
        return attn_mod.o_proj(o), knew, vnew

    # ---------- passage propre ----------
    def _causal_doc_mask(self, q_len, k_len, k_offset=0, doc_q=None,
                         doc_k=None, device=None):
        """Masque causal [B, Q, K] (+ isolation documents). `k_offset` :
        position absolue de la première clé."""
        allow = torch.ones(1, q_len, k_len, dtype=torch.bool, device=device)
        qp = torch.arange(q_len, device=device)[:, None] + k_offset
        kp = torch.arange(k_len, device=device)[None, :]
        allow = qp >= kp
        if doc_q is not None and doc_k is not None:
            allow = allow & (doc_q[:, :, None] == doc_k[:, None, :])
        return allow

    def _doc_segments(self, doc):
        """Listes (b, s, e) de runs contigus de même document.

        Étape D : frontières calculées en tenseurs (zéro `.item()` sur les
        jetons) ; seuls les quelques indices de frontière repassent en
        Python (nécessaires aux slices)."""
        B, L = doc.shape
        if L == 0:
            return []
        chg = torch.zeros(B, L, dtype=torch.bool, device=doc.device)
        chg[:, 0] = True
        if L > 1:
            chg[:, 1:] = doc[:, 1:] != doc[:, :-1]
        segs = []
        end = torch.full((1,), L, dtype=torch.long, device=doc.device)
        for b in range(B):
            # chg[b,0] est toujours vrai : premier départ = 0.
            bounds = torch.cat([torch.where(chg[b])[0], end]).tolist()
            for s, e in zip(bounds[:-1], bounds[1:]):
                if e > s:
                    segs.append((b, s, e))
        return segs

    def _clean_layers(self, h, pos, gstates, kvstates, doc, reset_rows):
        """Passe propre causale : h [B, L, D] aux positions absolues `pos`.
        `gstates` : états GDN d'entrée (RunnerGDNState, JAMAIS None —
        zéros si frais) ; `kvstates` : (K, V, doc) par couche attention
        (ou None) ; `reset_rows` [B] : nouveau document (zéro + passé
        ignoré). Retourne (h_out, gstates_out, kv_out) NEUFS. Résiduels
        standards : h + sous-couche(norm(h))."""
        B, L, _ = h.shape
        dev = h.device
        if doc is None:
            doc = torch.zeros(B, L, dtype=torch.long, device=dev)
        g_out, kv_out = [], []
        gi = ai = 0
        for entry in self.layers:
            kind, mod = entry["kind"], entry["mod"]
            hn = mod.input_layernorm(h)
            if kind == "attn":
                past = kvstates[ai] if ai < len(kvstates) else None
                if past is None:
                    kp = vp = None
                    klen, past_doc = 0, None
                else:
                    kp, vp, past_doc = past
                    klen = kp.shape[1]
                allow = self._causal_doc_mask(
                    L, klen + L, k_offset=klen, doc_q=doc,
                    doc_k=(torch.cat([past_doc, doc], dim=1)
                           if past_doc is not None else doc),
                    device=dev)
                if reset_rows is not None and klen:
                    new_only = torch.cat(
                        [torch.zeros(B, L, klen, dtype=torch.bool, device=dev),
                         self._causal_doc_mask(L, L, k_offset=0, doc_q=doc,
                                               doc_k=doc, device=dev)], dim=2)
                    allow = torch.where(
                        reset_rows[:, None, None].expand_as(allow),
                        new_only, allow)
                out, knew, vnew = self._attn_block(
                    mod.self_attn, hn, pos,
                    kp, vp, allow)
                h = h + out
                knew_full = (torch.cat([kp, knew], dim=1) if kp is not None
                             else knew)
                vnew_full = (torch.cat([vp, vnew], dim=1) if vp is not None
                             else vnew)
                doc_full = (torch.cat([past_doc, doc], dim=1)
                            if past_doc is not None else doc.clone())
                kv_out.append((knew_full, vnew_full, doc_full))
                ai += 1
            else:
                cell = entry["cell"]
                st = gstates[gi]
                S0 = st.S.clone()
                ch = st.conv.clone()
                if reset_rows is not None:
                    S0 = torch.where(
                        reset_rows[:, None, None, None].expand_as(S0),
                        torch.zeros_like(S0), S0)
                    ch = torch.where(
                        reset_rows[:, None, None].expand_as(ch),
                        torch.zeros_like(ch), ch)
                # Runs par document : le premier prolonge l'état d'entrée,
                # les suivants (nouveau doc) repartent de zéro.
                h_out = torch.zeros_like(hn)
                S_new = torch.zeros_like(S0)
                C_new = torch.zeros_like(ch)
                first = True
                for (b, s, e) in self._doc_segments(doc):
                    hb = hn[b:b + 1, s:e]
                    prolonge = first and (reset_rows is None
                                          or not bool(reset_rows[b]))
                    if prolonge:
                        s0, c0 = S0[b:b + 1], ch[b:b + 1]
                    else:
                        s0 = torch.zeros_like(S0[b:b + 1])
                        c0 = torch.zeros_like(ch[b:b + 1])
                    first = False
                    o, sn, ce = cell.forward_clean(hb, S0=s0, conv_hist=c0)
                    h_out[b:b + 1, s:e] = o
                    S_new[b:b + 1] = sn
                    C_new[b:b + 1] = ce[:, :, -(C_new.shape[-1]):]
                g_out.append(RunnerGDNState(S=S_new, conv=C_new))
                h = h + h_out
                gi += 1
            h = h + mod.mlp(mod.post_attention_layernorm(h))
        return h, g_out, kv_out

    # ---------- API publique ----------
    def fresh_states(self, batch, device):
        """États zéro (début de séquence/document)."""
        B = batch
        gstates = []
        for entry in self.layers:
            if entry["kind"] != "gdn":
                continue
            cell = entry["cell"]
            n_v = cell.W_b.shape[0]
            S = torch.zeros(B, n_v, cell.d_k, cell.d_v, device=device,
                            dtype=self.dtype)
            C = torch.zeros(B, cell.W_qkv.shape[0], cell.conv_k - 1,
                            device=device, dtype=self.dtype)
            gstates.append(RunnerGDNState(S=S, conv=C))
        return gstates, []

    def _positions(self, offset, length, batch, device):
        return (torch.arange(length, device=device) + offset
                ).unsqueeze(0).expand(batch, -1)

    def _doc_positions(self, doc, past_doc=None):
        """Positions redémarrées à 0 à chaque document (FLARE `_doc_ids_to_
        positions`, spec §10.2). `doc` [B, L] (nouveaux tokens), `past_doc`
        [B, P] ou None. Single-doc : identique à arange.

        Étape D : calcul 100 % tensoriel (zéro `.item()`, zéro
        synchronisation) — `pos[t] = t - s(t)` où `s(t)` = dernier départ
        de run (négatif si le run prolonge le passé : `s = -run0`)."""
        B, L = doc.shape
        dev = doc.device
        ar = torch.arange(L, device=dev).expand(B, L)
        if past_doc is not None and past_doc.shape[1] > 0:
            P = past_doc.shape[1]
            cont = (doc[:, :1] == past_doc[:, -1:]).squeeze(1)
            chg_p = torch.zeros(B, P, dtype=torch.bool, device=dev)
            if P > 1:
                chg_p[:, 1:] = past_doc[:, 1:] != past_doc[:, :-1]
            has = chg_p.any(dim=1)
            last = torch.where(
                has, chg_p.long().cumsum(dim=1).argmax(dim=1),
                torch.zeros(B, dtype=torch.long, device=dev))
            tail_len = torch.where(has, P - last,
                                   torch.full((B,), P, dtype=torch.long,
                                              device=dev))
            run0 = torch.where(cont, tail_len,
                               torch.zeros(B, dtype=torch.long, device=dev))
            chg0 = ~cont
        else:
            run0 = torch.zeros(B, dtype=torch.long, device=dev)
            chg0 = torch.ones(B, dtype=torch.bool, device=dev)
        shifted = torch.cat([doc[:, :1], doc[:, :-1]], dim=1)
        chg = doc != shifted
        chg[:, 0] = chg0
        neg_inf = torch.full((), -(2 ** 60), dtype=torch.long, device=dev)
        cands = torch.where(chg, ar, neg_inf)
        cands = torch.maximum(cands, (-run0)[:, None])
        s, _ = cands.cummax(dim=1)
        return ar - s

    def prefill(self, ids, doc_ids=None):
        """Hidden BRUT (pré-norme) + état propre après le préfixe causal."""
        B, L = ids.shape
        dev = ids.device
        h = self.embed(ids)
        doc = (torch.zeros(B, L, dtype=torch.long, device=dev)
               if doc_ids is None else doc_ids)
        pos = self._doc_positions(doc)
        gstates, kvstates = self.fresh_states(B, dev)
        h, gstates, kvstates = self._clean_layers(
            h, pos, gstates, kvstates, doc, reset_rows=None)
        return h, RunnerState(
            gdn=gstates, attn_k=[k for k, _, _ in kvstates],
            attn_v=[v for _, v, _ in kvstates],
            docs=[d for _, _, d in kvstates])

    def commit(self, state, block_ids, doc_ids=None):
        """Hidden BRUT + nouvel état (forward causal, SEUL écrivain)."""
        B, Lb = block_ids.shape
        dev = block_ids.device
        doc = (torch.zeros(B, Lb, dtype=torch.long, device=dev)
               if doc_ids is None else doc_ids)
        past_doc = (state.docs[0] if state.docs else None)
        pos = self._doc_positions(doc, past_doc=past_doc)
        h = self.embed(block_ids)
        kv_in = [(k, v, d) for k, v, d in zip(state.attn_k, state.attn_v,
                                              state.docs)]
        h, gstates, kvstates = self._clean_layers(
            h, pos, state.gdn, kv_in, doc, reset_rows=None)
        return h, RunnerState(
            gdn=gstates, attn_k=[k for k, _, _ in kvstates],
            attn_v=[v for _, v, _ in kvstates],
            docs=[d for _, _, d in kvstates])

    def denoise_block(self, state, block_ids, doc_ids=None):
        """Lecture depuis S̃_end (sémantique B0), SANS écriture d'état.

        Chaque bloc bruité démarre du snapshot propre courant (zéro si
        début de document) ; historique conv = copie du propre.
        Attention : TOUT le passé propre du même document est visible
        (le bloc est en tête de séquence) + bloc courant bidirectionnel.
        Garde non-écriture : `StateLeakError` si l'état est muté (T-STATE-02).
        Retourne les logits [B, Bb, V]."""
        from jainius.model.arch import StateLeakError

        B, Bb = block_ids.shape
        dev = block_ids.device
        fp_before = runner_state_fingerprint(state)
        h = self.embed(block_ids)
        doc = (torch.zeros(B, Bb, dtype=torch.long, device=dev)
               if doc_ids is None else doc_ids)
        past_doc = (state.docs[0] if state.docs else None)
        pos = self._doc_positions(doc, past_doc=past_doc)
        gi = ai = 0
        for entry in self.layers:
            kind, mod = entry["kind"], entry["mod"]
            hn = mod.input_layernorm(h)
            if kind == "attn":
                kp = state.attn_k[ai] if ai < len(state.attn_k) else None
                vp = state.attn_v[ai] if ai < len(state.attn_v) else None
                dp = state.docs[ai] if ai < len(state.docs) else None
                klen = kp.shape[1] if kp is not None else 0
                # CORRECTION (B2) : le bloc est en tête de séquence — TOUT
                # le passé propre du même document est visible (l'ancienne
                # version appliquait un masque causal sur le passé : la
                # position i ne voyait que les i+1 premières clés).
                # Sémantique identique à `_noisy_block_layers`.
                allow_past = None
                if kp is not None:
                    if dp is not None:
                        allow_past = (doc[:, :, None] == dp[:, None, :])
                    else:
                        allow_past = torch.ones(
                            B, Bb, klen, dtype=torch.bool, device=dev)
                allow_blk = torch.ones(B, Bb, Bb, dtype=torch.bool, device=dev)
                if dp is not None:
                    allow_blk = allow_blk & (doc[:, :, None] == doc[:, None, :])
                allow = (torch.cat([allow_past, allow_blk], dim=2)
                         if allow_past is not None else allow_blk)
                out, _, _ = self._attn_block(mod.self_attn, hn, pos,
                                             kp, vp, allow)
                h = h + out
                ai += 1
            else:
                cell = entry["cell"]
                st = state.gdn[gi]
                # Snapshot immuable : copies locales (jamais d'écriture).
                # La règle « zéro si début de document » est appliquée par
                # `forward_train` (qui connaît les frontières) ; ici
                # l'appelant fournit l'état de frontière exact.
                S_init = st.S.clone()
                C_init = st.conv.clone()
                out, _ = cell.forward_noisy_block(hn, S_init, C_init)
                h = h + out
                gi += 1
            h = h + mod.mlp(mod.post_attention_layernorm(h))
        logits = self.lm_head(self.final_norm(h))
        if runner_state_fingerprint(state) != fp_before:
            raise StateLeakError(
                "le forward de débruitage a écrit l'état (runner)")
        return logits

    def forward_train(self, x0, xt, unit_block, doc_ids=None, resp_start=None):
        """Two-stream complet : (h_x0, h_xt) [B, L, D] BRUTS (pré-norme).

        Reste dans le graphe autograd (P1) : aucun `detach`/`no_grad` sur le
        chemin étudiant ; les snapshots sont des `clone()` différentiables.

        Layout (B1/OPDLM) : le propre couvre toute la séquence ; les blocs
        bruités ne couvrent que la RÉPONSE `[resp_start, L)`, alignés sur
        `resp_start` (pas de bloc à cheval prompt/réponse). `resp_start`
        défaut = 0 (compatibilité : séquence entière).
        Chaque bloc bruité démarre du snapshot propre de sa frontière
        (zéro si début de document), lecture depuis S̃_end. `doc_ids`
        [B, L] (défaut : un seul document).

        P3 : snapshots du préfixe calculés en UN passage progressif (états
        propres étendus bloc par bloc, O(L)) puis réutilisés aux frontières
        — sans recalcul quadratique du préfixe. Sémantique identique à
        `_forward_train_quadratic` (cf. test d'équivalence P3).
        """
        B, L = x0.shape
        dev = x0.device
        doc = (torch.zeros(B, L, dtype=torch.long, device=dev)
               if doc_ids is None else doc_ids)
        rs = 0 if resp_start is None else resp_start
        # --- flux propre : un passage progressif, snapshots aux frontières ---
        h_emb_x0 = self.embed(x0)
        h_emb_xt = self.embed(xt)
        h_xt = torch.zeros_like(h_emb_xt)
        # États propres courants (après [0, cur)) ; snapshots par frontière.
        g_cur, kv_cur = self.fresh_states(B, dev)
        h_x0 = torch.zeros_like(h_emb_x0)
        # Préfixe [0, rs) une fois (peut être vide).
        cur = 0
        if rs > 0:
            seg = slice(0, rs)
            pos = self._doc_positions(doc[:, seg])
            h_seg, g_cur, kv_cur = self._clean_layers(
                h_emb_x0[:, seg], pos, g_cur, kv_cur, doc[:, seg],
                reset_rows=None)
            h_x0[:, seg] = h_seg
            cur = rs
        n_blocks = (L - rs + unit_block - 1) // unit_block
        for bb in range(n_blocks):
            s = rs + bb * unit_block
            e = min(s + unit_block, L)
            # Contrat : aucun bloc ne chevauche deux documents.
            assert bool(((doc[:, s:e] == doc[:, s:s + 1]).all())), (
                "bloc à cheval sur deux documents (hors contrat M1)")
            # Snapshot = états après le préfixe propre [0, s) (déjà calculés).
            # Étape D : pas de `clone()` ici — `_noisy_block_layers` et
            # `_clean_layers` clonent leurs entrées (vérifié : aucun
            # écrasement in-place des états d'entrée) ; `g_cur`/`kv_cur`
            # ne sont que réassignés, jamais mutés.
            snap_g, snap_kv = g_cur, kv_cur
            posb = self._doc_positions(doc[:, s:e], past_doc=doc[:, :s])
            zero_init = ((doc[:, s] != doc[:, s - 1]) if s > 0
                         else torch.zeros(B, dtype=torch.bool, device=dev))
            h_xt[:, s:e] = self._noisy_block_layers(
                h_emb_xt[:, s:e], posb, snap_g, snap_kv, doc[:, s:e], doc,
                zero_init)
            # Étendre le propre à travers [s, e) pour le bloc suivant.
            pos_c = self._doc_positions(doc[:, s:e], past_doc=doc[:, :s])
            h_seg, g_cur, kv_cur = self._clean_layers(
                h_emb_x0[:, s:e], pos_c, g_cur, kv_cur, doc[:, s:e],
                reset_rows=None)
            h_x0[:, s:e] = h_seg
            cur = e
        return h_x0, h_xt

    def _forward_train_quadratic(self, x0, xt, unit_block, doc_ids=None,
                                 resp_start=None):
        """Variante historique NAÏVE (P3) : snapshots recalculés depuis 0 à
        chaque bloc (O(n_blocs × L)). Conservée pour la mesure avant/après
        optimisation. Même sémantique, même contrat."""
        B, L = x0.shape
        dev = x0.device
        doc = (torch.zeros(B, L, dtype=torch.long, device=dev)
               if doc_ids is None else doc_ids)
        rs = 0 if resp_start is None else resp_start
        h = self.embed(x0)
        pos = self._doc_positions(doc)
        gstates, kvstates = self.fresh_states(B, dev)
        h_x0, gstates, kvstates = self._clean_layers(
            h, pos, gstates, kvstates, doc, reset_rows=None)
        h = self.embed(xt)
        h_xt = torch.zeros_like(h)
        n_blocks = (L - rs + unit_block - 1) // unit_block
        for bb in range(n_blocks):
            s = rs + bb * unit_block
            e = min(s + unit_block, L)
            assert bool(((doc[:, s:e] == doc[:, s:s + 1]).all())), (
                "bloc à cheval sur deux documents (hors contrat M1)")
            hb = h[:, s:e]
            posb = self._doc_positions(doc[:, s:e], past_doc=doc[:, :s])
            snap_g, snap_kv = self._clean_prefix_states(
                x0, s, doc, gstates=True)
            zero_init = ((doc[:, s] != doc[:, s - 1]) if s > 0
                         else torch.zeros(B, dtype=torch.bool, device=dev))
            h_xt[:, s:e] = self._noisy_block_layers(
                hb, posb, snap_g, snap_kv, doc[:, s:e], doc, zero_init)
        return h_x0, h_xt

    def _clean_prefix_states(self, x0, s, doc, gstates=True):
        """États propres après le préfixe [0, s) (snapshots par frontière)."""
        B, L = x0.shape
        dev = x0.device
        if s == 0:
            g, _ = self.fresh_states(B, dev)
            return g, []
        h = self.embed(x0[:, :s])
        pos = self._doc_positions(doc[:, :s])
        g0, _ = self.fresh_states(B, dev)
        h, g, kv = self._clean_layers(h, pos, g0, [], doc[:, :s],
                                      reset_rows=None)
        return g, kv

    def _noisy_block_layers(self, hb, posb, snap_g, snap_kv, doc_b,
                            doc_full, zero_init):
        """Un bloc bruité : init snapshots (zéro si `zero_init`), lecture
        S̃_end, sans écriture."""
        B, Bb, _ = hb.shape
        dev = hb.device
        h = hb
        gi = ai = 0
        for entry in self.layers:
            kind, mod = entry["kind"], entry["mod"]
            hn = mod.input_layernorm(h)
            if kind == "attn":
                past = snap_kv[ai] if ai < len(snap_kv) else None
                kp, vp, dp = (past if past is not None else (None, None, None))
                klen = kp.shape[1] if kp is not None else 0
                # Passé propre STRICT (positions < s, même doc) + bloc bidir.
                allow_past = None
                if kp is not None:
                    if dp is not None:
                        allow_past = (doc_b[:, :, None] == dp[:, None, :])
                    else:
                        allow_past = torch.ones(B, Bb, klen, dtype=torch.bool,
                                                device=dev)
                    if bool(zero_init.any()):
                        cut = zero_init[:, None, None].expand_as(allow_past)
                        allow_past = torch.where(
                            cut, torch.zeros_like(allow_past), allow_past)
                allow_blk = torch.ones(B, Bb, Bb, dtype=torch.bool, device=dev)
                allow_blk = (allow_blk
                             & (doc_b[:, :, None] == doc_b[:, None, :]))
                allow = (torch.cat([allow_past, allow_blk], dim=2)
                         if allow_past is not None else allow_blk)
                out, _, _ = self._attn_block(mod.self_attn, hn, posb,
                                             kp, vp, allow)
                h = h + out
                ai += 1
            else:
                cell = entry["cell"]
                st = snap_g[gi]
                S_init = st.S.clone()
                C_init = st.conv.clone()
                if bool(zero_init.any()):
                    cut_s = zero_init[:, None, None, None].expand_as(S_init)
                    S_init = torch.where(cut_s, torch.zeros_like(S_init),
                                         S_init)
                    cut_c = zero_init[:, None, None].expand_as(C_init)
                    C_init = torch.where(cut_c, torch.zeros_like(C_init),
                                         C_init)
                out, _ = cell.forward_noisy_block(hn, S_init, C_init)
                h = h + out
                gi += 1
            h = h + mod.mlp(mod.post_attention_layernorm(h))
        return h
