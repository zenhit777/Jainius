"""JAI-ROLL / JAI-GEN — rollout par blocs, référence CAUSALE (M1-B, B3).

Convention ``current`` : un bloc = ``[MASK] × B`` ; le logit à la position
*i* prédit le jeton *i*. Sémantique démontrée (test
`test_rollout_causal_intra_block`) : décodage causal par blocs —
préremplissage causal du prompt (écrit l'état), passes causales par bloc
(``denoise``, sans écriture), commit causal du bloc finalisé (écrit
l'état). Ce N'EST PAS du Diffusion-Trust : un vrai débruitage
bidirectionnel intra-bloc exige le contrat d'état GDN (M2), la pile HF
restant causale par construction. Séparation explicite : calcul des
logits (``JainiusModel``) / choix des tokens (``sampler``) / validation
d'état (``commit``).
"""

from __future__ import annotations

import torch

from jainius.decode.sampler import SamplerSpec, sample_token
from jainius.decode.trajectory import Round, Trajectory


def rollout(
    model,
    prompt_ids: list[int],
    response_len: int,
    sampler: SamplerSpec,
    *,
    eos_id: int | None = None,
) -> Trajectory:
    """Trajectoire de débruitage par blocs (T-ROLL-01)."""
    block = model.spec.block_size
    mask_id = model.spec.mask_token_id
    if response_len % block:
        raise ValueError(f"response_len={response_len} non multiple de B={block}")
    device = next(model.model.parameters()).device
    # Générateur sur le device des logits (B4.2) : un générateur CPU fait
    # échouer `multinomial` sur CUDA (RuntimeError constaté).
    rng = torch.Generator(device=device).manual_seed(sampler.seed)
    traj = Trajectory(
        prompt_ids=list(prompt_ids),
        x0=[],
        model_version=model.model_version(),
        seed=sampler.seed,
        sampler=sampler,
    )
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    _, cache = model.prefill(prompt)
    traj.n_forwards += 1
    for b in range(response_len // block):
        cur = [mask_id] * block
        for step in range(sampler.steps_per_block):
            if mask_id not in cur:
                break
            probe = torch.tensor([cur], dtype=torch.long, device=device)
            logits = model.denoise(probe, cache)[0]  # [B, V]
            traj.n_forwards += 1
            traj.rounds.append(
                Round(block=b, step=step, ids=list(cur), committed=[False] * block)
            )
            sampled, conf = [], []
            for i in range(block):
                if cur[i] == mask_id:
                    t, c = sample_token(logits[i], sampler, rng)
                    sampled.append(t)
                    conf.append(c)
                else:
                    sampled.append(cur[i])
                    conf.append(float("-inf"))
            if sampler.mode == "dynamic" and sampler.threshold <= 1.0:
                engage = [c >= sampler.threshold for c in conf]
                if not any(engage):
                    top = max(range(block), key=lambda i: conf[i])
                    engage[top] = True
            else:  # statique : 1 jeton par pas (le plus confiant)
                top = max(range(block), key=lambda i: conf[i])
                engage = [i == top for i in range(block)]
            for i in range(block):
                if engage[i] and cur[i] == mask_id:
                    cur[i] = sampled[i]
                    traj.rounds[-1].committed[i] = True
            if eos_id is not None and eos_id in cur:
                traj.eos_stop = True
                break
        if mask_id in cur:  # résidu : résolution argmax déterministe
            probe = torch.tensor([cur], dtype=torch.long, device=device)
            logits = model.denoise(probe, cache)[0]
            traj.n_forwards += 1
            for i in range(block):
                if cur[i] == mask_id:
                    cur[i] = int(logits[i].argmax().item())
        if traj.eos_stop:  # tronque au premier EOS après résolution
            cut = cur.index(eos_id) if eos_id in cur else block
            cur = cur[: cut + 1]
            traj.x0.extend(cur)
            block_t = torch.tensor([cur], dtype=torch.long, device=device)
            _, cache = model.commit(block_t, cache)
            traj.n_forwards += 1
            break
        traj.x0.extend(cur)
        block_t = torch.tensor([cur], dtype=torch.long, device=device)
        _, cache = model.commit(block_t, cache)
        traj.n_forwards += 1
        if traj.eos_stop:
            break
    return traj


def rollout_shifted(
    runner,
    prompt_ids: list[int],
    response_len: int,
    sampler: SamplerSpec,
    *,
    block_size: int,
    mask_id: int,
    eos_id: int | None = None,
) -> Trajectory:
    """Trajectoire Diffusion-Trust `shifted` (J3, on-policy réelle).

    Réplique d'entraînement de l'algorithme serving natif
    (`LowConfidenceShiftHybridDiffusion.run` de l'oracle FLARE, lu tel
    quel — D-04) : préremplissage causal (la graine 0 est échantillonnée
    sur son dernier logit), puis par bloc runtime `[graine, MASK×(Bb-1)]`
    — passes de débruitage `denoise_block_shifted` (GDN mode 2,
    graine causale), échantillonnage du jeton en position i≥1 depuis
    `logits[i-1]` (la graine n'est jamais rééchantillonnée), engagement
    par confiance (mode `dynamic` ≡ serving : seuil puis top-1 de repli ;
    mode `static` : 1 jeton par pas ; dernier pas : tout le résidu),
    commit causal du bloc complet (GDN mode 1) dont le dernier logit
    échantillonne la graine suivante.

    Différences training-side documentées (pas de portée serving) :
    `static` n'existe pas côté serving (seuil 0,95 par défaut) ; l'arrêt
    EOS tronque au bloc courant (queue < Bb ignorée côté perte).
    `response_len` doit être multiple de Bb. Retourne la trajectoire
    (x0 = réponse, graines incluses, Bb jetons par bloc).
    """
    Bb = int(block_size)
    if Bb < 2:
        raise ValueError(f"bloc shifted : Bb={Bb} < 2 (graine + masques)")
    if sampler.steps_per_block < 1:
        raise ValueError("steps_per_block >= 1 exigé (budget Diffusion-Trust)")
    if response_len % Bb:
        raise ValueError(
            f"response_len={response_len} non multiple de Bb={Bb}")
    device = next(runner.model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(sampler.seed)
    traj = Trajectory(
        prompt_ids=list(prompt_ids),
        x0=[],
        model_version=f"twostream-shifted-Bb{Bb}",
        seed=sampler.seed,
        sampler=sampler,
    )
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h_pre, state = runner.prefill(prompt)
        traj.n_forwards += 1
        pre_logits = runner.lm_head(runner.final_norm(h_pre))[:, -1, :]
        seed, _ = sample_token(pre_logits[0], sampler, rng)
        for j in range(response_len // Bb):
            cur = [seed] + [mask_id] * (Bb - 1)
            budget = sampler.steps_per_block
            for step in range(budget):
                if mask_id not in cur[1:]:
                    break
                probe = torch.tensor([cur], dtype=torch.long, device=device)
                logits = runner.denoise_block_shifted(state, probe)[0]
                traj.n_forwards += 1
                traj.rounds.append(
                    Round(block=j, step=step, ids=list(cur),
                          committed=[False] * Bb)
                )
                sampled, conf = [], []
                for i in range(Bb):
                    if i >= 1 and cur[i] == mask_id:
                        t, c = sample_token(logits[i - 1], sampler, rng)
                        sampled.append(t)
                        conf.append(c)
                    else:
                        sampled.append(cur[i])
                        conf.append(float("-inf"))
                is_last = step == budget - 1
                if is_last:
                    engage = [c > float("-inf") for c in conf]
                elif sampler.mode == "dynamic" and sampler.threshold <= 1.0:
                    engage = [c >= sampler.threshold for c in conf]
                    if not any(engage):
                        top = max(range(Bb), key=lambda i: conf[i])
                        engage[top] = True
                else:  # statique : 1 jeton par pas (le plus confiant)
                    top = max(range(Bb), key=lambda i: conf[i])
                    engage = [i == top for i in range(Bb)]
                engage[0] = False  # la graine n'est jamais rééchantillonnée
                for i in range(Bb):
                    if engage[i] and cur[i] == mask_id:
                        cur[i] = sampled[i]
                        traj.rounds[-1].committed[i] = True
            assert mask_id not in cur[1:], (
                f"bloc {j} non résolu après {budget} pas (serving : "
                "le dernier pas engage tout le résidu)")
            if eos_id is not None and eos_id in cur:
                cut = cur.index(eos_id) + 1
                cur = cur[:cut]
                traj.x0.extend(cur)
                block_t = torch.tensor([cur], dtype=torch.long, device=device)
                _, state = runner.commit(state, block_t)
                traj.n_forwards += 1
                traj.eos_stop = True
                break
            traj.x0.extend(cur)
            block_t = torch.tensor([cur], dtype=torch.long, device=device)
            h_c, state = runner.commit(state, block_t)
            traj.n_forwards += 1
            next_logits = runner.lm_head(runner.final_norm(h_c))[:, -1, :]
            seed, _ = sample_token(next_logits[0], sampler, rng)
    return traj


def generate(
    model,
    tokenizer,
    prompt_text: str,
    sampler: SamplerSpec,
    *,
    max_new_tokens: int = 32,
) -> tuple[str, Trajectory, dict]:
    """Génération complète de référence (T-GEN-01)."""
    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    traj = rollout(
        model, prompt_ids, max_new_tokens, sampler, eos_id=tokenizer.eos_token_id
    )
    out_text = tokenizer.decode(traj.x0, skip_special_tokens=True)
    stats = {
        "n_forwards": traj.n_forwards,
        "n_rounds": len(traj.rounds),
        "eos_stop": traj.eos_stop,
        "out_len": len(traj.x0),
    }
    return out_text, traj, stats
