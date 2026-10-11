"""JAI-TRAIN smoke (M1-D0) — câblage sans campagne d'entraînement.

Assemble : curriculum → rollouts étudiant → états par bloc (tour
all-mask dégénéré, documenté) → cibles enseignant (voie simple H-03 +
contrôle extended) → KL OPDLM → SGD/AdamW → checkpoint/reprise.
AUCUN terme FLARE. Convention `current` uniquement.

Config smoke (§11.3) : 16 prompts, 2 rollouts, L_max=128, 5 pas.
Micro-paramètres assumés et documentés : réponse plafonnée à
``response_cap=8`` (vitesse). CONSÉQUENCE ASSUMÉE : toutes les réponses
générées font 8 tokens — l'effet effectif du curriculum sur les longueurs
de rollout N'EST PAS mesuré ici (seules les valeurs L_s calculées sont
vérifiées) ; la mesure longueurs relève d'un pilote, pas du smoke.
``steps_per_block=1``, température 0.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch

from jainius.decode.rollout import rollout
from jainius.decode.sampler import SamplerSpec
from jainius.distill.curriculum import rollout_length
from jainius.distill.loss import kl_teacher_to_student
from jainius.model.arch import TwoStreamBatch
from jainius.train.master import MasterOptimizer

# Jeu SYNTHÉTIQUE déterministe (câblage uniquement — pas une évaluation).
# Substitution documentée : `divelab/opdlm_train_data/opdlm_train.json`
# pèse 1,72 Go (LFS) — téléchargement massif injustifié pour 16 prompts.
SYNTHETIC_PROMPTS = [
    "What is 2+2?",
    "Name a planet.",
    "Solve: x+3=7.",
    "Write a haiku about rain.",
    "What is the capital of France?",
    "Count to five.",
    "Define entropy briefly.",
    "Who wrote Hamlet?",
    "What is 9*8?",
    "Name a primary color.",
    "Say hello in Spanish.",
    "What is H2O?",
    "List two mammals.",
    "What day follows Monday?",
    "Spell the word cat backwards.",
    "What is 100/4?",
]


@dataclass
class SmokeConfig:
    n_prompts: int = 16
    n_rollouts: int = 2
    l_max: int = 128
    max_steps: int = 5
    l_min: int = 8
    warmup_rollouts: int = 10
    response_cap: int = 8
    seed: int = 0
    # lr de câblage (B4.3) : à 1e-5 + bf16 pur, les updates s'arrondissent
    # à zéro (ULP ≫ update) même avec maîtres côté recopie — 1e-3 rend les
    # mises à jour du modèle OBSERVABLES. Pas une recommandation
    # d'entraînement (le pilote utilisera 1e-5 + maîtres + accumulation).
    lr: float = 1e-3
    compat_tol: float = 5e-2  # bf16, parité extended/simple (H-03)


def param_fingerprint(model) -> str:
    views = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        digest.update(name.encode())
        t = param.detach().to("cpu").contiguous()
        digest.update(t.view(views[t.element_size()]).numpy().tobytes())
    return digest.hexdigest()


def train_step(student, teacher, optim, prompt_ids_list, trajs) -> float:
    """Un pas : 16 lignes (1 trajectoire ⇒ 1 ligne), KL moyenne, step.

    Alignement `current` (OPDLM `rl_sdar.py:1266-1272,1344-1348`) :
    l'étudiant n'est PAS décalé (`student_arm_shift=False`) — la position
    supervisée *i* porte la distribution du jeton *i* ; l'enseignant EST
    décalé (`Teacher.hidden_for_loss`, `roll(1)`) car sa tête AR prédit
    le jeton *i+1* à la position *i*. Sans ce décalage, l'entraînement
    superviserait « jeton suivant » de façon incohérente avec l'inférence
    (échantillonnage du jeton *i* depuis `logits[i]`). Début de séquence :
    position 0 = déchet recyclé, jamais supervisée. Supervisées : lignes
    réponse masquées-en-xt uniquement (`:1425-1428`).
    """
    device = next(student.model.parameters()).device
    mask_id = student.spec.mask_token_id
    optim.zero_grad()
    losses = []
    for pids, traj in zip(prompt_ids_list, trajs):
        x0 = torch.tensor([pids + traj.x0], dtype=torch.long, device=device)
        xt = x0.clone()
        resp = torch.zeros_like(x0, dtype=torch.bool)
        resp[0, len(pids):] = True
        xt[resp] = mask_id
        h_s = student.forward_two_stream(
            TwoStreamBatch(x0=x0, xt=xt, sup_mask=resp)
        )
        with torch.no_grad():
            h_t = teacher.hidden_for_loss(x0)
        out = kl_teacher_to_student(
            h_s.h_xt, lambda h: student.model.lm_head(h),
            h_t, lambda h: teacher.model.lm_head(h),
            resp, chunk_tokens=1024,
        )
        assert out.n_supervised == len(traj.x0)
        losses.append(out.loss)
    loss = torch.stack(losses).mean()
    assert torch.isfinite(loss).all(), "perte non finie"
    loss.backward()
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in student.model.parameters()
    ), "gradients non finis"
    optim.step()
    return float(loss.item())


def assert_traj_fresh(traj, model) -> None:
    """Lève si la trajectoire est obsolète (poids modifiés depuis sa
    génération) ou sans empreinte. À appeler avant toute réutilisation
    d'une trajectoire enregistrée (pas 2-4 du smoke, reprise)."""
    if not traj.weight_version:
        raise ValueError("trajectoire sans empreinte poids (stale inconnue)")
    if param_fingerprint(model) != traj.weight_version:
        raise ValueError("trajectoire obsolète : poids modifiés depuis le rollout")


def run_smoke(student, teacher, tokenizer, cfg: SmokeConfig, ckpt_dir) -> dict:
    """Exécute le smoke et retourne compteurs + preuves (lève si incohérent)."""
    from jainius.train.checkpoint import save_ckpt

    assert len(SYNTHETIC_PROMPTS) >= cfg.n_prompts
    prompts = SYNTHETIC_PROMPTS[: cfg.n_prompts]
    prompt_ids_list = []
    for text in prompts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids_list.append(
            tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        )
    assert teacher.is_frozen()
    optim = MasterOptimizer(student.model, lr=cfg.lr)
    student.model.train()
    rep = {
        "prompts_per_step": [],
        "lengths": [],
        "losses": [],
        "grads_finite": [],
        "params_changed": [],
        "params_fp": [],
        "student_train_fw": 0,
        "teacher_fw": 0,  # forward `hidden` uniquement (16/pas) ; le contrôle
        "rollout_fw": 0,  # extended vit dans le diagnostic séparé (pré-M2)
        "updates": 0,
        "trajectories": {},
        "gpu_seconds": {},
        "l_max_respected": True,
    }
    saved = {}
    for step in range(cfg.max_steps):
        length = rollout_length(
            step, l_min=cfg.l_min, l_max=cfg.l_max,
            warmup_rollouts=cfg.warmup_rollouts,
        )
        rep["lengths"].append(length)
        response_len = min(length, cfg.response_cap)
        assert response_len % student.spec.block_size == 0
        assert length <= cfg.l_max and response_len <= cfg.l_max
        t0 = time.time()
        trajs = None
        if step < cfg.n_rollouts:
            spec = SamplerSpec(mode="static", temperature=0.0,
                               steps_per_block=1, seed=cfg.seed + step)
            # Empreinte poids UNE fois par round (jamais par token) : toute
            # réutilisation ultérieure exige assert_traj_fresh (B4.1).
            round_fp = param_fingerprint(student.model)
            trajs = [
                rollout(student, pids, response_len, spec)
                for pids in prompt_ids_list
            ]
            for t in trajs:
                t.weight_version = round_fp
            rep["trajectories"][step] = trajs
            rep["rollout_fw"] += sum(t.n_forwards for t in trajs)
            assert len(trajs) == cfg.n_prompts
            # n_forwards exact : 1 prefill + 2 blocs × (1 denoise + 1 résidu + 1 commit).
            assert all(t.n_forwards == 7 for t in trajs), [
                t.n_forwards for t in trajs
            ]
            # Isolation inter-rollouts (poids identiques ici) : le prompt 0
            # rejoué après les 15 autres donne la même trajectoire.
            with torch.no_grad():
                replay = rollout(student, prompt_ids_list[0], response_len, spec)
            assert replay.x0 == trajs[0].x0, "fuite d'état entre rollouts"
            rep["rollout_fw"] += replay.n_forwards
        else:
            # Réutilisation après mise à jour : obsolète PAR CONSTRUCTION
            # dans ce smoke (style NPD single-rollout-multiple-updates) —
            # la politique de fraîcheur (fenêtre, re-génération) relève de
            # M2. Le test de reprise, lui, exige la fraîcheur exacte.
            trajs = rep["trajectories"][cfg.n_rollouts - 1]
        rep["gpu_seconds"][f"rollout_{step}"] = round(time.time() - t0, 2)
        # --- pas d'entraînement (factorisé, réutilisé par le test reprise) ---
        t0 = time.time()
        fp_before = param_fingerprint(student.model)
        loss_value = train_step(student, teacher, optim, prompt_ids_list, trajs)
        rep["student_train_fw"] += len(prompt_ids_list)
        rep["teacher_fw"] += len(prompt_ids_list)
        rep["grads_finite"].append(True)  # garanti par train_step (assert)
        rep["updates"] += 1
        rep["losses"].append(loss_value)
        changed = param_fingerprint(student.model) != fp_before
        assert changed, f"aucune mise à jour au pas {step}"
        rep["params_changed"].append(changed)
        rep["params_fp"].append(param_fingerprint(student.model))
        rep["prompts_per_step"].append(cfg.n_prompts)
        rep["gpu_seconds"][f"train_{step}"] = round(time.time() - t0, 2)
        if step == 1:
            rng_cuda = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None)
            saved["dir"] = save_ckpt(
                ckpt_dir, student.model, optim.optim, torch.get_rng_state(),
                step, {"smoke": True, "seed": cfg.seed, "lr": cfg.lr},
                masters=optim.masters, rng_cuda=rng_cuda,
                extra={"curriculum_step": step, "length": length,
                       "updates": rep["updates"],
                       "budget": {"prompts": sum(rep["prompts_per_step"]),
                                  "student_train_fw": rep["student_train_fw"],
                                  "teacher_fw": rep["teacher_fw"],
                                  "rollout_fw": rep["rollout_fw"]}})
            saved["params_fp"] = param_fingerprint(student.model)
    rep["saved"] = {k: (str(v) if not isinstance(v, str) else v)
                    for k, v in saved.items() if k == "dir"}
    rep["saved_fp"] = saved.get("params_fp", "")
    return rep
