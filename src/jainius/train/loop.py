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

from jainius.decode.rollout import rollout, rollout_shifted
from jainius.decode.sampler import SamplerSpec
from jainius.distill.curriculum import rollout_length
from jainius.distill.loss import kl_teacher_to_student
from jainius.distill.targets import (
    compute_teacher_targets,
    compute_teacher_targets_shifted,
)
from jainius.model.arch import LegacyLeakyError, TwoStreamBatch
from jainius.model.twostream import shifted_supervision_mask
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


def train_step(student, teacher, optim, prompt_ids_list, trajs,
               stats: dict | None = None, *, runner=None,
               allow_legacy_leaky: bool = False) -> float:
    """Un pas : 16 lignes (1 trajectoire ⇒ 1 ligne), KL moyenne, step.

    P1 — voie par défaut : `TwoStreamRunner.forward_train` (différentiable,
    sans fuite). `runner` explicite ou `student.runner` ; sinon `LegacyLeakyError`
    sauf `allow_legacy_leaky=True` (plumbing unitaire / caractérisation).
    Le runner reste dans le graphe autograd : aucun `detach`/`no_grad` sur
    le chemin étudiant. `h_xt` brut → `runner.final_norm` → `lm_head`.

    Voie historique `legacy_leaky` (`student.forward_two_stream`) : exclue du
    chemin d'entraînement par défaut, refusée explicitement, accessible
    uniquement avec `allow_legacy_leaky=True` (diagnostic isolé).

    Alignement `current` (OPDLM `rl_sdar.py:1266-1272,1344-1348`) :
    l'étudiant n'est PAS décalé (`student_arm_shift=False`) — la position
    supervisée *i* porte la distribution du jeton *i* ; l'enseignant EST
    décalé (`Teacher.hidden_for_loss`, `roll(1)`) car sa tête AR prédit
    le jeton *i+1* à la position *i*. Sans ce décalage, l'entraînement
    superviserait « jeton suivant » de façon incohérente avec l'inférence
    (échantillonnage du jeton *i* depuis `logits[i]`). Début de séquence :
    position 0 = déchet recyclé, jamais supervisée. Supervisées : lignes
    réponse masquées-en-xt uniquement (`:1425-1428`).
    Clipping de norme 1.0 (OPDLM `max_grad_norm`, cahier) avant le pas ;
    norme pré-clip journalisée dans `stats["grad_norm"]` si fourni.
    """
    device = next(student.model.parameters()).device
    mask_id = student.spec.mask_token_id
    if runner is None:
        runner = getattr(student, "runner", None)
    use_runner = runner is not None
    if not use_runner and not allow_legacy_leaky:
        raise LegacyLeakyError(
            "train_step sans runner : le chemin historique `legacy_leaky` "
            "(forward_two_stream) est refusé en entraînement (P1) ; fournir "
            "`runner=TwoStreamRunner(hf_model)` ou `student.runner`.")
    optim.zero_grad()
    losses = []
    for pids, traj in zip(prompt_ids_list, trajs):
        x0 = torch.tensor([pids + traj.x0], dtype=torch.long, device=device)
        xt = x0.clone()
        resp = torch.zeros_like(x0, dtype=torch.bool)
        resp[0, len(pids):] = True
        xt[resp] = mask_id
        if use_runner:
            # P1 : représentations étudiantes via le runner (graphe intact).
            unit_block = int(student.spec.unit_block)
            _, h_xt_raw = runner.forward_train(
                x0, xt, unit_block=unit_block, resp_start=len(pids))
            h_xt = runner.final_norm(h_xt_raw)
            lm_head = runner.lm_head
        else:
            h_s = student.forward_two_stream(
                TwoStreamBatch(x0=x0, xt=xt, sup_mask=resp),
                allow_legacy_leaky=True,
            )
            h_xt = h_s.h_xt
            lm_head = student.model.lm_head
        # Fonction UNIQUE des cibles (A1/B1 non miroir) : aussi appelée par
        # le test de régression — toute divergence du chemin échoue au test.
        # `output_hidden_states=True` OBLIGATOIRE : sans lui, HF renvoie
        # `hidden_states=None` et `hidden_states[-1]` lève TypeError
        # (régression corrigée, verrouillée par test_smoke + test_target_alignment).
        with torch.no_grad():
            tlog = compute_teacher_targets(
                x0,
                lambda ids: teacher.model(
                    ids, output_hidden_states=True).hidden_states[-1],
                teacher.model.lm_head,
            )
            tlog = tlog.reshape(-1, tlog.shape[-1])  # [B*L, V] pour la perte
        out = kl_teacher_to_student(
            h_xt, lambda h: lm_head(h),
            None, None, resp, chunk_tokens=1024, teacher_logits=tlog,
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
    params = [p for p in student.model.parameters() if p.grad is not None]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
    if stats is not None:
        stats["grad_norm"] = grad_norm
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
        # P1 : un runner est construit une fois sur le modèle étudiant et
        # branché dans train_step (le chemin legacy est refusé par défaut).
        t0 = time.time()
        fp_before = param_fingerprint(student.model)
        step_stats: dict = {}
        from jainius.model.twostream import TwoStreamRunner as _Runner
        _runner = getattr(student, "runner", None)
        if _runner is None:
            _runner = _Runner(student.model)
        loss_value = train_step(student, teacher, optim, prompt_ids_list,
                                trajs, stats=step_stats, runner=_runner)
        rep["student_train_fw"] += len(prompt_ids_list)
        rep["teacher_fw"] += len(prompt_ids_list)
        rep["grads_finite"].append(True)  # garanti par train_step (assert)
        assert step_stats["grad_norm"] == step_stats["grad_norm"]  # fini
        rep.setdefault("grad_norms", []).append(step_stats["grad_norm"])
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


def train_step_shifted(hf_model, teacher, optim, prompt_ids_list, trajs,
                       stats: dict | None = None, *, runner, block_size: int,
                       mask_id: int) -> float:
    """Un pas J3 : KL(enseignant ‖ étudiant) sous convention `shifted`.

    Pour chaque trajectoire Diffusion-Trust (`rollout_shifted`) : x0 =
    prompt + réponse terminale ; xt = x0 avec MASK aux positions
    supervisées (graines propres, queue < Bb propre et ignorée) ;
    `runner.forward_train_shifted` (GDN mode 2, graine causale, graphe
    intact) ; règle J2 des deux côtés — la position supervisée *i*
    (offset m>=1 du bloc) se prédit depuis la ligne *m-1* du bloc
    bruité, enseignant non décalé consommé au même indice (graines
    exclues, `compute_teacher_targets_shifted`).

    `runner` explicite exigé (aucune voie historique sous `shifted`).
    Enseignant gelé (vérifié : aucun gradient côté enseignant — ses
    forwards sont sous `no_grad`). Clipping de norme 1.0 avant le pas ;
    norme pré-clip journalisée dans `stats["grad_norm"]` si fourni.
    Les trajectoires sans bloc complet sont ignorées (documenté) ; si
    aucune n'est supervisable, lève.
    """
    Bb = int(block_size)
    if Bb < 2:
        raise ValueError(f"bloc shifted : Bb={Bb} < 2 (graine + masques)")
    if runner is None:
        raise ValueError("train_step_shifted exige runner=TwoStreamRunner")
    device = next(hf_model.parameters()).device
    optim.zero_grad()
    losses = []
    n_sup_total = 0
    for pids, traj in zip(prompt_ids_list, trajs):
        P = len(pids)
        R = len(traj.x0)
        L = P + R
        x0 = torch.tensor([pids + traj.x0], dtype=torch.long, device=device)
        sup1d = shifted_supervision_mask(L, P, Bb)
        nb = int((sup1d.sum().item()) // (Bb - 1)) if R else 0
        if nb == 0:
            continue  # queue seule / réponse vide : rien à superviser
        xt = x0.clone()
        xt[0][sup1d.to(device)] = mask_id
        _, h_noisy, starts = runner.forward_train_shifted(
            x0, xt, block_size=Bb, resp_start=P)
        assert len(starts) == nb, (len(starts), nb)
        # Vues compactes par bloc : rangs [0, Bb) ; supervisés : m>=1.
        h_comp = h_noisy.reshape(1, nb * Bb, -1)
        sup_comp = torch.zeros(1, nb * Bb, dtype=torch.bool, device=device)
        for k in range(nb):
            # Rang graine (offset 0) jamais supervisé (règle J2).
            sup_comp[0, k * Bb + 1: (k + 1) * Bb] = True
        with torch.no_grad():
            t_full = compute_teacher_targets_shifted(
                x0,
                lambda ids: teacher.model(
                    ids, output_hidden_states=True).hidden_states[-1],
                teacher.model.lm_head,
            )  # [1, L, V] log-probs NON décalées
        t_comp = torch.zeros(1, nb * Bb, t_full.shape[-1],
                             dtype=t_full.dtype, device=device)
        for k, s in enumerate(starts):
            t_comp[0, k * Bb: (k + 1) * Bb] = t_full[0, s: s + Bb]
        out = kl_teacher_to_student(
            runner.final_norm(h_comp), lambda h: runner.lm_head(h),
            None, None, sup_comp, chunk_tokens=1024,
            teacher_logits=t_comp.reshape(-1, t_comp.shape[-1]),
        )
        assert out.n_supervised == nb * (Bb - 1), (
            out.n_supervised, nb, Bb)
        n_sup_total += out.n_supervised
        losses.append(out.loss)
    if not losses:
        raise ValueError("aucune trajectoire supervisable (que des queues)")
    loss = torch.stack(losses).mean()
    assert torch.isfinite(loss).all(), "perte non finie"
    loss.backward()
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in hf_model.parameters()
    ), "gradients non finis"
    # Contrat enseignant gelé : aucun gradient côté enseignant.
    assert all(
        p.grad is None
        for p in teacher.model.parameters()
    ), "l'enseignant a reçu des gradients (contrat gelé violé)"
    params = [p for p in hf_model.parameters() if p.grad is not None]
    assert params, "aucun gradient côté étudiant"
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
    if stats is not None:
        stats["grad_norm"] = grad_norm
        stats["n_supervised"] = n_sup_total
    optim.step()
    return float(loss.item())


def train_step_shifted_batched(hf_model, teacher, optim, prompt_ids_list,
                               trajs, stats: dict | None = None, *,
                               runner, block_size: int,
                               mask_id: int) -> float:
    """Pas J4 : même mathématique que `train_step_shifted`, un seul forward
    batché (J4 : débit). Exige des lignes homogènes : même longueur de
    prompt P et même longueur de réponse R (multiple de Bb) — le pilote
    bucketise les prompts par longueur exacte (pas de padding, voie
    mono-document validée en J3 inchangée, B>1 natif du runner).
    Même asserts (finitude, gel enseignant, mise à jour, clip 1.0).
    """
    Bb = int(block_size)
    if Bb < 2:
        raise ValueError(f"bloc shifted : Bb={Bb} < 2 (graine + masques)")
    if runner is None:
        raise ValueError("train_step_shifted exige runner=TwoStreamRunner")
    Ps = {len(p) for p in prompt_ids_list}
    Rs = {len(t.x0) for t in trajs}
    if len(Ps) != 1 or len(Rs) != 1:
        raise ValueError(f"lignes hétérogènes : P={sorted(Ps)}, R={sorted(Rs)}")
    P, R = Ps.pop(), Rs.pop()
    if R % Bb:
        raise ValueError(f"R={R} non multiple de Bb={Bb} (pilote J4)")
    B = len(prompt_ids_list)
    device = next(hf_model.parameters()).device
    optim.zero_grad()
    L = P + R
    x0 = torch.tensor([p + t.x0 for p, t in zip(prompt_ids_list, trajs)],
                      dtype=torch.long, device=device)
    sup1d = shifted_supervision_mask(L, P, Bb)
    nb = int(sup1d.sum().item()) // (Bb - 1)
    assert nb * Bb == R, (nb, Bb, R)
    xt = x0.clone()
    xt[:, sup1d.to(device)] = mask_id
    _, h_noisy, starts = runner.forward_train_shifted(
        x0, xt, block_size=Bb, resp_start=P)
    assert list(starts) == [P + j * Bb for j in range(nb)], starts
    h_comp = h_noisy.reshape(B, nb * Bb, -1)
    sup_comp = torch.zeros(B, nb * Bb, dtype=torch.bool, device=device)
    for k in range(nb):
        sup_comp[:, k * Bb + 1: (k + 1) * Bb] = True
    with torch.no_grad():
        t_full = compute_teacher_targets_shifted(
            x0,
            lambda ids: teacher.model(
                ids, output_hidden_states=True).hidden_states[-1],
            teacher.model.lm_head,
        )
    t_comp = torch.zeros(B, nb * Bb, t_full.shape[-1],
                         dtype=t_full.dtype, device=device)
    for k, s in enumerate(starts):
        t_comp[:, k * Bb: (k + 1) * Bb] = t_full[:, s: s + Bb]
    out = kl_teacher_to_student(
        runner.final_norm(h_comp), lambda h: runner.lm_head(h),
        None, None, sup_comp, chunk_tokens=1024,
        teacher_logits=t_comp.reshape(-1, t_comp.shape[-1]),
    )
    assert out.n_supervised == B * nb * (Bb - 1), (
        out.n_supervised, B, nb, Bb)
    loss = out.loss
    assert torch.isfinite(loss).all(), "perte non finie"
    loss.backward()
    assert all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in hf_model.parameters()
    ), "gradients non finis"
    assert all(
        p.grad is None
        for p in teacher.model.parameters()
    ), "l'enseignant a reçu des gradients (contrat gelé violé)"
    params = [p for p in hf_model.parameters() if p.grad is not None]
    assert params, "aucun gradient côté étudiant"
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
    if stats is not None:
        stats["grad_norm"] = grad_norm
        stats["n_supervised"] = out.n_supervised
    optim.step()
    return float(loss.item())


@dataclass
class DistillConfig:
    """Boucle J3 : distillation on-policy Diffusion-Trust `shifted`."""

    n_prompts: int = 4
    max_steps: int = 3
    l_max: int = 64
    l_min: int = 8
    warmup_rollouts: int = 10
    block_size: int = 4  # Bb runtime ([graine, MASK×(Bb-1)])
    steps_per_block: int = 2
    sampler_mode: str = "static"  # "static" (1 jeton/pas) | "dynamic" (seuil)
    threshold: float = 0.95  # mode dynamic (défaut = seuil serving)
    temperature: float = 0.0
    seed: int = 0
    mask_token_id: int = 248077  # défaut DiffusionSpec (Qwen3.5)
    lr: float = 1e-3  # câblage observable (cf. SmokeConfig.lr) ; le pilote
    # utilisera 1e-5 + maîtres + accumulation.


def run_distill_loop(hf_model, teacher, tokenizer, runner, cfg: DistillConfig,
                     ckpt_dir) -> dict:
    """Boucle J3 complète : curriculum → rollouts Diffusion-Trust de
    l'étudiant → KL shifted → checkpoint/reprise. Prompts synthétiques
    (câblage, pas un corpus). Lève si incohérent."""
    from jainius.train.checkpoint import save_ckpt

    Bb = int(cfg.block_size)
    mask_id = int(cfg.mask_token_id)
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
    optim = MasterOptimizer(hf_model, lr=cfg.lr)
    hf_model.train()
    rep = {
        "lengths": [], "losses": [], "grad_norms": [], "n_supervised": [],
        "params_fp": [], "rollout_fw": 0, "student_train_fw": 0,
        "teacher_fw": 0, "updates": 0, "prompts_per_step": [],
        "gpu_seconds": {}, "convention": "shifted",
        "block_size": Bb,
    }
    saved = {}
    for step in range(cfg.max_steps):
        length = rollout_length(
            step, l_min=cfg.l_min, l_max=cfg.l_max,
            warmup_rollouts=cfg.warmup_rollouts, block_size=Bb,
        )
        rep["lengths"].append(length)
        t0 = time.time()
        spec = SamplerSpec(mode=cfg.sampler_mode, threshold=cfg.threshold,
                           temperature=cfg.temperature,
                           steps_per_block=cfg.steps_per_block,
                           seed=cfg.seed + step)
        round_fp = param_fingerprint(hf_model)
        trajs = [
            rollout_shifted(runner, pids, length, spec,
                            block_size=Bb, mask_id=mask_id)
            for pids in prompt_ids_list
        ]
        for t in trajs:
            t.weight_version = round_fp
        rep["rollout_fw"] += sum(t.n_forwards for t in trajs)
        assert len(trajs) == cfg.n_prompts
        assert all(len(t.x0) == length for t in trajs), (
            "rollout shifted exact (pas d'EOS sans eos_id) : "
            f"{[len(t.x0) for t in trajs]}")
        with torch.no_grad():
            replay = rollout_shifted(runner, prompt_ids_list[0], length,
                                     spec, block_size=Bb, mask_id=mask_id)
        assert replay.x0 == trajs[0].x0, "fuite d'état entre rollouts"
        rep["rollout_fw"] += replay.n_forwards
        rep["gpu_seconds"][f"rollout_{step}"] = round(time.time() - t0, 2)
        t0 = time.time()
        fp_before = param_fingerprint(hf_model)
        step_stats: dict = {}
        loss_value = train_step_shifted(
            hf_model, teacher, optim, prompt_ids_list, trajs,
            stats=step_stats, runner=runner, block_size=Bb, mask_id=mask_id)
        rep["student_train_fw"] += len(prompt_ids_list)
        rep["teacher_fw"] += len(prompt_ids_list)
        assert step_stats["grad_norm"] == step_stats["grad_norm"]  # fini
        rep["grad_norms"].append(step_stats["grad_norm"])
        rep["n_supervised"].append(step_stats["n_supervised"])
        rep["updates"] += 1
        rep["losses"].append(loss_value)
        changed = param_fingerprint(hf_model) != fp_before
        assert changed, f"aucune mise à jour au pas {step}"
        rep["params_fp"].append(param_fingerprint(hf_model))
        rep["prompts_per_step"].append(cfg.n_prompts)
        rep["gpu_seconds"][f"train_{step}"] = round(time.time() - t0, 2)
        if step == 1:
            rng_cuda = (torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available() else None)
            saved["dir"] = save_ckpt(
                ckpt_dir, hf_model, optim.optim, torch.get_rng_state(),
                step, {"distill": True, "shifted": True, "seed": cfg.seed,
                       "lr": cfg.lr, "block_size": Bb},
                masters=optim.masters, rng_cuda=rng_cuda,
                extra={"curriculum_step": step, "length": length,
                       "updates": rep["updates"],
                       "budget": {"prompts": sum(rep["prompts_per_step"]),
                                  "student_train_fw": rep["student_train_fw"],
                                  "teacher_fw": rep["teacher_fw"],
                                  "rollout_fw": rep["rollout_fw"]}})
            saved["params_fp"] = param_fingerprint(hf_model)
    rep["saved"] = {k: (str(v) if not isinstance(v, str) else v)
                    for k, v in saved.items() if k == "dir"}
    rep["saved_fp"] = saved.get("params_fp", "")
    return rep
