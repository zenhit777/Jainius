"""Runner B0 — mesure de la baseline qualité du modèle source.

Exécute le protocole figé P-G/1.0 (voir `harness.py`) sur le
modèle source Qwen3.5-0.8B servi par SGLang (B0-plain), sur
les 7 benchmarks du composite, avec `n_repetitions` répétitions
(temperature 0 : les répétitions mesurent le plancher de bruit
du système, pas l'échantillonnage).

Moteur de génération : SGLang HTTP (`/generate`). Aucun moteur
n'est implémenté ici (ADR-008/ADR-011 : le serving est celui
de SGLang ; ce runner est un client).

Plafonds de génération : le protocole P-G/1.0 fige
max_new_tokens=512, mais la mesure du taux de troncature
(2026-10-12) montre que 512 coupe 60 % de MATH-500 (sans
\\boxed{}) et 20 % de GSM8K — la baseline serait
systématiquement sous-estimée. Plafonds relevés pour les
datasets de raisonnement long (documentés dans la sortie
JSON ; déviation justifiée par preuve, réversible).

Scorers :
- GSM8K : nombre après «####» s'il existe, sinon dernier
  nombre (exact-match normalisé) ;
- MATH-500 : dernier \\boxed{} (parsing par profondeur
  pour les accolades imbriquées ; normalisation LaTeX
  minimale : espaces, virgules, \\left/\\right, \\( \\)) ;
- MMLU / MMLU-Pro : première lettre (A-D / A-J) ;
- IFEval : vérificateurs de contraintes prompt-level (toutes
  les contraintes d'un prompt doivent passer ; les contraintes
  non implémentées sont comptées comme échecs ET listées) ;
- HumanEval / MBPP : exécution des tests dans un sous-processus
  (timeout 10 s, pass@1). MBPP utilise la signature de
  référence du dataset (le nom de fonction est fixe dans
  les tests).

Usage :
  python -m jainius.eval.b0_runner --sglang-url http://127.0.0.1:30000 \
      --out _artefacts/b0/b0-results.json [--limit 5] [--concurrency 8]

Le mode `--limit N` exécute N exemples par dataset (smoke test
du runner ; ne constitue PAS une mesure B0).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import requests
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from jainius.eval.harness import COMPOSITE, DATASETS, GENERATION, protocol_hash

MODEL_ID = "Qwen/Qwen3.5-0.8B"

# Plafond de génération par dataset (voir docstring —
# déviation justifiée par taux de troncature mesuré).
MAX_NEW_TOKENS = {
    "gsm8k": 2048,
    "math500": 2048,
    "ifeval": 1024,
}
DEFAULT_MAX_NEW_TOKENS = 512

# Prompts système (documentés ; le protocole P-G/1.0 fige les
# datasets, la génération et le composite — les prompts système
# sont consignés dans la sortie JSON pour reproductibilité).
SYSTEM_PROMPTS = {
    "gsm8k": ("Solve the math word problem step by step. "
              "Conclude with the final number after «####»."),
    "math500": ("Solve the math problem step by step. "
                "Put the final answer in \\boxed{}."),
    "mmlu": ("Answer the multiple-choice question. "
             "Respond with a single letter (A, B, C or D) and nothing else."),
    "mmlu_pro": ("Answer the multiple-choice question. "
                 "Respond with a single letter (A through J) and nothing else."),
    "ifeval": ("Follow every instruction in the prompt exactly."),
    "humaneval": ("You are a Python assistant. Complete the function "
                  "described in the prompt. Output only the code."),
    "mbpp": ("You are a Python assistant. Write a function matching the "
             "description. Output only the code."),
}

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


# --------------------------------------------------------------------------
# Client SGLang (B0-plain : endpoint /generate, AR natif)
# --------------------------------------------------------------------------


class SGLangClient:
    """Client minimal du endpoint /generate de SGLang."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()
        self.failures = 0

    def generate(self, prompt: str,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "stop": STOP_TOKENS,
            },
        }
        for attempt in range(1, 6):
            try:
                r = requests.post(f"{self.base_url}/generate",
                                  json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["text"]
            except Exception as e:  # noqa: BLE001 — retry puis échec
                if attempt == 5:
                    with self._lock:
                        self.failures += 1
                    raise RuntimeError(
                        f"SGLang /generate échoué : {e}") from e
                # backoff croissant : les resets sous
                # charge sont transitoires (SGLang
                # récupère en quelques secondes)
                time.sleep(5 * attempt)
        raise RuntimeError("inaccessible")


def mbpp_signature(row: dict) -> str:
    """Signature de référence (nom de fonction fixe dans les tests)."""
    return row["code"].split("\n")[0].strip()


def build_prompt(tokenizer, dataset: str, row: dict) -> str:
    """Prompt (chat template Qwen3.5, thinking désactivé)."""
    system = SYSTEM_PROMPTS[dataset]
    if dataset == "gsm8k":
        user = f"Question: {row['question']}"
    elif dataset == "math500":
        user = f"Problem: {row['problem']}"
    elif dataset == "mmlu":
        choices = row["choices"]
        opts = ("\n".join(f"{k}. {v}" for k, v in choices.items())
                if isinstance(choices, dict)
                else "\n".join(f"{chr(65 + i)}. {v}"
                               for i, v in enumerate(choices)))
        user = f"Question: {row['question']}\n{opts}\nAnswer:"
    elif dataset == "mmlu_pro":
        choices = row["options"]
        opts = ("\n".join(f"{k}. {v}" for k, v in choices.items())
                if isinstance(choices, dict)
                else "\n".join(f"{chr(65 + i)}. {v}"
                               for i, v in enumerate(choices)))
        user = f"Question: {row['question']}\n{opts}\nAnswer:"
    elif dataset == "ifeval":
        user = row["prompt"]
    elif dataset == "humaneval":
        user = row["prompt"]
    elif dataset == "mbpp":
        user = (f"Write a Python function: {row['text']}\n"
                f"{mbpp_signature(row)}")
    else:
        raise ValueError(f"dataset inconnu : {dataset}")

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        # transformers plus ancien sans chat_template_kwargs
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------
# Extraction de réponses
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def extract_boxed(text: str) -> str | None:
    """Dernier \\boxed{} avec accolades imbriquées (parsing
    par compteur de profondeur — un regex [^}] s'arrête au
    premier } intérieur de \\frac{a}{b} etc.)."""
    last = None
    for m in re.finditer(r"\\boxed\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end():i - 1]
    return last


def normalize_math(expr: str) -> str:
    """Normalisation LaTeX minimale standard (MATH-500) :
    espaces, virgules, \\left/\\right, \\( \\) — les
    structures \\frac{}{} et symboles sont comparés tels quels."""
    return (expr.replace(" ", "").replace(",", "")
                .replace(r"\left", "").replace(r"\right", "")
                .replace(r"\(", "").replace(r"\)", ""))


def first_letter(text: str, n_choices: int) -> str | None:
    letters = "ABCDEFGHIJ"[:n_choices]
    m = re.search(rf"\b([{letters}])\b", text.strip())
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Scorers GSM8K / MATH / MMLU
# --------------------------------------------------------------------------


def score_gsm8k(row: dict, response: str) -> bool:
    ref = row["answer"].split("####")[-1].strip()
    m = re.search(r"####\s*(-?[\d,]*(?:\.\d+)?)", response)
    got = m.group(1) if m else last_number(response)
    return got is not None and normalize_math(got) == normalize_math(ref)


def score_math500(row: dict, response: str) -> bool:
    ref = row["answer"]
    got = extract_boxed(response)
    if got is None:
        got = last_number(response) or ""
    return normalize_math(got) == normalize_math(ref)


def score_mmlu(row: dict, response: str) -> bool:
    ans = row["answer"]
    ref = "ABCD"[ans] if isinstance(ans, int) else ans
    return first_letter(response, 4) == ref


def score_mmlu_pro(row: dict, response: str) -> bool:
    ans = row["answer"]
    ref = "ABCDEFGHIJ"[ans] if isinstance(ans, int) else ans
    return first_letter(response, 10) == ref


# --------------------------------------------------------------------------
# IFEval — vérificateurs de contraintes (IDs et kwargs réels
# du dataset google/IFEval, vérifiés le 2026-10-12).
#
# Un prompt est réussi ssi TOUTES ses contraintes sont
# satisfaites (prompt-level, standard IFEval). Les contraintes
# non implémentées sont comptées comme échecs et listées dans
# `unimplemented` (honnêteté du score, pas d'échec silencieux).
# --------------------------------------------------------------------------


def _words(t: str) -> list[str]:
    return t.split()


def _sentences(t: str) -> list[str]:
    return [s for s in re.split(r"[.!?]+", t) if s.strip()]


def _paragraphs(t: str) -> list[str]:
    return [p for p in t.strip().split("\n\n") if p.strip()]


def _rel_ok(value: int, target: int, relation: str | None) -> bool:
    if relation == "at_least":
        return value >= target
    if relation == "at_most":
        return value <= target
    return value == target  # "exactly" ou relation absente


def _bullet_groups(t: str) -> int:
    """Nombre de listes à puces (groupes de lignes * ou -)."""
    groups, in_group = 0, False
    for line in t.split("\n"):
        s = line.strip()
        if s.startswith(("* ", "- ")):
            if not in_group:
                groups += 1
                in_group = True
        else:
            in_group = False
    return groups


def _check_ifeval(t: str, iid: str, kw: dict) -> bool:
    category, _, key = iid.partition(":")

    if iid == "punctuation:no_comma":
        return "," not in t

    if category == "change_case":
        if key == "english_lowercase":
            return t == t.lower()
        if key == "english_capital":
            return t == t.upper()
        if key == "capital_word_frequency":
            caps = sum(1 for w in _words(t) if w[:1].isupper())
            return _rel_ok(caps, kw["capital_frequency"],
                           kw["capital_relation"])

    if category == "combination":
        if key == "repeat_prompt":
            return t.startswith(kw["prompt_to_repeat"])
        if key == "two_responses":
            parts = t.split("######")
            return len(parts) == 2 and all(p.strip() for p in parts)

    if category == "detectable_content":
        if key == "number_placeholders":
            n = len(re.findall(r"\[[^\]]+\]", t))
            return _rel_ok(n, kw["num_placeholders"], "exactly")
        if key == "postscript":
            return t.rstrip().endswith(kw.get("postscript_marker") or "P.S.")

    if category == "detectable_format":
        if key == "number_highlighted_sections":
            return _rel_ok(t.count("==") // 2, kw["num_highlights"],
                           "exactly")
        if key == "number_bullet_lists":
            return _rel_ok(_bullet_groups(t), kw["num_bullets"], "at_least")
        if key == "json_format":
            try:
                json.loads(t.strip().strip("`").removeprefix("json").strip())
                return True
            except Exception:  # noqa: BLE001
                return False
        if key == "title":
            return t == t.title()
        if key == "multiple_sections":
            n = t.count(kw["section_spliter"]) + 1
            return _rel_ok(n, kw["num_sections"], "at_least")
        if key == "constrained_response":
            # kwargs vides dans le dataset ; convention interne non
            # déductible — non implémenté (10/541 prompts).
            raise NotImplementedError(iid)

    if category == "keywords":
        if key == "existence":
            return all(k in t for k in kw["keywords"])
        if key == "forbidden_words":
            return not any(k in t for k in kw["forbidden_words"])
        if key == "frequency":
            n = len(re.findall(rf"\b{re.escape(kw['keyword'])}\b", t))
            return _rel_ok(n, kw["frequency"], kw["relation"])
        if key == "letter_frequency":
            n = t.lower().count(kw["letter"].lower())
            return _rel_ok(n, kw["let_frequency"], kw["let_relation"])

    if category == "language":
        if key == "response_language":
            # détection de langue hors périmètre — non implémenté
            # (31/541 prompts).
            raise NotImplementedError(iid)

    if category == "length_constraints":
        if key == "number_words":
            return _rel_ok(len(_words(t)), kw["num_words"], kw["relation"])
        if key == "number_sentences":
            return _rel_ok(len(_sentences(t)), kw["num_sentences"],
                           kw["relation"])
        if key == "number_paragraphs":
            return _rel_ok(len(_paragraphs(t)), kw["num_paragraphs"],
                           "exactly")
        if key == "nth_paragraph_first_word":
            paras = _paragraphs(t)
            n = kw["nth_paragraph"]
            if len(paras) < n:
                return False
            return paras[n - 1].split()[:1] == [kw["first_word"]]

    if category == "startend":
        if key == "quotation":
            return t.startswith('"') and t.endswith('"')
        if key == "end_checker":
            return t.endswith(kw["end_phrase"])
        if key == "start_checker":
            return t.startswith(kw["start_phrase"])

    raise NotImplementedError(iid)


def score_ifeval(row: dict, response: str) -> tuple[bool, list[str]]:
    """Retourne (toutes_satisfaites, contraintes_non_implémentées)."""
    unimplemented: list[str] = []
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        try:
            if not _check_ifeval(response, iid, kw):
                return False, unimplemented
        except NotImplementedError:
            unimplemented.append(iid)
            continue
    # contraintes non implémentées : validation impossible
    # → échec conservateur (pas de green-washing)
    return (len(unimplemented) == 0), unimplemented


# --------------------------------------------------------------------------
# Exécution de code (HumanEval / MBPP) — sous-processus, timeout
# --------------------------------------------------------------------------


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    i = text.find("def ")
    if i >= 0:
        return text[i:].strip()
    # corps sans déclaration : garder l'indentation
    # (le prompt se termine par la signature)
    return text


def run_code(prefix: str, completion: str, test_code: str,
             call: str | None = None, timeout: float = 10.0) -> bool:
    code = f"{prefix}{completion}\n\n{test_code}\n"
    if call:
        code += f"\n{call}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def score_humaneval(row: dict, response: str) -> bool:
    completion = extract_code(response)
    return run_code(row["prompt"], completion, row["test"],
                    call=f"check({row['entry_point']})")


def score_mbpp(row: dict, response: str) -> bool:
    sig = mbpp_signature(row)
    body = extract_code(response)
    if body.startswith("def "):
        # le modèle a ré-écrit la signature : garder le corps
        body = body.split("\n", 1)[1] if "\n" in body else ""
    tests = "\n".join(row.get("test_list", []))
    return run_code(f"{sig}\n", body, tests)


SCORERS = {
    "gsm8k": score_gsm8k,
    "math500": score_math500,
    "mmlu": score_mmlu,
    "mmlu_pro": score_mmlu_pro,
    "ifeval": score_ifeval,
    "humaneval": score_humaneval,
    "mbpp": score_mbpp,
}


def load_all(limit: int | None = None) -> dict[str, Dataset]:
    out = {}
    for name, spec in DATASETS.items():
        ds = load_dataset(spec["hf_id"], spec.get("config"),
                          split=spec["split"])
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        out[name] = ds
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_dataset(name: str, ds: Dataset, client: SGLangClient,
                tokenizer, concurrency: int) -> dict[str, Any]:
    scorer = SCORERS[name]
    max_new = MAX_NEW_TOKENS.get(name, DEFAULT_MAX_NEW_TOKENS)
    prompts = [build_prompt(tokenizer, name, row) for row in ds]

    responses: list[str] = [""] * len(ds)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, text in enumerate(ex.map(
                lambda p: client.generate(p, max_new), prompts)):
            responses[i] = text

    scores, unimplemented, failures_sample = [], set(), []
    for idx, (row, resp) in enumerate(zip(ds, responses)):
        try:
            r = scorer(row, resp)
            ok, unimpl = (r if name == "ifeval" else (bool(r), []))
            scores.append(1.0 if ok else 0.0)
            unimplemented.update(unimpl)
            if not ok and len(failures_sample) < 5:
                failures_sample.append({
                    "index": idx,
                    "response_head": resp[:200],
                })
        except Exception as e:  # noqa: BLE001 — scorer isolé
            scores.append(0.0)
            if len(failures_sample) < 5:
                failures_sample.append(
                    {"index": idx, "error": f"{type(e).__name__}: {e}"})

    n = len(scores)
    return {
        "rows": n,
        "score": sum(scores) / n if n else 0.0,
        "unimplemented_constraints": sorted(unimplemented),
        "generation_failures": client.failures,
        "failures_sample": failures_sample,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner B0 (protocole P-G/1.0)")
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repetitions", type=int,
                    default=GENERATION["n_repetitions"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="N exemples par dataset (smoke test uniquement)")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    client = SGLangClient(args.sglang_url)
    datasets = load_all(args.limit)
    names = args.datasets or list(DATASETS)

    per_dataset: dict[str, Any] = {}
    for name in names:
        t0 = time.time()
        reps = []
        unimpl: set[str] = set()
        for rep in range(args.repetitions):
            r = run_dataset(name, datasets[name], client, tokenizer,
                            args.concurrency)
            reps.append(r["score"])
            unimpl.update(r["unimplemented_constraints"])
            r.pop("unimplemented_constraints")
            per_dataset.setdefault(name, {})[f"rep{rep}"] = r
        mean = sum(reps) / len(reps)
        std = statistics.pstdev(reps) if len(reps) > 1 else 0.0
        per_dataset[name].update({
            "mean": mean, "std": std, "repetitions": reps,
            "unimplemented_constraints": sorted(unimpl),
            "max_new_tokens": MAX_NEW_TOKENS.get(
                name, DEFAULT_MAX_NEW_TOKENS),
        })
        # écriture incrémentale : résiste aux
        # interruptions longues (le JSON final
        # complet écrase à la fin)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"partial": True, "model": MODEL_ID,
                         "engine": f"SGLang {args.sglang_url}",
                         "datasets": per_dataset},
                        f, indent=2, ensure_ascii=False)
        print(f"[{name}] mean={mean:.4f} std={std:.2e} "
              f"({time.time() - t0:.0f}s)", flush=True)

    weights = COMPOSITE["weights"]
    composite = sum(per_dataset[n]["mean"] * weights[n] for n in names)
    result = {
        "protocol": {"version": "P-G/1.0", "hash": protocol_hash(),
                     "generation": GENERATION, "composite": COMPOSITE,
                     "system_prompts": SYSTEM_PROMPTS,
                     "stop_tokens": STOP_TOKENS,
                     "deviation_max_new_tokens": (
                         "P-G/1.0 fige 512 ; plafonds relevés pour "
                         "gsm8k/math500 (2048) et ifeval (1024) — "
                         "troncature mesurée 60% MATH-500, 20% GSM8K "
                         "à 512 (2026-10-12)")},
        "model": MODEL_ID,
        "engine": f"SGLang {args.sglang_url} (B0-plain, AR natif)",
        "environment": {
            "python": sys.version.split()[0],
            "concurrency": args.concurrency,
            "repetitions": args.repetitions,
            "limit": args.limit,
        },
        "datasets": per_dataset,
        "composite": composite,
        "noise_floor": {n: per_dataset[n]["std"] for n in names},
        "generation_failures": client.failures,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"écrit : {args.out}")
    print(f"composite B0 = {composite:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
