"""Scorers du runner B0 sur fixtures (CPU, sans GPU/SGLang).

Attrape les bugs réels trouvés en smoke test (2026-10-12) :
\\boxed{} à accolades imbriquées, normalisation \\left/\\right,
signature MBPP de référence, extraction «####», indentation
du corps sans déclaration, checkers IFEval, contraintes
non implémentées comptées échec (pas de green-washing).
Ne télécharge aucun dataset, n'appelle pas SGLang.
"""

from jainius.eval import b0_runner as br


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_extract_boxed_nested_braces():
    # bug smoke : le regex [^}] s'arrêtait au premier }
    # intérieur de \\frac{\\pi}{2}
    t = r"raisonnement \\boxed{(3, \frac{\pi}{2})} fin"
    assert br.extract_boxed(t) == r"(3, \frac{\pi}{2})"


def test_extract_boxed_last_of_multiple():
    assert br.extract_boxed(r"\boxed{1} puis \boxed{2}") == "2"


def test_extract_boxed_none_or_unclosed():
    assert br.extract_boxed("pas de boîte") is None
    assert br.extract_boxed(r"\boxed{incomplet") is None


def test_normalize_math_leftright_and_punctuation():
    got = br.normalize_math(r"\left( 3, \frac{\pi}{2} \right)")
    assert got == r"(3\frac{\pi}{2})"  # virgules ignorées


# --------------------------------------------------------------------------
# Scorers GSM8K / MATH-500
# --------------------------------------------------------------------------


def test_score_math500_leftright_boxed_parity():
    # bug smoke : \\boxed{(3, \\frac{\\pi}{2})} vs référence
    # \\left( 3, \\frac{\\pi}{2} \\right) — échouait avant
    # la normalisation \\left/\\right
    row = {"answer": r"\left( 3, \frac{\pi}{2} \right)"}
    resp = r"étapes \\boxed{(3, \frac{\pi}{2})}"
    assert br.score_math500(row, resp)


def test_score_math500_wrong_answer():
    assert not br.score_math500({"answer": "42"}, r"x \boxed{41}")


def test_score_gsm8k_hash_marker_priority():
    row = {"answer": "calculs #### 18"}
    assert br.score_gsm8k(row, "étapes #### 18")
    assert not br.score_gsm8k(row, "étapes #### 19")


def test_score_gsm8k_fallback_last_number():
    row = {"answer": "#### 18"}
    assert br.score_gsm8k(row, "la réponse est 18")
    assert not br.score_gsm8k(row, "la réponse est 19")


# --------------------------------------------------------------------------
# Scorers MMLU / MMLU-Pro
# --------------------------------------------------------------------------


def test_score_mmlu_letter():
    assert br.score_mmlu({"answer": 2}, "C")
    assert br.score_mmlu({"answer": 2}, "La réponse est C.")
    assert not br.score_mmlu({"answer": 2}, "D")


def test_score_mmlu_pro_letter():
    assert br.score_mmlu_pro({"answer": "I"}, "I")
    assert br.score_mmlu_pro({"answer": 8}, "I")
    assert not br.score_mmlu_pro({"answer": "I"}, "J")


# --------------------------------------------------------------------------
# Checkers IFEval
# --------------------------------------------------------------------------


def test_ifeval_punctuation_no_comma():
    assert br._check_ifeval("abc def", "punctuation:no_comma", {})
    assert not br._check_ifeval("a, b", "punctuation:no_comma", {})


def test_ifeval_number_words_relations():
    base = "length_constraints:number_words"
    assert br._check_ifeval(
        "a b c", base, {"num_words": 3, "relation": "exactly"})
    assert br._check_ifeval(
        "a b c d", base, {"num_words": 3, "relation": "at_least"})
    assert not br._check_ifeval(
        "a b c d", base, {"num_words": 3, "relation": "at_most"})


def test_ifeval_keywords_existence_forbidden():
    assert br._check_ifeval(
        "chat chien", "keywords:existence",
        {"keywords": ["chat", "chien"]})
    assert not br._check_ifeval(
        "chat", "keywords:existence", {"keywords": ["chat", "chien"]})
    assert br._check_ifeval(
        "ici pas", "keywords:forbidden_words",
        {"forbidden_words": ["chat"]})


def test_ifeval_keyword_frequency():
    kw = {"keyword": "mot", "frequency": 2, "relation": "exactly"}
    assert br._check_ifeval(
        "mot un mot deux", "keywords:frequency", kw)
    assert not br._check_ifeval(
        "mot un mot deux mot", "keywords:frequency", kw)


def test_ifeval_letter_frequency():
    kw = {"letter": "a", "let_frequency": 2, "let_relation": "exactly"}
    assert br._check_ifeval("abca", "keywords:letter_frequency", kw)
    assert not br._check_ifeval("abcde", "keywords:letter_frequency", kw)


def test_ifeval_case_and_title():
    assert br._check_ifeval("abc", "change_case:english_lowercase", {})
    assert not br._check_ifeval("Abc", "change_case:english_lowercase", {})
    assert br._check_ifeval("ABC", "change_case:english_capital", {})
    assert br._check_ifeval("A B C", "detectable_format:title", {})


def test_ifeval_startend():
    assert br._check_ifeval(
        "fini P.S.", "startend:end_checker", {"end_phrase": "P.S."})
    assert br._check_ifeval('"entre guillemets"', "startend:quotation", {})
    assert not br._check_ifeval("sans", "startend:quotation", {})


def test_ifeval_combination():
    assert br._check_ifeval(
        "REPEAT ensuite", "combination:repeat_prompt",
        {"prompt_to_repeat": "REPEAT"})
    assert not br._check_ifeval(
        "autre chose", "combination:repeat_prompt",
        {"prompt_to_repeat": "REPEAT"})
    assert br._check_ifeval(
        "réponse 1######réponse 2", "combination:two_responses", {})


def test_ifeval_content():
    assert br._check_ifeval(
        "[nom] [age]", "detectable_content:number_placeholders",
        {"num_placeholders": 2})
    assert br._check_ifeval("texte P.S.", "detectable_content:postscript", {})


def test_ifeval_format():
    assert br._check_ifeval('{"a": 1}', "detectable_format:json_format", {})
    assert not br._check_ifeval("pas json", "detectable_format:json_format", {})
    # deux groupes de puces séparés = 2 listes
    assert br._check_ifeval(
        "* a\n* b\n\n* c\n* d",
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2})
    assert br._check_ifeval(
        "==a== ==b==", "detectable_format:number_highlighted_sections",
        {"num_highlights": 2})


def test_ifeval_nth_paragraph_first_word():
    kw = {"nth_paragraph": 2, "first_word": "Deuxième",
          "num_paragraphs": 2}
    text = "Premier paragraphe.\n\nDeuxième paragraphe."
    assert br._check_ifeval(
        text, "length_constraints:nth_paragraph_first_word", kw)
    assert not br._check_ifeval(
        text, "length_constraints:nth_paragraph_first_word",
        {"nth_paragraph": 2, "first_word": "Autre",
         "num_paragraphs": 2})


def test_ifeval_unimplemented_flagged_not_silent():
    row = {"instruction_id_list": ["language:response_language"],
           "kwargs": [{"language": "français"}]}
    ok, unimpl = br.score_ifeval(row, "réponse")
    assert not ok
    assert unimpl == ["language:response_language"]


def test_ifeval_prompt_level_all_constraints_must_pass():
    row = {"instruction_id_list": ["punctuation:no_comma",
                                    "change_case:english_lowercase"],
           "kwargs": [{}, {}]}
    ok, _ = br.score_ifeval(row, "abc def")
    assert ok
    ok, _ = br.score_ifeval(row, "abc, def")  # virgule → tout échoue
    assert not ok


# --------------------------------------------------------------------------
# Exécution de code (HumanEval / MBPP)
# --------------------------------------------------------------------------


def test_score_humaneval_execution():
    row = {"prompt": 'def add(a, b):\n    """add"""\n',
           "test": "def check(candidate):\n"
                   "    assert candidate(1, 2) == 3\n",
           "entry_point": "add"}
    assert br.score_humaneval(row, "    return a + b")
    assert not br.score_humaneval(row, "    return a - b")


def test_extract_code_preserves_body_indentation():
    # le prompt se termine par la signature : le corps
    # généré est indenté et doit le rester
    body = "    return a + b"
    assert br.extract_code(body) == body
    fenced = "voici :\n```python\ndef f():\n    return 1\n```"
    assert br.extract_code(fenced) == "def f():\n    return 1"


def test_mbpp_signature_from_reference_code():
    row = {"code": "def remove_Occ(s,ch):\n    return s",
           "text": "desc", "test_list": []}
    assert br.mbpp_signature(row) == "def remove_Occ(s,ch):"


def test_score_mbpp_uses_reference_signature():
    # bug smoke : le modèle inventait un nom de fonction ≠
    # celui des tests (NameError) — le runner amorce la
    # signature de référence et garde le corps généré
    row = {"code": "def add(a,b):\n    return 0", "text": "add",
           "test_list": ["assert add(1,2) == 3"]}
    assert br.score_mbpp(row, "    return a + b")
    # le modèle a ré-écrit la signature : on garde le corps
    assert br.score_mbpp(row, "def add(a, b):\n    return a + b")
    assert not br.score_mbpp(row, "    return a - b")


# --------------------------------------------------------------------------
# Construction du prompt (tokenizer simulé)
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Capture les appels apply_chat_template."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False,
                              add_generation_prompt=False, **kwargs):
        self.calls.append({"messages": messages,
                           "add_generation_prompt": add_generation_prompt,
                           "kwargs": kwargs})
        return "PROMPT"


def test_build_prompt_sets_add_generation_prompt():
    tok = _FakeTokenizer()
    br.build_prompt(tok, "gsm8k", {"question": "Q"})
    assert tok.calls[0]["add_generation_prompt"] is True


def test_build_prompt_mbpp_includes_reference_signature():
    tok = _FakeTokenizer()
    br.build_prompt(
        tok, "mbpp",
        {"text": "remove", "code": "def remove_Occ(s,ch):\n    pass"})
    user = tok.calls[0]["messages"][1]["content"]
    assert "def remove_Occ(s,ch):" in user
