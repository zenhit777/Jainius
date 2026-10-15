"""J4-bis : remédiation format (CPU uniquement, sans modèle)."""

import inspect

from jainius.train import remediation as R


def test_suffix_par_domaine():
    assert "####" in R.FORMAT_SUFFIX["math"]
    assert "boxed" in R.FORMAT_SUFFIX["math"]
    assert "only the code" in R.FORMAT_SUFFIX["code"]
    assert "single letter" in R.FORMAT_SUFFIX["science"]
    assert "exactly" in R.FORMAT_SUFFIX["chat"]


def test_build_training_text_ne_recoit_que_question_et_domaine():
    sig = inspect.signature(R.build_training_text)
    assert list(sig.parameters) == ["question", "domain"]
    txt = R.build_training_text("What is 2+2?", "math")
    assert txt.startswith("What is 2+2?")
    assert txt.endswith("#### and put the final answer in \\boxed{}.")


def test_domaine_inconnu_repli_generique():
    assert R.build_training_text("q", "xxx") == R.build_training_text("q", None)
    assert R.build_training_text("q", None).endswith("answer concisely.")


def test_prompts_seuls_la_reponse_n_est_jamais_lue():
    src = inspect.getsource(R.build_training_text)
    assert "ground_truth" not in src
    assert "tests_json" not in src


def test_merge_budgets_cumulatif():
    resumed = {"tokens_response": 83200, "tokens_supervised": 28800,
               "tokens_seq": 152232, "updates": 300}
    fresh = {"tokens_response": 100, "tokens_supervised": 90,
             "tokens_seq": 200, "updates": 2}
    out = R.merge_budgets(resumed, fresh)
    assert out == {"tokens_response": 83300, "tokens_supervised": 28890,
                   "tokens_seq": 152432, "updates": 302}
    # Sans session courante : reprise à l'identique (pas de remise à zéro).
    assert R.merge_budgets(resumed) == resumed


def test_should_final_save():
    # Cas J4 : steps=700, ckpt_every=100 -> pas 699 déjà sauvé : pas de
    # double sauvegarde (FileExistsError + sortie 1).
    assert R.should_final_save(699, 699) is False
    assert R.should_final_save(649, 699) is True
    assert R.should_final_save(None, 2) is True
