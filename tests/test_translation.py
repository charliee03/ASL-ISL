import json
from types import SimpleNamespace

from src.translation.translator import ASLtoISLTranslator


def make_translator():
    return ASLtoISLTranslator(enable_llm=False)


def test_rule_based_translation_maps_known_glosses():
    translator = make_translator()

    assert translator.translate_gloss_string("HELLO I WATER") == "NAMASKAR MEIN PANI"


def test_rule_based_translation_ignores_surrounding_sentence_punctuation():
    translator = make_translator()

    assert translator.translate_gloss_string("Hello, I water!") == "NAMASKAR MEIN PANI"


def test_rule_based_translation_reorders_i_am_complements():
    translator = make_translator()

    assert translator.translate_gloss_string("Hello, I am Naman") == "NAMASKAR MEIN Naman HOON"


def test_rule_based_translation_maps_name_introduction():
    translator = make_translator()

    assert translator.translate_gloss_string("Hello my name is Charliee") == "NAMASKAR MERA NAAM Charliee HAI"


def test_curated_sentence_overrides_preserve_user_supplied_translations():
    translator = make_translator()

    assert translator.translate_gloss_string("Are you free today?") == "KYA TUM FREE HO"
    assert translator.translate_gloss_string("are you hiding something") == "TUM KUCH CHUPA RAHE HO"
    assert translator.translate_gloss_string("Bring water for me.") == "KRIPAYA MERE LIYE PANI LE AAO"
    assert translator.translate_gloss_string_with_mode("are you free today")[1] == "curated_sentence_override"


def test_approved_sentence_review_entries_are_used_offline():
    translator = make_translator()

    assert translator.translate_gloss_string("Nice to meet you!") == "Tumse milkar achha laga"
    assert translator.translate_gloss_string_with_mode("nice to meet you")[1] == "curated_sentence_override"


def test_rule_based_translation_keeps_third_person_glosses_unambiguous():
    translator = make_translator()

    assert translator.translate_gloss_string("HE IS GOING") == "HE IS GOING"


def test_rule_based_translation_removes_configured_fillers():
    translator = make_translator()

    assert translator.translate_gloss_string("UM THANK_YOU") == "SHUKRIYA"


def test_batch_translation_preserves_batch_boundaries():
    translator = make_translator()

    assert translator.translate_batch([["HELLO"], ["GOODBYE"]]) == [["NAMASKAR"], ["ALVIDA"]]


def test_gemini_refinement_accepts_only_a_permutation_of_rule_glosses():
    translator = make_translator()

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs):
            return SimpleNamespace(text=json.dumps({"isl_glosses": ["PANI", "NAMASKAR"]}))

    translator.gemini_client = SimpleNamespace(models=FakeModels())
    assert translator.translate_gloss_string("HELLO WATER") == "PANI NAMASKAR"


def test_gemini_refinement_rejects_invented_glosses():
    translator = make_translator()

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs):
            return SimpleNamespace(text=json.dumps({"isl_glosses": ["NAMASKAR", "INVENTED"]}))

    translator.gemini_client = SimpleNamespace(models=FakeModels())
    assert translator.translate_gloss_string("HELLO WATER") == "NAMASKAR PANI"
