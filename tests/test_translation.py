from src.translation.translator import ASLtoISLTranslator


def make_translator():
    return ASLtoISLTranslator(enable_llm=False)


def test_rule_based_translation_maps_known_glosses():
    translator = make_translator()

    assert translator.translate_gloss_string("HELLO I WATER") == "NAMASKAR MAIN PANI"


def test_rule_based_translation_keeps_third_person_glosses_unambiguous():
    translator = make_translator()

    assert translator.translate_gloss_string("HE IS GOING") == "HE IS GOING"


def test_rule_based_translation_removes_configured_fillers():
    translator = make_translator()

    assert translator.translate_gloss_string("UM THANK_YOU") == "SHUKRIYA"


def test_batch_translation_preserves_batch_boundaries():
    translator = make_translator()

    assert translator.translate_batch([["HELLO"], ["GOODBYE"]]) == [["NAMASKAR"], ["ALVIDA"]]
