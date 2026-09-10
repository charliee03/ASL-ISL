from src.utils.metrics import Metrics
from scripts.evaluate_cslrt_recognition import calibration_summary, classification_summary


def test_metrics_are_available_without_downloading_remote_evaluation_scripts():
    metrics = Metrics()

    bleu = metrics.compute_bleu(["NAMASKAR MAIN"], [["NAMASKAR MAIN"]])
    assert bleu["score"] > 99.99
    assert metrics.compute_wer(["NAMASKAR MAIN"], ["NAMASKAR MAIN"]) == 0.0


def test_classification_summary_reports_macro_f1_over_present_classes():
    predictions = [
        {"true_class_id": 0, "predicted_class_id": 0, "correct": True},
        {"true_class_id": 0, "predicted_class_id": 1, "correct": False},
        {"true_class_id": 1, "predicted_class_id": 1, "correct": True},
    ]
    result = classification_summary(predictions, ["A", "B"])
    assert result["per_class"]["A"]["recall"] == 0.5
    assert result["per_class"]["B"]["recall"] == 1.0
    assert result["balanced_accuracy"] == 0.75
    assert 0.0 < result["macro_f1"] < 1.0


def test_calibration_summary_is_zero_for_correct_certain_predictions():
    predictions = [
        {"confidence": 1.0, "true_label_confidence": 1.0, "correct": True},
        {"confidence": 1.0, "true_label_confidence": 1.0, "correct": True},
    ]
    result = calibration_summary(predictions)
    assert result["expected_calibration_error"] == 0.0
    assert result["negative_log_likelihood"] == 0.0
