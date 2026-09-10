from scripts.calibrate_msasl_recognition import select_threshold


def test_select_threshold_maximizes_coverage_that_meets_accuracy_target():
    rows = [
        {"confidence": 0.95, "correct": True},
        {"confidence": 0.90, "correct": True},
        {"confidence": 0.80, "correct": True},
        {"confidence": 0.70, "correct": False},
        {"confidence": 0.60, "correct": False},
    ]
    result = select_threshold(rows, min_selective_accuracy=0.75, min_coverage=0.20)
    assert result["target_met"] is True
    assert result["selected"]["threshold"] == 0.70
    assert result["selected"]["coverage"] == 0.80


def test_select_threshold_reports_when_target_is_unattainable():
    rows = [{"confidence": 0.9, "correct": False}, {"confidence": 0.8, "correct": True}]
    result = select_threshold(rows, min_selective_accuracy=1.0, min_coverage=1.0)
    assert result["target_met"] is False
