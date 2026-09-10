#!/usr/bin/env python3
"""Choose a recognition rejection threshold using validation predictions only."""

import argparse
import json
import math
from pathlib import Path


def select_threshold(
    predictions: list[dict], min_selective_accuracy: float, min_coverage: float
) -> dict:
    if not predictions:
        raise ValueError("No validation predictions were supplied")
    if not 0.0 <= min_selective_accuracy <= 1.0 or not 0.0 <= min_coverage <= 1.0:
        raise ValueError("Accuracy and coverage targets must be between zero and one")

    ranked = sorted(predictions, key=lambda row: float(row["confidence"]), reverse=True)
    minimum_accepted = max(1, math.ceil(len(ranked) * min_coverage))
    candidates = []
    for threshold in sorted({float(row["confidence"]) for row in ranked}, reverse=True):
        accepted = [row for row in ranked if float(row["confidence"]) >= threshold]
        correct = sum(bool(row["correct"]) for row in accepted)
        accuracy = correct / len(accepted)
        candidates.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": len(accepted) / len(ranked),
                "selective_accuracy": accuracy,
            }
        )
    eligible = [
        row for row in candidates
        if row["accepted"] >= minimum_accepted
        and row["selective_accuracy"] >= min_selective_accuracy
    ]
    selected = max(eligible, key=lambda row: (row["coverage"], row["selective_accuracy"])) if eligible else None
    return {
        "selection_split": "val",
        "samples": len(ranked),
        "target_selective_accuracy": min_selective_accuracy,
        "minimum_coverage": min_coverage,
        "selected": selected,
        "target_met": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MSASL confidence rejection on validation data")
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-selective-accuracy", type=float, default=0.80)
    parser.add_argument("--min-coverage", type=float, default=0.20)
    args = parser.parse_args()

    source = json.loads(Path(args.validation_results).read_text(encoding="utf-8"))
    if source.get("split") != "val":
        raise SystemExit("Calibration must use validation results, never test results")
    result = select_threshold(
        source.get("predictions", []), args.min_selective_accuracy, args.min_coverage
    )
    result["validation_results"] = args.validation_results
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["target_met"]:
        chosen = result["selected"]
        print(
            f"Threshold {chosen['threshold']:.6f}: "
            f"accuracy={chosen['selective_accuracy']:.2%}, coverage={chosen['coverage']:.2%}"
        )
    else:
        print("No validation threshold met both targets; recognition must remain disabled")
    print(f"Saved calibration to {output}")


if __name__ == "__main__":
    main()
