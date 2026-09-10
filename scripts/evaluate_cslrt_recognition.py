#!/usr/bin/env python3
"""Evaluate a trained ISL-CSLRT sentence classifier on a held-out signer."""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.dataset import CSLRTDataset, MSASLPoseDataset, collate_keypoints
from src.recognition.model import SignRecognitionTransformer


def classification_summary(predictions: list[dict], classes: list[str]) -> dict:
    """Compute class-balanced metrics without an external sklearn dependency."""
    true_counts = Counter(row["true_class_id"] for row in predictions)
    predicted_counts = Counter(row["predicted_class_id"] for row in predictions)
    true_positive_counts = Counter(
        row["true_class_id"] for row in predictions if row["correct"]
    )
    per_class = {}
    for class_id in sorted(true_counts):
        true_positives = true_positive_counts[class_id]
        recall = true_positives / true_counts[class_id]
        precision = true_positives / predicted_counts[class_id] if predicted_counts[class_id] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[classes[class_id]] = {
            "support": true_counts[class_id],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    count = len(per_class)
    return {
        "macro_precision": sum(row["precision"] for row in per_class.values()) / count if count else 0.0,
        "macro_recall": sum(row["recall"] for row in per_class.values()) / count if count else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / count if count else 0.0,
        "balanced_accuracy": sum(row["recall"] for row in per_class.values()) / count if count else 0.0,
        "per_class": per_class,
    }


def calibration_summary(predictions: list[dict], bins: int = 10) -> dict:
    """Compute equal-width expected calibration error and mean NLL."""
    if not predictions:
        return {"expected_calibration_error": 0.0, "negative_log_likelihood": 0.0, "bins": []}
    bin_rows = []
    weighted_gap = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        members = [
            row for row in predictions
            if lower <= float(row["confidence"])
            and (float(row["confidence"]) <= upper if bin_index == bins - 1 else float(row["confidence"]) < upper)
        ]
        if not members:
            continue
        accuracy = sum(bool(row["correct"]) for row in members) / len(members)
        confidence = sum(float(row["confidence"]) for row in members) / len(members)
        weighted_gap += len(members) / len(predictions) * abs(accuracy - confidence)
        bin_rows.append({
            "lower": lower, "upper": upper, "samples": len(members),
            "accuracy": accuracy, "mean_confidence": confidence,
        })
    nll = -sum(
        math.log(max(float(row["true_label_confidence"]), 1e-12)) for row in predictions
    ) / len(predictions)
    return {
        "expected_calibration_error": weighted_gap,
        "negative_log_likelihood": nll,
        "bins": bin_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a landmark-sequence classifier")
    parser.add_argument("--checkpoint", default="models/cslrt_recognition/best_model.pt")
    parser.add_argument("--keypoints-dir", default="data/isl/cslrt_keypoints")
    parser.add_argument("--splits-dir", default="data/isl/cslrt_splits")
    parser.add_argument("--output", default="models/cslrt_recognition/test_results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--calibration",
        help="Validation-only calibration JSON used to report selective accuracy/coverage",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes, config = checkpoint["classes"], checkpoint["model_config"]
    dataset_type = checkpoint.get("dataset_type", "cslrt_json")
    dataset_class = MSASLPoseDataset if dataset_type == "msasl_npz" else CSLRTDataset
    dataset = dataset_class(
        keypoints_dir=args.keypoints_dir,
        manifest_file=Path(args.splits_dir) / f"{args.split}.json",
        vocabulary_file=Path(args.splits_dir) / "vocabulary.json",
        num_frames=config["num_frames"],
        preload=True,
    )
    if dataset.classes != classes:
        raise ValueError("Checkpoint vocabulary does not match the test split vocabulary")
    if not dataset:
        raise ValueError(f"The {args.split} manifest contains no samples")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_keypoints)
    model = SignRecognitionTransformer(
        num_keypoints=config["num_keypoints"], d_model=config["d_model"],
        nhead=config["nhead"], num_encoder_layers=config["num_encoder_layers"],
        vocab_size=len(classes), dropout=config["dropout"],
        use_velocity=config.get("use_velocity", False),
        use_presence=config.get("use_presence", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    top1_correct = top5_correct = 0
    with torch.no_grad():
        offset = 0
        for batch in loader:
            logits = model(batch["keypoints"].to(device))
            probabilities = torch.softmax(logits, dim=1)
            top5 = probabilities.topk(min(5, len(classes)), dim=1)
            labels = batch["label"].to(device)
            top1_correct += (top5.indices[:, 0] == labels).sum().item()
            top5_correct += (top5.indices == labels.unsqueeze(1)).any(dim=1).sum().item()
            for row_index, true_id in enumerate(labels.tolist()):
                sample = dataset.samples[offset + row_index]
                predicted_ids = top5.indices[row_index].tolist()
                predictions.append({
                    "id": sample["id"], "signer": sample.get("signer"),
                    "true_class_id": true_id, "true_label": classes[true_id],
                    "predicted_class_id": predicted_ids[0], "predicted_label": classes[predicted_ids[0]],
                    "top5_labels": [classes[index] for index in predicted_ids],
                    "confidence": top5.values[row_index, 0].item(),
                    "true_label_confidence": probabilities[row_index, true_id].item(),
                    "correct": predicted_ids[0] == true_id,
                })
            offset += len(labels)

    total = len(dataset)
    confusion_counts = Counter(
        (row["true_label"], row["predicted_label"])
        for row in predictions if not row["correct"]
    )
    held_out_signers = sorted({str(row["signer"]) for row in dataset.samples if row.get("signer") is not None})
    class_metrics = classification_summary(predictions, classes)
    selective_evaluation = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if calibration.get("selection_split") != "val" or not calibration.get("target_met"):
            raise ValueError("Calibration must contain a threshold selected on validation data")
        threshold = float(calibration["selected"]["threshold"])
        accepted_predictions = [row for row in predictions if row["confidence"] >= threshold]
        selective_evaluation = {
            "threshold": threshold,
            "accepted": len(accepted_predictions),
            "rejected": total - len(accepted_predictions),
            "coverage": len(accepted_predictions) / total,
            "selective_accuracy": (
                sum(row["correct"] for row in accepted_predictions) / len(accepted_predictions)
                if accepted_predictions else None
            ),
        }
    coverage_rows = [row for row in dataset.samples if "pose_coverage" in row]
    result = {
        "checkpoint": args.checkpoint,
        "dataset_type": dataset_type,
        "split": args.split,
        "held_out_signers": held_out_signers,
        "samples": total,
        "classes_in_checkpoint": len(classes),
        "classes_present_in_test": len({row["true_label"] for row in predictions}),
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
        **class_metrics,
        "calibration": calibration_summary(predictions),
        "selective_evaluation": selective_evaluation,
        "feature_coverage": {
            "mean_pose_coverage": (
                sum(float(row["pose_coverage"]) for row in coverage_rows) / len(coverage_rows)
                if coverage_rows else None
            ),
            "mean_hand_coverage": (
                sum(float(row.get("hand_coverage", 0.0)) for row in coverage_rows) / len(coverage_rows)
                if coverage_rows else None
            ),
        },
        "most_common_confusions": [
            {"true_label": pair[0], "predicted_label": pair[1], "count": count}
            for pair, count in confusion_counts.most_common(15)
        ],
        "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    signer_description = ", ".join(held_out_signers) if held_out_signers else "not applicable (static word samples)"
    print(f"{args.split.title()} samples: {total}; signer(s): {signer_description}")
    print(f"Top-1 accuracy: {result['top1_accuracy']:.2%}")
    print(f"Top-5 accuracy: {result['top5_accuracy']:.2%}")
    print(f"Macro-F1: {result['macro_f1']:.2%}")
    print(f"Saved detailed results to {output}")


if __name__ == "__main__":
    main()
