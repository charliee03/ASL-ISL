#!/usr/bin/env python3
"""Evaluate the trained SignRecognitionTransformer on the WLASL-100 validation set."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── project root on sys.path so `recognition.*` and `src.*` imports both work ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recognition.dataset import WLASLDataset, collate_keypoints
from recognition.model import SignRecognitionTransformer
from src.utils.metrics import Metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASL Recognition Model")
    parser.add_argument("--config", default="configs/recognition.yaml",
                        help="Path to config file")
    parser.add_argument("--data-root", default="data/wlasl",
                        help="Root directory for dataset")
    parser.add_argument("--annotation-file", default="data/wlasl/nslt_100.json",
                        help="Path to annotation file")
    parser.add_argument("--checkpoint", default="models/recognition/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", default="models/recognition/eval_results.json",
                        help="Path to write evaluation results JSON")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_training_summary(log_path: str) -> dict:
    """Parse training_log.csv and return a training summary dict."""
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {
            "epochs_trained": 0,
            "best_val_accuracy": 0.0,
            "final_train_loss": 0.0,
            "final_val_loss": 0.0,
        }

    best_val_acc = max(float(r["val_acc"]) for r in rows)
    last = rows[-1]
    return {
        "epochs_trained": int(last["epoch"]),
        "best_val_accuracy": best_val_acc,
        "final_train_loss": float(last["train_loss"]),
        "final_val_loss": float(last["val_loss"]),
    }


def evaluate():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    num_frames = model_cfg.get("num_frames", 32)
    num_keypoints = model_cfg.get("num_keypoints", 27)
    dropout = model_cfg.get("dropout", 0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load checkpoint & infer architecture from tensor shapes ──────────
    checkpoint_path = Path(args.checkpoint)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # d_model from class_query shape: (1, 1, d_model)
    d_model = state_dict["class_query"].shape[2]

    # nhead: Cannot be directly inferred from weight shapes alone.
    # The checkpoint was trained with the model's class defaults (d_model=256,
    # num_encoder_layers=6), not the config YAML values, so we use the model
    # default of nhead=8.  Fall back to config if d_model matches the config.
    if d_model == model_cfg.get("d_model"):
        nhead = model_cfg.get("nhead", 8)
    else:
        nhead = 8  # SignRecognitionTransformer default

    # Validate: in_proj_weight should be (3*d_model, d_model)
    attn_key = "transformer_encoder.layers.0.self_attn.in_proj_weight"
    if attn_key in state_dict:
        assert state_dict[attn_key].shape == (3 * d_model, d_model), \
            f"Unexpected attention weight shape: {state_dict[attn_key].shape}"

    # num_encoder_layers: count layer keys
    layer_indices = set()
    for key in state_dict:
        if key.startswith("transformer_encoder.layers."):
            idx = int(key.split(".")[2])
            layer_indices.add(idx)
    num_encoder_layers = len(layer_indices) if layer_indices else model_cfg.get("num_encoder_layers", 6)

    # vocab_size from output_proj.weight: (vocab_size, d_model)
    vocab_size_from_ckpt = state_dict["output_proj.weight"].shape[0]

    print(f"Inferred from checkpoint: d_model={d_model}, nhead={nhead}, "
          f"num_encoder_layers={num_encoder_layers}, vocab_size={vocab_size_from_ckpt}")

    # ── Dataset ──────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    annotation_file = Path(args.annotation_file)

    val_dataset = WLASLDataset(
        data_root=data_root,
        annotation_file=str(annotation_file),
        split="val",
        num_frames=num_frames,
        transform=None,
    )

    num_classes = len(val_dataset.classes)
    num_samples = len(val_dataset)
    idx2class = {i: c for c, i in val_dataset.word2idx.items()}

    print(f"Validation samples (with existing videos): {num_samples}")
    print(f"Number of classes: {num_classes}")

    # Use the vocab_size that the checkpoint was actually trained with
    # (should match num_classes from the dataset)
    vocab_size = vocab_size_from_ckpt

    # ── Model ────────────────────────────────────────────────────────────
    model = SignRecognitionTransformer(
        num_keypoints=num_keypoints,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        vocab_size=vocab_size,
        dropout=dropout,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    # ── Inference ────────────────────────────────────────────────────────
    all_preds = []       # predicted class indices
    all_labels = []      # ground-truth class indices
    all_top5_preds = []  # top-5 predicted class indices
    all_pred_glosses = []
    all_ref_glosses = []

    # Per-class counters
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    if num_samples > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_keypoints,
        )

        with torch.no_grad():
            for batch in val_loader:
                keypoints = batch["keypoints"].to(device)
                labels = batch["label"].to(device)
                glosses = batch["gloss"]

                outputs = model(keypoints)  # (B, num_classes)

                # Top-1
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                # Top-5
                _, top5 = outputs.topk(min(5, num_classes), dim=1)
                all_top5_preds.extend(top5.cpu().tolist())

                # Glosses for WER
                for i in range(len(glosses)):
                    pred_idx = predicted[i].item()
                    all_pred_glosses.append(idx2class[pred_idx])
                    all_ref_glosses.append(glosses[i])

                    # Per-class tracking
                    gt_label = labels[i].item()
                    class_total[idx2class[gt_label]] += 1
                    if pred_idx == gt_label:
                        class_correct[idx2class[gt_label]] += 1

    # ── Compute metrics ──────────────────────────────────────────────────
    if num_samples > 0:
        # Top-1 accuracy
        correct_top1 = sum(p == l for p, l in zip(all_preds, all_labels))
        top1_accuracy = correct_top1 / num_samples

        # Top-5 accuracy
        correct_top5 = sum(
            l in t5 for l, t5 in zip(all_labels, all_top5_preds)
        )
        top5_accuracy = correct_top5 / num_samples

        # WER
        metrics_calc = Metrics()
        wer = metrics_calc.compute_wer(
            predictions=all_pred_glosses,
            references=all_ref_glosses,
        )
    else:
        top1_accuracy = 0.0
        top5_accuracy = 0.0
        wer = 1.0  # worst case when no data

    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Word Error Rate: {wer:.4f}")

    # ── Per-class accuracy ───────────────────────────────────────────────
    per_class_accuracy = {}
    for cls_name in sorted(class_total.keys()):
        total = class_total[cls_name]
        correct = class_correct.get(cls_name, 0)
        per_class_accuracy[cls_name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    # Sort by accuracy to get top/bottom 10
    sorted_classes = sorted(
        per_class_accuracy.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    top10 = {k: v for k, v in sorted_classes[:10]}
    bottom10 = {k: v for k, v in sorted_classes[-10:]}

    print(f"\n--- Top 10 classes ---")
    for cls, info in list(top10.items()):
        print(f"  {cls}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")
    print(f"\n--- Bottom 10 classes ---")
    for cls, info in list(bottom10.items()):
        print(f"  {cls}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

    # ── Training summary ─────────────────────────────────────────────────
    log_path = Path(args.checkpoint).parent / "training_log.csv"
    training_summary = load_training_summary(str(log_path))

    # ── Assemble results ─────────────────────────────────────────────────
    results = {
        "model": "SignRecognitionTransformer",
        "dataset": "WLASL-100",
        "split": "val",
        "num_samples": num_samples,
        "num_classes": num_classes,
        "metrics": {
            "top1_accuracy": round(top1_accuracy, 6),
            "top5_accuracy": round(top5_accuracy, 6),
            "word_error_rate": round(float(wer), 6),
        },
        "training_summary": training_summary,
        "per_class_accuracy": per_class_accuracy,
        "top_10_classes": top10,
        "bottom_10_classes": bottom10,
        "model_config": {
            "num_keypoints": num_keypoints,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "vocab_size": num_classes,
            "dropout": dropout,
            "num_frames": num_frames,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation results written to {output_path}")


if __name__ == "__main__":
    evaluate()
