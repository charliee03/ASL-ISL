#!/usr/bin/env python3
"""Train a sentence-classification baseline on extracted ISL-CSLRT landmarks."""
import argparse
import csv
import json
import platform
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recognition.dataset import (
    CSLRTDataset, MSASLPoseDataset, Compose, RandomFrameDropout, RandomKeypointMask,
    RandomLandmarkNoise, RandomLandmarkScale, RandomTemporalCrop, collate_keypoints,
)
from src.recognition.model import SignRecognitionTransformer


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch["keypoints"].to(device), batch["label"].to(device)
            logits = model(inputs)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a landmark-sequence recognition baseline")
    parser.add_argument("--config", default="configs/cslrt_recognition_v3.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int, help="Limit each split for a smoke test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-dir", help="Override model output directory")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    mc, dc, tc = config["model"], config["data"], config["training"]
    seed = tc.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    splits_dir, keypoints_dir = Path(dc["splits_dir"]), Path(dc["keypoints_dir"])
    common = dict(keypoints_dir=keypoints_dir, vocabulary_file=splits_dir / "vocabulary.json",
                  num_frames=mc["num_frames"], limit=args.limit, preload=True)
    train_transform = Compose([
        RandomTemporalCrop(dc.get("temporal_crop_min_ratio", 1.0)),
        RandomLandmarkScale(dc.get("scale_amount", 0.1)),
        RandomLandmarkNoise(dc.get("noise_std", 0.01)),
        RandomKeypointMask(dc.get("keypoint_mask_probability", 0.05)),
        RandomFrameDropout(dc.get("frame_dropout_probability", 0.05)),
    ])
    dataset_type = dc.get("dataset_type", "cslrt_json")
    dataset_class = MSASLPoseDataset if dataset_type == "msasl_npz" else CSLRTDataset
    train_data = dataset_class(manifest_file=splits_dir / "train.json", transform=train_transform, **common)
    val_data = dataset_class(manifest_file=splits_dir / "val.json", transform=None, **common)
    batch_size = args.batch_size or dc["batch_size"]
    sampler = None
    if dc.get("class_balanced_sampling", False):
        counts = torch.bincount(torch.tensor([row["class_id"] for row in train_data.samples]), minlength=len(train_data.classes))
        weights = torch.tensor([1.0 / counts[row["class_id"]].item() for row in train_data.samples], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=dc.get("num_workers", 0), collate_fn=collate_keypoints)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=dc.get("num_workers", 0), collate_fn=collate_keypoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignRecognitionTransformer(num_keypoints=mc["num_keypoints"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_encoder_layers=mc["num_encoder_layers"],
        vocab_size=len(train_data.classes), dropout=mc["dropout"],
        use_velocity=mc.get("use_velocity", False),
        use_presence=mc.get("use_presence", False)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=tc.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = args.epochs or tc["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    checkpoint_dir = Path(args.checkpoint_dir or tc["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {**config, "model": mc, "data": dc, "training": tc}
    (checkpoint_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    runtime = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "seed": seed,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "classes": len(train_data.classes),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    (checkpoint_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    log_path = checkpoint_dir / "training_log.csv"
    log_path.write_text("epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate\n")
    best_accuracy, stale = -1.0, 0
    print(f"Device: {device}; seed={seed}; train={len(train_data)}, val={len(val_data)}, classes={len(train_data.classes)}")
    for epoch in range(1, epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum = correct = total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs, labels = batch["keypoints"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        train_loss, train_accuracy = loss_sum / total, correct / total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow([
                epoch, train_loss, train_accuracy, val_loss, val_accuracy,
                epoch_learning_rate,
            ])
        scheduler.step()
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} val_loss={val_loss:.4f} val_acc={val_accuracy:.2%}")
        if val_accuracy > best_accuracy:
            best_accuracy, stale = val_accuracy, 0
            torch.save({"model_state_dict": model.state_dict(), "classes": train_data.classes,
                        "model_config": mc, "training_config": tc, "seed": seed,
                        "dataset_type": dataset_type,
                        "best_val_accuracy": best_accuracy}, checkpoint_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= tc["patience"]:
                print(f"Early stopping after {epoch} epochs")
                break
    (checkpoint_dir / "vocabulary.json").write_text(json.dumps(train_data.classes, indent=2))
    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
