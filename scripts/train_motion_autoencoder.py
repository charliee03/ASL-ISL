#!/usr/bin/env python3
"""Train/evaluate an ISL-CSLRT landmark-motion reconstruction baseline.

This is deliberately a motion autoencoder, not a text-to-pose generator.  The
split is signer-disjoint and the held-out test split is evaluated only after the
best validation checkpoint is selected.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.motion_autoencoder import TemporalMotionAutoencoder


def load_motion(path: Path, num_frames: int) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    if not frames:
        raise ValueError(f"No frames in {path}")
    motion = np.asarray(
        [frame["pose"] + frame["left_hand"] + frame["right_hand"] for frame in frames], dtype=np.float32
    )
    if motion.shape[1:] != (75, 3) or not np.isfinite(motion).all():
        raise ValueError(f"Invalid landmark tensor in {path}: {motion.shape}")
    values = torch.from_numpy(motion).permute(1, 2, 0).reshape(1, 225, len(motion))
    return F.interpolate(values, size=num_frames, mode="linear", align_corners=True).reshape(75, 3, num_frames).permute(2, 0, 1).numpy()


class MotionDataset(Dataset):
    def __init__(self, metadata: list[dict], root: Path, signers: set[str], num_frames: int):
        self.rows = [row for row in metadata if str(row["signer"]) in signers]
        self.items = [load_motion(root / row["landmarks"], num_frames) for row in self.rows]
        if not self.items:
            raise ValueError(f"No sequences for signers {sorted(signers)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return torch.from_numpy(self.items[index])


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, velocity_weight: float) -> tuple[torch.Tensor, dict]:
    position = F.mse_loss(prediction, target)
    velocity = F.mse_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    return position + velocity_weight * velocity, {"position_mse": position.item(), "velocity_mse": velocity.item()}


@torch.no_grad()
def evaluate(model, loader, device, velocity_weight):
    model.eval()
    total_loss = total_position = total_velocity = total_abs = total_values = 0.0
    for batch in loader:
        batch = batch.to(device)
        prediction, _ = model(batch)
        loss, details = reconstruction_loss(prediction, batch, velocity_weight)
        count = batch.numel()
        total_loss += loss.item() * len(batch)
        total_position += details["position_mse"] * len(batch)
        total_velocity += details["velocity_mse"] * len(batch)
        total_abs += (prediction - batch).abs().sum().item()
        total_values += count
    return {
        "loss": total_loss / len(loader.dataset),
        "position_mse": total_position / len(loader.dataset),
        "velocity_mse": total_velocity / len(loader.dataset),
        "mae": total_abs / total_values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/isl_motion_autoencoder.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-per-run", type=int,
                        help="Stop after N epochs and save a resumable training state")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config, data_config, train_config = config["model"], config["data"], config["training"]
    seed = train_config.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root = Path(data_config["keypoints_dir"])
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    num_frames = model_config["num_frames"]
    train_data = MotionDataset(metadata, root, set(data_config["train_signers"]), num_frames)
    val_data = MotionDataset(metadata, root, set(data_config["validation_signers"]), num_frames)
    test_data = MotionDataset(metadata, root, set(data_config["test_signers"]), num_frames)
    batch_size = train_config["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalMotionAutoencoder(**{key: model_config[key] for key in ("num_keypoints", "hidden_dim", "latent_dim")}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])
    checkpoint_dir = Path(args.checkpoint_dir or train_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = args.epochs or train_config["epochs"]
    state_path = checkpoint_dir / "training_state.pt"
    start_epoch, best, stale, history = 1, float("inf"), 0, []
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot resume: {state_path} is missing")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["total_epochs"] != total_epochs:
            raise ValueError(f"Cannot resume a {state['total_epochs']}-epoch run as {total_epochs} epochs")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, best, stale, history = state["next_epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng_state"]); random.setstate(state["python_rng_state"])
    else:
        (checkpoint_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if start_epoch > total_epochs:
        print(f"Training already completed through epoch {total_epochs}")
        return
    end_epoch = min(total_epochs, start_epoch + (args.max_epochs_per_run or total_epochs) - 1)
    for epoch in range(start_epoch, end_epoch + 1):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device); optimizer.zero_grad()
            prediction, _ = model(batch)
            loss, _ = reconstruction_loss(prediction, batch, train_config["velocity_loss_weight"])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item() * len(batch)
        validation = evaluate(model, val_loader, device, train_config["velocity_loss_weight"])
        row = {"epoch": epoch, "train_loss": train_loss / len(train_data), **{f"val_{key}": value for key, value in validation.items()}}
        history.append(row); print(json.dumps(row))
        if validation["loss"] < best:
            best, stale = validation["loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": model_config, "best_val": validation}, checkpoint_dir / "best_model.pt")
        else:
            stale += 1
        torch.save({
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "next_epoch": epoch + 1, "best": best, "stale": stale, "history": history,
            "total_epochs": total_epochs, "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }, state_path)
        if stale >= train_config["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
    # Do not touch the held-out test split until the planned run has reached a
    # terminal condition; intermediate CPU chunks are validation-only.
    terminal = epoch == total_epochs or stale >= train_config["patience"]
    if not terminal:
        print(f"Saved resumable state; continue with --resume at epoch {epoch + 1}")
        return
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    report = {
        "purpose": "Signer-disjoint motion reconstruction baseline; not text-conditioned pose generation.",
        "splits": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
        "best_validation": checkpoint["best_val"],
        "held_out_test": evaluate(model, test_loader, device, train_config["velocity_loss_weight"]),
        "epochs_completed": len(history), "history": history,
    }
    (checkpoint_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
