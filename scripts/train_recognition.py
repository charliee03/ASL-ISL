#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from recognition.dataset import WLASLDataset, MSASLDataset, RandomRotation, RandomSqueeze, RandomMirror, Compose, collate_keypoints
from recognition.model import SignRecognitionTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL Recognition Model")
    parser.add_argument("--config", default="configs/recognition.yaml", help="Path to config file")
    parser.add_argument("--data-root", default="data/wlasl", help="Root directory for dataset")
    parser.add_argument("--annotation-file", default="data/wlasl/nslt_100.json", help="Path to annotation file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of dataset samples")
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("training", {})
    
    # Hyperparameters
    num_frames = model_config.get("num_frames", 16)
    num_keypoints = model_config.get("num_keypoints", 27)
    d_model = model_config.get("d_model", 256)
    nhead = model_config.get("nhead", 8)
    num_encoder_layers = model_config.get("num_encoder_layers", 6)
    dropout = model_config.get("dropout", 0.1)
    
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    mirror_prob = data_config.get("mirror_prob", 0.5)
    rotation_range = data_config.get("rotation_range", 13)
    squeeze_ratio = data_config.get("squeeze_ratio", 0.15)
    
    epochs = args.epochs if args.epochs is not None else train_config.get("epochs", 100)
    lr = train_config.get("lr", 1e-4)
    weight_decay = train_config.get("weight_decay", 1e-4)
    scheduler_step = train_config.get("scheduler_step", 10)
    scheduler_gamma = train_config.get("scheduler_gamma", 0.9)
    checkpoint_dir = Path(train_config.get("checkpoint_dir", "models/recognition"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset Transforms & Loaders
    train_transforms = Compose([
        RandomMirror(p=mirror_prob),
        RandomRotation(degree_range=rotation_range),
        RandomSqueeze(squeeze_ratio=squeeze_ratio)
    ])
    
    data_root = Path(args.data_root)
    annotation_file = Path(args.annotation_file)
    
    # If annotation file doesn't exist, create a mock one for testing/verification purposes
    if not annotation_file.exists():
        print(f"Annotation file {annotation_file} not found. Creating mock annotations.")
        annotation_file.parent.mkdir(parents=True, exist_ok=True)
        mock_ann = []
        for i in range(10):
            mock_ann.append({
                "video": f"video_{i}.mp4",
                "gloss": f"word_{i % 3}",
                "split": "train"
            })
            mock_ann.append({
                "video": f"video_{i}.mp4",
                "gloss": f"word_{i % 3}",
                "split": "val"
            })
        with open(annotation_file, "w") as f:
            json.dump(mock_ann, f, indent=2)
            
    (data_root / "videos").mkdir(parents=True, exist_ok=True)
    
    dataset_type = data_config.get("dataset", "wlasl").lower()
    
    print(f"Loading {dataset_type.upper()} datasets...")
    if dataset_type == "msasl":
        DatasetClass = MSASLDataset
    else:
        DatasetClass = WLASLDataset
        
    train_dataset = DatasetClass(
        data_root=data_root,
        annotation_file=str(annotation_file),
        split="train",
        num_frames=num_frames,
        transform=train_transforms,
        limit=args.limit
    )
    
    val_dataset = DatasetClass(
        data_root=data_root,
        annotation_file=str(annotation_file),
        split="val",
        num_frames=num_frames,
        transform=None,
        limit=args.limit
    )
    
    vocab_size = len(train_dataset.classes)
    print(f"Vocabulary size: {vocab_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keypoints
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_keypoints
    )
    
    # 2. Instantiate Model
    model = SignRecognitionTransformer(
        num_keypoints=num_keypoints,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        vocab_size=vocab_size,
        dropout=dropout
    )
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count / 1e6:.2f}M parameters")
    
    # 3. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Prep CSV logging
    log_csv_path = checkpoint_dir / "training_log.csv"
    with open(log_csv_path, mode="w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        
    best_val_acc = -1.0
    epochs_no_improve = 0
    patience = train_config.get("patience", 50)
    
    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for idx, batch in enumerate(train_bar):
            keypoints = batch["keypoints"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * keypoints.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar description with running loss
            train_bar.set_postfix(loss=loss.item())
            if (idx + 1) % 5 == 0 or (idx + 1) == len(train_loader):
                print(f"Epoch {epoch}/{epochs} [Train] - Batch {idx+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)
            
        train_loss /= total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            for idx, batch in enumerate(val_bar):
                keypoints = batch["keypoints"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(keypoints)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * keypoints.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar description with running loss
                val_bar.set_postfix(loss=loss.item())
                if (idx + 1) % 5 == 0 or (idx + 1) == len(val_loader):
                    print(f"Epoch {epoch}/{epochs} [Val] - Batch {idx+1}/{len(val_loader)} | Batch Loss: {loss.item():.4f}", flush=True)
                
        if val_total > 0:
            val_loss /= val_total
            val_acc = val_correct / val_total
        else:
            val_loss = 0.0
            val_acc = 0.0
            
        scheduler.step()
        
        print(f"Epoch {epoch}/{epochs} Finished | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
              
        # Log to CSV
        with open(log_csv_path, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            
        # Save best checkpoint and Early Stopping
        if val_acc >= best_val_acc and val_total > 0:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"--> Saved best checkpoint to {checkpoint_dir / 'best_model.pt'} (acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve} out of {patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
    print("Training finished.")

    # Export gloss vocabulary JSON (required by translation module)
    gloss_vocab = {
        "version": "1.0",
        "dataset": "WLASL-100",
        "num_classes": vocab_size,
        "gloss_to_id": train_dataset.word2idx,
        "id_to_gloss": {str(v): k for k, v in train_dataset.word2idx.items()}
    }
    vocab_path = checkpoint_dir / "gloss_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(gloss_vocab, f, indent=2)
    print(f"--> Exported gloss vocabulary ({vocab_size} classes) to {vocab_path}")


if __name__ == "__main__":
    train()
