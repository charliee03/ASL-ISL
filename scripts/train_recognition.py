import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.recognition.dataset import WLASLDataset
from src.recognition.model import SignRecognitionTransformer
from src.utils.metrics import Metrics


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_checkpoint_dir(checkpoint_dir):
    """Create checkpoint directory if it doesn't exist."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        keypoints = batch['keypoints'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: reshape keypoints from (batch, frames, keypoints, 3) to (batch, frames, keypoints*3)
        batch_size, num_frames, num_keypoints, coords = keypoints.shape
        keypoints_flat = keypoints.reshape(batch_size, num_frames, num_keypoints * coords)
        
        logits = model(keypoints_flat)  # (batch, frames, vocab_size)
        
        # Use the last frame's prediction for classification
        logits = logits[:, -1, :]  # (batch, vocab_size)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            keypoints = batch['keypoints'].to(device)
            labels = batch['label'].to(device)
            
            # Reshape keypoints
            batch_size, num_frames, num_keypoints, coords = keypoints.shape
            keypoints_flat = keypoints.reshape(batch_size, num_frames, num_keypoints * coords)
            
            logits = model(keypoints_flat)  # (batch, frames, vocab_size)
            
            # Use the last frame's prediction for classification
            logits = logits[:, -1, :]  # (batch, vocab_size)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main(config_path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    create_checkpoint_dir(checkpoint_dir)
    
    # Load datasets
    print("Loading datasets...")
    # Assuming WLASL data is in data/wlasl/ directory
    data_root = "data/wlasl/processed"
    annotation_file = "data/wlasl/WLASL_v0.3.json"
    
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found at {annotation_file}")
        print("Please download WLASL dataset first.")
        return
    
    train_dataset = WLASLDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        split="train",
        num_frames=config['model']['num_frames']
    )
    val_dataset = WLASLDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        split="val",
        num_frames=config['model']['num_frames']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    print("Initializing model...")
    model = SignRecognitionTransformer(
        num_keypoints=config['model']['num_keypoints'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        vocab_size=len(train_dataset.classes)
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config['training']['scheduler_step'],
        gamma=config['training']['scheduler_gamma']
    )
    
    # Training loop
    best_val_loss = float('inf')
    training_log = []
    
    print(f"Training for {config['training']['epochs']} epochs...")
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log results
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'vocab_size': len(train_dataset.classes),
                'num_params': num_params
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    # Save training log
    log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    print(f"\nTraining log saved to {log_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(checkpoint_dir, 'gloss_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(train_dataset.word2idx, f, indent=2)
    print(f"Vocabulary saved to {vocab_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument("--config", type=str, default="configs/recognition.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    main(args.config)
