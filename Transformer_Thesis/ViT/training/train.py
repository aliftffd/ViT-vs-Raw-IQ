"""
Main Training Script for AMC Transformer
Modular, production-ready training with proper logging and checkpointing
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

from dataloader.dataset import SingleStreamImageDataset, worker_init_fn
from dataloader.utils import split_data
from models.amc_transformer import AMCTransformer
from training.utils import (
    save_checkpoint, 
    load_checkpoint,
    plot_training_history,
    EarlyStopping,
    get_lr,
    evaluate_model_with_confusion
)

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = Path("data")
    FILE_PATH = "C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\CNN model\\dataset\\radioml2018\\versions\\2\\classes-fixed.json' 
    CHECKPOINT_DIR = Path("result/checkpoints")
    LOG_DIR = Path("result/logs")
    
    # Data split
    TRAIN_SIZE = 0.7
    VALID_SIZE = 0.15
    TEST_SIZE = 0.15
    SPLIT_SEED = 42
    NORM_SEED = 49
    
    # Target modulations
    TARGET_MODULATIONS = [
        'OOK',
        '4ASK',
        '8ASK',
        'BPSK',
        'QPSK',
        '8PSK',
        '16PSK',
        '32PSK',
        '16APSK',
        '32APSK',
        '64APSK',
        '128APSK',
        '16QAM',
        '32QAM',
        '64QAM',
        '128QAM',
        '256QAM',
        'GMSK',
        'OQPSK'
    ]
    
    # Model architecture
    PATCH_SIZE = 4
    D_MODEL = 128
    N_HEAD = 8
    N_LAYERS = 6
    FFN_HIDDEN = D_MODEL * 4
    DROP_PROB = 0.1
    
    # Training hyperparameters
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.1
    
    # DataLoader settings
    NUM_WORKERS = 6
    PREFETCH_FACTOR = 3
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Early stopping
    PATIENCE = 10
    
    # Checkpointing
    SAVE_FREQ = 10  # Save checkpoint every N epochs
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def from_args(cls, args):
        """Update config from command line arguments"""
        for key, value in vars(args).items():
            if value is not None and hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
        return cls


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AMC Transformer')
    
    # Data arguments
    parser.add_argument('--file_path', type=str, help='Path to HDF5 data file')
    parser.add_argument('--json_path', type=str, help='Path to classes JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, help='Number of transformer layers')
    
    # Other
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    
    return parser.parse_args()


def get_config_dict(config):
    """Convert Config class to serializable dictionary."""
    return {
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'PATCH_SIZE': config.PATCH_SIZE,
        'D_MODEL': config.D_MODEL,
        'N_HEAD': config.N_HEAD,
        'N_LAYERS': config.N_LAYERS,
        'FFN_HIDDEN': config.FFN_HIDDEN,
        'DROP_PROB': config.DROP_PROB,
        'TARGET_MODULATIONS': config.TARGET_MODULATIONS,
        'TRAIN_SIZE': config.TRAIN_SIZE,
        'VALID_SIZE': config.VALID_SIZE,
        'TEST_SIZE': config.TEST_SIZE,
        'FILE_PATH': str(config.FILE_PATH),
        'JSON_PATH': str(config.JSON_PATH),
        'SPLIT_SEED': config.SPLIT_SEED,
        'NORM_SEED': config.NORM_SEED
    }


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, labels, snrs) in enumerate(pbar):
        # Move to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
    
    with torch.no_grad():
        for batch_idx, (images, labels, snrs) in enumerate(pbar):
            # Move to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ============================================
# MAIN TRAINING LOOP
# ============================================

def main():
    """Main training function"""
    
    # Parse arguments and setup config
    args = parse_args()
    config = Config.from_args(args)
    
    # Setup directories
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup experiment name
    experiment_name = args.experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_checkpoint_dir = config.CHECKPOINT_DIR / experiment_name
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    
    # Enable TF32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    print("="*70)
    print("AMC TRANSFORMER TRAINING")
    print("="*70)
    print(f"Experiment: {experiment_name}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Num workers: {config.NUM_WORKERS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*70)
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    print("\n📂 Loading data...")
    
    # Split data
    train_indices, valid_indices, test_indices, label_map = split_data(
        str(config.FILE_PATH),
        str(config.JSON_PATH),
        config.TARGET_MODULATIONS,
        config.TRAIN_SIZE,
        config.VALID_SIZE,
        config.TEST_SIZE,
        config.SPLIT_SEED
    )
    
    num_classes = len(config.TARGET_MODULATIONS)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SingleStreamImageDataset(
        file_path=str(config.FILE_PATH),
        json_path=str(config.JSON_PATH),
        target_modulations=config.TARGET_MODULATIONS,
        mode='train',
        indices=train_indices,
        label_map=label_map,
        seed=config.NORM_SEED
    )
    
    norm_stats = train_dataset.get_normalization_stats()
    
    valid_dataset = SingleStreamImageDataset(
        file_path=str(config.FILE_PATH),
        json_path=str(config.JSON_PATH),
        target_modulations=config.TARGET_MODULATIONS,
        mode='valid',
        indices=valid_indices,
        label_map=label_map,
        normalization_stats=norm_stats
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_loader):,} batches ({len(train_dataset):,} samples)")
    print(f"   Valid: {len(valid_loader):,} batches ({len(valid_dataset):,} samples)")
    
    # ========================================
    # MODEL SETUP
    # ========================================
    
    print("\n🤖 Initializing model...")
    
    model_params = {
        'in_channels': 1,
        'img_size_h': 32,
        'img_size_w': 64,
        'patch_size': config.PATCH_SIZE,
        'num_classes': num_classes,
        'd_model': config.D_MODEL,
        'n_head': config.N_HEAD,
        'n_layers': config.N_LAYERS,
        'ffn_hidden': config.FFN_HIDDEN,
        'drop_prob': config.DROP_PROB,
        'device': config.DEVICE
    }
    
    model = AMCTransformer(**model_params).to(config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created:")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    
    # ========================================
    # OPTIMIZER & CRITERION
    # ========================================
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.99)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        # verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # ========================================
    # RESUME FROM CHECKPOINT (if specified)
    # ========================================
    
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        print(f"   Resuming from epoch {start_epoch}")
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, valid_loader, criterion, config.DEVICE, epoch
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        current_lr = get_lr(optimizer)
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = exp_checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            
            # Convert config class to dict for saving
            config_dict = {
                'BATCH_SIZE': config.BATCH_SIZE,
                'NUM_EPOCHS': config.NUM_EPOCHS,
                'LEARNING_RATE': config.LEARNING_RATE,
                'NUM_WORKERS': config.NUM_WORKERS,
                'PATCH_SIZE': config.PATCH_SIZE,
                'D_MODEL': config.D_MODEL,
                'N_HEAD': config.N_HEAD,
                'N_LAYERS': config.N_LAYERS,
                'FFN_HIDDEN': config.FFN_HIDDEN,
                'DROP_PROB': config.DROP_PROB,
                'TARGET_MODULATIONS': config.TARGET_MODULATIONS,
                'TRAIN_SIZE': config.TRAIN_SIZE,
                'VALID_SIZE': config.VALID_SIZE,
                'TEST_SIZE': config.TEST_SIZE,
                'FILE_PATH': str(config.FILE_PATH),
                'JSON_PATH': str(config.JSON_PATH),
                'SPLIT_SEED': config.SPLIT_SEED,
                'NORM_SEED': config.NORM_SEED
            }
            
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                config_dict
            )
            print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Early stopping check
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("\n⏹️  Early stopping triggered!")
            # Save final model
            final_path = exp_checkpoint_dir / "model_final.pth"
            
            config_dict = {
                'BATCH_SIZE': config.BATCH_SIZE,
                'NUM_EPOCHS': config.NUM_EPOCHS,
                'LEARNING_RATE': config.LEARNING_RATE,
                'NUM_WORKERS': config.NUM_WORKERS,
                'PATCH_SIZE': config.PATCH_SIZE,
                'D_MODEL': config.D_MODEL,
                'N_HEAD': config.N_HEAD,
                'N_LAYERS': config.N_LAYERS,
                'FFN_HIDDEN': config.FFN_HIDDEN,
                'DROP_PROB': config.DROP_PROB,
                'TARGET_MODULATIONS': config.TARGET_MODULATIONS,
                'TRAIN_SIZE': config.TRAIN_SIZE,
                'VALID_SIZE': config.VALID_SIZE,
                'TEST_SIZE': config.TEST_SIZE,
                'FILE_PATH': str(config.FILE_PATH),
                'JSON_PATH': str(config.JSON_PATH),
                'SPLIT_SEED': config.SPLIT_SEED,
                'NORM_SEED': config.NORM_SEED
            }
            
            save_checkpoint(
                final_path,
                model,
                optimizer,
                scheduler,
                epoch,
                early_stopping.best_score,
                history,
                config_dict
            )
            break
        
        print("-" * 70)
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    
    total_training_time = time.time() - training_start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_training_time/3600:.2f} hours")
    print(f"Best val loss: {early_stopping.best_score:.4f}")
    print(f"Checkpoints saved in: {exp_checkpoint_dir}")
    
    # Create config dict once for final saves
    config_dict = {
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'NUM_WORKERS': config.NUM_WORKERS,
        'PATCH_SIZE': config.PATCH_SIZE,
        'D_MODEL': config.D_MODEL,
        'N_HEAD': config.N_HEAD,
        'N_LAYERS': config.N_LAYERS,
        'FFN_HIDDEN': config.FFN_HIDDEN,
        'DROP_PROB': config.DROP_PROB,
        'TARGET_MODULATIONS': config.TARGET_MODULATIONS,
        'TRAIN_SIZE': config.TRAIN_SIZE,
        'VALID_SIZE': config.VALID_SIZE,
        'TEST_SIZE': config.TEST_SIZE,
        'FILE_PATH': str(config.FILE_PATH),
        'JSON_PATH': str(config.JSON_PATH),
        'SPLIT_SEED': config.SPLIT_SEED,
        'NORM_SEED': config.NORM_SEED
    }
    
    # Save final model
    final_path = exp_checkpoint_dir / "model_final.pth"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        epoch,
        val_loss,
        history,
        config_dict
    )
    print(f"Final model saved: {final_path}")
    
    # Plot training history
    plot_path = config.LOG_DIR / f"{experiment_name}_training_history.png"
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved: {plot_path}")
    
    # ========================================
    # FINAL EVALUATION ON TEST SET
    # ========================================
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    # Load best model for evaluation
    best_checkpoint = load_checkpoint(
        exp_checkpoint_dir / "model_final.pth",
        model
    )
    
    # Create test dataset
    test_dataset = SingleStreamImageDataset(
        file_path=str(config.FILE_PATH),
        json_path=str(config.JSON_PATH),
        target_modulations=config.TARGET_MODULATIONS,
        mode='test',
        indices=test_indices,
        label_map=label_map,
        normalization_stats=norm_stats
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        worker_init_fn=worker_init_fn,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    # Evaluate and generate confusion matrices
    eval_results = evaluate_model_with_confusion(
        model=model,
        dataloader=test_loader,
        device=config.DEVICE,
        class_names=config.TARGET_MODULATIONS,
        save_dir=exp_checkpoint_dir / "evaluation",
        prefix='test'
    )
    
    # Cleanup
    train_dataset.close()
    valid_dataset.close()
    test_dataset.close()
    
    print("\n✅ All done!")


if __name__ == '__main__':
    main()
