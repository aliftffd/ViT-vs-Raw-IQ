"""
Training Script for HybridTransformer AMC
==========================================

Complete training pipeline with:
1. Training loop with validation
2. Comprehensive evaluation on test set
3. Visualization outputs:
   - Training loss/accuracy curves
   - Accuracy vs SNR plot
   - Overall confusion matrix (24 classes)
   - Confusion matrix for critical SNR range (4-8 dB)
   - Per-class accuracy bar chart

Usage:
    python train.py --config config.yaml
    python train.py --data_path /path/to/radioml.hdf5 --epochs 100

Author: Lipp
Project: HybridTransformer AMC Research
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from Model.HybridTransformer import HybridTransformer
from DataLoader.dataset import SingleStreamSignalDataset, worker_init_fn
from DataLoader.utils import split_data
from utils import (
    AverageMeter,
    EarlyStopping,
    get_lr,
    evaluate_model,
    compute_snr_range_metrics,
    plot_training_curves,
    plot_accuracy_vs_snr,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    save_checkpoint,
    load_checkpoint,
    save_results,
    print_evaluation_summary,
    RADIOML_MODULATIONS
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data paths
    'data_path': 'C:\\Users\\LippLopp\\Thesis\\Transformer_Thesis\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5',
    'json_path': 'C:\\Users\\LippLopp\\Thesis\\Transformer_Thesis\\radioml2018\\versions\\2\\classes-fixed.json',
    
    # Data split
    'train_ratio': 0.7,
    'valid_ratio': 0.2,
    'test_ratio': 0.1,
    
    # Model architecture
    'in_channels': 2,
    'seq_length': 1024,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dim_feedforward': 256,
    'n_classes': 24,
    'pool_size': 4,
    'dropout': 0.1,
    'classifier_dropout': 0.3,
    'pooling': 'gap',
    
    # Training hyperparameters
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',  # 'cosine', 'step', 'plateau'
    'warmup_epochs': 5,
    
    # Training settings
    'num_workers': 4,  # Use 0 on Windows to avoid DataLoader hanging
    'pin_memory': True,  # False for num_workers=0
    'mixed_precision': True,
    'gradient_clip': 1.0,
    
    # Early stopping
    'early_stopping': True,
    'patience': 15,
    
    # Checkpointing
    'save_dir': 'outputs',
    'experiment_name': 'hybrid_transformer_amc',
    'save_every': 10,
    
    # Random seed
    'seed': 42,
    
    # Critical SNR range for detailed analysis
    'critical_snr_min': 4,
    'critical_snr_max': 8,
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    gradient_clip: float,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Computation device
        scaler: Gradient scaler for mixed precision
        gradient_clip: Gradient clipping value
        epoch: Current epoch number
        total_epochs: Total number of epochs
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Accuracy')
    
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False
    )
    
    for batch_idx, (x_batch, y_batch, _) in enumerate(pbar):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with autocast('cuda'):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item()
        
        # Update meters
        batch_size = x_batch.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        total_epochs: Total number of epochs
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Accuracy')
    
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs} [Valid]",
        leave=False
    )
    
    for x_batch, y_batch, _ in pbar:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item()
        
        batch_size = x_batch.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: Dict):
    """
    Main training function.
    
    Args:
        config: Training configuration dictionary
    """
    # Setup
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(
        config['save_dir'],
        f"{config['experiment_name']}_{timestamp}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'figures'), exist_ok=True)
    
    # Save config
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  DATA LOADING")
    print("=" * 60)
    
    # All 24 modulation classes
    target_modulations = RADIOML_MODULATIONS
    print(f"📡 Target modulations: {len(target_modulations)} classes")
    
    # Split data
    train_indices, valid_indices, test_indices, label_map = split_data(
        file_path=config['data_path'],
        json_path=config['json_path'],
        target_mods=target_modulations,
        train_ratio=config['train_ratio'],
        valid_ratio=config['valid_ratio'],
        test_ratio=config['test_ratio'],
        seed=config['seed']
    )
    
    # Create datasets
    print("\n📂 Creating datasets...")
    
    train_dataset = SingleStreamSignalDataset(
        file_path=config['data_path'],
        json_path=config['json_path'],
        target_modulations=target_modulations,
        mode='train',
        indices=train_indices,
        label_map=label_map,
        normalization_stats=None,  # Will be calculated
        seed=config['seed']
    )
    
    # Get normalization stats from training set
    norm_stats = train_dataset.get_normalization_stats()
    
    valid_dataset = SingleStreamSignalDataset(
        file_path=config['data_path'],
        json_path=config['json_path'],
        target_modulations=target_modulations,
        mode='valid',
        indices=valid_indices,
        label_map=label_map,
        normalization_stats=norm_stats,
        seed=config['seed']
    )
    
    test_dataset = SingleStreamSignalDataset(
        file_path=config['data_path'],
        json_path=config['json_path'],
        target_modulations=target_modulations,
        mode='test',
        indices=test_indices,
        label_map=label_map,
        normalization_stats=norm_stats,
        seed=config['seed']
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,
        pin_memory=config['pin_memory'],
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn,
        pin_memory=config['pin_memory']
    )
    
    print(f"✅ DataLoaders created:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Valid: {len(valid_loader)} batches")
    print(f"   Test:  {len(test_loader)} batches")
    
    # ==========================================================================
    # Model Setup
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  MODEL SETUP")
    print("=" * 60)
    
    model = HybridTransformer(
        in_channels=config['in_channels'],
        seq_length=config['seq_length'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        n_classes=config['n_classes'],
        pool_size=config['pool_size'],
        dropout=config['dropout'],
        classifier_dropout=config['classifier_dropout'],
        pooling=config['pooling']
    )
    
    model = model.to(device)
    
    # Print model summary
    params = model.count_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"   Tokenizer:   {params['tokenizer']:>10,}")
    print(f"   Aggregator:  {params['aggregator']:>10,}")
    print(f"   Classifier:  {params['classifier']:>10,}")
    print(f"   ─────────────────────────")
    print(f"   Total:       {params['total']:>10,}")
    
    # ==========================================================================
    # Training Setup
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  TRAINING SETUP")
    print("=" * 60)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'] - config['warmup_epochs'],
            eta_min=1e-6
        )
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config['lr_scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    # Warmup scheduler
    if config['warmup_epochs'] > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config['warmup_epochs']
        )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config['mixed_precision'] and torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        mode='max'
    ) if config['early_stopping'] else None
    
    print(f"✅ Training configuration:")
    print(f"   Optimizer:     AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
    print(f"   Scheduler:     {config['lr_scheduler']}")
    print(f"   Warmup epochs: {config['warmup_epochs']}")
    print(f"   Mixed precision: {config['mixed_precision']}")
    print(f"   Early stopping: {config['early_stopping']} (patience={config['patience']})")
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  TRAINING")
    print("=" * 60)
    
    # History tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            gradient_clip=config['gradient_clip'],
            epoch=epoch,
            total_epochs=config['epochs']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=config['epochs']
        )
        
        # Update history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        current_lr = get_lr(optimizer)
        if epoch <= config['warmup_epochs']:
            warmup_scheduler.step()
        elif config['lr_scheduler'] == 'plateau':
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{config['epochs']} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            best_model_path = os.path.join(
                experiment_dir, 'checkpoints', 'best_model.pt'
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                train_accs=train_accs,
                val_accs=val_accs,
                best_val_acc=best_val_acc,
                config=config,
                save_path=best_model_path
            )
            print(f"  ⭐ New best model! Val Acc: {best_val_acc:.4f}")
        
        # Periodic checkpoint
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(
                experiment_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt'
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                train_accs=train_accs,
                val_accs=val_accs,
                best_val_acc=best_val_acc,
                config=config,
                save_path=checkpoint_path
            )
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_acc):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # ==========================================================================
    # Plot Training Curves
    # ==========================================================================
    print("\n📊 Generating training curves...")
    
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        save_path=os.path.join(experiment_dir, 'figures', 'training_curves.png')
    )
    
    # ==========================================================================
    # Final Evaluation on Test Set
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (Test Set)")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = load_checkpoint(
        checkpoint_path=os.path.join(experiment_dir, 'checkpoints', 'best_model.pt'),
        model=model,
        device=device
    )
    
    # Comprehensive evaluation
    print("\n🔍 Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        label_map=label_map,
        desc="Testing"
    )
    
    # Print summary
    print_evaluation_summary(test_results, "Test Set Results")
    
    # ==========================================================================
    # Generate All Visualizations
    # ==========================================================================
    print("\n📊 Generating evaluation visualizations...")
    
    figures_dir = os.path.join(experiment_dir, 'figures')
    
    # 1. Accuracy vs SNR plot
    plot_accuracy_vs_snr(
        per_snr_accuracy=test_results['per_snr_accuracy'],
        save_path=os.path.join(figures_dir, 'accuracy_vs_snr.png'),
        title="HybridTransformer: Classification Accuracy vs SNR"
    )
    
    # 2. Per-class accuracy bar chart
    plot_per_class_accuracy(
        per_class_accuracy=test_results['per_class_accuracy'],
        save_path=os.path.join(figures_dir, 'per_class_accuracy.png'),
        title="Per-Modulation Classification Accuracy"
    )
    
    # 3. Overall confusion matrix (all SNRs)
    cm_overall = confusion_matrix(
        test_results['labels'],
        test_results['predictions']
    )
    idx_to_mod = test_results['idx_to_mod']
    class_names_overall = [idx_to_mod[i] for i in range(len(idx_to_mod))]
    
    plot_confusion_matrix(
        cm=cm_overall,
        class_names=class_names_overall,
        save_path=os.path.join(figures_dir, 'confusion_matrix_overall.png'),
        title="Confusion Matrix (All SNR Levels)",
        normalize=True
    )
    
    # 4. Confusion matrix for critical SNR range (4-8 dB)
    critical_snr_results = compute_snr_range_metrics(
        predictions=test_results['predictions'],
        labels=test_results['labels'],
        snrs=test_results['snrs'],
        snr_min=config['critical_snr_min'],
        snr_max=config['critical_snr_max'],
        idx_to_mod=idx_to_mod
    )
    
    print(f"\n📊 Critical SNR Range ({config['critical_snr_min']}-{config['critical_snr_max']} dB):")
    print(f"   Samples: {critical_snr_results['num_samples']:,}")
    print(f"   Accuracy: {critical_snr_results['accuracy']:.4f} ({critical_snr_results['accuracy']*100:.2f}%)")
    
    if critical_snr_results['num_samples'] > 0:
        plot_confusion_matrix(
            cm=critical_snr_results['confusion_matrix'],
            class_names=critical_snr_results['class_names'],
            save_path=os.path.join(figures_dir, f'confusion_matrix_snr_{config["critical_snr_min"]}_{config["critical_snr_max"]}dB.png'),
            title=f"Confusion Matrix (SNR: {config['critical_snr_min']}-{config['critical_snr_max']} dB)",
            normalize=True
        )
    
    # ==========================================================================
    # Save Final Results
    # ==========================================================================
    print("\n💾 Saving results...")
    
    # Prepare final results
    final_results = {
        'config': config,
        'training': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'total_time_minutes': total_time / 60
        },
        'test': {
            'loss': test_results['loss'],
            'accuracy': test_results['accuracy'],
            'per_snr_accuracy': test_results['per_snr_accuracy'],
            'per_class_accuracy': test_results['per_class_accuracy']
        },
        'critical_snr': {
            'snr_range': f"{config['critical_snr_min']}-{config['critical_snr_max']} dB",
            'accuracy': critical_snr_results['accuracy'],
            'num_samples': critical_snr_results['num_samples']
        }
    }
    
    save_results(
        results=final_results,
        save_dir=experiment_dir,
        prefix='final_results'
    )
    
    # Save label map
    label_map_path = os.path.join(experiment_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Save normalization stats
    norm_stats_path = os.path.join(experiment_dir, 'normalization_stats.json')
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📁 All outputs saved to: {experiment_dir}")
    print(f"\n📊 Results Summary:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"   Test Accuracy (Overall):  {test_results['accuracy']:.4f}")
    print(f"   Test Accuracy (4-8 dB):   {critical_snr_results['accuracy']:.4f}")
    print(f"\n📈 Figures generated:")
    print(f"   - training_curves.png")
    print(f"   - accuracy_vs_snr.png")
    print(f"   - per_class_accuracy.png")
    print(f"   - confusion_matrix_overall.png")
    print(f"   - confusion_matrix_snr_4_8dB.png")
    print("\n" + "=" * 60)
    
    return final_results


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train HybridTransformer for Automatic Modulation Classification'
    )
    
    # Data paths
    parser.add_argument('--data_path', type=str, default=DEFAULT_CONFIG['data_path'],
                        help='Path to RadioML HDF5 file')
    parser.add_argument('--json_path', type=str, default=DEFAULT_CONFIG['json_path'],
                        help='Path to classes JSON file')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'],
                        help='Weight decay for AdamW')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=DEFAULT_CONFIG['d_model'],
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=DEFAULT_CONFIG['n_heads'],
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=DEFAULT_CONFIG['n_layers'],
                        help='Number of Transformer layers')
    parser.add_argument('--pooling', type=str, default=DEFAULT_CONFIG['pooling'],
                        choices=['gap', 'attention'],
                        help='Pooling strategy')
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=DEFAULT_CONFIG['num_workers'],
                        help='Number of DataLoader workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['patience'],
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--save_dir', type=str, default=DEFAULT_CONFIG['save_dir'],
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=DEFAULT_CONFIG['experiment_name'],
                        help='Experiment name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config from args
    config = DEFAULT_CONFIG.copy()
    config.update({
        'data_path': args.data_path,
        'json_path': args.json_path,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'pooling': args.pooling,
        'num_workers': args.num_workers,
        'mixed_precision': not args.no_amp,
        'patience': args.patience,
        'save_dir': args.save_dir,
        'experiment_name': args.experiment_name,
        'seed': args.seed,
    })
    
    # Print banner
    print("\n" + "=" * 60)
    print("  HybridTransformer AMC Training")
    print("  " + "=" * 56)
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run training
    results = train(config)
    
    return results


if __name__ == "__main__":
    main()