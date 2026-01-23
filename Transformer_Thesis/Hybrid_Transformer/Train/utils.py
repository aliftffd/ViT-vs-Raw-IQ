"""
Training Utilities for HybridTransformer AMC
=============================================

This module provides utility functions for:
1. Training loop helpers
2. Evaluation metrics computation
3. Visualization (loss curves, confusion matrices, accuracy plots)
4. Checkpoint management
5. Logging utilities

Author: Lipp
Project: HybridTransformer AMC Research
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from tqdm import tqdm


# =============================================================================
# RadioML 2018.01A Modulation Classes (24 classes)
# =============================================================================

RADIOML_MODULATIONS = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
    'FM', 'GMSK', 'OQPSK'
]

# SNR levels in RadioML 2018.01A
RADIOML_SNR_LEVELS = list(range(-20, 32, 2))  # -20 to +30 dB, step 2


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def get_lr(optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_map: Dict[str, int],
    desc: str = "Evaluating"
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        dataloader: Validation/Test DataLoader
        criterion: Loss function
        device: Computation device
        label_map: Mapping from modulation names to indices
        desc: Progress bar description
    
    Returns:
        Dictionary containing:
            - loss: Average loss
            - accuracy: Overall accuracy
            - predictions: All predictions
            - labels: All ground truth labels
            - snrs: All SNR values
            - per_class_accuracy: Accuracy per modulation class
            - per_snr_accuracy: Accuracy per SNR level
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_snrs = []
    total_loss = 0.0
    num_batches = 0
    
    # Reverse label map for reporting
    idx_to_mod = {v: k for k, v in label_map.items()}
    
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for x_batch, y_batch, snr_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        
        # Collect predictions
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = y_batch.cpu().numpy()
        snrs = snr_batch.numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_snrs.extend(snrs)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)
    
    # Overall metrics
    overall_accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / num_batches
    
    # Per-class accuracy
    per_class_accuracy = {}
    unique_labels = np.unique(all_labels)
    for label_idx in unique_labels:
        mask = all_labels == label_idx
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            mod_name = idx_to_mod.get(label_idx, f"Class_{label_idx}")
            per_class_accuracy[mod_name] = class_acc
    
    # Per-SNR accuracy
    per_snr_accuracy = {}
    unique_snrs = np.unique(all_snrs)
    for snr in sorted(unique_snrs):
        mask = all_snrs == snr
        if mask.sum() > 0:
            snr_acc = (all_preds[mask] == all_labels[mask]).mean()
            per_snr_accuracy[int(snr)] = snr_acc
    
    return {
        'loss': avg_loss,
        'accuracy': overall_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'snrs': all_snrs,
        'per_class_accuracy': per_class_accuracy,
        'per_snr_accuracy': per_snr_accuracy,
        'idx_to_mod': idx_to_mod
    }


def compute_snr_range_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    snr_min: float,
    snr_max: float,
    idx_to_mod: Dict[int, str]
) -> Dict[str, Any]:
    """
    Compute metrics for a specific SNR range.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        snrs: SNR values
        snr_min: Minimum SNR (inclusive)
        snr_max: Maximum SNR (inclusive)
        idx_to_mod: Mapping from index to modulation name
    
    Returns:
        Dictionary with accuracy, confusion matrix, and classification report
    """
    # Filter by SNR range
    mask = (snrs >= snr_min) & (snrs <= snr_max)
    filtered_preds = predictions[mask]
    filtered_labels = labels[mask]
    
    if len(filtered_labels) == 0:
        return {'accuracy': 0.0, 'num_samples': 0}
    
    accuracy = accuracy_score(filtered_labels, filtered_preds)
    
    # Get unique classes in this range
    unique_classes = np.unique(np.concatenate([filtered_labels, filtered_preds]))
    class_names = [idx_to_mod.get(c, f"Class_{c}") for c in unique_classes]
    
    # Confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=unique_classes)
    
    return {
        'accuracy': accuracy,
        'num_samples': len(filtered_labels),
        'confusion_matrix': cm,
        'class_names': class_names,
        'unique_classes': unique_classes
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved: {save_path}")


def plot_accuracy_vs_snr(
    per_snr_accuracy: Dict[int, float],
    save_path: str,
    title: str = "Classification Accuracy vs SNR"
):
    """
    Plot accuracy as a function of SNR.
    
    Args:
        per_snr_accuracy: Dictionary mapping SNR to accuracy
        save_path: Path to save the figure
        title: Plot title
    """
    snrs = sorted(per_snr_accuracy.keys())
    accuracies = [per_snr_accuracy[snr] for snr in snrs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(snrs, accuracies, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% threshold')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    
    # Highlight critical SNR range (4-8 dB)
    ax.axvspan(4, 8, alpha=0.2, color='red', label='Critical range (4-8 dB)')
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([min(snrs) - 1, max(snrs) + 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add SNR labels
    ax.set_xticks(snrs)
    ax.set_xticklabels([str(s) for s in snrs], rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 SNR accuracy plot saved: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: Tuple[int, int] = None
):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size (auto-calculated if None)
    """
    if normalize:
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        cm_display = cm_normalized
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Auto-calculate figure size based on number of classes
    n_classes = len(class_names)
    if figsize is None:
        figsize = (max(10, n_classes * 0.6), max(8, n_classes * 0.5))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
        annot_kws={'size': 8} if n_classes > 15 else {'size': 10}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix saved: {save_path}")


def plot_per_class_accuracy(
    per_class_accuracy: Dict[str, float],
    save_path: str,
    title: str = "Per-Class Accuracy"
):
    """
    Plot horizontal bar chart of per-class accuracy.
    
    Args:
        per_class_accuracy: Dictionary mapping class name to accuracy
        save_path: Path to save the figure
        title: Plot title
    """
    # Sort by accuracy
    sorted_items = sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_items]
    accuracies = [item[1] for item in sorted_items]
    
    # Color code by accuracy
    colors = ['green' if acc >= 0.9 else 'orange' if acc >= 0.7 else 'red' for acc in accuracies]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.4)))
    
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, accuracies, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1.05])
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90%')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='70%')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add accuracy values on bars
    for i, (acc, cls) in enumerate(zip(accuracies, classes)):
        ax.text(acc + 0.01, i, f'{acc:.2%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Per-class accuracy plot saved: {save_path}")


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    best_val_acc: float,
    config: Dict,
    save_path: str
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        train_losses: Training losses history
        val_losses: Validation losses history
        train_accs: Training accuracy history
        val_accs: Validation accuracy history
        best_val_acc: Best validation accuracy so far
        config: Training configuration
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load tensors to
    
    Returns:
        Checkpoint dictionary with training history
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return checkpoint


# =============================================================================
# Results Saving
# =============================================================================

def save_results(
    results: Dict,
    save_dir: str,
    prefix: str = "results"
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary of results
        save_dir: Directory to save results
        prefix: Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                str(k): (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    filepath = os.path.join(save_dir, f"{prefix}.json")
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved: {filepath}")


def print_evaluation_summary(results: Dict, title: str = "Evaluation Summary"):
    """
    Print formatted evaluation summary.
    
    Args:
        results: Evaluation results dictionary
        title: Summary title
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Loss:     {results['loss']:.4f}")
    print(f"   Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    print(f"\n📡 Per-SNR Accuracy (sample):")
    per_snr = results.get('per_snr_accuracy', {})
    snrs = sorted(per_snr.keys())
    for snr in snrs[:5]:  # Show first 5
        print(f"   SNR {snr:3d} dB: {per_snr[snr]:.4f}")
    if len(snrs) > 5:
        print(f"   ... ({len(snrs) - 5} more SNR levels)")
    
    print(f"\n📶 Per-Class Accuracy (top 5 / bottom 5):")
    per_class = results.get('per_class_accuracy', {})
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    
    print("   Top 5:")
    for mod, acc in sorted_classes[:5]:
        print(f"     {mod:12s}: {acc:.4f}")
    
    print("   Bottom 5:")
    for mod, acc in sorted_classes[-5:]:
        print(f"     {mod:12s}: {acc:.4f}")
    
    print("\n" + "=" * 60)