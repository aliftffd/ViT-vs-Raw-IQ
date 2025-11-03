"""
Training utilities: checkpointing, early stopping, logging, confusion matrices, etc.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import confusion_matrix, classification_report


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: If True, prints messages
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f})')
        self.val_loss_min = val_loss
        self.best_model = model.state_dict().copy()


def save_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_loss: float,
    history: Dict,
    config: Optional[Dict] = None
):
    """
    Save training checkpoint with better error handling.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        val_loss: Validation loss
        history: Training history
        config: Training configuration (will be converted to dict if it's a class)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': float(val_loss),  # Ensure it's a regular float
        'history': history
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        # Ensure config is a clean dictionary
        if isinstance(config, dict):
            # Clean the config dict - convert any non-serializable values
            clean_config = {}
            for key, value in config.items():
                try:
                    # Test if value is serializable
                    import pickle
                    pickle.dumps(value)
                    clean_config[key] = value
                except (TypeError, AttributeError, pickle.PicklingError):
                    # Convert to string if not serializable
                    clean_config[key] = str(value)
            checkpoint['config'] = clean_config
        else:
            # If it's a class, extract serializable attributes
            clean_config = {}
            if hasattr(config, '__dict__'):
                for key, value in config.__dict__.items():
                    if not key.startswith('_'):
                        try:
                            import pickle
                            pickle.dumps(value)
                            clean_config[key] = value
                        except (TypeError, AttributeError, pickle.PicklingError):
                            clean_config[key] = str(value)
            checkpoint['config'] = clean_config
    
    try:
        torch.save(checkpoint, filepath)
    except Exception as e:
        print(f"⚠️  Error saving checkpoint: {e}")
        print(f"   Attempting to save without config...")
        # Try saving without config as fallback
        checkpoint_minimal = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': float(val_loss),
            'history': history
        }
        if scheduler is not None:
            checkpoint_minimal['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_minimal, filepath)
        print(f"   ✅ Checkpoint saved without config")


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = 'Confusion Matrix',
    save_path: Optional[Path] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save plot
        normalize: If True, normalize confusion matrix
        figsize: Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm) if not normalize else (y_true == y_pred).mean()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(f'{title}\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
    
    plt.close()
    
    return cm, accuracy


def evaluate_model_with_confusion(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str],
    save_dir: Path,
    prefix: str = 'test'
):
    """
    Evaluate model and generate confusion matrices for overall and specific SNRs.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        class_names: List of modulation class names
        save_dir: Directory to save confusion matrices
        prefix: Prefix for saved files (e.g., 'test', 'val')
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_snrs = []
    
    print(f"\n🔍 Evaluating model on {prefix} set...")
    
    with torch.no_grad():
        for images, labels, snrs in dataloader:
            images = images.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_snrs.extend(snrs.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. OVERALL CONFUSION MATRIX
    # ========================================
    print("📊 Generating overall confusion matrix...")
    
    cm_overall, acc_overall = plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        title=f'Overall Confusion Matrix - {prefix.capitalize()} Set',
        save_path=save_dir / f'{prefix}_confusion_matrix_overall.png',
        normalize=True,
        figsize=(14, 12)
    )
    
    print(f"   Overall Accuracy: {acc_overall*100:.2f}%")
    
    # ========================================
    # 2. SNR-SPECIFIC CONFUSION MATRICES
    # ========================================
    target_snrs = [-8, 0, 8]
    snr_accuracies = {}
    
    for target_snr in target_snrs:
        # Find samples closest to target SNR (within ±0.5 dB)
        snr_mask = np.abs(all_snrs - target_snr) <= 0.5
        
        if snr_mask.sum() == 0:
            print(f"⚠️  No samples found for SNR = {target_snr} dB")
            continue
        
        snr_preds = all_preds[snr_mask]
        snr_labels = all_labels[snr_mask]
        
        print(f"\n📊 Generating confusion matrix for SNR = {target_snr} dB...")
        print(f"   Samples: {len(snr_preds):,}")
        
        cm_snr, acc_snr = plot_confusion_matrix(
            y_true=snr_labels,
            y_pred=snr_preds,
            class_names=class_names,
            title=f'Confusion Matrix - {prefix.capitalize()} Set (SNR = {target_snr} dB)',
            save_path=save_dir / f'{prefix}_confusion_matrix_snr_{target_snr}dB.png',
            normalize=True,
            figsize=(14, 12)
        )
        
        snr_accuracies[target_snr] = acc_snr
        print(f"   Accuracy @ {target_snr} dB: {acc_snr*100:.2f}%")
    
    # ========================================
    # 3. CLASSIFICATION REPORT
    # ========================================
    print(f"\n📋 Generating classification report...")
    
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
    
    # Save report to file
    report_path = save_dir / f'{prefix}_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {prefix.capitalize()} Set\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Accuracy: {acc_overall*100:.2f}%\n\n")
        f.write("Accuracy by SNR:\n")
        for snr, acc in snr_accuracies.items():
            f.write(f"  SNR {snr:+3d} dB: {acc*100:.2f}%\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write(report)
    
    print(f"📄 Classification report saved to {report_path}")
    
    # ========================================
    # 4. ACCURACY VS SNR PLOT
    # ========================================
    print(f"\n📊 Generating accuracy vs SNR plot...")
    
    unique_snrs = np.unique(all_snrs)
    snr_wise_acc = []
    
    for snr in sorted(unique_snrs):
        snr_mask = all_snrs == snr
        if snr_mask.sum() > 0:
            acc = (all_preds[snr_mask] == all_labels[snr_mask]).mean()
            snr_wise_acc.append((snr, acc * 100))
    
    if snr_wise_acc:
        snrs, accs = zip(*snr_wise_acc)
        
        plt.figure(figsize=(12, 6))
        plt.plot(snrs, accs, 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=acc_overall*100, color='r', linestyle='--', linewidth=2, label=f'Overall: {acc_overall*100:.2f}%')
        
        # Highlight target SNRs
        for target_snr in target_snrs:
            if target_snr in snr_accuracies:
                plt.axvline(x=target_snr, color='gray', linestyle=':', alpha=0.5)
                plt.text(target_snr, 5, f'{target_snr} dB', 
                        rotation=90, va='bottom', ha='right', fontsize=9)
        
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'Accuracy vs SNR - {prefix.capitalize()} Set', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        acc_vs_snr_path = save_dir / f'{prefix}_accuracy_vs_snr.png'
        plt.savefig(acc_vs_snr_path, dpi=300, bbox_inches='tight')
        print(f"📊 Accuracy vs SNR plot saved to {acc_vs_snr_path}")
        plt.close()
    
    # ========================================
    # 5. SUMMARY
    # ========================================
    print("\n" + "="*70)
    print(f"EVALUATION SUMMARY - {prefix.upper()} SET")
    print("="*70)
    print(f"Total samples: {len(all_labels):,}")
    print(f"Overall accuracy: {acc_overall*100:.2f}%")
    print(f"\nAccuracy by SNR:")
    for snr, acc in sorted(snr_accuracies.items()):
        print(f"  SNR {snr:+3d} dB: {acc*100:.2f}%")
    print(f"\nAll results saved to: {save_dir}")
    print("="*70)
    
    return {
        'overall_accuracy': acc_overall,
        'snr_accuracies': snr_accuracies,
        'confusion_matrix': cm_overall,
        'predictions': all_preds,
        'labels': all_labels,
        'snrs': all_snrs
    }


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: If True, prints messages
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f})')
        self.val_loss_min = val_loss
        self.best_model = model.state_dict().copy()


def save_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_loss: float,
    history: Dict,
    config: Optional[Dict] = None
):
    """
    Save training checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        val_loss: Validation loss
        history: Training history
        config: Training configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'history': history
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.close()


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 12s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"