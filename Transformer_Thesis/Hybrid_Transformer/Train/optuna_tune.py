"""
Hyperparameter Tuning with Optuna for HybridTransformer AMC
============================================================

This script uses Optuna to find optimal hyperparameters for the
HybridTransformer architecture on RadioML 2018.01A dataset.

Tunable Parameters:
    - Model: d_model, n_heads, n_layers, dim_feedforward, dropout, pooling
    - Training: learning_rate, weight_decay, batch_size
    - Architecture: pool_size, kernel_sizes

Features:
    - Pruning of unpromising trials (MedianPruner)
    - SQLite storage for resumable studies
    - Visualization of optimization history
    - Best trial export with full configuration

Usage:
    python Train/optuna_tune.py --n_trials 100 --study_name "hybrid_transformer_tuning"

Author: Lipp
Project: HybridTransformer AMC Research (Master's Thesis @ UGM)
"""

import os
import sys
import argparse
import json
import gc
import platform
from datetime import datetime
from typing import Dict, Any, Optional
from functools import partial

# Force unbuffered output for real-time progress
print = partial(print, flush=True)

import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import optuna
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from Model.HybridTransformer import HybridTransformer
from DataLoader.dataset import SingleStreamSignalDataset, worker_init_fn
from DataLoader.utils import split_data
from utils import (
    AverageMeter,
    RADIOML_MODULATIONS
)


# =============================================================================
# Configuration
# =============================================================================

# Detect OS for DataLoader settings
IS_WINDOWS = platform.system() == 'Windows'

TUNE_CONFIG = {
    # Data paths (RadioML 2018.01A)
    'data_path': r'C:\Users\LippLopp\Thesis\Transformer_Thesis\radioml2018\versions\2\GOLD_XYZ_OSC.0001_1024.hdf5',
    'json_path': r'C:\Users\LippLopp\Thesis\Transformer_Thesis\radioml2018\versions\2\classes-fixed.json',

    # Data split
    'train_ratio': 0.7,
    'valid_ratio': 0.15,
    'test_ratio': 0.15,

    # Subset for faster tuning (use fraction of data)
    # Set to 1.0 to use full dataset
    # RTX 5070 Ti (16GB) + 32GB RAM: 0.2-0.3 is good balance
    'tune_subset_fraction': 0.2,

    # Fixed parameters (not tuned)
    'in_channels': 2,
    'seq_length': 1024,
    'n_classes': 24,

    # Training settings - WINDOWS COMPATIBLE
    # On Windows, num_workers must be 0 due to HDF5 multiprocessing issues
    'num_workers': 0 if IS_WINDOWS else 4,
    'pin_memory': False if IS_WINDOWS else True,
    'mixed_precision': True,
    'gradient_clip': 1.0,

    # Tuning settings
    'tune_epochs': 30,          # Epochs per trial (reduced for speed)
    'early_stop_patience': 7,   # Early stopping within trial

    # Output
    'save_dir': r'C:\Users\LippLopp\Thesis\Transformer_Thesis\optuna_results',

    # Random seed
    'seed': 42,
}

# =============================================================================
# Hyperparameter Search Space
# =============================================================================

SEARCH_SPACE = {
    # Model Architecture
    'd_model': {
        'type': 'categorical',
        'choices': [32, 64, 128]
    },
    'n_heads': {
        'type': 'categorical',
        'choices': [2, 4, 8]
    },
    'n_layers': {
        'type': 'int',
        'low': 1,
        'high': 4
    },
    'dim_feedforward': {
        'type': 'categorical',
        'choices': [128, 256, 512]
    },
    'pool_size': {
        'type': 'categorical',
        'choices': [2, 4, 8]
    },
    'pooling': {
        'type': 'categorical',
        'choices': ['gap', 'attention']
    },
    
    # Regularization
    'dropout': {
        'type': 'float',
        'low': 0.05,
        'high': 0.3
    },
    'classifier_dropout': {
        'type': 'float',
        'low': 0.1,
        'high': 0.5
    },
    
    # Training Hyperparameters
    'learning_rate': {
        'type': 'float',
        'low': 1e-5,
        'high': 1e-2,
        'log': True
    },
    'weight_decay': {
        'type': 'float',
        'low': 1e-6,
        'high': 1e-2,
        'log': True
    },
    'batch_size': {
        'type': 'categorical',
        'choices': [256, 512, 1024]  # Larger batches = fewer file operations
    },
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Data Loading (Cached)
# =============================================================================

class DataCache:
    """Cache for datasets to avoid reloading for each trial."""
    
    _instance = None
    _datasets = None
    _norm_stats = None
    _label_map = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset_file_handles(self):
        """Reset HDF5 file handles between trials to prevent stale handles."""
        if self._datasets is not None:
            for dataset in self._datasets:
                if hasattr(dataset, 'reset_file_handle'):
                    dataset.reset_file_handle()
    
    def load_data(self, config: Dict) -> tuple:
        """Load and cache datasets."""
        if self._datasets is not None:
            return self._datasets, self._norm_stats, self._label_map

        print("\n📂 Loading datasets (one-time)...")

        target_modulations = RADIOML_MODULATIONS

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

        # Apply subset fraction for faster tuning
        subset_frac = config.get('tune_subset_fraction', 1.0)
        if subset_frac < 1.0:
            np.random.seed(config['seed'])
            n_train = int(len(train_indices) * subset_frac)
            n_valid = int(len(valid_indices) * subset_frac)
            train_indices = np.random.choice(train_indices, n_train, replace=False)
            valid_indices = np.random.choice(valid_indices, n_valid, replace=False)
            print(f"📉 Using {subset_frac*100:.0f}% subset: {n_train:,} train, {n_valid:,} valid")

        # Create training dataset (calculates normalization stats)
        train_dataset = SingleStreamSignalDataset(
            file_path=config['data_path'],
            json_path=config['json_path'],
            target_modulations=target_modulations,
            mode='train',
            indices=train_indices,
            label_map=label_map,
            normalization_stats=None,
            seed=config['seed']
        )

        norm_stats = train_dataset.get_normalization_stats()

        # Create validation dataset
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

        self._datasets = (train_dataset, valid_dataset)
        self._norm_stats = norm_stats
        self._label_map = label_map

        print(f"✅ Datasets cached: {len(train_dataset):,} train, {len(valid_dataset):,} valid")

        return self._datasets, self._norm_stats, self._label_map


# =============================================================================
# Objective Function
# =============================================================================

def create_model_from_trial(trial: optuna.Trial, config: Dict) -> HybridTransformer:
    """Create a HybridTransformer model with hyperparameters from trial."""
    
    # Sample hyperparameters
    d_model = trial.suggest_categorical('d_model', SEARCH_SPACE['d_model']['choices'])
    n_heads = trial.suggest_categorical('n_heads', SEARCH_SPACE['n_heads']['choices'])
    n_layers = trial.suggest_int('n_layers', SEARCH_SPACE['n_layers']['low'], SEARCH_SPACE['n_layers']['high'])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', SEARCH_SPACE['dim_feedforward']['choices'])
    pool_size = trial.suggest_categorical('pool_size', SEARCH_SPACE['pool_size']['choices'])
    pooling = trial.suggest_categorical('pooling', SEARCH_SPACE['pooling']['choices'])
    dropout = trial.suggest_float('dropout', SEARCH_SPACE['dropout']['low'], SEARCH_SPACE['dropout']['high'])
    classifier_dropout = trial.suggest_float('classifier_dropout', SEARCH_SPACE['classifier_dropout']['low'], SEARCH_SPACE['classifier_dropout']['high'])
    
    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        # Adjust n_heads to be valid
        valid_heads = [h for h in SEARCH_SPACE['n_heads']['choices'] if d_model % h == 0]
        if valid_heads:
            n_heads = valid_heads[0]
        else:
            n_heads = 1
    
    # Create model
    model = HybridTransformer(
        in_channels=config['in_channels'],
        seq_length=config['seq_length'],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        n_classes=config['n_classes'],
        pool_size=pool_size,
        dropout=dropout,
        classifier_dropout=classifier_dropout,
        pooling=pooling
    )
    
    return model


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    Trains a model with sampled hyperparameters and returns validation accuracy.
    """
    config = TUNE_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get cached datasets
    cache = DataCache.get_instance()
    (train_dataset, valid_dataset), norm_stats, label_map = cache.load_data(config)

    # Reset file handles at start of each trial (Windows HDF5 fix)
    cache.reset_file_handles()
    
    # Sample training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 
                                         SEARCH_SPACE['learning_rate']['low'],
                                         SEARCH_SPACE['learning_rate']['high'],
                                         log=SEARCH_SPACE['learning_rate']['log'])
    weight_decay = trial.suggest_float('weight_decay',
                                        SEARCH_SPACE['weight_decay']['low'],
                                        SEARCH_SPACE['weight_decay']['high'],
                                        log=SEARCH_SPACE['weight_decay']['log'])
    batch_size = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']['choices'])
    
    # Create DataLoaders - Windows compatible (no multiprocessing with HDF5)
    print(f"  📦 Creating DataLoaders (batch_size={batch_size})...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None,
        pin_memory=config['pin_memory'],
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        worker_init_fn=worker_init_fn if config['num_workers'] > 0 else None,
        pin_memory=config['pin_memory']
    )
    print(f"  ✓ DataLoaders created")

    # Create model
    print(f"  🔧 Creating model...")
    try:
        model = create_model_from_trial(trial, config)
        model = model.to(device)
        print(f"  ✓ Model created and moved to {device}")
    except Exception as e:
        print(f"  ⚠ Model creation failed: {e}")
        raise optuna.TrialPruned()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('total_params', total_params)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['tune_epochs'],
        eta_min=1e-6
    )
    
    # Mixed precision
    scaler = GradScaler('cuda') if config['mixed_precision'] and torch.cuda.is_available() else None
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    print(f"  🚀 Starting training ({config['tune_epochs']} epochs)...", flush=True)

    for epoch in range(1, config['tune_epochs'] + 1):
        # ===== Training =====
        model.train()
        train_loss = AverageMeter('Loss')

        for batch_idx, (x_batch, y_batch, _) in enumerate(train_loader):
            if batch_idx == 0 and epoch == 1:
                print(f"  ✓ First batch loaded, shape: {x_batch.shape}")

            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            if batch_idx == 0 and epoch == 1:
                print(f"  ✓ Data moved to GPU")

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    logits = model(x_batch)
                    if batch_idx == 0 and epoch == 1:
                        print(f"  ✓ Forward pass done")
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                if batch_idx == 0 and epoch == 1:
                    print(f"  ✓ Backward pass done")

                if config['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

                scaler.step(optimizer)
                scaler.update()
                if batch_idx == 0 and epoch == 1:
                    print(f"  ✓ Optimizer step done")
            else:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                if config['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

                optimizer.step()

            train_loss.update(loss.item(), x_batch.size(0))

            # Progress every 500 batches
            if batch_idx > 0 and batch_idx % 500 == 0:
                print(f"    Epoch {epoch}: {batch_idx}/{len(train_loader)} batches")
        
        scheduler.step()
        
        # ===== Validation =====
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch, _ in valid_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                logits = model(x_batch)
                preds = logits.argmax(dim=-1)
                
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_acc = val_correct / val_total
        
        # Report intermediate value for pruning
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            # Cleanup
            del model, optimizer, scheduler, train_loader, valid_loader
            gc.collect()
            torch.cuda.empty_cache()
            # Reset file handles for next trial
            cache.reset_file_handles()
            raise optuna.TrialPruned()
        
        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping within trial
        if patience_counter >= config['early_stop_patience']:
            break
    
    # Cleanup
    del model, optimizer, scheduler, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Reset file handles for next trial
    cache.reset_file_handles()

    return best_val_acc


# =============================================================================
# Visualization Functions
# =============================================================================

def save_optimization_plots(study: optuna.Study, save_dir: str):
    """Generate and save Optuna visualization plots."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n📊 Generating optimization plots...")
    
    try:
        # 1. Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(save_dir, 'optimization_history.png'))
        print("  ✓ optimization_history.png")
    except Exception as e:
        print(f"  ⚠ optimization_history failed: {e}")
    
    try:
        # 2. Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(save_dir, 'param_importances.png'))
        print("  ✓ param_importances.png")
    except Exception as e:
        print(f"  ⚠ param_importances failed: {e}")
    
    try:
        # 3. Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(save_dir, 'parallel_coordinate.png'))
        print("  ✓ parallel_coordinate.png")
    except Exception as e:
        print(f"  ⚠ parallel_coordinate failed: {e}")
    
    try:
        # 4. Slice plot for key parameters
        fig = plot_slice(study, params=['learning_rate', 'd_model', 'n_layers', 'dropout'])
        fig.write_image(os.path.join(save_dir, 'slice_plot.png'))
        print("  ✓ slice_plot.png")
    except Exception as e:
        print(f"  ⚠ slice_plot failed: {e}")


def save_study_results(study: optuna.Study, save_dir: str):
    """Save study results to JSON files."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Best trial
    best_trial = study.best_trial
    best_results = {
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'best_trial_number': best_trial.number,
        'user_attrs': best_trial.user_attrs,
        'datetime': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'best_trial.json'), 'w') as f:
        json.dump(best_results, f, indent=2)
    print(f"💾 Best trial saved: best_trial.json")
    
    # All trials
    trials_data = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs
            })
    
    with open(os.path.join(save_dir, 'all_trials.json'), 'w') as f:
        json.dump(trials_data, f, indent=2)
    print(f"💾 All trials saved: all_trials.json")
    
    # Generate training command for best params
    best_params = best_trial.params
    command = generate_training_command(best_params)
    
    with open(os.path.join(save_dir, 'best_training_command.txt'), 'w') as f:
        f.write("# PowerShell command to train with best hyperparameters\n\n")
        f.write(command)
    print(f"💾 Training command saved: best_training_command.txt")


def generate_training_command(params: Dict) -> str:
    """Generate PowerShell training command from best parameters."""
    
    command = 'python Train/train.py `\n'
    command += f'    --data_path "{TUNE_CONFIG["data_path"]}" `\n'
    command += f'    --json_path "{TUNE_CONFIG["json_path"]}" `\n'
    command += f'    --epochs 100 `\n'
    command += f'    --batch_size {params.get("batch_size", 128)} `\n'
    command += f'    --lr {params.get("learning_rate", 0.001)} `\n'
    command += f'    --weight_decay {params.get("weight_decay", 0.0001)} `\n'
    command += f'    --d_model {params.get("d_model", 64)} `\n'
    command += f'    --n_heads {params.get("n_heads", 4)} `\n'
    command += f'    --n_layers {params.get("n_layers", 2)} `\n'
    command += f'    --pooling {params.get("pooling", "gap")} `\n'
    command += f'    --experiment_name "best_optuna_config"'
    
    return command


# =============================================================================
# Main Tuning Function
# =============================================================================

def run_tuning(
    n_trials: int = 100,
    study_name: str = "hybrid_transformer_tuning",
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    n_jobs: int = 1
):
    """
    Run Optuna hyperparameter tuning.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
        storage: SQLite storage path (optional, for resumable studies)
        load_if_exists: Whether to load existing study
        n_jobs: Number of parallel jobs (1 for sequential)
    """
    
    set_seed(TUNE_CONFIG['seed'])
    
    # Create output directory
    save_dir = os.path.join(
        TUNE_CONFIG['save_dir'],
        f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup storage
    if storage is None:
        storage = f"sqlite:///{os.path.join(save_dir, 'optuna_study.db')}"
    
    print("\n" + "=" * 60)
    print("  Optuna Hyperparameter Tuning")
    print("=" * 60)
    print(f"\n📁 Results directory: {save_dir}")
    print(f"💾 Storage: {storage}")
    print(f"🔢 Trials: {n_trials}")
    print(f"⏱️  Epochs per trial: {TUNE_CONFIG['tune_epochs']}")
    print(f"🖥️  OS: {platform.system()} (num_workers={TUNE_CONFIG['num_workers']})")
    
    # Create sampler and pruner
    sampler = optuna.samplers.TPESampler(
        seed=TUNE_CONFIG['seed'],
        n_startup_trials=10,
        multivariate=True
    )
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1
    )
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',  # Maximize validation accuracy
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists
    )
    
    # Print search space
    print("\n📋 Search Space:")
    for param, config in SEARCH_SPACE.items():
        if config['type'] == 'categorical':
            print(f"   {param}: {config['choices']}")
        elif config['type'] == 'int':
            print(f"   {param}: [{config['low']}, {config['high']}]")
        elif config['type'] == 'float':
            log_str = " (log)" if config.get('log', False) else ""
            print(f"   {param}: [{config['low']}, {config['high']}]{log_str}")
    
    print("\n" + "=" * 60)
    print("  Starting Optimization")
    print("=" * 60 + "\n")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Results summary
    print("\n" + "=" * 60)
    print("  Optimization Complete")
    print("=" * 60)
    
    # Statistics
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
    
    print(f"\n📊 Trial Statistics:")
    print(f"   Completed: {len(completed_trials)}")
    print(f"   Pruned:    {len(pruned_trials)}")
    print(f"   Failed:    {len(failed_trials)}")
    
    # Best trial
    print(f"\n⭐ Best Trial:")
    print(f"   Trial #:   {study.best_trial.number}")
    print(f"   Value:     {study.best_value:.4f} (validation accuracy)")
    print(f"   Params:    {study.best_params}")
    
    if 'total_params' in study.best_trial.user_attrs:
        print(f"   Model size: {study.best_trial.user_attrs['total_params']:,} parameters")
    
    # Save results
    print("\n💾 Saving results...")
    save_study_results(study, save_dir)
    
    # Save plots (requires plotly and kaleido)
    try:
        save_optimization_plots(study, os.path.join(save_dir, 'figures'))
    except ImportError:
        print("  ⚠ Plotting requires: pip install plotly kaleido")
    
    # Save config
    config_path = os.path.join(save_dir, 'tune_config.json')
    with open(config_path, 'w') as f:
        json.dump(TUNE_CONFIG, f, indent=2)
    
    print(f"\n✅ All results saved to: {save_dir}")
    
    # Print best training command
    print("\n" + "=" * 60)
    print("  Recommended Training Command")
    print("=" * 60)
    print(generate_training_command(study.best_params))
    
    return study


# =============================================================================
# Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for HybridTransformer using Optuna'
    )
    
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials (default: 100)')
    parser.add_argument('--study_name', type=str, default='hybrid_transformer_tuning',
                        help='Study name (default: hybrid_transformer_tuning)')
    parser.add_argument('--tune_epochs', type=int, default=30,
                        help='Epochs per trial (default: 30)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Parallel jobs (default: 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing study')
    parser.add_argument('--storage', type=str, default=None,
                        help='SQLite storage path (optional)')
    parser.add_argument('--subset', type=float, default=0.1,
                        help='Fraction of data to use for tuning (default: 0.1 = 10%%)')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Update config
    TUNE_CONFIG['tune_epochs'] = args.tune_epochs
    TUNE_CONFIG['tune_subset_fraction'] = args.subset
    
    # Print banner
    print("\n" + "=" * 60)
    print("  HybridTransformer AMC - Optuna Hyperparameter Tuning")
    print("  " + "=" * 56)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  CUDA not available, using CPU (will be slow)")
    
    # Print Windows warning
    if IS_WINDOWS:
        print(f"⚠️  Windows detected: Using num_workers=0 (HDF5 compatibility)")
    
    # Run tuning
    study = run_tuning(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        n_jobs=args.n_jobs
    )
    
    return study


if __name__ == "__main__":
    main()