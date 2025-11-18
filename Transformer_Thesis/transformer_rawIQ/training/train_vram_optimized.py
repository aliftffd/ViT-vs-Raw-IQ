"""
Main Training Script for AMC Transformer
AGGRESSIVELY OPTIMIZED FOR 16GB VRAM - Maximum Utilization (80-85%)
Based on VRAM_OPTIMIZATION_GUIDE.md recommendations

Key Optimizations:
- BATCH_SIZE: 128 → 256 (2x increase)
- D_MODEL: 256 → 384 (+50% capacity)
- N_LAYERS: 9 → 12 (+33% depth)
- FFN_HIDDEN: 512 → 768 (2× d_model standard)
- NUM_WORKERS: 8 → 12 (better CPU utilization)
- PREFETCH_FACTOR: 3 → 5 (more aggressive prefetching)
- CACHE_CLEAR_FREQUENCY: 50 → 100 (less overhead)

Expected Results:
- VRAM Usage: 5GB → 12-14GB (80-85% utilization)
- Training Speed: 35 it/s → 65-80 it/s (2x faster)
- Epoch Time: ~40 min → ~18-22 min (2x faster)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.storages import InMemoryStorage

warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# Add parent directory to path to import custom modules
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from dataloader.dataset import SingleStreamImageDataset, worker_init_fn
from dataloader.utils import split_data
from models.transformer_rawIQ import AMCTransformer
from training.utils import (
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    EarlyStopping,
    get_lr,
    evaluate_model_with_confusion
)

# ============================================
# AGGRESSIVELY OPTIMIZED CONFIGURATION FOR 16GB VRAM
# Target: 80-85% VRAM Utilization (12-14GB)
# ============================================

class Config:
    """Training configuration AGGRESSIVELY optimized for 16GB VRAM (80-85% target utilization)"""

    # Paths
    DATA_DIR = Path("data")
    FILE_PATH = "C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\classes-fixed.json'
    CHECKPOINT_DIR = Path("result/checkpoints")
    LOG_DIR = Path("result/logs")
    DB_DIR = Path("result/optuna_db")

    # Data split
    TRAIN_SIZE = 0.7
    VALID_SIZE = 0.15
    TEST_SIZE = 0.15
    SPLIT_SEED = 42
    NORM_SEED = 49

    # Target modulations
    TARGET_MODULATIONS = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
        '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
        '128QAM', '256QAM', 'GMSK', 'OQPSK'
    ]

    # --- Model architecture (AGGRESSIVELY OPTIMIZED FOR 16GB VRAM) ---
    SEQ_LENGTH = 1024
    EMBEDDING_TYPE = 'segment'
    SEGMENT_SIZE = 16
    USE_CLS_TOKEN = True
    D_MODEL = 384        # ← INCREASED from 256 (+50% capacity)
    N_HEAD = 16          # ← Unchanged (divides 384)
    N_LAYERS = 12        # ← INCREASED from 9 (+33% depth)
    FFN_HIDDEN = 768     # ← INCREASED to 2× d_model (standard practice)
    DROP_PROB = 0.1

    # Training hyperparameters (OPTIMIZED)
    BATCH_SIZE = 256     # ← DOUBLED from 128 (biggest impact!)
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_MAX_NORM = 1.0

    # DataLoader settings (AGGRESSIVELY OPTIMIZED)
    NUM_WORKERS = 12     # ← INCREASED from 8 (better CPU utilization)
    PREFETCH_FACTOR = 5  # ← INCREASED from 3 (more aggressive prefetching)
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    # Mixed precision training (ENABLED - saves VRAM and speeds up)
    USE_AMP = True

    # Memory management (OPTIMIZED)
    CACHE_CLEAR_FREQUENCY = 100  # ← INCREASED from 50 (less overhead)
    EMPTY_CACHE_BETWEEN_TRIALS = True

    # Gradient accumulation (Optional - for even larger effective batch sizes)
    ACCUMULATION_STEPS = 1  # Set to 2 for effective batch=512

    # Early stopping & checkpointing
    PATIENCE = 10
    SAVE_FREQ = 5

    # Optuna settings (Academically rigorous + aggressive search)
    N_EPOCHS_PER_TRIAL = 20
    N_TRIALS = 50
    PRUNE_STARTUP_TRIALS = 5
    PRUNE_WARMUP_STEPS = 5

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        errors = []

        if not Path(cls.FILE_PATH).exists():
            errors.append(f"HDF5 file not found: {cls.FILE_PATH}")
        if not Path(cls.JSON_PATH).exists():
            errors.append(f"JSON file not found: {cls.JSON_PATH}")

        split_sum = cls.TRAIN_SIZE + cls.VALID_SIZE + cls.TEST_SIZE
        if not np.isclose(split_sum, 1.0):
            errors.append(f"Data splits must sum to 1.0, got {split_sum}")

        if cls.D_MODEL % cls.N_HEAD != 0:
            errors.append(f"D_MODEL ({cls.D_MODEL}) must be divisible by N_HEAD ({cls.N_HEAD})")

        if cls.BATCH_SIZE <= 0: errors.append("BATCH_SIZE must be positive")
        if cls.NUM_EPOCHS <= 0: errors.append("NUM_EPOCHS must be positive")
        if cls.LEARNING_RATE <= 0: errors.append("LEARNING_RATE must be positive")

        if cls.NUM_WORKERS < 0:
            errors.append(f"NUM_WORKERS cannot be negative, got {cls.NUM_WORKERS}")

        # Warnings for aggressive settings
        if cls.NUM_WORKERS > 10:
            warnings.warn(f"NUM_WORKERS={cls.NUM_WORKERS} is aggressive. Monitor RAM usage.")
        if cls.BATCH_SIZE > 200:
            warnings.warn(f"BATCH_SIZE={cls.BATCH_SIZE} is large. Monitor VRAM usage.")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

    @classmethod
    def from_args(cls, args):
        """Update config from command line arguments with validation"""
        for key, value in vars(args).items():
            if value is not None and hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)

        if not args.tune:
            cls.validate()
        return cls


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train or Tune AMC Transformer (AGGRESSIVELY optimized for 16GB VRAM)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--file_path', type=str, help='Path to HDF5 data file')
    parser.add_argument('--json_path', type=str, help='Path to classes JSON file')

    # Training arguments
    parser.add_argument('--batch_size', type=int, help='Batch size (default: 256 for max VRAM usage)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers (default: 12)')

    # Model arguments
    parser.add_argument('--d_model', type=int, help='Model dimension (default: 384)')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, help='Number of transformer layers (default: 12)')

    # Memory optimization
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false', help='Disable automatic mixed precision')

    # Other
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (single run only)')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')

    # Optuna arguments
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, help='Number of Optuna trials to run')
    parser.add_argument('--study_name', type=str, default=f"amc_vram_optimized_{datetime.now().strftime('%Y%m%d')}")

    return parser.parse_args()


def get_config_dict(config):
    """Convert Config class to serializable dictionary"""
    return {
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'LABEL_SMOOTHING': config.LABEL_SMOOTHING,
        'GRAD_CLIP_MAX_NORM': config.GRAD_CLIP_MAX_NORM,
        'NUM_WORKERS': config.NUM_WORKERS,
        'PREFETCH_FACTOR': config.PREFETCH_FACTOR,
        'USE_AMP': config.USE_AMP,
        'ACCUMULATION_STEPS': config.ACCUMULATION_STEPS,
        'SEQ_LENGTH': config.SEQ_LENGTH,
        'EMBEDDING_TYPE': config.EMBEDDING_TYPE,
        'SEGMENT_SIZE': config.SEGMENT_SIZE,
        'USE_CLS_TOKEN': config.USE_CLS_TOKEN,
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
        'NORM_SEED': config.NORM_SEED,
        'PATIENCE': config.PATIENCE,
        'SAVE_FREQ': config.SAVE_FREQ
    }


# ============================================
# MEMORY-EFFICIENT TRAINING FUNCTIONS WITH AMP + GRADIENT ACCUMULATION
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler=None, trial=None):
    """Train for one epoch with mixed precision support and optional gradient accumulation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    accumulation_steps = getattr(config, 'ACCUMULATION_STEPS', 1)

    desc = f'Epoch {epoch+1} [Train]'
    if trial:
        desc = f'Trial {trial.number} Epoch {epoch+1} [Train]'

    pbar = tqdm(train_loader, desc=desc, leave=False)
    use_amp = config.USE_AMP and scaler is not None

    for batch_idx, (images, labels, snrs) in enumerate(pbar):
        # Less frequent cache clearing for better performance
        if batch_idx % config.CACHE_CLEAR_FREQUENCY == 0 and batch_idx > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if use_amp:
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Scale loss for gradient accumulation

            # Mixed precision backward pass
            scaler.scale(loss).backward()

            # Only step optimizer every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            # Standard precision
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch, config, trial=None):
    """Validate for one epoch with mixed precision support"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = f'Epoch {epoch+1} [Valid]'
    if trial:
        desc = f'Trial {trial.number} Epoch {epoch+1} [Valid]'

    pbar = tqdm(val_loader, desc=desc, leave=False)
    use_amp = config.USE_AMP and torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (images, labels, snrs) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


# ============================================
# AGGRESSIVE OPTUNA SEARCH SPACE FOR MAXIMUM VRAM UTILIZATION
# ============================================

def get_memory_safe_constraints(d_model, n_layers, batch_size):
    """
    Estimate VRAM usage for a given configuration.

    Based on empirical testing and optimization guide:
    - d_model=384, n_layers=12, batch=256 should use ~12-13GB VRAM
    - Target: 80-85% of 16GB = 12.8-13.6GB

    Returns (is_safe, estimated_vram_gb)
    """

    seq_length = 1024
    num_classes = 19

    # Model parameter count estimation
    avg_ffn_multiplier = 3
    params_per_layer = (
        4 * d_model * d_model +  # Attention
        2 * d_model * (d_model * avg_ffn_multiplier) +  # FFN
        4 * d_model  # Layer norms
    )

    total_params = (
        params_per_layer * n_layers +
        2 * seq_length * d_model +  # Embeddings
        d_model * num_classes  # Classifier
    )

    # Memory estimation (in GB)
    param_memory = (total_params * 4) / (1024**3)
    optimizer_memory = param_memory * 2

    activation_memory = (
        batch_size * seq_length * d_model * n_layers * 12 * 4
    ) / (1024**3)

    base_estimate = param_memory + optimizer_memory + activation_memory
    estimated_vram = base_estimate * 1.3

    # With AMP, activations use less memory
    if hasattr(Config, 'USE_AMP') and Config.USE_AMP:
        estimated_vram_amp = param_memory + optimizer_memory + (activation_memory * 0.6) * 1.3
    else:
        estimated_vram_amp = estimated_vram

    # More aggressive threshold: allow up to 15GB estimated
    # (estimates tend to be conservative)
    is_safe = estimated_vram_amp < 15.0

    return is_safe, estimated_vram_amp


def suggest_safe_hyperparameters(trial):
    """
    AGGRESSIVE hyperparameter suggestions for maximum VRAM utilization.

    Based on VRAM_OPTIMIZATION_GUIDE.md:
    - Larger d_model values (256-448)
    - More layers (9-15)
    - Larger batch sizes (128-256)
    """

    embedding_type = trial.suggest_categorical('embedding_type', ['segment', 'conv1d'])
    segment_size = trial.suggest_categorical('segment_size', [16, 32])

    # === AGGRESSIVE ARCHITECTURE PARAMETERS ===
    # Increased from conservative [128, 192, 256, 320]
    d_model = trial.suggest_categorical("d_model", [256, 320, 384, 448])

    # Number of heads must divide d_model
    n_head = trial.suggest_categorical("n_head", [8, 16])

    if d_model % n_head != 0:
        raise optuna.exceptions.TrialPruned()

    # Increased layer range from [6, 12] to [9, 15]
    n_layers = trial.suggest_int("n_layers", 9, 15)

    # FFN multiplier
    ffn_multiplier = trial.suggest_categorical("ffn_multiplier", [2, 3, 4])
    ffn_hidden = d_model * ffn_multiplier

    # === AGGRESSIVE BATCH SIZE ===
    # FIX: Optuna requires fixed categorical choices across all trials
    # We use a unified aggressive range and prune unsafe combinations
    batch_size = trial.suggest_categorical("batch_size", [96, 128, 160, 192, 224, 256])

    # Memory safety check with aggressive model-size-aware pruning
    is_safe, estimated_vram = get_memory_safe_constraints(d_model, n_layers, batch_size)

    # Prune unsafe configurations aggressively but smartly
    # Strategy: Larger models need smaller batches to fit in 16GB
    if d_model <= 256:
        # Smaller models: can use largest batches
        if estimated_vram > 15.0:
            print(f"Trial {trial.number}: d={d_model}, batch={batch_size} - pruning (VRAM: {estimated_vram:.1f}GB)")
            raise optuna.exceptions.TrialPruned()
    elif d_model <= 384:
        # Medium-large models: avoid very large batches
        if batch_size > 192 or estimated_vram > 15.0:
            print(f"Trial {trial.number}: d={d_model}, batch={batch_size} - pruning (VRAM: {estimated_vram:.1f}GB)")
            raise optuna.exceptions.TrialPruned()
    else:  # d_model >= 448
        # Very large models: only moderate batches
        if batch_size > 160 or estimated_vram > 15.0:
            print(f"Trial {trial.number}: d={d_model}, batch={batch_size} - pruning (VRAM: {estimated_vram:.1f}GB)")
            raise optuna.exceptions.TrialPruned()

    print(f"Trial {trial.number}: d={d_model}, L={n_layers}, B={batch_size} | Est. VRAM: {estimated_vram:.1f}GB")

    # === REGULARIZATION ===
    drop_prob = trial.suggest_float("drop_prob", 0.05, 0.25)
    use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # === OPTIMIZATION ===
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ['AdamW', 'Adam'])

    return {
        'embedding_type': embedding_type,
        'segment_size': segment_size,
        'd_model': d_model,
        'n_head': n_head,
        'n_layers': n_layers,
        'ffn_multiplier': ffn_multiplier,
        'ffn_hidden': ffn_hidden,
        'drop_prob': drop_prob,
        'use_cls_token': use_cls_token,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'optimizer_name': optimizer_name,
        'label_smoothing': label_smoothing
    }


# Global variables for datasets (reused across trials)
g_train_dataset = None
g_valid_dataset = None
g_config = None
g_device = None
g_train_loader = None
g_valid_loader = None

def objective(trial: optuna.trial.Trial) -> float:
    """Optimized Optuna objective function with OOM error handling"""

    global g_train_dataset, g_valid_dataset, g_config, g_device, g_train_loader, g_valid_loader
    if g_train_dataset is None or g_valid_dataset is None:
        raise ValueError("Global datasets not set. Run data loading first.")

    try:
        # --- 1. Suggest Hyperparameters ---
        params = suggest_safe_hyperparameters(trial)

        _, est_vram = get_memory_safe_constraints(
            params['d_model'], params['n_layers'], params['batch_size']
        )
        print(f"Trial {trial.number}: Est. VRAM {est_vram:.1f}GB | "
              f"d={params['d_model']}, L={params['n_layers']}, B={params['batch_size']}")

        # --- 2. Create DataLoaders (reuse if batch_size unchanged) ---
        current_batch_size = params['batch_size']

        need_new_loaders = (
            g_train_loader is None or
            g_train_loader.batch_size != current_batch_size
        )

        if need_new_loaders:
            if g_train_loader is not None:
                del g_train_loader, g_valid_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            use_persistent = g_config.PERSISTENT_WORKERS and g_config.NUM_WORKERS > 0

            g_train_loader = DataLoader(
                g_train_dataset,
                batch_size=current_batch_size,
                shuffle=True,
                num_workers=g_config.NUM_WORKERS,
                pin_memory=g_config.PIN_MEMORY and torch.cuda.is_available(),
                worker_init_fn=worker_init_fn if g_config.NUM_WORKERS > 0 else None,
                persistent_workers=use_persistent,
                prefetch_factor=g_config.PREFETCH_FACTOR if g_config.NUM_WORKERS > 0 else None
            )

            g_valid_loader = DataLoader(
                g_valid_dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=g_config.NUM_WORKERS,
                pin_memory=g_config.PIN_MEMORY and torch.cuda.is_available(),
                worker_init_fn=worker_init_fn if g_config.NUM_WORKERS > 0 else None,
                persistent_workers=use_persistent,
                prefetch_factor=g_config.PREFETCH_FACTOR if g_config.NUM_WORKERS > 0 else None
            )

        # --- 3. Initialize Model ---
        model_params = {
            'in_channels': 2,
            'seq_length': g_config.SEQ_LENGTH,
            'num_classes': len(g_config.TARGET_MODULATIONS),
            'd_model': params['d_model'],
            'n_head': params['n_head'],
            'n_layers': params['n_layers'],
            'ffn_hidden': params['ffn_hidden'],
            'drop_prob': params['drop_prob'],
            'device': g_device,
            'use_cls_token': params['use_cls_token'],
            'embedding_type': params['embedding_type'],
            'segment_size': params['segment_size']
        }

        model = AMCTransformer(**model_params).to(g_device)

        # --- 4. Setup Optimizer & Criterion ---
        criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])

        if params['optimizer_name'] == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay'],
                betas=(0.9, 0.999)
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        scaler = torch.amp.GradScaler('cuda') if g_config.USE_AMP and torch.cuda.is_available() else None
        early_stopping = EarlyStopping(patience=5)

        # --- 5. Training Loop ---
        best_val_acc = 0.0

        for epoch in range(g_config.N_EPOCHS_PER_TRIAL):
            train_loss, train_acc = train_epoch(
                model, g_train_loader, criterion, optimizer, g_device,
                epoch, g_config, scaler, trial
            )

            val_loss, val_acc = validate_epoch(
                model, g_valid_loader, criterion, g_device, epoch, g_config, trial
            )

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            trial.report(val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Trial {trial.number}: Early stopping at epoch {epoch+1}")
                break

        # --- 6. Cleanup ---
        del model, optimizer, criterion, scheduler
        if scaler is not None:
            del scaler

        if torch.cuda.is_available() and g_config.EMPTY_CACHE_BETWEEN_TRIALS:
            torch.cuda.empty_cache()
        gc.collect()

        return best_val_acc

    except RuntimeError as e:
        # OOM Error Handler (from optimization guide)
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number}: OOM Error - pruning trial")
            # Clean up
            if 'model' in locals():
                del model
            if 'optimizer' in locals():
                del optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()
        else:
            raise

    except optuna.exceptions.TrialPruned:
        # Clean up before re-raising
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise

    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()

        # Clean up on error
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return 0.0


# ============================================
# OPTIMIZED OPTUNA STUDY RUNNER
# ============================================

def run_study(args, config, test_dataset, label_map, norm_stats):
    """Run Optuna study with aggressive VRAM optimization"""

    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{config.DB_DIR}/{args.study_name}.db"

    print("\n" + "="*70)
    print("🚀 AGGRESSIVE VRAM OPTIMIZATION - HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Study Name: {args.study_name}")
    print(f"Database: {storage_name}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per Trial: {config.N_EPOCHS_PER_TRIAL}")
    print("\n📊 Aggressive Search Space (80-85% VRAM Target):")
    print("  d_model:        [256, 320, 384, 448]")
    print("  n_layers:       [9, 10, 11, 12, 13, 14, 15]")
    print("  batch_size:     [96, 128, 160, 192, 224, 256] (adaptive)")
    print("\n🎯 Target Performance:")
    print("  VRAM Usage:     12-14GB (80-85% of 16GB)")
    print("  Training Speed: 65-80 it/s (2× faster than baseline)")
    print("  Expected Speedup: 1.8-2.3×")
    print("="*70)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=InMemoryStorage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.PRUNE_STARTUP_TRIALS,
            n_warmup_steps=config.PRUNE_WARMUP_STEPS
        ),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
        load_if_exists=False
    )

    # Enqueue optimized baseline configuration
    print("\n🎯 Enqueuing optimized baseline (d=384, L=12, B=256)...")
    try:
        study.enqueue_trial({
            'embedding_type': 'segment',
            'segment_size': 16,
            'd_model': 384,
            'n_head': 16,
            'n_layers': 12,
            'ffn_multiplier': 2,
            'drop_prob': 0.1,
            'use_cls_token': True,
            'batch_size': 256,
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'optimizer': 'AdamW',
            'label_smoothing': 0.1
        })
        print("✅ Optimized baseline configuration enqueued as Trial 0")
    except Exception as e:
        print(f"⚠️  Could not enqueue baseline: {e}")

    def memory_callback(study, trial):
        """Aggressive memory cleanup between trials"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=None,
            catch=(Exception,),
            callbacks=[memory_callback],
            gc_after_trial=True,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Study interrupted by user!")

    elapsed_time = time.time() - start_time

    print("\n" + "="*70)
    print("✅ STUDY COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Completed trials: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    _, est_vram = get_memory_safe_constraints(
        study.best_params['d_model'],
        study.best_params['n_layers'],
        study.best_params['batch_size']
    )
    print(f"\n📊 Estimated VRAM for best model: {est_vram:.1f} GB")

    # Final evaluation
    print("\n" + "="*70)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*70)

    best_params = study.best_params

    model_params = {
        'in_channels': 2,
        'seq_length': config.SEQ_LENGTH,
        'num_classes': len(config.TARGET_MODULATIONS),
        'd_model': best_params['d_model'],
        'n_head': best_params['n_head'],
        'n_layers': best_params['n_layers'],
        'ffn_hidden': best_params['d_model'] * best_params['ffn_multiplier'],
        'drop_prob': best_params['drop_prob'],
        'device': config.DEVICE,
        'use_cls_token': best_params['use_cls_token'],
        'embedding_type': best_params['embedding_type'],
        'segment_size': best_params.get('segment_size')
    }

    best_model = AMCTransformer(**model_params).to(config.DEVICE)

    use_persistent = config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(best_params['batch_size'], 64),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if config.NUM_WORKERS > 0 else None,
        persistent_workers=use_persistent,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    print(f"Test set: {len(test_loader):,} batches ({len(test_dataset):,} samples)")

    eval_dir = config.LOG_DIR / f"study_{args.study_name}_best_eval"
    eval_results = evaluate_model_with_confusion(
        model=best_model,
        dataloader=test_loader,
        device=config.DEVICE,
        class_names=config.TARGET_MODULATIONS,
        save_dir=eval_dir,
        prefix='best_trial_test'
    )

    print(f"\n✅ Best Model Test Accuracy: {eval_results.get('accuracy', 'N/A')}")

    # Save results
    results_path = eval_dir / "study_results.json"
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'total_time_hours': elapsed_time / 3600,
        'estimated_vram_gb': est_vram,
        'test_accuracy': eval_results.get('accuracy')
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nStudy results saved to: {results_path}")

    # Cleanup
    del best_model, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================
# OPTIMIZED SINGLE TRAINING
# ============================================

def run_single_training(config, args, experiment_name, exp_checkpoint_dir,
                        train_dataset, valid_dataset, test_indices, label_map, norm_stats):
    """Run single training with aggressive VRAM optimization"""

    print("\n" + "="*70)
    print("🚀 AGGRESSIVE VRAM OPTIMIZATION - SINGLE TRAINING")
    print("="*70)
    print(f"Configuration:")
    print(f"  d_model:     {config.D_MODEL}")
    print(f"  n_layers:    {config.N_LAYERS}")
    print(f"  batch_size:  {config.BATCH_SIZE}")
    print(f"  workers:     {config.NUM_WORKERS}")
    print(f"  prefetch:    {config.PREFETCH_FACTOR}")
    print("="*70)

    use_persistent = config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if config.NUM_WORKERS > 0 else None,
        persistent_workers=use_persistent,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if config.NUM_WORKERS > 0 else None,
        persistent_workers=use_persistent,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    print(f"\n✅ Data loaded:")
    print(f"   Train: {len(train_loader):,} batches ({len(train_dataset):,} samples)")
    print(f"   Valid: {len(valid_loader):,} batches ({len(valid_dataset):,} samples)")

    # Model setup
    print(f"\n🤖 Initializing model...")

    model_params = {
        'in_channels': 2,
        'seq_length': config.SEQ_LENGTH,
        'num_classes': len(config.TARGET_MODULATIONS),
        'd_model': config.D_MODEL,
        'n_head': config.N_HEAD,
        'n_layers': config.N_LAYERS,
        'ffn_hidden': config.FFN_HIDDEN,
        'drop_prob': config.DROP_PROB,
        'device': config.DEVICE,
        'use_cls_token': config.USE_CLS_TOKEN,
        'embedding_type': config.EMBEDDING_TYPE,
        'segment_size': config.SEGMENT_SIZE
    }

    model = AMCTransformer(**model_params).to(config.DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {num_params:,}")

    _, est_vram = get_memory_safe_constraints(
        config.D_MODEL, config.N_LAYERS, config.BATCH_SIZE
    )
    print(f"   Estimated VRAM: {est_vram:.1f} GB")

    # Optimizer & criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda') if config.USE_AMP and torch.cuda.is_available() else None
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    # Training loop
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if args.resume:
        print(f"\n📥 Resuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    training_start_time = time.time()

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            epoch, config, scaler
        )

        val_loss, val_acc = validate_epoch(
            model, valid_loader, criterion, config.DEVICE, epoch, config
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start_time
        current_lr = get_lr(optimizer)

        memory_info = ""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            memory_info = f" | GPU: {memory_allocated:.1f}/{memory_cached:.1f}GB"

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"   Train: {train_loss:.4f} / {train_acc:.2f}%")
        print(f"   Val:   {val_loss:.4f} / {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {current_lr:.2e}{memory_info}")

        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = exp_checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'config': get_config_dict(config)
            }
            if scaler:
                checkpoint_dict['scaler'] = scaler.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"   💾 Saved: {checkpoint_path.name}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\n⏹️  Early stopping!")
            break

    total_time = time.time() - training_start_time
    print(f"\n✅ Training complete! Total time: {total_time/3600:.2f} hours")

    # Save final model
    final_path = exp_checkpoint_dir / "model_final.pth"
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'history': history,
        'config': get_config_dict(config)
    }
    if scaler:
        checkpoint_dict['scaler'] = scaler.state_dict()
    torch.save(checkpoint_dict, final_path)

    # Plot history
    plot_path = config.LOG_DIR / f"{experiment_name}_training_history.png"
    plot_training_history(history, save_path=plot_path)

    # Test evaluation
    print("\n" + "="*70)
    print("TEST EVALUATION")
    print("="*70)

    test_dataset = SingleStreamImageDataset(
        file_path=str(config.FILE_PATH),
        json_path=str(config.JSON_PATH),
        target_modulations=config.TARGET_MODULATIONS,
        mode='test',
        indices=test_indices,
        label_map=label_map,
        normalization_stats=norm_stats
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if config.NUM_WORKERS > 0 else None,
        persistent_workers=use_persistent,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )

    eval_results = evaluate_model_with_confusion(
        model=model,
        dataloader=test_loader,
        device=config.DEVICE,
        class_names=config.TARGET_MODULATIONS,
        save_dir=exp_checkpoint_dir / "evaluation",
        prefix='test'
    )

    print(f"\n✅ Test Accuracy: {eval_results.get('accuracy', 'N/A')}")
    return test_dataset


# ============================================
# MAIN
# ============================================

def main():
    """Main function"""

    try:
        args = parse_args()
        config = Config.from_args(args)

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

        experiment_name = args.experiment_name or f"vram_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_checkpoint_dir = config.CHECKPOINT_DIR / experiment_name
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

        # PyTorch optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        print("="*70)
        print("AMC TRANSFORMER - AGGRESSIVE VRAM OPTIMIZATION")
        print("="*70)
        print(f"Mode: {'TUNING' if args.tune else 'SINGLE RUN'}")
        print(f"Device: {config.DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {total_memory:.1f} GB")
            print(f"Target: 80-85% (~{total_memory * 0.825:.1f} GB)")

        # Load data
        print("\n📂 Loading data...")

        train_indices, valid_indices, test_indices, label_map = split_data(
            str(config.FILE_PATH), str(config.JSON_PATH), config.TARGET_MODULATIONS,
            config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE, config.SPLIT_SEED
        )

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

        # Route to tuning or single run
        if args.tune:
            global g_train_dataset, g_valid_dataset, g_config, g_device
            g_train_dataset = train_dataset
            g_valid_dataset = valid_dataset
            g_config = config
            g_device = config.DEVICE

            test_dataset = SingleStreamImageDataset(
                file_path=str(config.FILE_PATH),
                json_path=str(config.JSON_PATH),
                target_modulations=config.TARGET_MODULATIONS,
                mode='test',
                indices=test_indices,
                label_map=label_map,
                normalization_stats=norm_stats
            )

            run_study(args, config, test_dataset, label_map, norm_stats)
        else:
            test_dataset = run_single_training(
                config, args, experiment_name, exp_checkpoint_dir,
                train_dataset, valid_dataset, test_indices,
                label_map, norm_stats
            )

        # Cleanup
        print("\nCleaning up...")
        train_dataset.close()
        valid_dataset.close()
        if 'test_dataset' in locals():
            test_dataset.close()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("\n✅ All done!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
