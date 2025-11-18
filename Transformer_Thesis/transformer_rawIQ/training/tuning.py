"""
Main Training Script for AMC Transformer
MODIFIED FOR 16GB VRAM OPTIMIZATION
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
# CONFIGURATION FOR 16GB VRAM
# ============================================

class Config:
    """Training configuration optimized for 16GB VRAM"""
    
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
    
    # --- Model architecture (Reduced for 16GB VRAM) ---
    SEQ_LENGTH = 1024
    EMBEDDING_TYPE = 'segment'
    SEGMENT_SIZE = 16
    USE_CLS_TOKEN = True
    D_MODEL = 128  # Reduced from 256/512
    N_HEAD = 8
    N_LAYERS = 4   # Reduced from 6
    FFN_HIDDEN = 512  # Reduced from 1024
    DROP_PROB = 0.2
    
    # Training hyperparameters (Optimized for memory)
    BATCH_SIZE = 32  # Increased but manageable
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_MAX_NORM = 1.0
    
    # DataLoader settings (Reduced for memory)
    NUM_WORKERS = 16  # Reduced from higher values
    PREFETCH_FACTOR = 4  # Reduced from 3
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True  # Disabled to save memory
    
    # Early stopping & checkpointing
    PATIENCE = 10
    SAVE_FREQ = 5
    
    # Optuna settings (Reduced for memory constraints)
    N_EPOCHS_PER_TRIAL = 20  # Reduced from 30
    N_TRIALS = 50  # Reduced from 100
    
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
        
        # Memory constraints
        if cls.BATCH_SIZE > 64:
            warnings.warn(f"Batch size {cls.BATCH_SIZE} might be too large for 16GB VRAM")
        if cls.D_MODEL > 256:
            warnings.warn(f"d_model {cls.D_MODEL} might be too large for 16GB VRAM")
        if cls.N_LAYERS > 6:
            warnings.warn(f"n_layers {cls.N_LAYERS} might be too large for 16GB VRAM")
            
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
        description='Train or Tune AMC Transformer (16GB VRAM optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--file_path', type=str, help='Path to HDF5 data file')
    parser.add_argument('--json_path', type=str, help='Path to classes JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, help='Batch size (suggested: 16-64 for 16GB VRAM)')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers (suggested: 1-2 for 16GB VRAM)')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, help='Model dimension (suggested: 64-256 for 16GB VRAM)')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, help='Number of transformer layers (suggested: 2-6 for 16GB VRAM)')
    
    # Other
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (single run only)')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    
    # Optuna arguments
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter tuning')
    parser.add_argument('--n_trials', type=int, help='Number of Optuna trials to run')
    parser.add_argument('--study_name', type=str, default=f"amc_study_{datetime.now().strftime('%Y%m%d')}")
    
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
# MEMORY-EFFICIENT TRAINING FUNCTIONS
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, trial=None):
    """Train for one epoch with memory optimization"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f'Epoch {epoch+1} [Train]'
    if trial:
        desc = f'Trial {trial.number} Epoch {epoch+1} [Train]'
        
    pbar = tqdm(train_loader, desc=desc, leave=False)
    
    for batch_idx, (images, labels, snrs) in enumerate(pbar):
        # Clear cache periodically to prevent memory fragmentation
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.GRAD_CLIP_MAX_NORM
        )
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch, trial=None):
    """Validate for one epoch with memory optimization"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f'Epoch {epoch+1} [Valid]'
    if trial:
        desc = f'Trial {trial.number} Epoch {epoch+1} [Valid]'

    pbar = tqdm(val_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels, snrs) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
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
# FIXED OPTUNA SEARCH SPACE
# ============================================

# Define fixed search spaces that won't change between trials
EMBEDDING_TYPES = ['segment', 'conv1d']
SEGMENT_SIZES = [16, 32, 64]
D_MODELS = [64, 128, 256]
N_HEADS = [4, 8]
N_LAYERS_RANGE = (2, 6)  # Using integer range instead of categorical
FFN_MULTIPLIERS = [2, 4]
BATCH_SIZES = [32, 64, 128]
OPTIMIZERS = ['AdamW', 'Adam']

g_train_dataset = None
g_valid_dataset = None
g_config = None
g_device = None

def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function with fixed search space for 16GB VRAM"""
    
    global g_train_dataset, g_valid_dataset, g_config, g_device
    if g_train_dataset is None or g_valid_dataset is None:
        raise ValueError("Global datasets not set. Run data loading first.")

    try:
        # --- 1. Define Hyperparameters (FIXED search space) ---
        embedding_type = trial.suggest_categorical('embedding_type', EMBEDDING_TYPES)
        segment_size = trial.suggest_categorical('segment_size', SEGMENT_SIZES)
        
        # Reduced search space for 16GB VRAM
        d_model = trial.suggest_categorical("d_model", D_MODELS)
        n_head = trial.suggest_categorical("n_head", N_HEADS)
        
        # Constraint: n_head must divide d_model
        if d_model % n_head != 0:
            raise optuna.exceptions.TrialPruned()

        n_layers = trial.suggest_int("n_layers", N_LAYERS_RANGE[0], N_LAYERS_RANGE[1])
        ffn_multiplier = trial.suggest_categorical("ffn_multiplier", FFN_MULTIPLIERS)
        ffn_hidden = d_model * ffn_multiplier
        drop_prob = trial.suggest_float("drop_prob", 0.1, 0.4)
        use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZERS)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)

        # --- 2. Create DataLoaders ---
        use_persistent = g_config.PERSISTENT_WORKERS and g_config.NUM_WORKERS > 0
        
        train_loader = DataLoader(
            g_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=g_config.NUM_WORKERS,
            pin_memory=g_config.PIN_MEMORY and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn if g_config.NUM_WORKERS > 0 else None,
            persistent_workers=use_persistent,
            prefetch_factor=g_config.PREFETCH_FACTOR if g_config.NUM_WORKERS > 0 else None
        )
        
        valid_loader = DataLoader(
            g_valid_dataset,
            batch_size=batch_size,
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
            'd_model': d_model,
            'n_head': n_head,
            'n_layers': n_layers,
            'ffn_hidden': ffn_hidden,
            'drop_prob': drop_prob,
            'device': g_device,
            'use_cls_token': use_cls_token,
            'embedding_type': embedding_type,
            'segment_size': segment_size
        }
        
        model = AMCTransformer(**model_params).to(g_device)

        # --- 4. Setup Optimizer & Criterion ---
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        if optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99)
            )
        else: # Adam
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        early_stopping = EarlyStopping(patience=g_config.PATIENCE)
        
        # --- 5. Training Loop with Memory Management ---
        best_val_acc = 0.0
        
        for epoch in range(g_config.N_EPOCHS_PER_TRIAL):
            # Clear cache at start of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, g_device, epoch, g_config, trial
            )
            
            val_loss, val_acc = validate_epoch(
                model, valid_loader, criterion, g_device, epoch, trial
            )
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            trial.report(val_acc, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Trial {trial.number}: Early stopping triggered at epoch {epoch+1}")
                break
        
        # --- 6. Clean up aggressively ---
        del model, optimizer, criterion, train_loader, valid_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return best_val_acc

    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return 0.0


# ============================================
# OPTUNA STUDY RUNNER (MEMORY OPTIMIZED)
# ============================================

def run_study(args, config, test_dataset, label_map, norm_stats):
    """Create and run the Optuna study with memory optimization"""
    
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{config.DB_DIR}/{args.study_name}.db"
    
    print("\n" + "="*70)
    print("🚀 STARTING OPTUNA STUDY (16GB VRAM OPTIMIZED)")
    print("="*70)
    print(f"Study Name: {args.study_name}")
    print(f"Database: {storage_name}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per Trial: {config.N_EPOCHS_PER_TRIAL}")
    print("Fixed Search Space:")
    print(f"  Embedding types: {EMBEDDING_TYPES}")
    print(f"  Segment sizes: {SEGMENT_SIZES}")
    print(f"  d_models: {D_MODELS}")
    print(f"  n_heads: {N_HEADS}")
    print(f"  n_layers: {N_LAYERS_RANGE}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print("="*70)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=InMemoryStorage(),
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=False
    )
    
    def gc_callback(study, trial):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=None,
            catch=(Exception,),
            callbacks=[gc_callback],
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Study interrupted by user!")
        
    print("\n" + "="*70)
    print("✅ STUDY COMPLETE!")
    print("="*70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Final evaluation with best model
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
        batch_size=min(best_params['batch_size'], 64),  # Cap batch size for evaluation
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
    
    best_params_path = eval_dir / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best params saved to: {best_params_path}")
    
    # Clean up
    del best_model, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================
# MEMORY-EFFICIENT SINGLE TRAINING
# ============================================

def run_single_training(config, args, experiment_name, exp_checkpoint_dir,
                        train_dataset, valid_dataset, test_indices, label_map, norm_stats):
    """Run a single training loop optimized for 16GB VRAM"""
    
    print("\n" + "="*70)
    print("🚀 STARTING SINGLE TRAINING RUN (16GB VRAM OPTIMIZED)")
    print("="*70)

    # --- Create dataloaders ---
    print("Creating dataloaders...")
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
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_loader):,} batches ({len(train_dataset):,} samples)")
    print(f"   Valid: {len(valid_loader):,} batches ({len(valid_dataset):,} samples)")
    
    # --- Model Setup ---
    print("\n🤖 Initializing model...")
    
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
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created:")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Trainable parameters: {num_trainable:,}")
    
    # Estimate memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"   GPU Memory allocated: {memory_allocated:.2f} GB")
    
    # --- Optimizer & Criterion ---
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.99)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # --- Resume from Checkpoint ---
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        try:
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            print(f"   ✅ Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch...")
    
    # --- Training Loop ---
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Clear cache at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch, config
        )
        
        val_loss, val_acc = validate_epoch(
            model, valid_loader, criterion, config.DEVICE, epoch
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = get_lr(optimizer)
        
        # Print memory usage
        memory_info = ""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            memory_info = f" | GPU Mem: {memory_allocated:.1f}/{memory_cached:.1f} GB"
        
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {current_lr:.2e}{memory_info}")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.NUM_EPOCHS:
            checkpoint_path = exp_checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, epoch,
                val_loss, history, get_config_dict(config)
            )
            print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
        
        # Early stopping
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("\n⏹️  Early stopping triggered!")
            final_path = exp_checkpoint_dir / "model_best.pth"
            save_checkpoint(
                final_path, model, optimizer, scheduler, epoch,
                early_stopping.best_score, history, get_config_dict(config)
            )
            print(f"   💾 Best model saved: {final_path}")
            break
        
        print("-" * 70)
    
    # --- Training Complete ---
    total_training_time = time.time() - training_start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total time: {total_training_time/3600:.2f} hours")
    print(f"Best val loss: {early_stopping.best_score:.4f}")
    
    if not early_stopping.early_stop:
        final_path = exp_checkpoint_dir / "model_final.pth"
        save_checkpoint(
            final_path, model, optimizer, scheduler, epoch,
            val_loss, history, get_config_dict(config)
        )
        print(f"Final model saved: {final_path}")

    plot_path = config.LOG_DIR / f"{experiment_name}_training_history.png"
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved: {plot_path}")
    
    # --- Final Evaluation ---
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    best_model_path = exp_checkpoint_dir / "model_best.pth"
    if not best_model_path.exists():
        best_model_path = exp_checkpoint_dir / "model_final.pth"
    
    print(f"Loading model from: {best_model_path}")
    load_checkpoint(best_model_path, model)
    
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
    
    print(f"Test set: {len(test_loader):,} batches ({len(test_dataset):,} samples)")
    
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
# MAIN ROUTER
# ============================================

def main():
    """Main function to route to single run or Optuna study"""
    
    config = None
    experiment_name = None
    exp_checkpoint_dir = None
    
    try:
        # --- 1. Initial Setup ---
        args = parse_args()
        config = Config.from_args(args)
        
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        experiment_name = args.experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_checkpoint_dir = config.CHECKPOINT_DIR / experiment_name
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        print("="*70)
        print("AMC TRANSFORMER (16GB VRAM OPTIMIZED)")
        print("="*70)
        print(f"Mode: {'TUNING' if args.tune else 'SINGLE RUN'}")
        print(f"Device: {config.DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # --- 2. Load Data ---
        print("\n📂 Loading data...")
        
        train_indices, valid_indices, test_indices, label_map = split_data(
            str(config.FILE_PATH), str(config.JSON_PATH), config.TARGET_MODULATIONS,
            config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE, config.SPLIT_SEED
        )
        
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
        print(f"Normalization stats: I μ={norm_stats['i_mean']:.4f}, σ={norm_stats['i_std']:.4f} | "
              f"Q μ={norm_stats['q_mean']:.4f}, σ={norm_stats['q_std']:.4f}")

        valid_dataset = SingleStreamImageDataset(
            file_path=str(config.FILE_PATH),
            json_path=str(config.JSON_PATH),
            target_modulations=config.TARGET_MODULATIONS,
            mode='valid',
            indices=valid_indices,
            label_map=label_map,
            normalization_stats=norm_stats
        )
        
        # --- 3. Route to Tuning or Single Run ---
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

        # --- 4. Cleanup ---
        print("\nCleaning up datasets...")
        train_dataset.close()
        valid_dataset.close()
        if 'test_dataset' in locals():
            test_dataset.close()
        
        print("\n✅ All done!")

    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user!")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()