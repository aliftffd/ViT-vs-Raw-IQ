"""
AMC Transformer Hyperparameter Tuning with Optuna
OPTIMIZED FOR 16GB VRAM with Dynamic Memory Management
WITH STRATIFIED SAMPLING - FIXED VERSION
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")
import gc
import pickle
import psutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import optuna
from optuna.storages import RDBStorage
import h5py

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent))

from dataloader.dataset import SingleStreamImageDataset, worker_init_fn
from dataloader.utils import split_data
from models.transformer_rawIQ import AMCTransformer
from training.utils import (
    save_checkpoint, 
    plot_training_history,
    EarlyStopping,
    get_lr,
    evaluate_model_with_confusion
)
import torch
torch.cuda.empty_cache()

# ============================================
# STRATIFIED SAMPLING FUNCTIONS - FIXED
# ============================================

def get_dataset_labels_and_snrs(file_path, json_path, target_modulations):
    """Extract labels and SNRs from dataset for stratified sampling - FIXED VERSION"""
    
    # Load JSON labels - it's a list, not a dict
    with open(json_path, 'r') as f:
        modulation_names = json.load(f)  # This is the list of 24 modulation names
    
    print(f"📋 Loaded {len(modulation_names)} modulation names from JSON")
    
    # Load HDF5 file to get actual labels and SNR values
    with h5py.File(file_path, 'r') as f:
        Y_one_hot = f['Y'][:]  # One-hot encoded labels (2555904, 24)
        Z = f['Z'][:]  # SNR values (2555904, 1)
    
    print(f"📊 HDF5 shapes - Y: {Y_one_hot.shape}, Z: {Z.shape}")
    
    # Convert one-hot to class indices and then to modulation names
    Y_indices = np.argmax(Y_one_hot, axis=1)
    Y_strings = [modulation_names[idx] for idx in Y_indices]
    Y_strings = np.array(Y_strings)
    Z_values = Z.flatten()  # Convert (N,1) to (N,)
    
    # Filter for target modulations only
    target_mask = np.isin(Y_strings, target_modulations)
    all_indices = np.where(target_mask)[0]
    filtered_Y = Y_strings[target_mask]
    filtered_Z = Z_values[target_mask]
    
    print(f"✅ Filtered for target modulations: {len(all_indices):,} samples")
    print(f"   Target modulations: {target_modulations}")
    
    return all_indices, filtered_Y, filtered_Z

def stratified_split_data(file_path, json_path, target_modulations, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    """
    Perform stratified sampling by modulation and SNR - FIXED VERSION
    """
    np.random.seed(seed)
    
    # Get all indices with their labels and SNRs
    all_indices, Y_strings, Z_values = get_dataset_labels_and_snrs(file_path, json_path, target_modulations)
    
    print(f"🎯 Starting stratified split with {len(all_indices):,} samples")
    
    # Create strata: group by modulation and binned SNR (2dB bins for better grouping)
    strata = defaultdict(list)
    
    # FIX: Create a lookup map for the print function later
    # Maps Absolute HDF5 Index -> (Modulation, SNR)
    metadata_lookup = {}
    
    for idx, mod, snr in zip(all_indices, Y_strings, Z_values):
        # Store metadata for lookup
        metadata_lookup[idx] = (mod, snr)
        
        # Bin SNR in 2dB steps
        snr_bin = int(np.round(snr / 2) * 2)
        key = f"{mod}_SNR{snr_bin:+d}"
        strata[key].append(idx)
    
    print(f"📈 Created {len(strata)} strata (modulation × SNR bins)")
    
    # Split each stratum proportionally
    train_indices = []
    val_indices = []
    test_indices = []
    
    strata_used = 0
    for stratum_key, stratum_indices in strata.items():
        n_stratum = len(stratum_indices)
        
        # Skip strata with too few samples
        if n_stratum < 3:
            continue
            
        strata_used += 1
        
        # Shuffle stratum
        np.random.shuffle(stratum_indices)
        
        # Calculate split sizes for this stratum
        n_train = int(n_stratum * train_size)
        n_val = int(n_stratum * val_size)
        n_test = n_stratum - n_train - n_val
        
        # Ensure we have at least 1 sample in each split
        if n_train > 0 and n_val > 0 and n_test > 0:
            train_indices.extend(stratum_indices[:n_train])
            val_indices.extend(stratum_indices[n_train:n_train + n_val])
            test_indices.extend(stratum_indices[n_train + n_val:])
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)
    
    # Create label map
    label_map = {mod: idx for idx, mod in enumerate(target_modulations)}
    
    print(f"✅ Stratified split completed:")
    print(f"   Used {strata_used}/{len(strata)} strata")
    print(f"   Train: {len(train_indices):,} samples")
    print(f"   Validation: {len(val_indices):,} samples")
    print(f"   Test: {len(test_indices):,} samples")
    
    # FIX: Pass the lookup dictionary instead of the raw arrays
    print_distribution_stats(train_indices, val_indices, test_indices, metadata_lookup, "Stratified Split")
    
    return train_indices, val_indices, test_indices, label_map

def stratified_sampling(indices, Y_strings, Z_values, target_modulations, ratio=0.1, seed=42):
    """Create stratified subset for fast tuning - FIXED VERSION"""
    np.random.seed(seed)
    
    print(f"🎯 Creating stratified subset with {ratio*100:.1f}% ratio")
    
    # Create strata
    strata = defaultdict(list)
    for idx, mod, snr in zip(indices, Y_strings, Z_values):
        snr_bin = int(np.round(snr / 2) * 2)  # 2dB bins
        key = f"{mod}_SNR{snr_bin:+d}"
        strata[key].append(idx)
    
    print(f"📈 Sampling from {len(strata)} strata")
    
    # Sample from each stratum
    subset = []
    strata_used = 0
    for key, stratum_indices in strata.items():
        n_samples = max(1, int(len(stratum_indices) * ratio))
        if n_samples <= len(stratum_indices):
            sampled = np.random.choice(stratum_indices, n_samples, replace=False)
            subset.extend(sampled)
            strata_used += 1
    
    subset = np.array(subset)
    
    print(f"✅ Stratified subset created: {len(subset):,} samples from {strata_used} strata (was {len(indices):,})")
    
    return subset

def print_distribution_stats(train_indices, val_indices, test_indices, metadata_lookup, split_name):
    """Print distribution statistics for splits - FIXED VERSION"""
    
    def get_split_stats(indices, lookup_dict, split_name):
        if len(indices) == 0:
            return {'total': 0, 'mod_counts': {}, 'snr_range': "N/A"}
            
        mod_counts = defaultdict(int)
        snr_values = []
        
        for idx in indices:
            # FIX: Use dictionary lookup instead of array indexing
            # idx is the HDF5 absolute index
            if idx in lookup_dict:
                mod, snr = lookup_dict[idx]
                mod_counts[mod] += 1
                snr_values.append(snr)
        
        return {
            'total': len(indices),
            'mod_counts': mod_counts,
            'snr_range': f"{min(snr_values):.0f} to {max(snr_values):.0f} dB" if snr_values else "N/A"
        }
    
    train_stats = get_split_stats(train_indices, metadata_lookup, "Train")
    val_stats = get_split_stats(val_indices, metadata_lookup, "Validation")
    test_stats = get_split_stats(test_indices, metadata_lookup, "Test")
    
    print(f"\n📊 {split_name} Distribution:")
    print(f"   Train: {train_stats['total']:,} samples, SNR: {train_stats['snr_range']}")
    print(f"   Validation: {val_stats['total']:,} samples, SNR: {val_stats['snr_range']}") 
    print(f"   Test: {test_stats['total']:,} samples, SNR: {test_stats['snr_range']}")
    
    # Show modulation distribution
    print(f"\n   Modulation Distribution:")
    all_mods = sorted(set(train_stats['mod_counts'].keys()) | set(val_stats['mod_counts'].keys()) | set(test_stats['mod_counts'].keys()))
    
    print(f"     {'Modulation':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"     {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    
    for mod in all_mods:
        train_count = train_stats['mod_counts'].get(mod, 0)
        val_count = val_stats['mod_counts'].get(mod, 0)
        test_count = test_stats['mod_counts'].get(mod, 0)
        total = train_count + val_count + test_count
        if total > 0:
            print(f"     {mod:<12} {train_count:>8} {val_count:>8} {test_count:>8}")

# ============================================
# MEMORY MONITORING (unchanged)
# ============================================

class MemoryMonitor:
    """Monitor GPU and system memory"""
    
    def __init__(self, vram_threshold=0.90, ram_threshold=0.85):
        self.vram_threshold = vram_threshold
        self.ram_threshold = ram_threshold
        self.has_cuda = torch.cuda.is_available()
        
    def get_memory_status(self):
        """Get current memory usage"""
        status = {
            'ram_used_gb': 0,
            'ram_total_gb': 0,
            'ram_percent': 0,
            'vram_used_gb': 0,
            'vram_total_gb': 0,
            'vram_percent': 0,
            'vram_available_gb': 0
        }
        
        # System RAM
        ram = psutil.virtual_memory()
        status['ram_used_gb'] = ram.used / (1024**3)
        status['ram_total_gb'] = ram.total / (1024**3)
        status['ram_percent'] = ram.percent / 100
        
        # GPU VRAM
        if self.has_cuda:
            vram_used = torch.cuda.memory_allocated() / (1024**3)
            vram_reserved = torch.cuda.memory_reserved() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            status['vram_used_gb'] = vram_reserved
            status['vram_total_gb'] = vram_total
            status['vram_percent'] = vram_reserved / vram_total
            status['vram_available_gb'] = vram_total - vram_reserved
        
        return status
    
    def is_memory_critical(self):
        """Check if memory usage is critical"""
        status = self.get_memory_status()
        
        ram_critical = status['ram_percent'] > self.ram_threshold
        vram_critical = status['vram_percent'] > self.vram_threshold if self.has_cuda else False
        
        return ram_critical or vram_critical, status
    
    def clear_memory(self):
        """Aggressive memory clearing"""
        gc.collect()
        if self.has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def print_status(self, prefix=""):
        """Print current memory status"""
        status = self.get_memory_status()
        print(f"{prefix}RAM: {status['ram_used_gb']:.1f}/{status['ram_total_gb']:.1f}GB "
              f"({status['ram_percent']*100:.1f}%) | "
              f"VRAM: {status['vram_used_gb']:.1f}/{status['vram_total_gb']:.1f}GB "
              f"({status['vram_percent']*100:.1f}%)")

# ============================================
# CONFIGURATION (unchanged)
# ============================================

class Config:
    """Tuning configuration with memory management"""
    
    # Paths
    DATA_DIR = Path("data")
    FILE_PATH = "C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\classes-fixed.json' 
    CHECKPOINT_DIR = Path("result/checkpoints")
    LOG_DIR = Path("result/logs")
    STUDY_DIR = Path("result/optuna_studies")
    
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
    
    # Fixed model parameters
    SEQ_LENGTH = 1024
    
    # DataLoader settings
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Mixed precision
    USE_AMP = True
    
    # Memory management
    CACHE_CLEAR_FREQUENCY = 30
    EMPTY_CACHE_BETWEEN_TRIALS = True
    VRAM_SAFETY_MARGIN = 2.0
    ENABLE_MEMORY_FALLBACK = True
    
    # Optuna settings
    N_EPOCHS_PER_TRIAL = 50
    N_TRIALS = 50
    PATIENCE_PER_TRIAL = 7
    PRUNE_STARTUP_TRIALS = 5
    PRUNE_WARMUP_STEPS = 5
    SAVE_BEST_MODELS = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [Rest of the code remains the same - Memory tiers, AdaptiveHyperparameters, StudyState, training functions, etc.]
# ============================================
# MEMORY-AWARE HYPERPARAMETER SPACE
# ============================================

CONFIGURATION_TIERS = [
    # Tier 0: Minimal
    {
        'd_model': [32, 48],
        'n_layers': [2, 3],
        'batch_size': [8, 16],
        'tier_name': 'ultra_minimal',
        'max_vram': 2.0
    },
    # Tier 1: Very Small
    {
        'd_model': [64],
        'n_layers': [3, 4],
        'batch_size': [16, 24],
        'tier_name': 'very_small',
        'max_vram': 3.0
    },
    # Tier 2: Small
    {
        'd_model': [96, 128],
        'n_layers': [4, 5],
        'batch_size': [32, 48],
        'tier_name': 'small',
        'max_vram': 5.0
    },
    # Tier 3: Medium
    {
        'd_model': [128, 160],
        'n_layers': [6, 7],
        'batch_size': [48, 64],
        'tier_name': 'medium',
        'max_vram': 7.0
    },
    # Tier 4: Large
    {
        'd_model': [192, 224, 256],
        'n_layers': [8, 9],
        'batch_size': [64, 96],
        'tier_name': 'large',
        'max_vram': 10.0
    }
]

def get_memory_constraints(d_model, n_layers, batch_size, use_amp=True):
    """Estimate VRAM usage"""
    seq_length = 1024
    num_classes = 19
    
    # Parameter estimation
    params_per_layer = 4 * d_model * d_model + 8 * d_model * d_model + 4 * d_model
    total_params = params_per_layer * n_layers + 2 * seq_length * d_model + d_model * num_classes
    
    # Memory in GB
    param_memory = (total_params * 4) / (1024**3)
    optimizer_memory = param_memory * 2
    
    # Activation memory (reduced with AMP)
    activation_memory = (batch_size * seq_length * d_model * n_layers * 12 * 4) / (1024**3)
    if use_amp:
        activation_memory *= 0.6
    
    # Total with overhead
    estimated_vram = (param_memory + optimizer_memory + activation_memory) * 1.3
    
    return estimated_vram

class AdaptiveHyperparameters:
    """Manages hyperparameter selection with memory fallback"""
    
    def __init__(self, memory_monitor, config):
        self.memory_monitor = memory_monitor
        self.config = config
        self.current_tier = len(CONFIGURATION_TIERS) - 1
        self.tier_history = {}
        
    def get_available_tier(self):
        """Get the highest tier that fits in available memory"""
        status = self.memory_monitor.get_memory_status()
        available_vram = status['vram_available_gb']
        
        # Account for safety margin
        safe_vram = available_vram - self.config.VRAM_SAFETY_MARGIN
        
        # Find appropriate tier
        for i in range(self.current_tier, -1, -1):
            tier = CONFIGURATION_TIERS[i]
            if tier['max_vram'] <= safe_vram:
                return i
        
        return 0  # Fallback to minimal
    
    def suggest_with_fallback(self, trial, forced_tier=None):
        """Suggest hyperparameters with automatic tier fallback"""
        
        # Determine tier to use
        if forced_tier is not None:
            tier_idx = forced_tier
        else:
            tier_idx = self.get_available_tier()
        
        tier = CONFIGURATION_TIERS[tier_idx]
        
        # Record tier usage
        self.tier_history[trial.number] = tier_idx
        
        # Architecture parameters from tier
        d_model = trial.suggest_categorical(f"d_model", tier['d_model'])
        n_layers = trial.suggest_int(f"n_layers", min(tier['n_layers']), max(tier['n_layers']))
        batch_size = trial.suggest_categorical(f"batch_size", tier['batch_size'])
        
        # Fixed or universal parameters
        embedding_type = trial.suggest_categorical('embedding_type', ['segment', 'conv1d'])
        segment_size = trial.suggest_categorical('segment_size', [16, 32]) if embedding_type == 'segment' else 16
        n_head = trial.suggest_categorical("n_head", [8, 16])
        
        # Ensure n_head divides d_model
        if d_model % n_head != 0:
            n_head = 8 if d_model % 8 == 0 else 4
        
        ffn_multiplier = trial.suggest_categorical("ffn_multiplier", [2, 3, 4])
        
        # Regularization
        drop_prob = trial.suggest_float("drop_prob", 0.05, 0.25)
        use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
        
        # Optimization
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ['AdamW', 'Adam'])
        
        # Calculate estimated VRAM
        estimated_vram = get_memory_constraints(d_model, n_layers, batch_size, self.config.USE_AMP)
        
        return {
            'embedding_type': embedding_type,
            'segment_size': segment_size,
            'd_model': d_model,
            'n_head': n_head,
            'n_layers': n_layers,
            'ffn_multiplier': ffn_multiplier,
            'drop_prob': drop_prob,
            'use_cls_token': use_cls_token,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer_name': optimizer_name,
            'label_smoothing': label_smoothing,
            'estimated_vram': estimated_vram,
            'tier': tier_idx,
            'tier_name': tier['tier_name']
        }

# ============================================
# STUDY STATE MANAGEMENT
# ============================================

class StudyState:
    """Manages persistent state across study interruptions"""
    
    def __init__(self, study_dir: Path):
        self.study_dir = study_dir
        self.state_file = study_dir / "study_state.pkl"
        self.best_models_dir = study_dir / "best_models"
        self.best_models_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, state_dict):
        """Save study state"""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state_dict, f)
    
    def load(self):
        """Load study state if exists"""
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_trial_checkpoint(self, trial_number, model_state, params, metrics):
        """Save best model from a trial"""
        checkpoint_path = self.best_models_dir / f"trial_{trial_number}_best.pth"
        torch.save({
            'trial_number': trial_number,
            'model_state_dict': model_state,
            'params': params,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        return checkpoint_path

# ============================================
# TRAINING WITH MEMORY RECOVERY
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                    epoch, config, scaler=None, trial=None):
    """Train for one epoch - fail fast on OOM"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f'Trial {trial.number} Epoch {epoch+1}' if trial else f'Epoch {epoch+1}'
    pbar = tqdm(train_loader, desc=f'{desc} [Train]', leave=False)
    
    use_amp = config.USE_AMP and scaler is not None
    
    for batch_idx, (images, labels, snrs) in enumerate(pbar):
        # Don't skip batches - fail fast on OOM
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device, epoch, config, trial=None):
    """Validate epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f'Trial {trial.number} Epoch {epoch+1}' if trial else f'Epoch {epoch+1}'
    pbar = tqdm(val_loader, desc=f'{desc} [Valid]', leave=False)
    
    use_amp = config.USE_AMP and torch.cuda.is_available()
    
    with torch.no_grad():
        for images, labels, snrs in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
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
            
            pbar.set_postfix({
                'loss': f'{running_loss/(len(pbar)):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total

# ============================================
# OBJECTIVE WITH MEMORY FALLBACK
# ============================================

g_datasets = None
g_config = None
g_study_state = None
g_memory_monitor = None
g_adaptive_hp = None

def objective_with_fallback(trial: optuna.trial.Trial) -> float:
    """Objective function with automatic memory fallback"""
    
    global g_datasets, g_config, g_study_state, g_memory_monitor, g_adaptive_hp
    
    # Start with smallest possible configuration
    tier_attempts = 0
    max_tier_attempts = 5
    current_tier = 0  # START WITH SMALLEST TIER!
    
    while tier_attempts < max_tier_attempts:
        try:
            # Clear memory before starting
            g_memory_monitor.clear_memory()
            torch.cuda.synchronize()
            
            # Get hyperparameters for current tier
            params = g_adaptive_hp.suggest_with_fallback(trial, forced_tier=current_tier)
            
            # Start with even smaller batch if previous attempts failed
            if tier_attempts > 0:
                params['batch_size'] = max(8, params['batch_size'] // (2 ** tier_attempts))
            
            print(f"\nTrial {trial.number} Attempt {tier_attempts+1}:")
            print(f"  Tier: {params['tier_name']}")
            print(f"  Config: d={params['d_model']}, L={params['n_layers']}, B={params['batch_size']}")
            print(f"  Est. VRAM: {params['estimated_vram']:.1f}GB")
            
            # Try to create model
            try:
                model = AMCTransformer(
                    in_channels=2,
                    seq_length=g_config.SEQ_LENGTH,
                    num_classes=len(g_config.TARGET_MODULATIONS),
                    d_model=params['d_model'],
                    n_head=params['n_head'],
                    n_layers=params['n_layers'],
                    ffn_hidden=params['d_model'] * params['ffn_multiplier'],
                    drop_prob=params['drop_prob'],
                    device=g_config.DEVICE,
                    use_cls_token=params['use_cls_token'],
                    embedding_type=params['embedding_type'],
                    segment_size=params['segment_size']
                ).to(g_config.DEVICE)
            except torch.cuda.OutOfMemoryError:
                print(f"  ❌ OOM creating model! Reducing tier...")
                tier_attempts += 1
                continue
            
            # Setup training
            criterion = nn.CrossEntropyLoss(label_smoothing=params['label_smoothing'])
            
            if params['optimizer_name'] == "AdamW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
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
            
            scaler = torch.amp.GradScaler('cuda') if g_config.USE_AMP else None
            early_stopping = EarlyStopping(patience=g_config.PATIENCE_PER_TRIAL)
            
            best_val_acc = 0.0
            best_model_state = None
            current_batch_size = params['batch_size']
            
            # Training loop
            epoch_oom_count = 0
            for epoch in range(g_config.N_EPOCHS_PER_TRIAL):
                # Create fresh dataloaders each epoch
                train_loader = DataLoader(
                    g_datasets['train'],
                    batch_size=current_batch_size,
                    shuffle=True,
                    num_workers=g_config.NUM_WORKERS,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn
                )
                
                valid_loader = DataLoader(
                    g_datasets['valid'],
                    batch_size=min(current_batch_size, 16),
                    shuffle=False,
                    num_workers=g_config.NUM_WORKERS,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn
                )
                
                try:
                    # Try training
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer,
                        g_config.DEVICE, epoch, g_config, scaler, trial
                    )
                    
                    val_loss, val_acc = validate_epoch(
                        model, valid_loader, criterion,
                        g_config.DEVICE, epoch, g_config, trial
                    )
                    
                    scheduler.step(val_loss)
                    
                    # Success! Reset OOM counter
                    epoch_oom_count = 0
                    
                    # Track best
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = model.state_dict().copy()
                    
                    # Report
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                    # Early stopping
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
                        
                except torch.cuda.OutOfMemoryError:
                    epoch_oom_count += 1
                    print(f"  ❌ OOM at epoch {epoch+1}!")
                    
                    # Clean up
                    del train_loader, valid_loader
                    g_memory_monitor.clear_memory()
                    torch.cuda.synchronize()
                    
                    # Reduce batch size
                    new_batch_size = max(1, current_batch_size // 2)
                    if new_batch_size < current_batch_size:
                        current_batch_size = new_batch_size
                        print(f"  📉 Reducing batch size to {current_batch_size}")
                    else:
                        # Can't reduce further, need smaller model
                        raise torch.cuda.OutOfMemoryError("Cannot reduce batch size further")
            
            # If we got here, training succeeded!
            if best_model_state is not None:
                g_study_state.save_trial_checkpoint(
                    trial.number,
                    best_model_state,
                    params,
                    {'best_val_acc': best_val_acc, 'final_batch_size': current_batch_size}
                )
                print(f"  ✅ Completed with {best_val_acc:.2f}%")
            
            # Cleanup
            del model, optimizer, criterion, scheduler
            g_memory_monitor.clear_memory()
            gc.collect()
            
            return best_val_acc
            
        except torch.cuda.OutOfMemoryError:
            print(f"  ❌ OOM with tier {current_tier}. Trying smaller configuration...")
            
            # Cleanup everything
            if 'model' in locals(): del model
            if 'optimizer' in locals(): del optimizer
            if 'train_loader' in locals(): del train_loader
            if 'valid_loader' in locals(): del valid_loader
            
            g_memory_monitor.clear_memory()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(2)
            
            tier_attempts += 1
            continue
            
        except optuna.exceptions.TrialPruned:
            raise
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            tier_attempts += 1
            continue
    
    print(f"  ❌ Failed after {max_tier_attempts} attempts. Pruning trial.")
    raise optuna.exceptions.TrialPruned()

# ============================================
# MAIN STUDY RUNNER - UPDATED
# ============================================

def run_optuna_study(args, config):
    """Run Optuna study with memory management - UPDATED VERSION"""
    
    global g_datasets, g_config, g_study_state, g_memory_monitor, g_adaptive_hp
    g_config = config
    
    # Initialize memory monitor
    g_memory_monitor = MemoryMonitor(vram_threshold=0.85, ram_threshold=0.80)
    
    # Setup directories
    config.STUDY_DIR.mkdir(parents=True, exist_ok=True)
    study_name = args.study_name or f"amc_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_dir = config.STUDY_DIR / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize study state
    g_study_state = StudyState(study_dir)
    
    # Initialize adaptive hyperparameters
    g_adaptive_hp = AdaptiveHyperparameters(g_memory_monitor, config)
    
    # Database for persistence
    db_path = study_dir / f"{study_name}.db"
    storage = RDBStorage(f"sqlite:///{db_path}")
    
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER TUNING WITH STRATIFIED SAMPLING - FIXED")
    print("="*70)
    print(f"Study: {study_name}")
    print(f"Database: {db_path}")
    print(f"Target trials: {args.n_trials}")
    print(f"Epochs per trial: {config.N_EPOCHS_PER_TRIAL}")
    
    # Memory status
    print("\n📊 System Resources:")
    g_memory_monitor.print_status("  ")
    print(f"  VRAM Safety Margin: {config.VRAM_SAFETY_MARGIN}GB")
    print(f"  Memory Fallback: {'Enabled' if config.ENABLE_MEMORY_FALLBACK else 'Disabled'}")
    
    # Load data with stratified sampling - USING FIXED VERSION
    print("\n📂 Loading datasets with stratified sampling...")
    
    # Perform stratified split using the FIXED function
    train_indices, valid_indices, test_indices, label_map = stratified_split_data(
        str(config.FILE_PATH),
        str(config.JSON_PATH),
        config.TARGET_MODULATIONS,
        config.TRAIN_SIZE,
        config.VALID_SIZE,
        config.TEST_SIZE,
        config.SPLIT_SEED
    )
    
    # For fast tuning, create stratified subsets
    if args.fast_tuning:
        print("\n⚡ Fast tuning mode: Using 10% stratified subsets")
        
        # Get the full labels and SNRs for sampling
        all_indices, Y_strings, Z_values = get_dataset_labels_and_snrs(
            str(config.FILE_PATH),
            str(config.JSON_PATH),
            config.TARGET_MODULATIONS
        )
        
        train_subset = stratified_sampling(
            train_indices, Y_strings, Z_values, config.TARGET_MODULATIONS, 
            ratio=0.1, seed=42
        )
        valid_subset = stratified_sampling(
            valid_indices, Y_strings, Z_values, config.TARGET_MODULATIONS,
            ratio=0.1, seed=43
        )
        
        train_indices = train_subset
        valid_indices = valid_subset
        
        print(f"   Train subset: {len(train_indices):,} samples")
        print(f"   Valid subset: {len(valid_indices):,} samples")
    
    # Create datasets
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
    
    test_dataset = SingleStreamImageDataset(
        file_path=str(config.FILE_PATH),
        json_path=str(config.JSON_PATH),
        target_modulations=config.TARGET_MODULATIONS,
        mode='test',
        indices=test_indices,
        label_map=label_map,
        normalization_stats=norm_stats
    )
    
    g_datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }
    
    print(f"\n✅ Datasets loaded:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Valid: {len(valid_dataset):,} samples")
    print(f"   Test: {len(test_dataset):,} samples")
    
    # Memory after loading
    g_memory_monitor.print_status("  After loading - ")
    
    # Create or load study
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        n_previous_trials = len(study.trials)
        print(f"\n✅ Resuming study with {n_previous_trials} completed trials")
    except:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=config.PRUNE_STARTUP_TRIALS,
                n_warmup_steps=config.PRUNE_WARMUP_STEPS
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        n_previous_trials = 0
        print("\n✅ Created new study")
    
    # Run optimization
    remaining_trials = max(0, args.n_trials - n_previous_trials)
    if remaining_trials == 0:
        print(f"\n✅ Study already completed {args.n_trials} trials")
    else:
        print(f"\n🚀 Running {remaining_trials} more trials...")
        
        def callback(study, trial):
            """Callback after each trial"""
            g_study_state.save({
                'best_value': study.best_value if study.best_trial else None,
                'best_params': study.best_params if study.best_trial else None,
                'best_trial_number': study.best_trial.number if study.best_trial else None,
                'n_trials': len(study.trials),
                'tier_history': g_adaptive_hp.tier_history,
                'timestamp': datetime.now().isoformat()
            })
            
            g_memory_monitor.clear_memory()
            gc.collect()
        
        start_time = time.time()
        
        try:
            study.optimize(
                objective_with_fallback,
                n_trials=remaining_trials,
                callbacks=[callback],
                gc_after_trial=True,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️ Study interrupted! Progress saved.")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Session complete! Time: {elapsed/3600:.2f} hours")
    
    # Final evaluation and reporting
    if study.best_trial:
        print("\n" + "="*70)
        print("BEST TRIAL RESULTS")
        print("="*70)
        print(f"Trial #{study.best_trial.number}")
        print(f"Accuracy: {study.best_value:.4f}")
        print("\nParameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    
    # Cleanup
    for dataset in g_datasets.values():
        dataset.close()
    
    print("\n✅ Study complete!")

# ============================================
# MAIN ENTRY POINT
# ============================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AMC Transformer Hyperparameter Tuning with Memory Management',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--study_name', type=str, help='Study name (for resuming)')
    parser.add_argument('--n_trials', type=int, default=50, help='Total number of trials')
    parser.add_argument('--epochs_per_trial', type=int, default=50, help='Epochs per trial')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--no_fallback', action='store_true', help='Disable memory fallback')
    parser.add_argument('--fast_tuning', action='store_true', help='Use 10% subsets for faster tuning')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    try:
        args = parse_args()
        
        # Setup environment
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Configure
        config = Config()
        config.N_TRIALS = args.n_trials
        config.N_EPOCHS_PER_TRIAL = args.epochs_per_trial
        config.NUM_WORKERS = args.num_workers
        config.USE_AMP = not args.no_amp
        config.ENABLE_MEMORY_FALLBACK = not args.no_fallback
        
        print("="*70)
        print("AMC TRANSFORMER HYPERPARAMETER TUNING WITH STRATIFIED SAMPLING - FIXED")
        print("="*70)
        print(f"Device: {config.DEVICE}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Fast tuning: {args.fast_tuning}")
        
        # Run study
        run_optuna_study(args, config)
        
        print("\n✅ Complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()