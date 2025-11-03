"""
Main Training Script for AMC Transformer
Enhanced with improved error handling, validation, and flexibility
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", message="h5py is running against HDF5")

# Add parent directory to path to import custom modules
sys.path.append(str(Path(__file__).parent.parent))

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
# CONFIGURATION
# ============================================

class Config:
    """Training configuration with validation"""
    
    # Paths
    DATA_DIR = Path("data")
    FILE_PATH = "C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\classes-fixed.json' 
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
    
    # --- Model architecture (1D Transformer) ---
    SEQ_LENGTH = 1024               # Length of the raw sequence
    EMBEDDING_TYPE = 'segment'      # 'segment' (1D ViT) or 'conv1d' (Pure Transformer)
    SEGMENT_SIZE = 16          # 1D Patch size (1024 / 64 = 16 tokens)
    USE_CLS_TOKEN = True
    D_MODEL = 128
    N_HEAD = 8
    N_LAYERS = 6
    FFN_HIDDEN = 1024            # D_MODEL * 4 (Note: 128*4 = 512)
    DROP_PROB = 0.2
    
    # Training hyperparameters
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_MAX_NORM = 1.0
    
    # DataLoader settings
    NUM_WORKERS = 6
    PREFETCH_FACTOR = 3
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Early stopping & checkpointing
    PATIENCE = 10
    SAVE_FREQ = 5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        errors = []
        
        # Check file paths exist
        if not Path(cls.FILE_PATH).exists():
            errors.append(f"HDF5 file not found: {cls.FILE_PATH}")
        if not Path(cls.JSON_PATH).exists():
            errors.append(f"JSON file not found: {cls.JSON_PATH}")
        
        # Check data splits sum to 1.0
        split_sum = cls.TRAIN_SIZE + cls.VALID_SIZE + cls.TEST_SIZE
        if not np.isclose(split_sum, 1.0):
            errors.append(f"Data splits must sum to 1.0, got {split_sum}")
        
        # Check model architecture compatibility
        if cls.D_MODEL % cls.N_HEAD != 0:
            errors.append(f"D_MODEL ({cls.D_MODEL}) must be divisible by N_HEAD ({cls.N_HEAD})")
        
        # Check positive values
        if cls.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive, got {cls.BATCH_SIZE}")
        if cls.NUM_EPOCHS <= 0:
            errors.append(f"NUM_EPOCHS must be positive, got {cls.NUM_EPOCHS}")
        if cls.LEARNING_RATE <= 0:
            errors.append(f"LEARNING_RATE must be positive, got {cls.LEARNING_RATE}")
        
        # Check NUM_WORKERS
        if cls.NUM_WORKERS < 0:
            errors.append(f"NUM_WORKERS cannot be negative, got {cls.NUM_WORKERS}")
        if cls.NUM_WORKERS == 0:
            warnings.warn(
                "NUM_WORKERS is 0. The dataset requires num_workers > 0 for proper HDF5 handling. "
                "Set NUM_WORKERS to at least 1 for production use.",
                UserWarning
            )
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        return True
    
    @classmethod
    def from_args(cls, args):
        """Update config from command line arguments with validation"""
        for key, value in vars(args).items():
            if value is not None and hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
        
        # Validate after updating
        cls.validate()
        return cls


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train AMC Transformer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--file_path', type=str, help='Path to HDF5 data file')
    parser.add_argument('--json_path', type=str, help='Path to classes JSON file')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers (use >0 for production)')
    parser.add_argument('--grad_clip_max_norm', type=float, help='Max norm for gradient clipping')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, help='Number of transformer layers')
    parser.add_argument('--drop_prob', type=float, help='Dropout probability')
    
    # Other
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, help='Experiment name for logging')
    parser.add_argument('--no_validate_config', action='store_true', help='Skip configuration validation')
    
    return parser.parse_args()


def get_config_dict(config):
    """Convert Config class to serializable dictionary for 1D Transformer"""
    return {
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'WEIGHT_DECAY': config.WEIGHT_DECAY,
        'LABEL_SMOOTHING': config.LABEL_SMOOTHING,
        'GRAD_CLIP_MAX_NORM': config.GRAD_CLIP_MAX_NORM,
        'NUM_WORKERS': config.NUM_WORKERS,
        
        # --- Parameter 1D Model (Menggantikan PATCH_SIZE) ---
        'SEQ_LENGTH': config.SEQ_LENGTH,
        'EMBEDDING_TYPE': config.EMBEDDING_TYPE,
        'SEGMENT_SIZE': config.SEGMENT_SIZE,
        'USE_CLS_TOKEN': config.USE_CLS_TOKEN,
        # ------------------------------------------------
        
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
# TRAINING FUNCTIONS
# ============================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch with gradient clipping"""
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
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.GRAD_CLIP_MAX_NORM
        )
        
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
    """Main training function with enhanced error handling"""
    
    try:
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
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        print("="*70)
        print("AMC TRANSFORMER TRAINING")
        print("="*70)
        print(f"Experiment: {experiment_name}")
        print(f"Device: {config.DEVICE}")
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Num workers: {config.NUM_WORKERS}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print("="*70)
        
        # Save configuration
        config_save_path = exp_checkpoint_dir / "config.json"
        with open(config_save_path, 'w') as f:
            json.dump(get_config_dict(config), f, indent=4)
        print(f"Configuration saved to: {config_save_path}")
        
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
        print(f"Number of classes: {num_classes}")
        
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
        print(f"Normalization stats: I μ={norm_stats['i_mean']:.4f}, σ={norm_stats['i_std']:.4f} | "
              f"Q μ={norm_stats['q_mean']:.4f}, σ={norm_stats['q_std']:.4f}")
        
        # Check for potential issues with normalization
        if norm_stats['i_std'] < 1e-6 or norm_stats['q_std'] < 1e-6:
            warnings.warn(
                f"Very low standard deviation detected (I: {norm_stats['i_std']:.2e}, Q: {norm_stats['q_std']:.2e}). "
                "Data might be constant or normalization may cause numerical issues.",
                UserWarning
            )
        
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
        
        # Adjust persistent_workers based on num_workers
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
        
        # ========================================
        # MODEL SETUP
        # ========================================
        
        print("\n🤖 Initializing model...")
        
        model_params = {
            'in_channels': 2,  # CRITICAL: 2 for I and Q
            'seq_length': config.SEQ_LENGTH,
            'num_classes': num_classes,
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
        print(f"   Model size: ~{num_params * 4 / 1024**2:.2f} MB (fp32)")
        
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
            try:
                checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
                start_epoch = checkpoint['epoch'] + 1
                history = checkpoint.get('history', history)
                print(f"   ✅ Resuming from epoch {start_epoch}")
            except Exception as e:
                print(f"   ⚠️  Failed to load checkpoint: {e}")
                print("   Starting training from scratch...")
        
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
                model, train_loader, criterion, optimizer, config.DEVICE, epoch, config
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
            if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.NUM_EPOCHS:
                checkpoint_path = exp_checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
                
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                    history,
                    get_config_dict(config)
                )
                print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Early stopping check
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print("\n⏹️  Early stopping triggered!")
                final_path = exp_checkpoint_dir / "model_best.pth"
                
                save_checkpoint(
                    final_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    early_stopping.best_score,
                    history,
                    get_config_dict(config)
                )
                print(f"   💾 Best model saved: {final_path}")
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
        
        # Save final model if not early stopped
        if not early_stopping.early_stop:
            final_path = exp_checkpoint_dir / "model_final.pth"
            save_checkpoint(
                final_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                get_config_dict(config)
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
        best_model_path = exp_checkpoint_dir / "model_best.pth"
        if not best_model_path.exists():
            best_model_path = exp_checkpoint_dir / "model_final.pth"
        
        print(f"Loading model from: {best_model_path}")
        best_checkpoint = load_checkpoint(best_model_path, model)
        
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
            pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
            worker_init_fn=worker_init_fn if config.NUM_WORKERS > 0 else None,
            persistent_workers=use_persistent,
            prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
        )
        
        print(f"Test set: {len(test_loader):,} batches ({len(test_dataset):,} samples)")
        
        # Evaluate and generate confusion matrices
        eval_results = evaluate_model_with_confusion(
            model=model,
            dataloader=test_loader,
            device=config.DEVICE,
            class_names=config.TARGET_MODULATIONS,
            save_dir=exp_checkpoint_dir / "evaluation",
            prefix='test'
        )
        
        print(f"\n✅ Test Accuracy: {eval_results.get('accuracy', 'N/A')}")
        
        # Cleanup
        print("\nCleaning up...")
        train_dataset.close()
        valid_dataset.close()
        test_dataset.close()
        
        print("\n✅ All done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print("Attempting to save checkpoint...")
        try:
            interrupt_path = exp_checkpoint_dir / "checkpoint_interrupted.pth"
            save_checkpoint(
                interrupt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                get_config_dict(config)
            )
            print(f"Checkpoint saved to: {interrupt_path}")
        except:
            print("Failed to save checkpoint.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()