"""
Evaluation Script for AMC Transformer
Generates confusion matrices and comprehensive performance metrics
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataloader.dataset import SingleStreamImageDataset, worker_init_fn
from dataloader.utils import split_data
from models.amc_transformer import AMCTransformer
from training.utils import load_checkpoint, evaluate_model_with_confusion


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate AMC Transformer')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='result/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['test', 'valid', 'train'],
                        help='Which dataset to evaluate on')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    
    args = parse_args()
    
    print("="*70)
    print("AMC TRANSFORMER EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # Load checkpoint
    print("\n📥 Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("✅ Config loaded from checkpoint")
    else:
        print("⚠️  No config in checkpoint, using defaults")
        # Default config
        config = {
            'FILE_PATH': 'data/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
            'JSON_PATH': 'data/2018.01/classes-fixed.json',
            'TARGET_MODULATIONS': [
                'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
                '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
                'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
            ],
            'TRAIN_SIZE': 0.7,
            'VALID_SIZE': 0.15,
            'TEST_SIZE': 0.15,
            'SPLIT_SEED': 42,
            'NORM_SEED': 49,
            'PATCH_SIZE': 4,
            'D_MODEL': 128,
            'N_HEAD': 8,
            'N_LAYERS': 6,
            'FFN_HIDDEN': 512,
            'DROP_PROB': 0.1,
            'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
    
    # Setup
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    device = config.get('DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print(f"Device: {device}")
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    print("\n📂 Loading data...")
    
    # Split data
    train_indices, valid_indices, test_indices, label_map = split_data(
        config['FILE_PATH'],
        config['JSON_PATH'],
        config['TARGET_MODULATIONS'],
        config.get('TRAIN_SIZE', 0.7),
        config.get('VALID_SIZE', 0.15),
        config.get('TEST_SIZE', 0.15),
        config.get('SPLIT_SEED', 42)
    )
    
    # Select indices based on dataset
    if args.dataset == 'test':
        indices = test_indices
    elif args.dataset == 'valid':
        indices = valid_indices
    else:
        indices = train_indices
    
    print(f"Using {args.dataset} set: {len(indices):,} samples")
    
    # Create dataset
    # First create train dataset to get normalization stats
    train_dataset = SingleStreamImageDataset(
        file_path=config['FILE_PATH'],
        json_path=config['JSON_PATH'],
        target_modulations=config['TARGET_MODULATIONS'],
        mode='train',
        indices=train_indices,
        label_map=label_map,
        seed=config.get('NORM_SEED', 49)
    )
    
    norm_stats = train_dataset.get_normalization_stats()
    train_dataset.close()
    
    # Create evaluation dataset
    eval_dataset = SingleStreamImageDataset(
        file_path=config['FILE_PATH'],
        json_path=config['JSON_PATH'],
        target_modulations=config['TARGET_MODULATIONS'],
        mode=args.dataset,
        indices=indices,
        label_map=label_map,
        normalization_stats=norm_stats
    )
    
    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"✅ DataLoader created: {len(eval_loader)} batches")
    
    # ========================================
    # MODEL SETUP
    # ========================================
    
    print("\n🤖 Loading model...")
    
    num_classes = len(config['TARGET_MODULATIONS'])
    
    model_params = {
        'in_channels': 1,
        'img_size_h': 32,
        'img_size_w': 64,
        'patch_size': config.get('PATCH_SIZE', 4),
        'num_classes': num_classes,
        'd_model': config.get('D_MODEL', 128),
        'n_head': config.get('N_HEAD', 8),
        'n_layers': config.get('N_LAYERS', 6),
        'ffn_hidden': config.get('FFN_HIDDEN', 512),
        'drop_prob': config.get('DROP_PROB', 0.1),
        'device': device
    }
    
    model = AMCTransformer(**model_params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'unknown')}")
    
    # ========================================
    # EVALUATION
    # ========================================
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_model_with_confusion(
        model=model,
        dataloader=eval_loader,
        device=device,
        class_names=config['TARGET_MODULATIONS'],
        save_dir=output_dir,
        prefix=args.dataset
    )
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    
    print("\n💾 Saving detailed results...")
    
    import pickle
    results_file = output_dir / f'{args.dataset}_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✅ Results saved to {results_file}")
    
    # Cleanup
    eval_dataset.close()
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    main()