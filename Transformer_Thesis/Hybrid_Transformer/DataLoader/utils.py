"""
Utility functions for data loading and preprocessing
"""

import json
from typing import Tuple, List, Dict
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


def load_dataset_metadata(file_path: str, json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Load only metadata and labels from HDF5 file (memory efficient).
    
    Args:
        file_path: Path to HDF5 file
        json_path: Path to classes JSON file
        
    Returns:
        (Y_strings, Z_data, available_modulations, total_samples)
    """
    print("📂 Loading dataset metadata (memory efficient)...")
    
    with h5py.File(file_path, 'r') as hdf5_file:
        # Get dataset shape without loading data
        total_samples = hdf5_file['X'].shape[0]
        signal_length = hdf5_file['X'].shape[1]
        num_channels = hdf5_file['X'].shape[2]
        
        print(f"📊 Dataset shape: ({total_samples:,} × {signal_length} × {num_channels})")
        
        # Load only labels (much smaller than signal data)
        Y_int = np.argmax(hdf5_file['Y'][:], axis=1)
        Z_data = hdf5_file['Z'][:, 0]
        
    # Load modulation classes
    with open(json_path, 'r') as f:
        modulation_classes = json.load(f)
    
    # Convert integer labels to string labels
    Y_strings = np.array([modulation_classes[i] for i in Y_int])
    
    # Get available modulations
    available_modulations = list(np.unique(Y_strings))
    
    print(f"✅ Metadata loaded: {total_samples:,} samples")
    print(f"📡 Available modulations: {len(available_modulations)}")
    print(f"📊 SNR range: {np.min(Z_data):.1f} to {np.max(Z_data):.1f} dB")
    
    # Memory usage estimate
    data_size_gb = (total_samples * signal_length * num_channels * 4) / (1024**3)
    print(f"💾 Full dataset size: ~{data_size_gb:.2f} GB")
    
    return Y_strings, Z_data, available_modulations, total_samples


def split_data(
    file_path: str,
    json_path: str,
    target_mods: List[str],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Split data into train/valid/test sets with stratification by modulation and SNR.
    
    Args:
        file_path: Path to HDF5 file
        json_path: Path to classes JSON file
        target_mods: List of target modulation types
        train_ratio: Ratio for training set
        valid_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
        
    Returns:
        (train_indices, valid_indices, test_indices, label_map)
    """
    print("📂 Splitting data...")
    
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    # Create label map
    label_map = {mod: i for i, mod in enumerate(target_mods)}
    
    # Load labels and SNR
    with h5py.File(file_path, 'r') as f:
        Y_int = np.argmax(f['Y'][:], axis=1)
        Z = f['Z'][:, 0]
    
    with open(json_path, 'r') as file:
        all_mod_classes = json.load(file)
    
    Y_strings = np.array([all_mod_classes[i] for i in Y_int])
    
    train_indices, valid_indices, test_indices = [], [], []
    
    # Stratify by modulation AND SNR
    for mod in target_mods:
        for snr in np.unique(Z):
            # Get all indices for this (modulation, SNR) pair
            idx = np.where((Y_strings == mod) & (Z == snr))[0]
            
            if len(idx) == 0:
                continue
            
            # First split: separate test set
            idx_train_val, idx_test = train_test_split(
                idx, test_size=test_ratio, random_state=seed, shuffle=True
            )
            
            # Second split: separate train and validation
            relative_valid_ratio = valid_ratio / (train_ratio + valid_ratio)
            
            if len(idx_train_val) > 1:
                idx_train, idx_valid = train_test_split(
                    idx_train_val,
                    test_size=relative_valid_ratio,
                    random_state=seed,
                    shuffle=True
                )
            else:
                # Only 1 sample left, put in training
                idx_train, idx_valid = idx_train_val, []
            
            train_indices.extend(idx_train)
            valid_indices.extend(idx_valid)
            test_indices.extend(idx_test)
    
    # Shuffle final results
    np.random.seed(seed)
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)
    
    print(f"✅ Data split: {len(train_indices):,} train, "
          f"{len(valid_indices):,} valid, {len(test_indices):,} test")
    
    return (
        np.array(train_indices),
        np.array(valid_indices),
        np.array(test_indices),
        label_map
    )