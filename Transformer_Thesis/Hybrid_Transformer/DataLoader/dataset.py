"""
PyTorch Dataset and DataLoader worker definitions for HDF5 files.

This module defines:
1.  SingleStreamSignalDataset: A Dataset class for loading I/Q signal data.
2.  worker_init_fn: A function for torch.utils.data.DataLoader to ensure
    multiprocessing-safe HDF5 file access.

Output Format: (2, 1024) - Compatible with HybridTransformer architecture
    - Channel 0: I (In-phase) component
    - Channel 1: Q (Quadrature) component
"""

import gc
import json
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


def worker_init_fn(worker_id: int):
    """
    PyTorch DataLoader worker initialization function.

    This function is called by the DataLoader in each worker process.
    It gets the dataset instance for that worker and calls its
    open_hdf5() method, ensuring that each worker has its own
    file handle to the HDF5 file.

    Args:
        worker_id: The ID of the worker process (unused, but required by API).
    """
    info = get_worker_info()
    if info is not None:
        dataset = info.dataset
        dataset.open_hdf5()


class SingleStreamSignalDataset(Dataset):
    """
    A multiprocessing-safe HDF5 Dataset for I/Q signal classification.

    Output Format: (2, 1024) for HybridTransformer
        - Shape: (Channels, Sequence_Length)
        - Channel 0: Normalized I component
        - Channel 1: Normalized Q component

    The HDF5 file is NOT opened in __init__. Instead, it is opened by
    each worker process via `worker_init_fn` to prevent file handle
    conflicts when using num_workers > 0.

    Args:
        file_path: Path to the HDF5 file containing RadioML data.
        json_path: Path to the JSON file containing class mappings.
        target_modulations: List of modulation strings to include.
        mode: Dataset mode ('train', 'valid', or 'test').
        indices: Array of integer indices for this dataset split.
        label_map: Dictionary mapping modulation strings to integer labels.
        normalization_stats: Optional dict with 'i_mean', 'i_std', 'q_mean', 'q_std'.
                             If None and mode='train', stats are calculated.
                             Required if mode is 'valid' or 'test'.
        seed: Random seed for reproducibility.

    Example:
        >>> dataset = SingleStreamSignalDataset(
        ...     file_path='radioml2018.hdf5',
        ...     json_path='classes.json',
        ...     target_modulations=['QPSK', '16QAM', '64QAM'],
        ...     mode='train',
        ...     indices=train_indices,
        ...     label_map={'QPSK': 0, '16QAM': 1, '64QAM': 2}
        ... )
        >>> x, y, snr = dataset[0]
        >>> print(x.shape)  # torch.Size([2, 1024])
    """

    def __init__(
        self,
        file_path: str,
        json_path: str,
        target_modulations: List[str],
        mode: str,
        indices: np.ndarray,
        label_map: Dict[str, int],
        normalization_stats: Optional[Dict[str, float]] = None,
        seed: int = 49,
    ):
        super(SingleStreamSignalDataset, self).__init__()

        # Store paths and parameters (DO NOT open file here)
        self.file_path = file_path
        self.json_path = json_path
        self.target_modulations = target_modulations
        self.mode = mode
        self.indices = np.array(indices, dtype=int)
        self.label_map = label_map
        self.seed = seed
        self.norm_stats = normalization_stats

        # HDF5 handles (initialized by worker_init_fn)
        self.hdf5_file = None
        self.X_h5 = None

        # Read metadata ONCE in main process (safe: read-only, closes handle)
        # Use contiguous reads for speed (fancy indexing on HDF5 is very slow)
        with h5py.File(self.file_path, "r") as f:
            self.seq_length = f["X"].shape[1]
            total_samples = f["Y"].shape[0]

            print(f"   Loading labels ({total_samples:,} total samples)...")

            # Read Y in contiguous chunks with progress
            chunk_size = 500000
            Y_int_list = []
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                Y_chunk = f["Y"][start:end]
                Y_int_list.append(np.argmax(Y_chunk, axis=1))
                print(f"   ... loaded {end:,}/{total_samples:,}", end="\r")
                del Y_chunk

            self.Y_int = np.concatenate(Y_int_list)
            del Y_int_list
            print(f"   ✓ Y labels loaded ({total_samples:,} samples)      ")

            # Z is small, read directly
            self.Z = f["Z"][:, 0]
            print(f"   ✓ SNR values loaded")

            gc.collect()

        with open(self.json_path, "r") as f:
            self.modulation_classes = json.load(f)

        self.Y_strings = np.array([self.modulation_classes[i] for i in self.Y_int])

        # Calculate or validate normalization stats
        if mode == "train":
            if self.norm_stats is None:
                print(f"📊 Calculating normalization stats for {mode} mode...")
                self.norm_stats = self._calculate_normalization_stats()
                print(f"✅ Stats calculated:")
                print(
                    f"   I: mean={self.norm_stats['i_mean']:.6f}, std={self.norm_stats['i_std']:.6f}"
                )
                print(
                    f"   Q: mean={self.norm_stats['q_mean']:.6f}, std={self.norm_stats['q_std']:.6f}"
                )
        else:
            if self.norm_stats is None:
                raise ValueError(f"normalization_stats are required for '{mode}' mode")

        print(
            f"✅ {mode.capitalize()} dataset initialized: {len(self.indices):,} samples"
        )
        print(f"   Output shape: (2, {self.seq_length}) - [I/Q channels, samples]")

    def _calculate_normalization_stats(self) -> Dict[str, float]:
        """
        Calculate normalization statistics using chunked processing.

        Uses a random subset of training data for efficiency.
        Statistics are calculated separately for I and Q channels.

        Returns:
            Dict with 'i_mean', 'i_std', 'q_mean', 'q_std'
        """
        with h5py.File(self.file_path, "r") as temp_file:
            X_temp = temp_file["X"]

            # Use subset for efficiency
            num_samples = min(5000, len(self.indices))
            np.random.seed(self.seed)
            sample_indices = np.random.choice(self.indices, num_samples, replace=False)
            sorted_indices = np.sort(sample_indices)

            chunk_size = min(500, num_samples)
            i_vals, q_vals = [], []

            num_chunks = int(np.ceil(len(sorted_indices) / chunk_size))
            print(f"   Processing {num_samples} samples in {num_chunks} chunks...")

            for i in range(0, len(sorted_indices), chunk_size):
                chunk_indices = sorted_indices[i : i + chunk_size]
                chunk_data = X_temp[chunk_indices, ...]
                chunk_tensor = torch.from_numpy(chunk_data).float()

                # RadioML format: (samples, seq_length, 2) where dim 2 is [I, Q]
                i_vals.append(chunk_tensor[:, :, 0].flatten())
                q_vals.append(chunk_tensor[:, :, 1].flatten())

                del chunk_data, chunk_tensor

            i_all = torch.cat(i_vals)
            q_all = torch.cat(q_vals)

            stats = {
                "i_mean": i_all.mean().item(),
                "i_std": max(i_all.std().item(), 1e-8),
                "q_mean": q_all.mean().item(),
                "q_std": max(q_all.std().item(), 1e-8),
            }

            del i_vals, q_vals, i_all, q_all
            gc.collect()

            return stats

    def open_hdf5(self):
        """
        Open the HDF5 file handle.

        CRITICAL: Must be called AFTER process fork, inside worker process.
        This is handled automatically by worker_init_fn.
        """
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.file_path, "r", swmr=False, libver="latest")
            self.X_h5 = self.hdf5_file["X"]

    def get_normalization_stats(self) -> Dict[str, float]:
        """Returns a copy of the normalization statistics."""
        return self.norm_stats.copy()

    def __len__(self) -> int:
        """Returns the number of samples in this dataset split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get a single data sample.

        Args:
            idx: Index into the dataset (0 to len-1)

        Returns:
            Tuple of:
                - x: I/Q signal tensor, shape (2, 1024)
                - y: Integer class label
                - snr: Signal-to-noise ratio in dB
        """
        # Map to true dataset index
        true_index = int(self.indices[idx])

        # Windows-safe: open file, read, close immediately to avoid locking
        # This is slower but prevents file handle conflicts
        with h5py.File(self.file_path, "r") as f:
            x_raw = f["X"][true_index]

        # Get label and SNR (already in memory)
        y_string = self.Y_strings[true_index]
        z_snr = float(self.Z[true_index])
        y_label = self.label_map[y_string]

        # Convert to tensor: (1024, 2)
        x_tensor = torch.from_numpy(x_raw).float()

        # Extract and normalize I/Q channels
        i_normalized = (x_tensor[:, 0] - self.norm_stats["i_mean"]) / self.norm_stats["i_std"]
        q_normalized = (x_tensor[:, 1] - self.norm_stats["q_mean"]) / self.norm_stats["q_std"]

        # Stack to (2, 1024) format for HybridTransformer
        x_output = torch.stack([i_normalized, q_normalized], dim=0)

        return x_output, y_label, z_snr

    def close(self):
        """Close the HDF5 file handle."""
        if hasattr(self, "hdf5_file") and self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except Exception:
                pass
            finally:
                self.hdf5_file = None
                self.X_h5 = None

    def reset_file_handle(self):
        """Reset HDF5 file handle for reuse across trials."""
        self.close()
        # Will be reopened on next __getitem__ call

    def __del__(self):
        """Destructor to ensure file is closed."""
        self.close()