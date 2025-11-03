"""
PyTorch Dataset and DataLoader worker definitions for HDF5 files.

This module defines:
1.  SingleStreamImageDataset: A Dataset class for loading I/Q signal data.
2.  worker_init_fn: A function for torch.utils.data.DataLoader to ensure
    multiprocessing-safe HDF5 file access.
"""

import json
import gc
from typing import Tuple, List, Dict, Optional

import numpy as np
import h5py
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
    # Get worker info
    info = get_worker_info()
    if info is not None:
        # Get the dataset instance for this specific worker
        dataset = info.dataset
        # Call the dataset's method to open the HDF5 file
        dataset.open_hdf5()


class SingleStreamImageDataset(Dataset):
    """
    A truly multiprocessing-safe HDF5 Dataset implementation.

    The HDF5 file is NOT opened in the __init__ method. Instead,
    it is opened by each worker process via the `worker_init_fn`.
    This prevents file handle conflicts when using num_workers > 0.
    """

    def __init__(self,
                 file_path: str,
                 json_path: str,
                 target_modulations: List[str],
                 mode: str,
                 indices: np.ndarray,
                 label_map: Dict[str, int],
                 normalization_stats: Optional[Dict[str, float]] = None,
                 seed: int = 49):
        """
        Initializes the dataset.

        Args:
            file_path: Path to the HDF5 file.
            json_path: Path to the JSON file containing class mappings.
            target_modulations: List of modulation strings to be used.
            mode: Dataset mode ('train', 'valid', or 'test').
            indices: Array of integer indices for this dataset split.
            label_map: Dictionary mapping modulation strings to integer labels.
            normalization_stats: Optional dict with 'i_mean', 'i_std', 'q_mean', 'q_std'.
                                 If None and mode='train', they are calculated.
                                 Required if mode is 'valid' or 'test'.
            seed: Random seed for reproducibility (e.g., for stat calculation).
        """
        super(SingleStreamImageDataset, self).__init__()

        # Store paths and parameters, DO NOT open the file here
        self.file_path = file_path
        self.json_path = json_path
        self.target_modulations = target_modulations
        self.mode = mode
        self.indices = np.array(indices, dtype=int)
        self.label_map = label_map
        self.seed = seed
        self.norm_stats = normalization_stats

        # These will be initialized by open_hdf5() in each worker
        self.hdf5_file = None
        self.X_h5 = None

        # Read metadata (labels/SNR) ONCE in the main process
        # This is safe as it's read-only and closes the file handle
        with h5py.File(self.file_path, 'r') as f:
            self.Y_int = np.argmax(f['Y'][:], axis=1)
            self.Z = f['Z'][:, 0]

        with open(self.json_path, 'r') as f:
            self.modulation_classes = json.load(f)

        self.Y_strings = np.array([self.modulation_classes[i] for i in self.Y_int])
        
        # Define the target "image" dimensions
        self.H, self.W = 32, 64

        # Calculate normalization stats if they are not provided (for training)
        if mode == 'train':
            if self.norm_stats is None:
                print(f"📊 Calculating normalization stats for {mode} mode...")
                self.norm_stats = self._calculate_normalization_stats()
                print(f"✅ Stats calculated: {self.norm_stats}")
        else:
            if self.norm_stats is None:
                raise ValueError(f"normalization_stats are required for '{mode}' mode")

        print(f"✅ {mode.capitalize()} dataset: {len(self.indices):,} samples")

    def _calculate_normalization_stats(self) -> Dict[str, float]:
        """
        Calculates normalization stats (mean/std for I/Q) using chunked processing
        on a subset of the training data.
        """
        with h5py.File(self.file_path, 'r') as temp_file:
            X_temp = temp_file['X']
            
            # Use a subset of indices to calculate stats
            num_samples = min(5000, len(self.indices))
            np.random.seed(self.seed)
            sample_indices = np.random.choice(self.indices, num_samples, replace=False)
            sorted_indices = np.sort(sample_indices)

            chunk_size = min(500, num_samples)
            i_vals, q_vals = [], []

            print(f"  Processing {num_samples} samples in {int(np.ceil(len(sorted_indices)/chunk_size))} chunks...")
            for i in range(0, len(sorted_indices), chunk_size):
                chunk_indices = sorted_indices[i:i+chunk_size]
                # Read data chunk
                chunk_data = X_temp[chunk_indices, ...]
                chunk_tensor = torch.from_numpy(chunk_data).float()

                # Append I and Q channels separately
                i_vals.append(chunk_tensor[:, :, 0].flatten())
                q_vals.append(chunk_tensor[:, :, 1].flatten())
                del chunk_data, chunk_tensor

            # Concatenate all values and calculate stats
            i_all = torch.cat(i_vals)
            q_all = torch.cat(q_vals)

            stats = {
                'i_mean': i_all.mean().item(),
                'i_std': max(i_all.std().item(), 1e-8), # Avoid division by zero
                'q_mean': q_all.mean().item(),
                'q_std': max(q_all.std().item(), 1e-8)  # Avoid division by zero
            }

            del i_vals, q_vals, i_all, q_all
            gc.collect()
            return stats

    def open_hdf5(self):
        """
        Opens the HDF5 file. This method is called by worker_init_fn.
        CRITICAL: This must be called AFTER the process fork,
        inside the worker process.
        """
        if self.hdf5_file is None:
            # Open the file in read-only mode
            # swmr=False and libver='latest' are often good for performance
            self.hdf5_file = h5py.File(self.file_path, 'r', swmr=False, libver='latest')
            # Get a handle to the 'X' dataset
            self.X_h5 = self.hdf5_file['X']

    def get_normalization_stats(self) -> Dict[str, float]:
        """Returns a copy of the normalization stats."""
        return self.norm_stats.copy()

    def __len__(self) -> int:
        """Returns the number of samples in this dataset split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Gets a single data sample.
        Assumes the HDF5 file has already been opened by worker_init_fn.
        """

        # This check is crucial. If the file isn't open, it means
        # worker_init_fn wasn't called (e.g., num_workers=0).
        if self.hdf5_file is None:
            # If you must support num_workers=0, you could call self.open_hdf5() here.
            # But the reference code implies raising an error is desired.
            raise RuntimeError(
                "HDF5 file is not open! "
                "Ensure DataLoader uses worker_init_fn=worker_init_fn and num_workers > 0."
            )

        # Get the "true" index from the dataset's index list
        true_index = int(self.indices[idx])

        # Read data from the open HDF5 file handle
        x_raw = self.X_h5[true_index]
        
        # Get corresponding labels and SNR
        y_string = self.Y_strings[true_index]
        z_snr = float(self.Z[true_index])
        
        # Map string label to integer label
        y_label = self.label_map[y_string]

        # Convert to tensor and apply normalization
        iq_sequence = torch.from_numpy(x_raw.copy()).float()
        iq_sequence[:, 0] = (iq_sequence[:, 0] - self.norm_stats['i_mean']) / self.norm_stats['i_std']
        iq_sequence[:, 1] = (iq_sequence[:, 1] - self.norm_stats['q_mean']) / self.norm_stats['q_std']

        # Reshape into [C, H, W] format
        i_signal = iq_sequence[:, 0] # [1024]
        q_signal = iq_sequence[:, 1] # [1024]
        
        # Concatenate I and Q to form a [2048] vector
        iq_concat = torch.cat((i_signal, q_signal), dim=0)
        
        # Reshape the [2048] vector into a [1, 32, 64] "image"
        # 1 = channel, 32 = height, 64 = width
        iq_image = iq_concat.view(1, self.H, self.W)

        return iq_image, y_label, z_snr

    def close(self):
        """Closes the HDF5 file handle."""
        if hasattr(self, 'hdf5_file') and self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
            except Exception:
                pass  # Ignore errors on close
            finally:
                self.hdf5_file = None
                self.X_h5 = None

    def __del__(self):
        """Destructor to ensure the file is closed when the object is deleted."""
        self.close()