"""
Preprocessing Visualization Script for Paper

This script visualizes the actual preprocessing pipeline used in:
1. ViT approach: I/Q data reshaped to 2D images [1, 32, 64]
2. Transformer approach: I/Q data as raw sequences [2, 1024]
Both approaches use z-score normalization on I and Q channels separately.
High-resolution output (600 DPI) for publication quality.
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for data visualization"""

    # Paths - Update these to match your system
    FILE_PATH = "/home/lipplopp/research/AMC_Repository/dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = '/home/lipplopp/research/AMC_Repository/dataset/classes-fixed.json'

    # Output directory
    OUTPUT_DIR = Path("visualization_results")

    # Visualization settings
    NUM_SAMPLES_PER_MOD = 3  # Number of samples to visualize per modulation
    SEQUENCE_LENGTH = 1024

    # ViT image dimensions (matches ViT/dataloader/dataset.py)
    VIT_HEIGHT = 32
    VIT_WIDTH = 64

    # Output DPI for publication quality
    OUTPUT_DPI = 600

    # Target modulations to visualize
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


# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

def calculate_normalization_stats(file_path, indices, num_samples=5000):
    """
    Calculate normalization statistics (mean/std) for I/Q channels.
    This matches the preprocessing in both dataset implementations.

    Args:
        file_path: Path to HDF5 file
        indices: Array of indices to use for calculation
        num_samples: Number of samples to use for stats calculation

    Returns:
        Dictionary with i_mean, i_std, q_mean, q_std
    """
    with h5py.File(file_path, 'r') as f:
        X = f['X']

        # Use subset for efficiency
        num_samples = min(num_samples, len(indices))
        np.random.seed(42)
        sample_indices = np.random.choice(indices, num_samples, replace=False)

        # Collect I and Q values
        i_vals = []
        q_vals = []

        for idx in sample_indices[:1000]:  # Use first 1000 for quick calculation
            x_raw = X[idx]
            i_vals.append(x_raw[:, 0])
            q_vals.append(x_raw[:, 1])

        i_all = np.concatenate(i_vals)
        q_all = np.concatenate(q_vals)

        stats = {
            'i_mean': np.mean(i_all),
            'i_std': max(np.std(i_all), 1e-8),
            'q_mean': np.mean(q_all),
            'q_std': max(np.std(q_all), 1e-8)
        }

        return stats


def apply_normalization(i_signal, q_signal, stats):
    """
    Apply z-score normalization to I/Q signals.
    Matches the preprocessing in dataset.__getitem__()

    Args:
        i_signal: In-phase signal
        q_signal: Quadrature signal
        stats: Dictionary with i_mean, i_std, q_mean, q_std

    Returns:
        Normalized I and Q signals
    """
    i_normalized = (i_signal - stats['i_mean']) / stats['i_std']
    q_normalized = (q_signal - stats['q_mean']) / stats['q_std']
    return i_normalized, q_normalized


def preprocess_for_vit(i_signal, q_signal, stats, H=32, W=64):
    """
    Preprocess I/Q data for ViT approach.
    Matches ViT/dataloader/dataset.py __getitem__() method.

    Steps:
    1. Normalize I and Q channels
    2. Concatenate [I, Q] to form [2048] vector
    3. Reshape to [1, 32, 64] image

    Args:
        i_signal: In-phase signal [1024]
        q_signal: Quadrature signal [1024]
        stats: Normalization statistics
        H, W: Image height and width

    Returns:
        Image array [1, H, W]
    """
    # Normalize
    i_norm, q_norm = apply_normalization(i_signal, q_signal, stats)

    # Concatenate I and Q to form [2048] vector
    iq_concat = np.concatenate([i_norm, q_norm])

    # Reshape to [1, H, W] image
    iq_image = iq_concat.reshape(1, H, W)

    return iq_image


def preprocess_for_transformer(i_signal, q_signal, stats):
    """
    Preprocess I/Q data for Transformer approach.
    Matches transformer_rawIQ/dataloader/dataset.py __getitem__() method.

    Steps:
    1. Normalize I and Q channels
    2. Stack to [2, sequence_length] format

    Args:
        i_signal: In-phase signal [1024]
        q_signal: Quadrature signal [1024]
        stats: Normalization statistics

    Returns:
        Stacked array [2, 1024]
    """
    # Normalize
    i_norm, q_norm = apply_normalization(i_signal, q_signal, stats)

    # Stack to [2, sequence_length]
    iq_data = np.stack([i_norm, q_norm], axis=0)

    return iq_data


def load_dataset_info(file_path, json_path):
    """
    Load dataset information and metadata

    Args:
        file_path: Path to HDF5 file
        json_path: Path to JSON class file

    Returns:
        Dictionary containing dataset info
    """
    print(f"📂 Loading dataset from: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # Get dataset dimensions
        X_shape = f['X'].shape
        Y_shape = f['Y'].shape
        Z_shape = f['Z'].shape

        print(f"\nDataset shapes:")
        print(f"  X (IQ data): {X_shape}")
        print(f"  Y (labels):  {Y_shape}")
        print(f"  Z (SNR):     {Z_shape}")

        # Load labels and SNR
        Y_int = np.argmax(f['Y'][:], axis=1)
        Z = f['Z'][:, 0]

    # Load class names
    with open(json_path, 'r') as f:
        modulation_classes = json.load(f)

    Y_strings = np.array([modulation_classes[i] for i in Y_int])

    # Get unique SNR values
    unique_snr = np.unique(Z)

    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(Y_strings):,}")
    print(f"  Modulation types: {len(modulation_classes)}")
    print(f"  SNR range: {unique_snr.min():.1f} to {unique_snr.max():.1f} dB")
    print(f"  SNR values: {sorted(unique_snr)}")

    return {
        'file_path': file_path,
        'Y_strings': Y_strings,
        'Z': Z,
        'modulation_classes': modulation_classes,
        'unique_snr': unique_snr
    }


def plot_preprocessing_pipeline(i_signal, q_signal, stats, title, save_path, config):
    """
    Visualize the complete preprocessing pipeline for both ViT and Transformer approaches.

    Args:
        i_signal: Raw in-phase signal [1024]
        q_signal: Raw quadrature signal [1024]
        stats: Normalization statistics
        title: Plot title
        save_path: Path to save the figure
        config: Configuration object
    """
    # Apply preprocessing
    i_norm, q_norm = apply_normalization(i_signal, q_signal, stats)
    vit_image = preprocess_for_vit(i_signal, q_signal, stats, config.VIT_HEIGHT, config.VIT_WIDTH)
    transformer_data = preprocess_for_transformer(i_signal, q_signal, stats)

    # Create figure with better layout for paper
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Time axis
    time_axis = np.arange(len(i_signal))

    # Row 1: Raw signals
    # 1a. Raw I/Q time series
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(time_axis, i_signal, label='I (In-phase)', alpha=0.8, linewidth=1, color='#1f77b4')
    ax1.plot(time_axis, q_signal, label='Q (Quadrature)', alpha=0.8, linewidth=1, color='#ff7f0e')
    ax1.set_xlabel('Sample Index', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('(a) Raw I/Q Signal', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # 1b. Raw constellation
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(i_signal, q_signal, alpha=0.5, s=10, color='#1f77b4', edgecolors='none')
    ax2.set_xlabel('I (In-phase)', fontsize=10)
    ax2.set_ylabel('Q (Quadrature)', fontsize=10)
    ax2.set_title('(b) Raw Constellation', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')

    # 1c. Signal statistics
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    stats_text = (
        f"Raw Statistics:\n"
        f"I mean: {np.mean(i_signal):.4f}\n"
        f"I std: {np.std(i_signal):.4f}\n"
        f"Q mean: {np.mean(q_signal):.4f}\n"
        f"Q std: {np.std(q_signal):.4f}\n\n"
        f"Normalization:\n"
        f"I: (x - {stats['i_mean']:.4f}) / {stats['i_std']:.4f}\n"
        f"Q: (x - {stats['q_mean']:.4f}) / {stats['q_std']:.4f}"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax3.set_title('(c) Preprocessing Stats', fontsize=11, fontweight='bold')

    # Row 2: Normalized signals
    # 2a. Normalized I/Q time series
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(time_axis, i_norm, label='I normalized', alpha=0.8, linewidth=1, color='#2ca02c')
    ax4.plot(time_axis, q_norm, label='Q normalized', alpha=0.8, linewidth=1, color='#d62728')
    ax4.set_xlabel('Sample Index', fontsize=10)
    ax4.set_ylabel('Normalized Amplitude', fontsize=10)
    ax4.set_title('(d) Normalized I/Q Signal (Common to both approaches)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3, linewidth=0.5)

    # 2b. Normalized constellation
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(i_norm, q_norm, alpha=0.5, s=10, color='#2ca02c', edgecolors='none')
    ax5.set_xlabel('I (normalized)', fontsize=10)
    ax5.set_ylabel('Q (normalized)', fontsize=10)
    ax5.set_title('(e) Normalized Constellation', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, linewidth=0.5)
    ax5.axhline(y=0, color='k', linewidth=0.5)
    ax5.axvline(x=0, color='k', linewidth=0.5)
    ax5.axis('equal')

    # 2c. Normalized statistics
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    norm_stats_text = (
        f"Normalized Statistics:\n"
        f"I mean: {np.mean(i_norm):.4f}\n"
        f"I std: {np.std(i_norm):.4f}\n"
        f"Q mean: {np.mean(q_norm):.4f}\n"
        f"Q std: {np.std(q_norm):.4f}\n\n"
        f"Shape: [1024, 2]\n"
        f"(1024 samples, 2 channels)"
    )
    ax6.text(0.1, 0.5, norm_stats_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax6.set_title('(f) Normalized Stats', fontsize=11, fontweight='bold')

    # Row 3: Final representations
    # 3a. ViT: 2D image representation
    ax7 = fig.add_subplot(gs[2, 0:2])
    im = ax7.imshow(vit_image[0], aspect='auto', cmap='viridis', interpolation='nearest')
    ax7.set_xlabel('Width (64)', fontsize=10)
    ax7.set_ylabel('Height (32)', fontsize=10)
    ax7.set_title('(g) ViT Approach: Reshaped to 2D Image [1, 32, 64]', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Amplitude', fontsize=9)
    # Add annotation
    ax7.text(0.5, -0.15, 'Concatenate [I, Q] → [2048] → Reshape to [1, 32, 64]',
             transform=ax7.transAxes, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 3b. Transformer: Sequence representation
    ax8 = fig.add_subplot(gs[2, 2:4])
    # Show both channels as separate lines
    ax8.plot(time_axis, transformer_data[0], label='I channel', alpha=0.8, linewidth=1, color='#9467bd')
    ax8.plot(time_axis, transformer_data[1], label='Q channel', alpha=0.8, linewidth=1, color='#8c564b')
    ax8.set_xlabel('Sequence Position', fontsize=10)
    ax8.set_ylabel('Normalized Amplitude', fontsize=10)
    ax8.set_title('(h) Transformer Approach: Raw Sequence [2, 1024]', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9, loc='upper right')
    ax8.grid(True, alpha=0.3, linewidth=0.5)
    # Add annotation
    ax8.text(0.5, -0.15, 'Stack [I, Q] → [2, 1024] (2 channels, 1024 time steps)',
             transform=ax8.transAxes, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=config.OUTPUT_DPI, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path} (DPI={config.OUTPUT_DPI})")

    plt.close()


def visualize_modulation_samples(file_path, dataset_info, modulation, config, stats, num_samples=3, snr_value=None):
    """
    Visualize preprocessing pipeline for samples of a specific modulation type

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        modulation: Modulation type to visualize
        config: Configuration object
        stats: Normalization statistics
        num_samples: Number of samples to visualize
        snr_value: Specific SNR value to filter (optional, uses highest if None)
    """
    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    # Filter indices for this modulation
    mod_indices = np.where(Y_strings == modulation)[0]

    if len(mod_indices) == 0:
        print(f"  ⚠️  No samples found for modulation: {modulation}")
        return

    # Filter by SNR if specified, otherwise use highest SNR
    if snr_value is None:
        snr_value = dataset_info['unique_snr'].max()

    snr_mask = np.abs(Z[mod_indices] - snr_value) < 0.1
    filtered_indices = mod_indices[snr_mask]

    if len(filtered_indices) == 0:
        print(f"  ⚠️  No samples found for {modulation} at SNR={snr_value}dB")
        return

    # Select random samples
    num_samples = min(num_samples, len(filtered_indices))
    sample_indices = np.random.choice(filtered_indices, num_samples, replace=False)

    print(f"\n📊 Visualizing {modulation} (SNR={snr_value}dB) - {num_samples} samples")

    # Create output directory for this modulation
    mod_dir = config.OUTPUT_DIR / modulation
    mod_dir.mkdir(parents=True, exist_ok=True)

    # Load and plot each sample
    with h5py.File(file_path, 'r') as f:
        X = f['X']

        for i, idx in enumerate(sample_indices):
            # Load raw IQ data
            x_raw = X[idx]  # Shape: [1024, 2]

            i_signal = x_raw[:, 0]  # In-phase
            q_signal = x_raw[:, 1]  # Quadrature

            # Get actual SNR for this sample
            actual_snr = Z[idx]

            # Create plot
            title = f"{modulation} - Preprocessing Pipeline (SNR={actual_snr:.1f}dB)"
            save_path = mod_dir / f"{modulation}_preprocessing_sample_{i+1}_snr{int(actual_snr)}.png"

            plot_preprocessing_pipeline(i_signal, q_signal, stats, title, save_path, config)


def create_overview_plot(file_path, dataset_info, config, stats):
    """
    Create an overview showing preprocessing for multiple modulation types

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        config: Configuration object
        stats: Normalization statistics
    """
    print(f"\n📊 Creating modulation overview")

    # Select a subset of modulations for overview
    overview_mods = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM']
    available_mods = [m for m in overview_mods if m in config.TARGET_MODULATIONS]

    if len(available_mods) == 0:
        print("  ⚠️  No modulations available for overview")
        return

    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    # Use highest SNR for clearest signals
    snr_value = dataset_info['unique_snr'].max()

    with h5py.File(file_path, 'r') as f:
        X = f['X']

        fig = plt.figure(figsize=(18, 3*len(available_mods)))
        gs = fig.add_gridspec(len(available_mods), 4, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Preprocessing Overview - Multiple Modulations (SNR={snr_value:.1f}dB)',
                     fontsize=14, fontweight='bold')

        for row, mod in enumerate(available_mods):
            # Find sample
            mod_indices = np.where(Y_strings == mod)[0]
            snr_mask = np.abs(Z[mod_indices] - snr_value) < 0.1
            filtered_indices = mod_indices[snr_mask]

            if len(filtered_indices) == 0:
                continue

            idx = filtered_indices[0]
            x_raw = X[idx]

            i_signal = x_raw[:, 0]
            q_signal = x_raw[:, 1]
            time_axis = np.arange(len(i_signal))

            # Apply preprocessing
            i_norm, q_norm = apply_normalization(i_signal, q_signal, stats)
            vit_image = preprocess_for_vit(i_signal, q_signal, stats, config.VIT_HEIGHT, config.VIT_WIDTH)

            # Column 0: Raw I/Q
            ax0 = fig.add_subplot(gs[row, 0])
            ax0.plot(time_axis, i_signal, alpha=0.7, linewidth=0.8, color='blue')
            ax0.plot(time_axis, q_signal, alpha=0.7, linewidth=0.8, color='orange')
            ax0.set_ylabel(mod, fontsize=11, fontweight='bold')
            ax0.grid(True, alpha=0.3)
            if row == 0:
                ax0.set_title('Raw I/Q', fontsize=10, fontweight='bold')
            if row == len(available_mods) - 1:
                ax0.set_xlabel('Sample', fontsize=9)

            # Column 1: Raw constellation
            ax1 = fig.add_subplot(gs[row, 1])
            ax1.scatter(i_signal, q_signal, alpha=0.5, s=8, color='blue', edgecolors='none')
            ax1.axhline(y=0, color='k', linewidth=0.5)
            ax1.axvline(x=0, color='k', linewidth=0.5)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            if row == 0:
                ax1.set_title('Raw Constellation', fontsize=10, fontweight='bold')
            if row == len(available_mods) - 1:
                ax1.set_xlabel('I', fontsize=9)

            # Column 2: Normalized constellation
            ax2 = fig.add_subplot(gs[row, 2])
            ax2.scatter(i_norm, q_norm, alpha=0.5, s=8, color='green', edgecolors='none')
            ax2.axhline(y=0, color='k', linewidth=0.5)
            ax2.axvline(x=0, color='k', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            if row == 0:
                ax2.set_title('Normalized', fontsize=10, fontweight='bold')
            if row == len(available_mods) - 1:
                ax2.set_xlabel('I (norm)', fontsize=9)

            # Column 3: ViT image
            ax3 = fig.add_subplot(gs[row, 3])
            im = ax3.imshow(vit_image[0], aspect='auto', cmap='viridis', interpolation='nearest')
            if row == 0:
                ax3.set_title('ViT Image [32×64]', fontsize=10, fontweight='bold')
            if row == len(available_mods) - 1:
                ax3.set_xlabel('Width', fontsize=9)
                ax3.set_ylabel('Height', fontsize=9)

        plt.tight_layout()
        save_path = config.OUTPUT_DIR / "modulation_overview.png"
        plt.savefig(save_path, dpi=config.OUTPUT_DPI, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path} (DPI={config.OUTPUT_DPI})")
        plt.close()


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(
        description='Visualize preprocessing pipeline for ViT and Transformer approaches')
    parser.add_argument('--file_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--json_path', type=str, help='Path to JSON file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--modulations', type=str, nargs='+', help='Specific modulations to visualize')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per modulation')
    parser.add_argument('--create_overview', action='store_true', help='Create overview plot')
    parser.add_argument('--dpi', type=int, default=600, help='Output DPI (default: 600 for publication)')

    args = parser.parse_args()

    # Update config from args
    config = Config()
    if args.file_path:
        config.FILE_PATH = args.file_path
    if args.json_path:
        config.JSON_PATH = args.json_path
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    if args.num_samples:
        config.NUM_SAMPLES_PER_MOD = args.num_samples
    if args.dpi:
        config.OUTPUT_DPI = args.dpi

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PREPROCESSING PIPELINE VISUALIZATION")
    print("="*70)
    print(f"Output DPI: {config.OUTPUT_DPI} (publication quality)")
    print(f"ViT image dimensions: {config.VIT_HEIGHT} × {config.VIT_WIDTH}")
    print(f"Transformer sequence length: {config.SEQUENCE_LENGTH}")

    # Load dataset info
    dataset_info = load_dataset_info(config.FILE_PATH, config.JSON_PATH)

    # Calculate normalization statistics
    print(f"\n📊 Calculating normalization statistics...")
    all_indices = np.arange(len(dataset_info['Y_strings']))
    stats = calculate_normalization_stats(config.FILE_PATH, all_indices)
    print(f"✅ Normalization stats:")
    print(f"   I: mean={stats['i_mean']:.6f}, std={stats['i_std']:.6f}")
    print(f"   Q: mean={stats['q_mean']:.6f}, std={stats['q_std']:.6f}")

    # Determine which modulations to visualize
    if args.modulations:
        modulations_to_viz = args.modulations
    else:
        modulations_to_viz = config.TARGET_MODULATIONS[:5]  # First 5 by default

    print(f"\nModulations to visualize: {modulations_to_viz}")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create overview plot
    if args.create_overview or not args.modulations:
        create_overview_plot(config.FILE_PATH, dataset_info, config, stats)

    # Visualize each modulation
    for modulation in modulations_to_viz:
        if modulation in config.TARGET_MODULATIONS:
            visualize_modulation_samples(
                config.FILE_PATH,
                dataset_info,
                modulation,
                config,
                stats,
                num_samples=config.NUM_SAMPLES_PER_MOD
            )
        else:
            print(f"  ⚠️  Skipping unknown modulation: {modulation}")

    print("\n" + "="*70)
    print(f"✅ Visualization complete! Results saved to: {config.OUTPUT_DIR}")
    print(f"   High-resolution figures ready for publication (DPI={config.OUTPUT_DPI})")
    print("="*70)


if __name__ == '__main__':
    main()
