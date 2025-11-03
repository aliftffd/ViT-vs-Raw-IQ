"""
Raw IQ Signal Visualization Script

This script loads and visualizes raw I/Q signal data from the RadioML 2018 dataset.
It displays:
1. Time-domain I and Q channel signals
2. Constellation diagrams (I vs Q)
3. Magnitude and phase representations
4. Signals from different modulation types and SNR levels
"""

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import signal


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration for data visualization"""

    # Paths - Update these to match your system
    FILE_PATH = "C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\GOLD_XYZ_OSC.0001_1024.hdf5"
    JSON_PATH = 'C:\\workarea\\Research and Thesis\\dataset\\radioml2018\\versions\\2\\classes-fixed.json'

    # Output directory
    OUTPUT_DIR = Path("visualization_results")

    # Visualization settings
    NUM_SAMPLES_PER_MOD = 3  # Number of samples to visualize per modulation
    SEQUENCE_LENGTH = 1024

    # DSP settings
    # NOTE: RadioML 2018.01A uses 1 sample per symbol (no oversampling)
    # Set to 1 to disable DSP processing and treat each sample as a symbol
    SAMPLES_PER_SYMBOL = 1  # Samples per symbol (1 = no oversampling, >1 = requires timing recovery)
    TIMING_METHOD = 'simple_energy'  # Only used if SPS > 1

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
# DSP UTILITY FUNCTIONS
# ============================================

def rrc_filter(alpha, span, sps):
    """
    Generate Root Raised Cosine (RRC) filter

    Args:
        alpha: Roll-off factor (0 to 1)
        span: Filter span in symbols
        sps: Samples per symbol

    Returns:
        Normalized RRC filter coefficients
    """
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)

    # Handle special cases
    h = np.zeros(len(n))

    for i, t in enumerate(n):
        if t == 0:
            h[i] = (1 + alpha * (4 / np.pi - 1))
        elif abs(abs(t) - sps / (4 * alpha)) < 1e-10:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            numerator = np.sin(np.pi * t / sps * (1 - alpha)) + \
                       4 * alpha * t / sps * np.cos(np.pi * t / sps * (1 + alpha))
            denominator = np.pi * t / sps * (1 - (4 * alpha * t / sps) ** 2)
            h[i] = numerator / denominator

    # Normalize
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def matched_filter(i_signal, q_signal, alpha=0.35, span=8, sps=2):
    """
    Apply matched filtering to I/Q signals

    Args:
        i_signal: In-phase component
        q_signal: Quadrature component
        alpha: Roll-off factor
        span: Filter span in symbols
        sps: Samples per symbol

    Returns:
        Filtered I and Q signals
    """
    rrc = rrc_filter(alpha, span, sps)

    # Apply filter
    i_filtered = signal.convolve(i_signal, rrc, mode='same')
    q_filtered = signal.convolve(q_signal, rrc, mode='same')

    return i_filtered, q_filtered


def timing_recovery_gardner(i_signal, q_signal, sps=2):
    """
    Gardner timing recovery algorithm

    Args:
        i_signal: In-phase component
        q_signal: Quadrature component
        sps: Samples per symbol (initial estimate)

    Returns:
        Symbol indices (timing instants)
    """
    # Combine I and Q into complex signal
    sig = i_signal + 1j * q_signal

    # Initialize
    mu = 0  # Fractional delay
    mu_history = []
    symbol_indices = []

    # Loop gain
    K = 0.1

    # Start after some initial samples to avoid edge effects
    idx = sps

    while idx < len(sig) - sps:
        # Get samples for timing error calculation
        if int(idx) >= len(sig) - 1:
            break

        # Sample at current estimate
        symbol_indices.append(int(idx))

        # Gardner TED: error = real(mid_sample * conj(prev_symbol - next_symbol))
        if int(idx - sps/2) >= 0 and int(idx + sps/2) < len(sig):
            prev_sym = sig[int(idx - sps/2)]
            mid_sample = sig[int(idx)]
            next_sym = sig[int(idx + sps/2)]

            # Timing error
            error = np.real((mid_sample.conjugate() * (next_sym - prev_sym)))

            # Update mu
            mu = mu + K * error
            mu_history.append(mu)

        # Advance by sps + mu adjustment
        idx = idx + sps + mu
        mu = 0  # Reset mu after applying

    return np.array(symbol_indices)


def timing_recovery_mueller_muller(i_signal, q_signal, sps=2):
    """
    Mueller and Müller timing recovery algorithm

    Args:
        i_signal: In-phase component
        q_signal: Quadrature component
        sps: Samples per symbol (initial estimate)

    Returns:
        Symbol indices (timing instants)
    """
    # Combine into complex signal
    sig = i_signal + 1j * q_signal

    # Initialize
    mu = 0
    symbol_indices = []

    # Loop parameters
    K = 0.05  # Timing loop gain

    # Start after initial samples
    idx = sps
    prev_symbol = sig[0]

    while idx < len(sig) - sps:
        if int(idx) >= len(sig):
            break

        # Sample current symbol
        current_idx = int(idx)
        symbol_indices.append(current_idx)
        current_symbol = sig[current_idx]

        # M&M timing error detector
        # error = real(current * conj(prev)) - real(prev * conj(current))
        error = np.real(current_symbol) * np.real(prev_symbol) + \
                np.imag(current_symbol) * np.imag(prev_symbol)

        # Update timing
        mu = mu + K * error

        # Advance
        idx = idx + sps + mu
        mu = 0
        prev_symbol = current_symbol

    return np.array(symbol_indices)


def simple_timing_recovery(i_signal, q_signal, sps=2, method='energy'):
    """
    Simple timing recovery based on signal energy or correlation

    Args:
        i_signal: In-phase component
        q_signal: Quadrature component
        sps: Samples per symbol
        method: 'energy' or 'correlation'

    Returns:
        Symbol indices
    """
    sig = i_signal + 1j * q_signal
    energy = np.abs(sig) ** 2

    if method == 'energy':
        # Find peaks in energy
        # Use a sliding window to find local maxima
        window_size = max(1, int(sps * 0.5))
        peaks = []

        for i in range(window_size, len(energy) - window_size, int(sps)):
            # Find max in window
            window = energy[i-window_size:i+window_size]
            local_max_idx = np.argmax(window)
            peaks.append(i - window_size + local_max_idx)

        return np.array(peaks)

    else:  # correlation
        # Simple fixed-rate sampling with offset estimation
        # Find best offset by maximizing energy
        best_offset = 0
        best_energy = 0

        for offset in range(sps):
            indices = np.arange(offset, len(sig), sps)
            total_energy = np.sum(energy[indices])
            if total_energy > best_energy:
                best_energy = total_energy
                best_offset = offset

        return np.arange(best_offset, len(sig), sps)


def extract_symbols(i_signal, q_signal, sps=1, method='gardner', alpha=0.35):
    """
    Extract symbol decision points from raw I/Q waveform

    Args:
        i_signal: In-phase component
        q_signal: Quadrature component
        sps: Samples per symbol (1 = no oversampling, >1 = requires timing recovery)
        method: 'gardner', 'mueller_muller', 'simple_energy', 'simple_correlation'
        alpha: RRC roll-off factor for matched filtering

    Returns:
        Dictionary containing:
            - symbol_i: I values at decision points
            - symbol_q: Q values at decision points
            - symbol_indices: Timing indices
            - filtered_i: Matched filtered I signal (same as input for sps=1)
            - filtered_q: Matched filtered Q signal (same as input for sps=1)
    """
    # Special case: sps=1 means data is already at symbol rate
    # No DSP processing needed - every sample IS a symbol
    if sps == 1:
        symbol_indices = np.arange(len(i_signal))
        return {
            'symbol_i': i_signal.copy(),
            'symbol_q': q_signal.copy(),
            'symbol_indices': symbol_indices,
            'filtered_i': i_signal.copy(),
            'filtered_q': q_signal.copy()
        }

    # For sps > 1: Apply DSP processing
    # Apply matched filtering
    i_filtered, q_filtered = matched_filter(i_signal, q_signal, alpha=alpha, sps=sps)

    # Timing recovery
    if method == 'gardner':
        symbol_indices = timing_recovery_gardner(i_filtered, q_filtered, sps=sps)
    elif method == 'mueller_muller':
        symbol_indices = timing_recovery_mueller_muller(i_filtered, q_filtered, sps=sps)
    elif method == 'simple_energy':
        symbol_indices = simple_timing_recovery(i_filtered, q_filtered, sps=sps, method='energy')
    elif method == 'simple_correlation':
        symbol_indices = simple_timing_recovery(i_filtered, q_filtered, sps=sps, method='correlation')
    else:
        raise ValueError(f"Unknown timing recovery method: {method}")

    # Clip indices to valid range
    symbol_indices = symbol_indices[symbol_indices < len(i_filtered)]

    # Extract symbols at decision points
    symbol_i = i_filtered[symbol_indices]
    symbol_q = q_filtered[symbol_indices]

    return {
        'symbol_i': symbol_i,
        'symbol_q': symbol_q,
        'symbol_indices': symbol_indices,
        'filtered_i': i_filtered,
        'filtered_q': q_filtered
    }


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


def plot_iq_signal(i_signal, q_signal, title, save_path=None, sps=1, timing_method='simple_energy'):
    """
    Plot I/Q signal in multiple representations with proper DSP processing

    Args:
        i_signal: In-phase component (numpy array)
        q_signal: Quadrature component (numpy array)
        title: Plot title
        save_path: Path to save the figure (optional)
        sps: Samples per symbol (1 = no oversampling, >1 = requires timing recovery)
        timing_method: Timing recovery method (only used if sps > 1)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Update title based on SPS
    if sps == 1:
        full_title = f"{title}\n(1 sample/symbol - No DSP processing)"
    else:
        full_title = f"{title}\n(DSP: {sps} samples/symbol, {timing_method})"

    fig.suptitle(full_title, fontsize=16, fontweight='bold')

    # Time axis
    time_axis = np.arange(len(i_signal))

    # Extract symbols using DSP processing
    try:
        symbols = extract_symbols(i_signal, q_signal, sps=sps, method=timing_method)
        dsp_success = len(symbols['symbol_i']) > 0
    except Exception as e:
        print(f"  Warning: Symbol extraction failed ({e}), showing raw signals only")
        dsp_success = False

    # 1. Time domain - Raw I and Q signals
    ax1 = axes[0, 0]
    ax1.plot(time_axis, i_signal, label='I (In-phase)', alpha=0.7, linewidth=0.8, color='blue')
    ax1.plot(time_axis, q_signal, label='Q (Quadrature)', alpha=0.7, linewidth=0.8, color='orange')
    if dsp_success:
        # Mark symbol decision points
        ax1.scatter(symbols['symbol_indices'], symbols['symbol_i'],
                   color='blue', s=20, marker='o', zorder=5, label='I symbols', edgecolors='black', linewidths=0.5)
        ax1.scatter(symbols['symbol_indices'], symbols['symbol_q'],
                   color='orange', s=20, marker='o', zorder=5, label='Q symbols', edgecolors='black', linewidths=0.5)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain - Raw I/Q Signals')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Constellation diagram - Raw trajectory (all samples)
    ax2 = axes[0, 1]
    if sps == 1:
        # For sps=1, all samples are symbols - show differently
        ax2.scatter(i_signal, q_signal, alpha=0.4, s=8, color='blue',
                   edgecolors='darkblue', linewidths=0.3, label=f'All samples (n={len(i_signal)})')
        ax2.set_title('Constellation - All Samples\n(Each sample is a symbol)')
    else:
        ax2.scatter(i_signal, q_signal, alpha=0.2, s=3, color='gray', label='Raw trajectory')
        ax2.set_title('Constellation - Raw Trajectory\n(includes inter-symbol transitions)')
    ax2.set_xlabel('I (In-phase)')
    ax2.set_ylabel('Q (Quadrature)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.axis('equal')
    ax2.legend(fontsize=8)

    # 3. Constellation diagram - Recovered/extracted symbols
    ax3 = axes[0, 2]
    if dsp_success:
        if sps == 1:
            # Same as raw for sps=1
            ax3.scatter(symbols['symbol_i'], symbols['symbol_q'],
                       alpha=0.6, s=15, color='red', marker='o',
                       edgecolors='darkred', linewidths=0.5, label=f'Symbols (n={len(symbols["symbol_i"])})')
            ax3.set_title('Constellation - Symbol Points\n(No processing needed for 1 SPS)')
        else:
            ax3.scatter(symbols['symbol_i'], symbols['symbol_q'],
                       alpha=0.6, s=15, color='red', marker='o',
                       edgecolors='darkred', linewidths=0.5, label=f'Recovered (n={len(symbols["symbol_i"])})')
            ax3.set_title(f'Constellation - Recovered Symbols\n(Method: {timing_method})')
    else:
        ax3.text(0.5, 0.5, 'Symbol extraction\nfailed',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Constellation - Symbol Points')
    ax3.set_xlabel('I (In-phase)')
    ax3.set_ylabel('Q (Quadrature)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    ax3.axis('equal')
    if dsp_success:
        ax3.legend(fontsize=8)

    # 4. Magnitude over time
    ax4 = axes[1, 0]
    magnitude = np.sqrt(i_signal**2 + q_signal**2)
    ax4.plot(time_axis, magnitude, color='purple', linewidth=0.8, alpha=0.7, label='Raw magnitude')
    if dsp_success:
        filtered_magnitude = np.sqrt(symbols['filtered_i']**2 + symbols['filtered_q']**2)
        ax4.plot(time_axis, filtered_magnitude, color='green', linewidth=0.8, alpha=0.7, label='Filtered magnitude')
        ax4.scatter(symbols['symbol_indices'], filtered_magnitude[symbols['symbol_indices']],
                   color='red', s=15, marker='o', zorder=5, label='Symbol timing', edgecolors='darkred', linewidths=0.5)
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Signal Magnitude')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Phase over time
    ax5 = axes[1, 1]
    phase = np.arctan2(q_signal, i_signal)
    ax5.plot(time_axis, phase, color='orange', linewidth=0.8, alpha=0.7, label='Raw phase')
    if dsp_success:
        filtered_phase = np.arctan2(symbols['filtered_q'], symbols['filtered_i'])
        ax5.plot(time_axis, filtered_phase, color='brown', linewidth=0.8, alpha=0.7, label='Filtered phase')
        ax5.scatter(symbols['symbol_indices'], filtered_phase[symbols['symbol_indices']],
                   color='red', s=15, marker='o', zorder=5, label='Symbol timing', edgecolors='darkred', linewidths=0.5)
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Phase (radians)')
    ax5.set_title('Signal Phase')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linewidth=0.5)

    # 6. Eye diagram (if DSP successful)
    ax6 = axes[1, 2]
    if dsp_success and sps > 1:
        # Create eye diagram
        eye_period = sps * 2  # Show 2 symbol periods
        i_filt = symbols['filtered_i']

        # Overlay multiple traces
        for start_idx in range(0, len(i_filt) - eye_period, sps):
            segment = i_filt[start_idx:start_idx + eye_period]
            if len(segment) == eye_period:
                ax6.plot(range(eye_period), segment, alpha=0.1, color='blue', linewidth=0.5)

        ax6.set_xlabel('Samples (2 symbol periods)')
        ax6.set_ylabel('Amplitude')
        ax6.set_title('Eye Diagram (I channel)')
        ax6.grid(True, alpha=0.3)
        ax6.axvline(x=sps, color='red', linewidth=1, linestyle='--', label='Symbol boundary')
        ax6.legend(fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Eye diagram\nnot available',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Eye Diagram')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")

    plt.close()


def visualize_modulation_samples(file_path, dataset_info, modulation, config, num_samples=3, snr_value=None, sps=2, timing_method='simple_energy'):
    """
    Visualize samples for a specific modulation type

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        modulation: Modulation type to visualize
        config: Configuration object
        num_samples: Number of samples to visualize
        snr_value: Specific SNR value to filter (optional, uses highest if None)
        sps: Samples per symbol
        timing_method: Timing recovery method
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
            title = f"{modulation} - Sample {i+1} (SNR={actual_snr:.1f}dB)"
            save_path = mod_dir / f"{modulation}_sample_{i+1}_snr{int(actual_snr)}.png"

            plot_iq_signal(i_signal, q_signal, title, save_path, sps=sps, timing_method=timing_method)


def visualize_snr_comparison(file_path, dataset_info, modulation, config, snr_values=None, sps=1, timing_method='simple_energy'):
    """
    Visualize the same modulation type at different SNR levels

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        modulation: Modulation type to visualize
        config: Configuration object
        snr_values: List of SNR values to compare (optional)
        sps: Samples per symbol (1 = no oversampling)
        timing_method: Timing recovery method (only used if sps > 1)
    """
    if snr_values is None:
        # Select low, medium, and high SNR
        unique_snr = dataset_info['unique_snr']
        snr_values = [unique_snr.min(), unique_snr[len(unique_snr)//2], unique_snr.max()]

    Y_strings = dataset_info['Y_strings']
    Z = dataset_info['Z']

    print(f"\n📊 SNR Comparison for {modulation}")

    # Create comparison directory
    comp_dir = config.OUTPUT_DIR / "snr_comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, 'r') as f:
        X = f['X']

        fig, axes = plt.subplots(len(snr_values), 4, figsize=(20, 4*len(snr_values)))
        if len(snr_values) == 1:
            axes = axes.reshape(1, -1)

        if sps == 1:
            title_suffix = '(1 sample/symbol - No DSP)'
        else:
            title_suffix = f'(DSP: {sps} SPS, {timing_method})'
        fig.suptitle(f'{modulation} - SNR Comparison {title_suffix}', fontsize=16, fontweight='bold')

        for row, snr in enumerate(snr_values):
            # Find sample at this SNR
            mod_indices = np.where(Y_strings == modulation)[0]
            snr_mask = np.abs(Z[mod_indices] - snr) < 0.1
            filtered_indices = mod_indices[snr_mask]

            if len(filtered_indices) == 0:
                print(f"  ⚠️  No samples found at SNR={snr}dB")
                continue

            # Get one sample
            idx = filtered_indices[0]
            x_raw = X[idx]

            i_signal = x_raw[:, 0]
            q_signal = x_raw[:, 1]
            time_axis = np.arange(len(i_signal))

            # Extract symbols
            try:
                symbols = extract_symbols(i_signal, q_signal, sps=sps, method=timing_method)
                dsp_success = len(symbols['symbol_i']) > 0
            except:
                dsp_success = False

            # Plot I/Q time series
            axes[row, 0].plot(time_axis, i_signal, label='I', alpha=0.7, linewidth=0.8, color='blue')
            axes[row, 0].plot(time_axis, q_signal, label='Q', alpha=0.7, linewidth=0.8, color='orange')
            if dsp_success:
                axes[row, 0].scatter(symbols['symbol_indices'], symbols['symbol_i'],
                                    color='blue', s=15, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
                axes[row, 0].scatter(symbols['symbol_indices'], symbols['symbol_q'],
                                    color='orange', s=15, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            axes[row, 0].set_ylabel(f'SNR={snr:.1f}dB', fontsize=12, fontweight='bold')
            axes[row, 0].legend(fontsize=8)
            axes[row, 0].grid(True, alpha=0.3)
            if row == 0:
                axes[row, 0].set_title('Time Domain')
            if row == len(snr_values) - 1:
                axes[row, 0].set_xlabel('Sample Index')

            # Plot constellation
            if sps == 1:
                axes[row, 1].scatter(i_signal, q_signal, alpha=0.4, s=8, color='blue',
                                    edgecolors='darkblue', linewidths=0.3)
            else:
                axes[row, 1].scatter(i_signal, q_signal, alpha=0.2, s=3, color='gray')
            axes[row, 1].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 1].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].axis('equal')
            if row == 0:
                axes[row, 1].set_title('Constellation (All Samples)' if sps == 1 else 'Raw Trajectory')
            if row == len(snr_values) - 1:
                axes[row, 1].set_xlabel('I (In-phase)')
            axes[row, 1].set_ylabel('Q (Quadrature)')

            # Plot symbols (same as all samples for sps=1)
            if dsp_success:
                axes[row, 2].scatter(symbols['symbol_i'], symbols['symbol_q'],
                                    alpha=0.6, s=15, color='red', marker='o',
                                    edgecolors='darkred', linewidths=0.5)
            axes[row, 2].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 2].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 2].grid(True, alpha=0.3)
            axes[row, 2].axis('equal')
            if row == 0:
                axes[row, 2].set_title('Symbol Points' if sps == 1 else 'Recovered Symbols')
            if row == len(snr_values) - 1:
                axes[row, 2].set_xlabel('I (In-phase)')
            axes[row, 2].set_ylabel('Q (Quadrature)')

            # Plot magnitude
            magnitude = np.sqrt(i_signal**2 + q_signal**2)
            axes[row, 3].plot(time_axis, magnitude, color='purple', linewidth=0.8, alpha=0.7, label='Raw')
            if dsp_success:
                filtered_magnitude = np.sqrt(symbols['filtered_i']**2 + symbols['filtered_q']**2)
                axes[row, 3].plot(time_axis, filtered_magnitude, color='green', linewidth=0.8, alpha=0.7, label='Filtered')
                axes[row, 3].scatter(symbols['symbol_indices'], filtered_magnitude[symbols['symbol_indices']],
                                    color='red', s=10, marker='o', zorder=5, edgecolors='darkred', linewidths=0.5)
            axes[row, 3].grid(True, alpha=0.3)
            axes[row, 3].legend(fontsize=8)
            if row == 0:
                axes[row, 3].set_title('Magnitude')
            if row == len(snr_values) - 1:
                axes[row, 3].set_xlabel('Sample Index')
            axes[row, 3].set_ylabel('Magnitude')

        plt.tight_layout()
        save_path = comp_dir / f"{modulation}_snr_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()


def create_overview_plot(file_path, dataset_info, config, sps=1, timing_method='simple_energy'):
    """
    Create an overview plot showing multiple modulation types

    Args:
        file_path: Path to HDF5 file
        dataset_info: Dictionary containing dataset metadata
        config: Configuration object
        sps: Samples per symbol (1 = no oversampling)
        timing_method: Timing recovery method (only used if sps > 1)
    """
    print(f"\n📊 Creating overview plot")

    # Select a subset of modulations for overview
    overview_mods = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'GMSK']
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

        fig, axes = plt.subplots(len(available_mods), 3, figsize=(18, 3*len(available_mods)))
        if len(available_mods) == 1:
            axes = axes.reshape(1, -1)

        if sps == 1:
            title_suffix = '1 sample/symbol'
        else:
            title_suffix = f'DSP: {sps} SPS, {timing_method}'
        fig.suptitle(f'Modulation Overview (SNR={snr_value:.1f}dB) - {title_suffix}', fontsize=16, fontweight='bold')

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

            # Extract symbols
            try:
                symbols = extract_symbols(i_signal, q_signal, sps=sps, method=timing_method)
                dsp_success = len(symbols['symbol_i']) > 0
            except:
                dsp_success = False

            # Time domain
            axes[row, 0].plot(time_axis, i_signal, label='I', alpha=0.7, linewidth=0.8, color='blue')
            axes[row, 0].plot(time_axis, q_signal, label='Q', alpha=0.7, linewidth=0.8, color='orange')
            if dsp_success:
                axes[row, 0].scatter(symbols['symbol_indices'], symbols['symbol_i'],
                                    color='blue', s=10, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
                axes[row, 0].scatter(symbols['symbol_indices'], symbols['symbol_q'],
                                    color='orange', s=10, marker='o', zorder=5, edgecolors='black', linewidths=0.5)
            axes[row, 0].set_ylabel(mod, fontsize=12, fontweight='bold')
            axes[row, 0].legend(fontsize=8)
            axes[row, 0].grid(True, alpha=0.3)
            if row == 0:
                axes[row, 0].set_title('Time Domain')
            if row == len(available_mods) - 1:
                axes[row, 0].set_xlabel('Sample Index')

            # Constellation
            if sps == 1:
                axes[row, 1].scatter(i_signal, q_signal, alpha=0.4, s=8, color='blue',
                                    edgecolors='darkblue', linewidths=0.3)
            else:
                axes[row, 1].scatter(i_signal, q_signal, alpha=0.2, s=3, color='gray')
            axes[row, 1].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 1].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].axis('equal')
            if row == 0:
                axes[row, 1].set_title('Constellation (All Samples)' if sps == 1 else 'Raw Trajectory')
            if row == len(available_mods) - 1:
                axes[row, 1].set_xlabel('I (In-phase)')
            axes[row, 1].set_ylabel('Q (Quadrature)')

            # Symbols
            if dsp_success:
                axes[row, 2].scatter(symbols['symbol_i'], symbols['symbol_q'],
                                    alpha=0.6, s=15, color='red', marker='o',
                                    edgecolors='darkred', linewidths=0.5)
            axes[row, 2].axhline(y=0, color='k', linewidth=0.5)
            axes[row, 2].axvline(x=0, color='k', linewidth=0.5)
            axes[row, 2].grid(True, alpha=0.3)
            axes[row, 2].axis('equal')
            if row == 0:
                axes[row, 2].set_title('Symbol Points' if sps == 1 else 'Recovered Symbols')
            if row == len(available_mods) - 1:
                axes[row, 2].set_xlabel('I (In-phase)')
            axes[row, 2].set_ylabel('Q (Quadrature)')

        plt.tight_layout()
        save_path = config.OUTPUT_DIR / "modulation_overview.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()


def main():
    """Main visualization function"""
    parser = argparse.ArgumentParser(
        description='Visualize Raw IQ Signals (RadioML 2018.01A uses 1 sample/symbol by default)')
    parser.add_argument('--file_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--json_path', type=str, help='Path to JSON file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--modulations', type=str, nargs='+', help='Specific modulations to visualize')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per modulation')
    parser.add_argument('--create_overview', action='store_true', help='Create overview plot')
    parser.add_argument('--snr_comparison', action='store_true', help='Create SNR comparison plots')
    parser.add_argument('--sps', type=int, default=1,
                       help='Samples per symbol (default: 1 for RadioML 2018.01A, >1 enables DSP processing)')
    parser.add_argument('--timing_method', type=str, default='simple_energy',
                       choices=['gardner', 'mueller_muller', 'simple_energy', 'simple_correlation'],
                       help='Timing recovery method for SPS>1 (default: simple_energy)')

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
    if args.sps:
        config.SAMPLES_PER_SYMBOL = args.sps
    if args.timing_method:
        config.TIMING_METHOD = args.timing_method

    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RAW IQ SIGNAL VISUALIZATION")
    print("="*70)
    print(f"Processing Mode:")
    if config.SAMPLES_PER_SYMBOL == 1:
        print(f"  - 1 sample per symbol (NO DSP processing)")
        print(f"  - Each sample is already a symbol decision point")
    else:
        print(f"  - {config.SAMPLES_PER_SYMBOL} samples per symbol (DSP processing enabled)")
        print(f"  - Timing recovery method: {config.TIMING_METHOD}")

    # Load dataset info
    dataset_info = load_dataset_info(config.FILE_PATH, config.JSON_PATH)

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
        create_overview_plot(config.FILE_PATH, dataset_info, config,
                           sps=config.SAMPLES_PER_SYMBOL,
                           timing_method=config.TIMING_METHOD)

    # Visualize each modulation
    for modulation in modulations_to_viz:
        if modulation in config.TARGET_MODULATIONS:
            visualize_modulation_samples(
                config.FILE_PATH,
                dataset_info,
                modulation,
                config,
                num_samples=config.NUM_SAMPLES_PER_MOD,
                sps=config.SAMPLES_PER_SYMBOL,
                timing_method=config.TIMING_METHOD
            )

            # Create SNR comparison if requested
            if args.snr_comparison:
                visualize_snr_comparison(
                    config.FILE_PATH,
                    dataset_info,
                    modulation,
                    config,
                    sps=config.SAMPLES_PER_SYMBOL,
                    timing_method=config.TIMING_METHOD
                )
        else:
            print(f"  ⚠️  Skipping unknown modulation: {modulation}")

    print("\n" + "="*70)
    print(f"✅ Visualization complete! Results saved to: {config.OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
