"""
Test script for DSP functions in plot_preprocessing_signal.py
Tests RRC filtering, timing recovery, and symbol extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import DSP functions from the visualization script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_preprocessing_signal import (
    rrc_filter,
    matched_filter,
    extract_symbols,
    timing_recovery_gardner,
    timing_recovery_mueller_muller,
    simple_timing_recovery
)


def generate_test_signal(modulation='QPSK', num_symbols=100, sps=2, snr_db=20):
    """
    Generate a simple test signal with known symbol timing

    Args:
        modulation: Modulation type ('QPSK', 'BPSK', '16QAM')
        num_symbols: Number of symbols to generate
        sps: Samples per symbol
        snr_db: Signal-to-noise ratio in dB

    Returns:
        i_signal, q_signal, true_symbol_indices
    """
    # Generate symbols based on modulation
    if modulation == 'BPSK':
        symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
        symbols_i = symbols
        symbols_q = np.zeros(num_symbols)
    elif modulation == 'QPSK':
        # QPSK constellation points
        constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbol_indices = np.random.randint(0, 4, num_symbols)
        symbols_complex = constellation[symbol_indices]
        symbols_i = np.real(symbols_complex)
        symbols_q = np.imag(symbols_complex)
    elif modulation == '16QAM':
        # 16-QAM constellation
        points = [-3, -1, 1, 3]
        i_vals = np.random.choice(points, num_symbols)
        q_vals = np.random.choice(points, num_symbols)
        symbols_i = i_vals / np.sqrt(10)
        symbols_q = q_vals / np.sqrt(10)
    else:
        raise ValueError(f"Unknown modulation: {modulation}")

    # Upsample to create waveform
    i_upsampled = np.zeros(num_symbols * sps)
    q_upsampled = np.zeros(num_symbols * sps)

    true_indices = []
    for i in range(num_symbols):
        idx = i * sps
        i_upsampled[idx] = symbols_i[i]
        q_upsampled[idx] = symbols_q[i]
        true_indices.append(idx)

    # Apply RRC pulse shaping
    rrc = rrc_filter(alpha=0.35, span=8, sps=sps)
    i_shaped = np.convolve(i_upsampled, rrc, mode='same')
    q_shaped = np.convolve(q_upsampled, rrc, mode='same')

    # Add noise
    signal_power = np.mean(i_shaped**2 + q_shaped**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise_i = np.sqrt(noise_power/2) * np.random.randn(len(i_shaped))
    noise_q = np.sqrt(noise_power/2) * np.random.randn(len(q_shaped))

    i_signal = i_shaped + noise_i
    q_signal = q_shaped + noise_q

    return i_signal, q_signal, np.array(true_indices)


def test_timing_recovery():
    """Test all timing recovery methods"""
    print("="*70)
    print("TESTING DSP FUNCTIONS")
    print("="*70)

    # Test parameters
    modulation = 'QPSK'
    num_symbols = 100
    sps = 2
    snr_db = 20

    print(f"\nGenerating test signal:")
    print(f"  Modulation: {modulation}")
    print(f"  Symbols: {num_symbols}")
    print(f"  Samples per symbol: {sps}")
    print(f"  SNR: {snr_db} dB")

    # Generate test signal
    i_signal, q_signal, true_indices = generate_test_signal(
        modulation=modulation,
        num_symbols=num_symbols,
        sps=sps,
        snr_db=snr_db
    )

    print(f"\nSignal generated:")
    print(f"  Length: {len(i_signal)} samples")
    print(f"  True symbol positions: {len(true_indices)} symbols")

    # Test each timing recovery method
    methods = ['simple_energy', 'simple_correlation', 'gardner', 'mueller_muller']

    results = {}

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing method: {method}")
        print(f"{'='*70}")

        try:
            symbols = extract_symbols(i_signal, q_signal, sps=sps, method=method)

            num_recovered = len(symbols['symbol_i'])
            print(f"  ✅ Success!")
            print(f"  Recovered symbols: {num_recovered}")
            print(f"  Expected symbols: {num_symbols}")
            print(f"  Recovery rate: {num_recovered/num_symbols*100:.1f}%")

            # Calculate timing accuracy (if we recovered similar number of symbols)
            if num_recovered > 0:
                # Compare recovered indices with true indices
                recovered_indices = symbols['symbol_indices']

                # Find closest matches
                errors = []
                for rec_idx in recovered_indices[:min(len(recovered_indices), len(true_indices))]:
                    closest_true = true_indices[np.argmin(np.abs(true_indices - rec_idx))]
                    errors.append(abs(rec_idx - closest_true))

                mean_error = np.mean(errors) if errors else float('inf')
                print(f"  Mean timing error: {mean_error:.2f} samples")

                results[method] = {
                    'success': True,
                    'num_recovered': num_recovered,
                    'mean_error': mean_error
                }
            else:
                results[method] = {'success': False}

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[method] = {'success': False, 'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for method, result in results.items():
        if result.get('success'):
            print(f"  {method:20s}: ✅ Recovered {result['num_recovered']:3d} symbols, "
                  f"timing error = {result['mean_error']:.2f} samples")
        else:
            print(f"  {method:20s}: ❌ Failed")

    return results


def test_visual_comparison():
    """Create visual comparison of timing recovery methods"""
    print(f"\n{'='*70}")
    print("CREATING VISUAL COMPARISON")
    print(f"{'='*70}")

    # Generate test signal
    i_signal, q_signal, true_indices = generate_test_signal(
        modulation='QPSK', num_symbols=50, sps=2, snr_db=15
    )

    methods = ['simple_energy', 'gardner']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DSP Processing Comparison - QPSK Signal', fontsize=16, fontweight='bold')

    # Plot raw constellation
    axes[0, 0].scatter(i_signal, q_signal, alpha=0.2, s=3, color='gray', label='Raw samples')
    axes[0, 0].scatter(i_signal[true_indices], q_signal[true_indices],
                      alpha=0.6, s=30, color='green', marker='x', label='True symbols')
    axes[0, 0].set_title('Raw Trajectory with True Symbols')
    axes[0, 0].set_xlabel('I')
    axes[0, 0].set_ylabel('Q')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # Plot time domain
    time_axis = np.arange(len(i_signal))
    axes[0, 1].plot(time_axis, i_signal, alpha=0.7, linewidth=0.8, label='I')
    axes[0, 1].plot(time_axis, q_signal, alpha=0.7, linewidth=0.8, label='Q')
    axes[0, 1].scatter(true_indices, i_signal[true_indices], s=20, color='red', marker='o', zorder=5)
    axes[0, 1].scatter(true_indices, q_signal[true_indices], s=20, color='red', marker='o', zorder=5)
    axes[0, 1].set_title('Time Domain with True Symbol Timing')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot recovered constellations for different methods
    for idx, method in enumerate(methods):
        ax = axes[1, idx]
        try:
            symbols = extract_symbols(i_signal, q_signal, sps=2, method=method)
            ax.scatter(symbols['symbol_i'], symbols['symbol_q'],
                      alpha=0.6, s=20, color='red', marker='o', label=f'Recovered ({len(symbols["symbol_i"])})')
            ax.scatter(i_signal[true_indices], q_signal[true_indices],
                      alpha=0.6, s=30, color='green', marker='x', label=f'True ({len(true_indices)})')
            ax.set_title(f'Recovered Symbols - {method}')
        except Exception as e:
            ax.text(0.5, 0.5, f'Failed:\n{str(e)}', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'Recovered Symbols - {method} (Failed)')

        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.tight_layout()

    # Save figure
    output_path = Path('dsp_test_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved visual comparison to: {output_path}")
    plt.close()


if __name__ == '__main__':
    # Run tests
    results = test_timing_recovery()
    test_visual_comparison()

    print(f"\n{'='*70}")
    print("✅ DSP TESTING COMPLETE")
    print(f"{'='*70}")
