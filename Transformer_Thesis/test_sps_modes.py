"""
Quick comparison test: SPS=1 vs SPS=2 to demonstrate the difference
Shows how RadioML 2018.01A should be processed with sps=1
"""

import numpy as np
import matplotlib.pyplot as plt
from plot_preprocessing_signal import extract_symbols

# Generate a simple QPSK signal at 1 sample per symbol (like RadioML)
np.random.seed(42)
num_symbols = 100

# QPSK constellation
constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
symbol_indices = np.random.randint(0, 4, num_symbols)
symbols_complex = constellation[symbol_indices]

# Add some noise
snr_db = 15
signal_power = np.mean(np.abs(symbols_complex)**2)
noise_power = signal_power / (10**(snr_db/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))
signal_complex = symbols_complex + noise

i_signal = np.real(signal_complex)
q_signal = np.imag(signal_complex)

# Test both modes
print("="*70)
print("COMPARISON: SPS=1 vs SPS=2 Processing")
print("="*70)

# Mode 1: SPS=1 (correct for RadioML)
print("\n🟢 Mode 1: SPS=1 (RadioML 2018.01A mode)")
print("-"*70)
result_sps1 = extract_symbols(i_signal, q_signal, sps=1)
print(f"  Input samples: {len(i_signal)}")
print(f"  Output symbols: {len(result_sps1['symbol_i'])}")
print(f"  Matched filtering: {'NO' if np.array_equal(result_sps1['filtered_i'], i_signal) else 'YES'}")
print(f"  Timing recovery: NO (bypass mode)")
print(f"  ✅ Correct: Every sample IS a symbol")

# Mode 2: SPS=2 (wrong for RadioML, right for oversampled signals)
print("\n🔴 Mode 2: SPS=2 (Oversampled signal mode - WRONG for RadioML)")
print("-"*70)
try:
    result_sps2 = extract_symbols(i_signal, q_signal, sps=2, method='simple_correlation')
    print(f"  Input samples: {len(i_signal)}")
    print(f"  Output symbols: {len(result_sps2['symbol_i'])}")
    print(f"  Matched filtering: YES")
    print(f"  Timing recovery: YES (simple_correlation)")
    print(f"  ❌ Wrong: Lost ~50% of symbols ({len(result_sps2['symbol_i'])}/{len(i_signal)})")
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Correct vs Incorrect Processing of RadioML Data', fontsize=16, fontweight='bold')

# Plot 1: SPS=1 constellation (CORRECT)
axes[0, 0].scatter(result_sps1['symbol_i'], result_sps1['symbol_q'],
                  alpha=0.6, s=30, color='green', marker='o',
                  edgecolors='darkgreen', linewidths=0.8)
axes[0, 0].set_title(f'✅ CORRECT: SPS=1\n({len(result_sps1["symbol_i"])} symbols preserved)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('I')
axes[0, 0].set_ylabel('Q')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
axes[0, 0].axis('equal')

# Plot 2: SPS=2 constellation (WRONG)
if len(result_sps2['symbol_i']) > 0:
    axes[0, 1].scatter(result_sps2['symbol_i'], result_sps2['symbol_q'],
                      alpha=0.6, s=30, color='red', marker='x',
                      linewidths=1.5)
    axes[0, 1].set_title(f'❌ WRONG: SPS=2\n({len(result_sps2["symbol_i"])} symbols, lost {len(i_signal)-len(result_sps2["symbol_i"])})',
                        fontsize=12, fontweight='bold')
else:
    axes[0, 1].text(0.5, 0.5, '❌ WRONG\nProcessing failed',
                   ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
    axes[0, 1].set_title('❌ WRONG: SPS=2', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('I')
axes[0, 1].set_ylabel('Q')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
axes[0, 1].axis('equal')

# Plot 3: Time domain comparison
time_axis = np.arange(len(i_signal))
axes[1, 0].plot(time_axis, i_signal, alpha=0.7, linewidth=0.8, label='I signal', color='blue')
axes[1, 0].scatter(result_sps1['symbol_indices'], result_sps1['symbol_i'],
                  s=40, color='green', marker='o', zorder=5, label='SPS=1 samples', edgecolors='darkgreen', linewidths=0.8)
axes[1, 0].set_title('Time Domain - SPS=1 (all samples used)', fontsize=11)
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Data summary
summary_text = f"""
RadioML 2018.01A Dataset Characteristics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ CORRECT Processing (SPS=1):
   • No pulse shaping
   • No oversampling
   • 1 sample = 1 symbol
   • 1024 samples = 1024 symbols
   • NO matched filtering needed
   • NO timing recovery needed

❌ WRONG Processing (SPS=2):
   • Assumes oversampling (WRONG!)
   • Tries to recover timing (unnecessary!)
   • Discards ~50% of data
   • Applies matched filtering (wrong!)

Conclusion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For RadioML 2018.01A, ALWAYS use:
    --sps 1

This treats each sample as a symbol.
"""
axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('sps_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print("✅ Comparison plot saved to: sps_comparison.png")
print(f"{'='*70}")
plt.close()
