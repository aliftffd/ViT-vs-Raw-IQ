# AMC Model Comparison Results

This directory contains comprehensive comparison results between two Automatic Modulation Classification approaches:

1. **ViT (Vision Transformer)**: Located at `ViT/result/checkpoints/production_v2/`
2. **Transformer (Raw IQ)**: Located at `transformer_rawIQ/result/checkpoints/exp_L9_H8_F1024_W1e-3/`

## Generated Files

### CSV Files
- `summary_comparison.csv` - High-level summary comparing overall accuracy and SNR-specific accuracies
- `detailed_comparison.csv` - Detailed per-class comparison including precision, recall, and F1-scores for all 19 modulation types

### Visualizations

#### 1. overall_comparison.png
Four-panel visualization showing:
- Overall accuracy comparison
- Average metrics (Precision, Recall, F1-Score)
- SNR performance curves
- Pie chart of improved/degraded modulations

#### 2. snr_comparison.png
Bar chart comparing model performance across different Signal-to-Noise Ratio (SNR) levels:
- SNR -8 dB (low quality signal)
- SNR 0 dB (medium quality signal)
- SNR +8 dB (high quality signal)

#### 3. per_class_metrics.png
Three subplots showing precision, recall, and F1-score for each of the 19 modulation types

#### 4. f1_difference_heatmap.png
Horizontal bar chart showing F1-score improvements (positive = Transformer better, negative = ViT better)

## Key Findings

### Overall Performance
- **Transformer (Raw IQ)** achieves **1.42% higher overall accuracy** (63.44% vs 62.02%)
- Most significant improvement at SNR 0 dB: **+4.77%**

### Top Improvements (Transformer vs ViT)
1. **64QAM**: +18.66% F1-score improvement
2. **16APSK**: +16.35% F1-score improvement
3. **8ASK**: +6.23% F1-score improvement

### Notable Degradations
1. **OQPSK**: -8.71% F1-score
2. **32QAM**: -8.39% F1-score
3. **QPSK**: -8.14% F1-score

### SNR-Specific Insights
- **Low SNR (-8 dB)**: Minimal improvement (+0.42%)
- **Medium SNR (0 dB)**: Significant improvement (+4.77%)
- **High SNR (+8 dB)**: Moderate improvement (+2.47%)

## How to Regenerate

Run the comparison script from the project root:

```bash
python compare_models.py
```

This will regenerate all visualizations and CSV files.

## Interpretation

### When to Use ViT
- QPSK, OQPSK, 32QAM modulations
- Scenarios where precision is critical
- When working with PSK family modulations

### When to Use Transformer (Raw IQ)
- 64QAM, 16APSK, 8ASK modulations
- Medium SNR conditions (0 dB)
- When working with APSK and high-order QAM modulations
- Overall better average performance

## Notes

- Both models show excellent performance at high SNR (+8 dB): >96% accuracy
- Both models struggle at low SNR (-8 dB): ~13-14% accuracy
- The Transformer approach shows more balanced performance across different modulation families
- ViT excels at specific PSK modulation types
