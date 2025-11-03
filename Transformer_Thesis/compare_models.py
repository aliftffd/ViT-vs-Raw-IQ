#!/usr/bin/env python3
"""
Comparison script for two AMC approaches:
1. Vision Transformer (ViT)
2. Transformer with Raw IQ sequences

This script parses classification reports and generates comprehensive comparisons.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class ClassificationReportParser:
    """Parser for classification report text files."""

    def __init__(self, report_path: str):
        self.report_path = Path(report_path)
        self.overall_accuracy = None
        self.snr_accuracies = {}
        self.class_metrics = {}
        self.parse_report()

    def parse_report(self):
        """Parse the classification report."""
        with open(self.report_path, 'r') as f:
            content = f.read()

        # Extract overall accuracy
        overall_match = re.search(r'Overall Accuracy:\s+([\d.]+)%', content)
        if overall_match:
            self.overall_accuracy = float(overall_match.group(1))

        # Extract SNR accuracies
        snr_matches = re.findall(r'SNR\s+([-+]\d+)\s+dB:\s+([\d.]+)%', content)
        for snr, acc in snr_matches:
            self.snr_accuracies[int(snr)] = float(acc)

        # Extract per-class metrics
        class_pattern = r'^\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
        for line in content.split('\n'):
            match = re.match(class_pattern, line)
            if match:
                mod_type, precision, recall, f1, support = match.groups()
                if mod_type not in ['accuracy', 'macro', 'weighted']:
                    self.class_metrics[mod_type] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1-score': float(f1),
                        'support': int(support)
                    }


class ModelComparison:
    """Comparison between two AMC models."""

    def __init__(self, vit_report_path: str, transformer_report_path: str,
                 output_dir: str = "comparison_results"):
        self.vit_parser = ClassificationReportParser(vit_report_path)
        self.transformer_parser = ClassificationReportParser(transformer_report_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Model names for display
        self.vit_name = "ViT (Vision Transformer)"
        self.transformer_name = "Transformer (Raw IQ)"

    def create_summary_table(self) -> pd.DataFrame:
        """Create summary comparison table."""
        summary_data = {
            'Metric': ['Overall Accuracy (%)', 'SNR -8 dB (%)', 'SNR 0 dB (%)', 'SNR +8 dB (%)'],
            self.vit_name: [
                self.vit_parser.overall_accuracy,
                self.vit_parser.snr_accuracies.get(-8, 0),
                self.vit_parser.snr_accuracies.get(0, 0),
                self.vit_parser.snr_accuracies.get(8, 0)
            ],
            self.transformer_name: [
                self.transformer_parser.overall_accuracy,
                self.transformer_parser.snr_accuracies.get(-8, 0),
                self.transformer_parser.snr_accuracies.get(0, 0),
                self.transformer_parser.snr_accuracies.get(8, 0)
            ]
        }

        df = pd.DataFrame(summary_data)
        df['Difference'] = df[self.transformer_name] - df[self.vit_name]
        df['Improvement (%)'] = (df['Difference'] / df[self.vit_name] * 100).round(2)

        return df

    def plot_snr_comparison(self):
        """Plot SNR-wise accuracy comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        snr_values = sorted(self.vit_parser.snr_accuracies.keys())
        vit_accs = [self.vit_parser.snr_accuracies[snr] for snr in snr_values]
        trans_accs = [self.transformer_parser.snr_accuracies[snr] for snr in snr_values]

        x = np.arange(len(snr_values))
        width = 0.35

        bars1 = ax.bar(x - width/2, vit_accs, width, label=self.vit_name, alpha=0.8)
        bars2 = ax.bar(x + width/2, trans_accs, width, label=self.transformer_name, alpha=0.8)

        ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Comparison Across Different SNR Levels', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{snr:+d}' for snr in snr_values])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'snr_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved SNR comparison plot to {self.output_dir / 'snr_comparison.png'}")
        plt.close()

    def plot_per_class_metrics(self):
        """Plot per-class F1-score comparison."""
        # Get all modulation types
        mod_types = sorted(set(self.vit_parser.class_metrics.keys()) |
                          set(self.transformer_parser.class_metrics.keys()))

        # Create subplots for different metrics
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        metrics = ['precision', 'recall', 'f1-score']

        for ax, metric in zip(axes, metrics):
            vit_values = [self.vit_parser.class_metrics.get(mod, {}).get(metric, 0) * 100
                         for mod in mod_types]
            trans_values = [self.transformer_parser.class_metrics.get(mod, {}).get(metric, 0) * 100
                           for mod in mod_types]

            x = np.arange(len(mod_types))
            width = 0.35

            bars1 = ax.bar(x - width/2, vit_values, width, label=self.vit_name, alpha=0.8)
            bars2 = ax.bar(x + width/2, trans_values, width, label=self.transformer_name, alpha=0.8)

            ax.set_ylabel(f'{metric.title()} (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.title()} Comparison by Modulation Type',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(mod_types, rotation=45, ha='right')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 100])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics plot to {self.output_dir / 'per_class_metrics.png'}")
        plt.close()

    def plot_f1_difference_heatmap(self):
        """Plot heatmap showing F1-score differences."""
        mod_types = sorted(set(self.vit_parser.class_metrics.keys()) |
                          set(self.transformer_parser.class_metrics.keys()))

        differences = []
        for mod in mod_types:
            vit_f1 = self.vit_parser.class_metrics.get(mod, {}).get('f1-score', 0)
            trans_f1 = self.transformer_parser.class_metrics.get(mod, {}).get('f1-score', 0)
            diff = (trans_f1 - vit_f1) * 100
            differences.append(diff)

        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Modulation': mod_types,
            'F1-Score Difference (%)': differences
        })
        df = df.sort_values('F1-Score Difference (%)')

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in df['F1-Score Difference (%)']]
        bars = ax.barh(df['Modulation'], df['F1-Score Difference (%)'], color=colors, alpha=0.7)

        ax.set_xlabel('F1-Score Difference (%)\n(Positive = Transformer better, Negative = ViT better)',
                     fontsize=11, fontweight='bold')
        ax.set_title('F1-Score Improvement: Transformer vs ViT', fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.2 if width > 0 else width - 0.2
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%',
                   ha='left' if width > 0 else 'right',
                   va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'f1_difference_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved F1-score difference plot to {self.output_dir / 'f1_difference_heatmap.png'}")
        plt.close()

    def create_detailed_comparison_table(self) -> pd.DataFrame:
        """Create detailed per-class comparison table."""
        mod_types = sorted(set(self.vit_parser.class_metrics.keys()) |
                          set(self.transformer_parser.class_metrics.keys()))

        rows = []
        for mod in mod_types:
            vit_metrics = self.vit_parser.class_metrics.get(mod, {})
            trans_metrics = self.transformer_parser.class_metrics.get(mod, {})

            row = {
                'Modulation': mod,
                'ViT Precision': vit_metrics.get('precision', 0) * 100,
                'Trans Precision': trans_metrics.get('precision', 0) * 100,
                'ViT Recall': vit_metrics.get('recall', 0) * 100,
                'Trans Recall': trans_metrics.get('recall', 0) * 100,
                'ViT F1-Score': vit_metrics.get('f1-score', 0) * 100,
                'Trans F1-Score': trans_metrics.get('f1-score', 0) * 100,
            }

            # Calculate improvements
            row['Precision Diff'] = row['Trans Precision'] - row['ViT Precision']
            row['Recall Diff'] = row['Trans Recall'] - row['ViT Recall']
            row['F1 Diff'] = row['Trans F1-Score'] - row['ViT F1-Score']

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_overall_comparison(self):
        """Create overall comparison visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall accuracy comparison
        models = [self.vit_name, self.transformer_name]
        accuracies = [self.vit_parser.overall_accuracy, self.transformer_parser.overall_accuracy]
        colors = ['#3498db', '#e74c3c']

        bars = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Overall Accuracy Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylim([0, 100])
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Average metrics comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        vit_avg = [
            np.mean([m['precision'] for m in self.vit_parser.class_metrics.values()]) * 100,
            np.mean([m['recall'] for m in self.vit_parser.class_metrics.values()]) * 100,
            np.mean([m['f1-score'] for m in self.vit_parser.class_metrics.values()]) * 100
        ]
        trans_avg = [
            np.mean([m['precision'] for m in self.transformer_parser.class_metrics.values()]) * 100,
            np.mean([m['recall'] for m in self.transformer_parser.class_metrics.values()]) * 100,
            np.mean([m['f1-score'] for m in self.transformer_parser.class_metrics.values()]) * 100
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width/2, vit_avg, width, label=self.vit_name, alpha=0.7, color='#3498db')
        ax2.bar(x + width/2, trans_avg, width, label=self.transformer_name, alpha=0.7, color='#e74c3c')
        ax2.set_ylabel('Score (%)', fontweight='bold')
        ax2.set_title('Average Metrics Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 100])

        # 3. SNR performance line plot
        snr_values = sorted(self.vit_parser.snr_accuracies.keys())
        vit_snr = [self.vit_parser.snr_accuracies[snr] for snr in snr_values]
        trans_snr = [self.transformer_parser.snr_accuracies[snr] for snr in snr_values]

        ax3.plot(snr_values, vit_snr, marker='o', linewidth=2,
                label=self.vit_name, color='#3498db')
        ax3.plot(snr_values, trans_snr, marker='s', linewidth=2,
                label=self.transformer_name, color='#e74c3c')
        ax3.set_xlabel('SNR (dB)', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.set_title('SNR Performance Curve', fontweight='bold', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])

        # 4. Number of better/worse/equal classes
        better_count = 0
        worse_count = 0
        equal_count = 0

        for mod in self.vit_parser.class_metrics.keys():
            vit_f1 = self.vit_parser.class_metrics.get(mod, {}).get('f1-score', 0)
            trans_f1 = self.transformer_parser.class_metrics.get(mod, {}).get('f1-score', 0)
            diff = trans_f1 - vit_f1

            if abs(diff) < 0.001:
                equal_count += 1
            elif diff > 0:
                better_count += 1
            else:
                worse_count += 1

        categories = ['Better', 'Worse', 'Equal']
        counts = [better_count, worse_count, equal_count]
        colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']

        wedges, texts, autotexts = ax4.pie(counts, labels=categories, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90)
        ax4.set_title(f'Transformer vs ViT\n(F1-Score Comparison by Class)',
                     fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved overall comparison plot to {self.output_dir / 'overall_comparison.png'}")
        plt.close()

    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("AUTOMATIC MODULATION CLASSIFICATION - MODEL COMPARISON")
        print("="*80)
        print(f"\nModel 1: {self.vit_name}")
        print(f"Model 2: {self.transformer_name}")
        print("\n" + "-"*80)

        # Summary table
        print("\nSUMMARY COMPARISON:")
        print("-"*80)
        summary_df = self.create_summary_table()
        print(summary_df.to_string(index=False))
        summary_df.to_csv(self.output_dir / 'summary_comparison.csv', index=False)

        # Detailed table
        print("\n" + "-"*80)
        print("\nDETAILED PER-CLASS COMPARISON:")
        print("-"*80)
        detailed_df = self.create_detailed_comparison_table()
        print(detailed_df.to_string(index=False))
        detailed_df.to_csv(self.output_dir / 'detailed_comparison.csv', index=False)

        # Key insights
        print("\n" + "-"*80)
        print("\nKEY INSIGHTS:")
        print("-"*80)

        overall_diff = self.transformer_parser.overall_accuracy - self.vit_parser.overall_accuracy
        print(f"1. Overall Accuracy Improvement: {overall_diff:+.2f}%")

        for snr in sorted(self.vit_parser.snr_accuracies.keys()):
            diff = self.transformer_parser.snr_accuracies[snr] - self.vit_parser.snr_accuracies[snr]
            print(f"2. SNR {snr:+d} dB Improvement: {diff:+.2f}%")

        # Best and worst improvements
        improvements = detailed_df[['Modulation', 'F1 Diff']].sort_values('F1 Diff', ascending=False)
        print(f"\n3. Top 3 Improved Modulations (F1-Score):")
        for idx, row in improvements.head(3).iterrows():
            print(f"   - {row['Modulation']}: {row['F1 Diff']:+.2f}%")

        print(f"\n4. Top 3 Degraded Modulations (F1-Score):")
        for idx, row in improvements.tail(3).iterrows():
            print(f"   - {row['Modulation']}: {row['F1 Diff']:+.2f}%")

        print("\n" + "="*80)
        print(f"\nAll results saved to: {self.output_dir.absolute()}")
        print("="*80 + "\n")

    def run_comparison(self):
        """Run complete comparison analysis."""
        print("\nRunning comprehensive model comparison...")

        self.generate_report()
        self.plot_overall_comparison()
        self.plot_snr_comparison()
        self.plot_per_class_metrics()
        self.plot_f1_difference_heatmap()

        print("\nComparison complete! Check the output directory for visualizations.")


def main():
    """Main function."""
    # Define paths
    vit_report = "ViT/result/checkpoints/production_v2/evaluation/test_classification_report.txt"
    transformer_report = "transformer_rawIQ/result/checkpoints/exp_L9_H8_F1024_W1e-3/evaluation/test_classification_report.txt"

    # Create comparison object
    comparison = ModelComparison(
        vit_report_path=vit_report,
        transformer_report_path=transformer_report,
        output_dir="comparison_results"
    )

    # Run comparison
    comparison.run_comparison()


if __name__ == "__main__":
    main()
