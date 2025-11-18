# Optuna Hyperparameter Tuning for Transformer RawIQ - Jupyter Notebook Guide

This guide provides comprehensive instructions for using Optuna to perform hyperparameter tuning on the Transformer for Raw I/Q Signal Classification in a Jupyter Notebook environment.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Notebook Setup](#notebook-setup)
- [Implementation Steps](#implementation-steps)
- [Running the Study](#running-the-study)
- [Analyzing Results](#analyzing-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Optuna** is an automatic hyperparameter optimization framework that helps find optimal hyperparameters for machine learning models through:
- Efficient search algorithms (TPE, CMA-ES, Grid Search)
- Automatic pruning of unpromising trials
- Rich visualization capabilities
- Easy integration with PyTorch

**Goal**: Optimize hyperparameters for the Transformer-based Raw I/Q signal classifier to maximize validation accuracy.

**Key Difference from ViT**: This model includes additional hyperparameters for embedding configuration:
- `embedding_type`: 'segment' or 'conv1d'
- `segment_size`: Size of segments for segment-based embedding

---

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- Jupyter Lab or Jupyter Notebook
- Existing transformer_rawIQ project structure (see `transformer_rawIQ/README_RAWIQ.md`)

---

## Installation

### Install Required Packages

```bash
pip install optuna jupyterlab pandas numpy torch torchvision
pip install matplotlib seaborn plotly scikit-learn
pip install h5py tqdm
```

### Verify Installation

```python
import optuna
import torch
print(f"Optuna version: {optuna.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Notebook Setup

### 1. Create Optuna Tuning Notebook

Create a new Jupyter notebook in the `transformer_rawIQ` directory:

```bash
cd /path/to/transformer_rawIQ
jupyter lab
# Create new notebook: optuna_tuning_rawiq.ipynb
```

### 2. Import Libraries

```python
# Cell 1: Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)
import h5py
import numpy as np
import json
import os
import sys
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

print(f"✅ Optuna version: {optuna.__version__}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
```

### 3. Setup Project Paths

```python
# Cell 2: Setup paths and import project modules
# Add project root to path
notebook_dir = os.getcwd()
if notebook_dir not in sys.path:
    sys.path.append(notebook_dir)

# Import project-specific modules
from models.transformer_rawIQ import AMCTransformer
from dataloader.dataset import RawIQDataset  # Or your dataset class name
from dataloader.utils import split_data

print("✅ Project modules imported successfully")
```

---

## Implementation Steps

### Step 1: Configuration

```python
# Cell 3: Configuration
# Data paths
FILE_PATH = "/path/to/GOLD_XYZ_OSC.0001_1024.hdf5"
JSON_PATH = "/path/to/classes-fixed.json"

# Target modulations (19 classes)
TARGET_MODULATIONS = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'GMSK', 'OQPSK'
]
NUM_CLASSES = len(TARGET_MODULATIONS)

# Data split configuration
TRAIN_SIZE = 0.7
VALID_SIZE = 0.15
TEST_SIZE = 0.15
SPLIT_SEED = 42

# Optuna tuning configuration
N_TRIALS = 50               # Number of hyperparameter combinations to try
N_EPOCHS_PER_TRIAL = 20     # Train each trial for 20 epochs
BATCH_SIZE = 256
NUM_WORKERS = 4
PATIENCE = 5                # Early stopping patience for trials

# Fixed model parameters
SIGNAL_LENGTH = 1024
NUM_CHANNELS = 2  # I and Q channels

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Configuration loaded")
print(f"📊 Dataset: {FILE_PATH}")
print(f"🎯 Classes: {NUM_CLASSES}")
print(f"🔬 Trials: {N_TRIALS}")
print(f"📈 Epochs per trial: {N_EPOCHS_PER_TRIAL}")
print(f"🖥️ Device: {device}")
```

### Step 2: Data Preparation

```python
# Cell 4: Prepare dataset
print("📂 Preparing datasets...")

# Split data
train_indices, valid_indices, test_indices, label_map = split_data(
    file_path=FILE_PATH,
    json_path=JSON_PATH,
    target_modulations=TARGET_MODULATIONS,
    train_size=TRAIN_SIZE,
    valid_size=VALID_SIZE,
    test_size=TEST_SIZE,
    seed=SPLIT_SEED
)

# Create train dataset and calculate normalization stats
train_dataset = RawIQDataset(
    file_path=FILE_PATH,
    json_path=JSON_PATH,
    target_modulations=TARGET_MODULATIONS,
    mode='train',
    indices=train_indices,
    label_map=label_map,
    seed=SPLIT_SEED
)

norm_stats = train_dataset.get_normalization_stats()
print(f"📊 Normalization stats: {norm_stats}")

# Create validation dataset
valid_dataset = RawIQDataset(
    file_path=FILE_PATH,
    json_path=JSON_PATH,
    target_modulations=TARGET_MODULATIONS,
    mode='valid',
    indices=valid_indices,
    label_map=label_map,
    normalization_stats=norm_stats
)

print(f"✅ Train samples: {len(train_dataset):,}")
print(f"✅ Valid samples: {len(valid_dataset):,}")
```

### Step 3: Helper Functions

```python
# Cell 5: Define helper functions

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_data in dataloader:
        # Handle different dataset return formats
        if len(batch_data) == 3:
            iq_data, labels, _ = batch_data  # (iq, label, snr)
        else:
            iq_data, labels = batch_data  # (iq, label)

        iq_data, labels = iq_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(iq_data)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in dataloader:
            # Handle different dataset return formats
            if len(batch_data) == 3:
                iq_data, labels, _ = batch_data
            else:
                iq_data, labels = batch_data

            iq_data, labels = iq_data.to(device), labels.to(device)

            outputs = model(iq_data)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

print("✅ Helper functions defined")
```

### Step 4: Define Objective Function

```python
# Cell 6: Define Optuna objective function

def objective_rawiq(trial: Trial) -> float:
    """
    Optuna objective function for Transformer RawIQ hyperparameter tuning.

    Args:
        trial: Optuna trial object

    Returns:
        Best validation accuracy achieved
    """

    # 1. Suggest hyperparameters

    # Embedding configuration (unique to RawIQ transformer)
    embedding_type = trial.suggest_categorical("embedding_type", ["segment", "conv1d"])

    # Segment size only matters for segment embedding
    segment_size = None
    if embedding_type == "segment":
        segment_size = trial.suggest_categorical("segment_size", [16, 32, 64])

    # Model architecture
    d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
    n_head = trial.suggest_categorical("n_head", [4, 8, 16])
    n_layers = trial.suggest_int("n_layers", 2, 8)
    ffn_hidden = trial.suggest_categorical("ffn_hidden", [
        d_model * 2, d_model * 4, d_model * 8
    ])
    drop_prob = trial.suggest_float("drop_prob", 0.05, 0.4)

    # CLS token usage
    use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])

    # Optimizer hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    # Training hyperparameters
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # Constraint: n_head must divide d_model evenly
    if d_model % n_head != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 3. Initialize model
    model_params = {
        'num_channels': NUM_CHANNELS,
        'signal_length': SIGNAL_LENGTH,
        'num_classes': NUM_CLASSES,
        'embedding_type': embedding_type,
        'd_model': d_model,
        'n_head': n_head,
        'n_layers': n_layers,
        'ffn_hidden': ffn_hidden,
        'drop_prob': drop_prob,
        'use_cls_token': use_cls_token,
        'device': device
    }

    # Add segment_size only if using segment embedding
    if embedding_type == "segment" and segment_size is not None:
        model_params['segment_size'] = segment_size

    try:
        model = AMCTransformer(**model_params).to(device)
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    # 4. Setup optimizer and loss
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.99)
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 5. Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(N_EPOCHS_PER_TRIAL):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)

        # Track best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Report intermediate value to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning (stop unpromising trials early)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if patience_counter >= PATIENCE:
            break

    # Clean up
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_acc

print("✅ Objective function defined")
```

---

## Running the Study

### Execute Optimization

```python
# Cell 7: Create and run Optuna study

# Create study
study = optuna.create_study(
    study_name="rawiq_amc_tuning",
    direction="maximize",  # Maximize validation accuracy
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials
        n_warmup_steps=5,      # Don't prune first 5 epochs
        interval_steps=1
    ),
    sampler=optuna.samplers.TPESampler(seed=42)  # Use TPE algorithm
)

print(f"🚀 Starting Optuna optimization with {N_TRIALS} trials...")
print(f"⏱️ This may take several hours depending on your hardware.")
print(f"💡 Trials with segment embedding will be faster than conv1d")
print("-" * 70)

# Run optimization
study.optimize(
    objective_rawiq,
    n_trials=N_TRIALS,
    timeout=None,  # No timeout
    catch=(Exception,),  # Catch exceptions and continue
    show_progress_bar=True
)

print("\n" + "=" * 70)
print("✅ Optimization completed!")
print("=" * 70)
```

### Display Best Results

```python
# Cell 8: Display best trial results

print("\n🏆 BEST TRIAL RESULTS")
print("=" * 70)

trial = study.best_trial
print(f"🎯 Best Validation Accuracy: {trial.value:.2f}%")
print(f"📊 Trial Number: {trial.number}")
print(f"\n⚙️ Best Hyperparameters:")
print("-" * 70)

for key, value in trial.params.items():
    print(f"  {key:20s}: {value}")

print("=" * 70)

# Save best parameters to JSON
best_params_path = "result/best_hyperparameters_rawiq.json"
os.makedirs("result", exist_ok=True)

with open(best_params_path, 'w') as f:
    json.dump({
        'best_accuracy': trial.value,
        'trial_number': trial.number,
        'params': trial.params
    }, f, indent=4)

print(f"💾 Best parameters saved to: {best_params_path}")
```

---

## Analyzing Results

### Comprehensive Visualization

```python
# Cell 9: Visualize optimization history

# Plot 1: Optimization History
fig1 = plot_optimization_history(study)
fig1.update_layout(title="Optimization History - RawIQ Transformer", width=900, height=500)
fig1.show()
fig1.write_html("result/optuna_optimization_history_rawiq.html")

# Plot 2: Parameter Importances
fig2 = plot_param_importances(study)
fig2.update_layout(title="Hyperparameter Importances", width=900, height=500)
fig2.show()
fig2.write_html("result/optuna_param_importances_rawiq.html")

# Plot 3: Parallel Coordinate Plot
fig3 = plot_parallel_coordinate(study)
fig3.update_layout(title="Parallel Coordinate Plot", width=1000, height=600)
fig3.show()
fig3.write_html("result/optuna_parallel_coordinate_rawiq.html")

# Plot 4: Slice Plot
fig4 = plot_slice(study)
fig4.update_layout(width=1200, height=800)
fig4.show()
fig4.write_html("result/optuna_slice_plot_rawiq.html")

# Plot 5: Contour Plot (for parameter interactions)
fig5 = plot_contour(study, params=["d_model", "n_layers"])
fig5.update_layout(title="Contour Plot: d_model vs n_layers", width=800, height=600)
fig5.show()
fig5.write_html("result/optuna_contour_rawiq.html")

print("✅ All plots saved to result/ directory")
```

### Embedding Type Analysis

```python
# Cell 10: Compare embedding types

print("\n📊 EMBEDDING TYPE COMPARISON")
print("=" * 70)

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# Separate by embedding type
segment_trials = [t for t in completed_trials if t.params.get('embedding_type') == 'segment']
conv1d_trials = [t for t in completed_trials if t.params.get('embedding_type') == 'conv1d']

if segment_trials:
    segment_accs = [t.value for t in segment_trials]
    print(f"\n🔹 Segment Embedding:")
    print(f"   Trials: {len(segment_trials)}")
    print(f"   Best:   {max(segment_accs):.2f}%")
    print(f"   Mean:   {np.mean(segment_accs):.2f}%")
    print(f"   Std:    {np.std(segment_accs):.2f}%")

if conv1d_trials:
    conv1d_accs = [t.value for t in conv1d_trials]
    print(f"\n🔸 Conv1D Embedding:")
    print(f"   Trials: {len(conv1d_trials)}")
    print(f"   Best:   {max(conv1d_accs):.2f}%")
    print(f"   Mean:   {np.mean(conv1d_accs):.2f}%")
    print(f"   Std:    {np.std(conv1d_accs):.2f}%")

# Plot comparison
if segment_trials and conv1d_trials:
    plt.figure(figsize=(10, 6))
    plt.boxplot([segment_accs, conv1d_accs], labels=['Segment', 'Conv1D'])
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Embedding Type Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig('result/embedding_comparison_rawiq.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Comparison plot saved")

print("=" * 70)
```

### Study Statistics

```python
# Cell 11: Display study statistics

print("\n📊 STUDY STATISTICS")
print("=" * 70)

# Trial statistics
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

print(f"Total trials:     {len(study.trials)}")
print(f"Completed trials: {len(completed_trials)}")
print(f"Pruned trials:    {len(pruned_trials)}")
print(f"Failed trials:    {len(failed_trials)}")

# Pruning efficiency
if pruned_trials:
    pruning_rate = len(pruned_trials) / len(study.trials) * 100
    print(f"Pruning rate:     {pruning_rate:.1f}%")

# Top 10 trials
print(f"\n🏅 TOP 10 TRIALS")
print("-" * 70)

top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
for i, trial in enumerate(top_trials, 1):
    emb_type = trial.params.get('embedding_type', 'N/A')
    seg_size = trial.params.get('segment_size', 'N/A')
    print(f"{i:2d}. Trial {trial.number:3d}: {trial.value:.2f}% "
          f"(emb={emb_type}, seg={seg_size})")

# Parameter distribution summary
print(f"\n📈 PARAMETER RANGES (Completed Trials)")
print("-" * 70)

if completed_trials:
    param_names = list(completed_trials[0].params.keys())
    for param in param_names:
        values = [t.params[param] for t in completed_trials]
        if isinstance(values[0], (int, float)):
            print(f"  {param:20s}: [{min(values)}, {max(values)}]")
        else:
            unique_vals = set(values)
            print(f"  {param:20s}: {unique_vals}")

print("=" * 70)
```

### Save Study

```python
# Cell 12: Save study for later analysis

# Save study to database and CSV
study_path = "result/optuna_study_rawiq.db"
study.trials_dataframe().to_csv("result/optuna_trials_rawiq.csv", index=False)

print(f"💾 Study trials saved to: result/optuna_trials_rawiq.csv")
print(f"💾 Study database: sqlite:///{study_path}")
print("\nℹ️ To load study later:")
print(f"   study = optuna.load_study(")
print(f"       study_name='rawiq_amc_tuning',")
print(f"       storage='sqlite:///{study_path}'")
print(f"   )")
```

---

## Best Practices

### 1. Start with Segment Embedding
- Segment embedding is faster and uses less memory
- Good for initial exploration
- Once you find promising hyperparameters, try conv1d for fine-grained modeling

### 2. Hyperparameter Ranges
- **embedding_type**: Start with 'segment', add 'conv1d' if time allows
- **segment_size**: [16, 32, 64] for segment embedding
- **d_model**: Start with [64, 128, 256], expand if needed
- **n_layers**: 2-8 layers typically sufficient
- **use_cls_token**: Try both True and False

### 3. Memory Management
```python
# Monitor GPU memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Clear cache between trials
torch.cuda.empty_cache()
gc.collect()
```

### 4. Segment Size Impact
- **Smaller segments (16)**: More tokens, finer detail, slower, more memory
- **Larger segments (64)**: Fewer tokens, coarser detail, faster, less memory
- Trade-off between detail and efficiency

### 5. Conv1D vs Segment
```python
# Conv1D: 1024 tokens (one per time step)
# Segment (64): 16 tokens (1024/64)
# Segment is ~64x faster but may lose fine temporal detail
```

---

## Troubleshooting

### Issue 1: Out of Memory with Conv1D
**Solution:**
```python
# Force segment embedding only
embedding_type = trial.suggest_categorical("embedding_type", ["segment"])

# Or reduce batch size for conv1d
if embedding_type == "conv1d":
    BATCH_SIZE = 64  # Much smaller for conv1d
```

### Issue 2: Trials Too Slow
**Solution:**
```python
# Prioritize segment embedding
embedding_type = trial.suggest_categorical("embedding_type", ["segment"])

# Use larger segment sizes
segment_size = trial.suggest_categorical("segment_size", [64])

# Reduce model size
d_model = trial.suggest_categorical("d_model", [64, 128])
n_layers = trial.suggest_int("n_layers", 2, 4)
```

### Issue 3: Poor Performance with CLS Token
**Solution:**
```python
# Try global average pooling instead
use_cls_token = False

# Or make it a hyperparameter
use_cls_token = trial.suggest_categorical("use_cls_token", [True, False])
```

### Issue 4: Segment Size Conflicts
**Solution:**
```python
# Ensure signal_length is divisible by segment_size
valid_segment_sizes = []
for size in [16, 32, 64, 128]:
    if SIGNAL_LENGTH % size == 0:
        valid_segment_sizes.append(size)

segment_size = trial.suggest_categorical("segment_size", valid_segment_sizes)
```

### Issue 5: Model Architecture Mismatch
**Solution:**
```python
# Check your model's __init__ parameters
# Ensure model_params dict matches exactly

# Common issues:
# - Missing parameters
# - Wrong parameter names
# - Incompatible parameter combinations
```

---

## Advanced Techniques

### 1. Multi-Objective Optimization
Optimize for both accuracy and inference speed:

```python
def objective_multi(trial):
    # ... train model ...

    # Measure inference time
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            dummy_input = torch.randn(1, NUM_CHANNELS, SIGNAL_LENGTH).to(device)
            _ = model(dummy_input)
    inference_time = (time.time() - start) / 100

    # Return both metrics
    return val_acc, -inference_time  # Negative for minimization

# Create multi-objective study
study = optuna.create_study(
    directions=["maximize", "maximize"],  # Max accuracy, min time
    sampler=optuna.samplers.NSGAIISampler()
)
```

### 2. Conditional Hyperparameters
```python
# Tune embedding-specific parameters
if embedding_type == "segment":
    segment_size = trial.suggest_categorical("segment_size", [16, 32, 64])
    # Segment-specific: Try different aggregation methods
    aggregation = trial.suggest_categorical("aggregation", ["mean", "max", "attention"])
elif embedding_type == "conv1d":
    # Conv1D-specific: Try different kernel sizes
    kernel_size = trial.suggest_int("kernel_size", 3, 9, step=2)
```

### 3. Learning Rate Scheduling
```python
# Add scheduler as hyperparameter
scheduler_type = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])

if scheduler_type == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
elif scheduler_type == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_PER_TRIAL
    )
```

---

## Next Steps

### 1. Train Final Model
Use best hyperparameters for full training:

```python
# Load best parameters
with open("result/best_hyperparameters_rawiq.json", 'r') as f:
    best_params = json.load(f)['params']

# Train with full epochs (100-300)
# See transformer_rawIQ/training/train.py
```

### 2. Compare Embedding Types
If both performed well:
- Use **segment** for faster inference in production
- Use **conv1d** for maximum accuracy when speed is not critical

### 3. Ensemble Methods
```python
# Train multiple models with top-k hyperparameters
# Ensemble predictions for better robustness
```

### 4. Test Set Evaluation
```python
# Evaluate best model on held-out test set
# Report final performance metrics
```

---

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Optuna Examples**: https://github.com/optuna/optuna-examples
- **Transformer Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Project README**: `transformer_rawIQ/README_RAWIQ.md`
- **Architecture Details**: `transformer_rawIQ/ARCHITECTURE.md`

---

## Performance Expectations

Based on typical results:

| Embedding | Accuracy | Speed | Memory |
|-----------|----------|-------|--------|
| Segment (64) | 80-85% | Fast | Low |
| Segment (32) | 82-87% | Medium | Medium |
| Segment (16) | 84-88% | Slow | High |
| Conv1D | 85-90% | Very Slow | Very High |

*Note: Actual performance depends on dataset and configuration*

---

## Support

For issues specific to:
- **Optuna**: [Optuna GitHub Issues](https://github.com/optuna/optuna/issues)
- **Transformer Model**: See `transformer_rawIQ/ARCHITECTURE.md`
- **Training**: See `transformer_rawIQ/training/train.py`

---

**Happy Tuning! 🚀**
