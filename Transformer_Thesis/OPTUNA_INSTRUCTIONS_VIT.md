# Optuna Hyperparameter Tuning for ViT - Jupyter Notebook Guide

This guide provides step-by-step instructions for using Optuna to perform hyperparameter tuning on the Vision Transformer (ViT) model for Automatic Modulation Classification in a Jupyter Notebook environment.

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

**Optuna** is an automatic hyperparameter optimization framework designed for machine learning. It provides:
- Efficient search algorithms (TPE, CMA-ES, Grid Search)
- Pruning strategies to stop unpromising trials early
- Visualization tools for analysis
- Easy integration with existing training code

**Goal**: Find optimal hyperparameters to maximize validation accuracy for the ViT-based AMC model.

---

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- Jupyter Lab or Jupyter Notebook
- Existing ViT project structure (see `ViT/README_ViT.md`)

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
print(f"Optuna version: {optuna.__version__}")
```

---

## Notebook Setup

### 1. Create Optuna Tuning Notebook

Create a new Jupyter notebook in the `ViT` directory:

```bash
cd /path/to/ViT
jupyter lab
# Create new notebook: optuna_tuning_vit.ipynb
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
    plot_slice
)
import h5py
import numpy as np
import json
import os
import sys
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"✅ Optuna version: {optuna.__version__}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
```

### 3. Setup Project Paths

```python
# Cell 2: Setup paths and import project modules
# Add project root to path
notebook_dir = os.getcwd()
if notebook_dir not in sys.path:
    sys.path.append(notebook_dir)

# Import project-specific modules
from models.amc_transformer import AMCTransformer
from dataloader.dataset import SingleStreamImageDataset
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
N_EPOCHS_PER_TRIAL = 20     # Train each trial for 20 epochs (faster than full training)
BATCH_SIZE = 256
NUM_WORKERS = 4
PATIENCE = 5                # Early stopping patience for trials

# Fixed model parameters
IN_CHANNELS = 1
IMG_SIZE_H = 32
IMG_SIZE_W = 64

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
train_dataset = SingleStreamImageDataset(
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
valid_dataset = SingleStreamImageDataset(
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

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
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

def objective_vit(trial: Trial) -> float:
    """
    Optuna objective function for ViT hyperparameter tuning.

    Args:
        trial: Optuna trial object

    Returns:
        Best validation accuracy achieved
    """

    # 1. Suggest hyperparameters
    d_model = trial.suggest_categorical("d_model", [64, 128, 256, 512])
    n_head = trial.suggest_categorical("n_head", [4, 8, 16])
    n_layers = trial.suggest_int("n_layers", 3, 8)
    patch_size = trial.suggest_categorical("patch_size", [2, 4, 8])
    ffn_hidden = trial.suggest_categorical("ffn_hidden", [
        d_model * 2, d_model * 4, d_model * 8
    ])
    drop_prob = trial.suggest_float("drop_prob", 0.05, 0.3)

    # Optimizer hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])

    # Label smoothing
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
        'in_channels': IN_CHANNELS,
        'img_size_h': IMG_SIZE_H,
        'img_size_w': IMG_SIZE_W,
        'patch_size': patch_size,
        'num_classes': NUM_CLASSES,
        'd_model': d_model,
        'n_head': n_head,
        'n_layers': n_layers,
        'ffn_hidden': ffn_hidden,
        'drop_prob': drop_prob,
        'device': device
    }

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
    study_name="vit_amc_tuning",
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
print("-" * 70)

# Run optimization
study.optimize(
    objective_vit,
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
best_params_path = "result/best_hyperparameters_vit.json"
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

### Visualization

```python
# Cell 9: Visualize optimization history

# Plot 1: Optimization History
fig1 = plot_optimization_history(study)
fig1.update_layout(title="Optimization History", width=900, height=500)
fig1.show()
fig1.write_html("result/optuna_optimization_history_vit.html")

# Plot 2: Parameter Importances
fig2 = plot_param_importances(study)
fig2.update_layout(title="Hyperparameter Importances", width=900, height=500)
fig2.show()
fig2.write_html("result/optuna_param_importances_vit.html")

# Plot 3: Parallel Coordinate Plot
fig3 = plot_parallel_coordinate(study)
fig3.update_layout(title="Parallel Coordinate Plot", width=1000, height=600)
fig3.show()
fig3.write_html("result/optuna_parallel_coordinate_vit.html")

# Plot 4: Slice Plot
fig4 = plot_slice(study)
fig4.update_layout(width=1200, height=800)
fig4.show()
fig4.write_html("result/optuna_slice_plot_vit.html")

print("✅ All plots saved to result/ directory")
```

### Study Statistics

```python
# Cell 10: Display study statistics

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

# Top 10 trials
print(f"\n🏅 TOP 10 TRIALS")
print("-" * 70)

top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
for i, trial in enumerate(top_trials, 1):
    print(f"{i:2d}. Trial {trial.number:3d}: {trial.value:.2f}%")

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
# Cell 11: Save study for later analysis

# Save study to database
study_path = "result/optuna_study_vit.db"
study.trials_dataframe().to_csv("result/optuna_trials_vit.csv", index=False)

print(f"💾 Study trials saved to: result/optuna_trials_vit.csv")
print(f"💾 Study database: sqlite:///{study_path}")
print("\nℹ️ To load study later:")
print(f"   study = optuna.load_study(")
print(f"       study_name='vit_amc_tuning',")
print(f"       storage='sqlite:///{study_path}'")
print(f"   )")
```

---

## Best Practices

### 1. Start Small
- Begin with fewer trials (10-20) to test the pipeline
- Use fewer epochs per trial (10-15) initially
- Gradually increase based on results

### 2. Hyperparameter Ranges
- **d_model**: Start with [64, 128, 256], expand to 512 if needed
- **n_layers**: 3-8 layers usually sufficient
- **learning_rate**: Log scale from 1e-5 to 1e-3
- **drop_prob**: 0.1-0.3 is typical

### 3. Pruning Strategy
- Use `MedianPruner` to stop unpromising trials early
- Saves significant computation time
- Adjust `n_startup_trials` and `n_warmup_steps` based on convergence speed

### 4. Resource Management
```python
# Monitor GPU memory
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
torch.cuda.empty_cache()  # Clear cache between trials
```

### 5. Parallel Execution
For multiple GPUs:
```python
# Run in parallel (requires separate processes)
study.optimize(objective_vit, n_trials=50, n_jobs=2)  # 2 parallel workers
```

---

## Troubleshooting

### Issue 1: Out of Memory
**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 128  # or 64

# Reduce model size in search space
d_model = trial.suggest_categorical("d_model", [64, 128])  # Remove 256, 512
```

### Issue 2: Trials Taking Too Long
**Solution:**
```python
# Reduce epochs per trial
N_EPOCHS_PER_TRIAL = 10

# Use more aggressive pruning
pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)

# Subsample training data
train_indices_subset = np.random.choice(train_indices, size=int(len(train_indices)*0.2))
```

### Issue 3: All Trials Being Pruned
**Solution:**
```python
# Increase warmup period
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)

# Or disable pruning temporarily
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
```

### Issue 4: Study Not Improving
**Solution:**
- Check that data loading is correct
- Verify model trains without Optuna
- Expand hyperparameter search ranges
- Increase number of trials

### Issue 5: Restarting Study
```python
# Load existing study and continue
study = optuna.create_study(
    study_name="vit_amc_tuning",
    direction="maximize",
    storage="sqlite:///result/optuna_study_vit.db",
    load_if_exists=True  # Continue existing study
)

# Add more trials
study.optimize(objective_vit, n_trials=20)
```

---

## Next Steps

### 1. Train Final Model
Use best hyperparameters for full training:

```python
# Load best parameters
with open("result/best_hyperparameters_vit.json", 'r') as f:
    best_params = json.load(f)['params']

# Train with full epochs (100-300)
# See ViT/training/train.py for full training pipeline
```

### 2. Fine-tune Further
- Narrow search space around best parameters
- Run more trials in promising regions
- Test different optimizers (SGD with momentum, AdamW variants)

### 3. Cross-validation
- Implement k-fold cross-validation for robust estimates
- Average results across folds

### 4. Test Set Evaluation
```python
# Evaluate best model on held-out test set
# Only do this once to avoid overfitting to test set
```

---

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Optuna Examples**: https://github.com/optuna/optuna-examples
- **Vision Transformer Paper**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- **Project README**: `ViT/README_ViT.md`

---

## Support

For issues specific to:
- **Optuna**: Check [Optuna GitHub Issues](https://github.com/optuna/optuna/issues)
- **ViT Model**: See `ViT/ARCHITECTURE_VIT.md`
- **Training**: See `ViT/training/train.py`

---

**Happy Tuning! 🚀**
