# Vision Transformer for Automatic Modulation Classification (AMC)

A Vision Transformer-based deep learning model for classifying radio signal modulation schemes from I/Q data. This project applies transformer architecture to automatic modulation classification (AMC) tasks.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training Workflow](#training-workflow)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration](#configuration)

## Overview

This project implements a Vision Transformer (ViT)-inspired architecture for Automatic Modulation Classification. The model processes I/Q signal data by treating the concatenated I and Q components as a 2D image, which is then divided into patches and processed through transformer encoder layers.

### Key Features

- **Vision Transformer Architecture**: Adapts ViT principles for radio signal classification
- **19 Modulation Classes**: Supports OOK, ASK, PSK, APSK, QAM, GMSK, and OQPSK variants
- **Efficient Training**: Multi-worker data loading with HDF5 optimization
- **Production-Ready**: Includes checkpointing, early stopping, and comprehensive logging
- **Evaluation Tools**: Confusion matrices, classification reports, and training history plots

### Supported Modulation Types

The model classifies 19 different modulation schemes:
- **ASK**: OOK, 4ASK, 8ASK
- **PSK**: BPSK, QPSK, 8PSK, 16PSK, 32PSK
- **APSK**: 16APSK, 32APSK, 64APSK, 128APSK
- **QAM**: 16QAM, 32QAM, 64QAM, 128QAM, 256QAM
- **Other**: GMSK, OQPSK

## Model Architecture

### AMC Transformer Overview

The model consists of three main components:

```
Input (I/Q Signal) → Patch Embedding → Transformer Encoder → MLP Head → Classification
```

### Detailed Architecture

#### 1. Input Processing
- **Input Shape**: `(batch_size, 1, 32, 64)`
  - 1 channel (I and Q signals concatenated and reshaped)
  - 32×64 "image" representation of the I/Q signal
  - Original I/Q data: 1024 samples per channel → reshaped to [2048] → view as [1, 32, 64]

#### 2. Patch Embedding (`models/embedding/patch_embedding.py`)
```python
Conv2d(in_channels=1, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
```
- Divides the input image into non-overlapping patches
- Default patch size: 4×4
- Number of patches: (32÷4) × (64÷4) = 8 × 16 = 128 patches
- Each patch is projected to `d_model` dimensions (default: 128)

#### 3. Positional Encoding (`models/embedding/positional_encoding.py`)
- Learnable positional embeddings for each patch position
- Adds positional information to maintain spatial relationships
- Total sequence length: 129 (128 patches + 1 CLS token)

#### 4. CLS Token
- Learnable classification token prepended to the sequence
- Shape: `(1, 1, d_model)`
- Used for final classification output

#### 5. Transformer Encoder (`models/encoder.py`)
Consists of `n_layers` (default: 6) identical encoder layers, each containing:

##### Multi-Head Self-Attention (`models/layers/multi_head_attention.py`)
- Number of heads: 8 (default)
- Each head dimension: d_model ÷ n_head = 128 ÷ 8 = 16
- Scaled Dot-Product Attention mechanism
- Layer normalization applied before attention (Pre-LN)

##### Position-wise Feed-Forward Network (`models/layers/position_wise_feed_forward.py`)
```python
Linear(d_model, ffn_hidden) → GELU → Dropout → Linear(ffn_hidden, d_model)
```
- FFN hidden size: 512 (4 × d_model)
- Activation: GELU
- Dropout rate: 0.1

##### Residual Connections & Layer Normalization
- Residual connections around both sub-layers
- Layer normalization applied before each sub-layer (Pre-LN architecture)

#### 6. Classification Head (`models/amc_transformer.py`)
```python
Linear(d_model, num_classes)
```
- Takes the CLS token output: `encoder_output[:, 0]`
- Projects to 19 classes (number of modulation types)

### Model Parameters

Default configuration:
```python
Model Hyperparameters:
├── Input: (1, 32, 64)
├── Patch Size: 4
├── d_model: 128
├── Number of Heads: 8
├── Number of Layers: 6
├── FFN Hidden: 512 (4 × d_model)
├── Dropout: 0.1
└── Output Classes: 19
```

**Total Parameters**: ~1-2M (depends on exact configuration)

## Project Structure

```
ViT/
├── models/                          # Model architecture
│   ├── amc_transformer.py          # Main transformer model
│   ├── encoder.py                  # Transformer encoder
│   ├── blocks/
│   │   └── encoder_layer.py        # Single encoder layer
│   ├── layers/
│   │   ├── multi_head_attention.py # Multi-head attention
│   │   ├── scale_dot_product_attention.py
│   │   ├── position_wise_feed_forward.py
│   │   └── layers_norm.py          # Layer normalization
│   └── embedding/
│       ├── patch_embedding.py      # Patch embedding (Conv2d)
│       └── positional_encoding.py  # Positional encoding
│
├── dataloader/                      # Data loading utilities
│   ├── dataset.py                  # HDF5 dataset class
│   └── utils.py                    # Data splitting utilities
│
├── training/                        # Training scripts
│   ├── train.py                    # Main training script
│   ├── evaluate.py                 # Evaluation utilities
│   └── utils.py                    # Training utilities
│
├── result/                          # Training outputs
│   ├── checkpoints/                # Model checkpoints
│   │   ├── production_v1/
│   │   ├── production_v2/
│   │   └── quick_test/
│   └── logs/                       # Training logs and plots
│
├── main.ipynb                       # Interactive notebook
├── MDF_NET.ipynb                    # Multi-Domain Fusion experiments
├── test_model.py                    # Model sanity check script
└── README.md                        # This file
```

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
h5py>=3.8.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ViT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy h5py scikit-learn matplotlib seaborn tqdm
```

## Dataset

### Data Format

The project uses HDF5 format for efficient data loading:

```
HDF5 File Structure:
├── X: (N, 1024, 2) - I/Q signal data
│   └── [:, :, 0] - In-phase (I) component
│   └── [:, :, 1] - Quadrature (Q) component
├── Y: (N, num_classes) - One-hot encoded labels
└── Z: (N, 1) - SNR values
```

### Data Processing Pipeline

1. **Loading**: Read I/Q data from HDF5 file
2. **Normalization**: Per-channel mean/std normalization
   ```python
   I_normalized = (I - I_mean) / I_std
   Q_normalized = (Q - Q_mean) / Q_std
   ```
3. **Reshaping**: Concatenate and reshape to image format
   ```python
   [I(1024), Q(1024)] → [2048] → view(1, 32, 64)
   ```
4. **Batching**: Group into batches for training

### Dataset Configuration

```python
# In training/train.py Config class
FILE_PATH = "path/to/GOLD_XYZ_OSC.0001_1024.hdf5"
JSON_PATH = "path/to/classes-fixed.json"

# Data split ratios
TRAIN_SIZE = 0.7   # 70% training
VALID_SIZE = 0.15  # 15% validation
TEST_SIZE = 0.15   # 15% testing
```

## Usage

### Quick Start

#### 1. Test Model Architecture

```bash
python test_model.py
```

This script performs a sanity check to verify the model architecture.

#### 2. Train the Model

```bash
cd training
python train.py --experiment_name my_experiment
```

#### 3. Train with Custom Parameters

```bash
python train.py \
    --experiment_name production_v3 \
    --batch_size 256 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --num_workers 6 \
    --d_model 128 \
    --n_layers 6
```

#### 4. Resume Training from Checkpoint

```bash
python train.py \
    --resume result/checkpoints/production_v1/checkpoint_epoch_50.pth \
    --experiment_name production_v1_continued
```

### Command-Line Arguments

```bash
Training Arguments:
  --file_path PATH          Path to HDF5 data file
  --json_path PATH          Path to classes JSON file
  --batch_size INT          Batch size (default: 256)
  --num_epochs INT          Number of epochs (default: 100)
  --learning_rate FLOAT     Learning rate (default: 1e-4)
  --num_workers INT         Data loading workers (default: 6)
  --experiment_name STR     Experiment name for logging

Model Arguments:
  --d_model INT             Model dimension (default: 128)
  --n_head INT              Number of attention heads (default: 8)
  --n_layers INT            Number of transformer layers (default: 6)
  --patch_size INT          Patch size (default: 4)

Other Arguments:
  --resume PATH             Path to checkpoint to resume from
```

## Training Workflow

### 1. Data Preparation

```python
# Data is split into train/validation/test sets
train_indices, valid_indices, test_indices, label_map = split_data(
    file_path, json_path, target_modulations,
    train_size=0.7, valid_size=0.15, test_size=0.15, seed=42
)

# Normalization statistics calculated from training set
train_dataset.calculate_normalization_stats()  # Mean and std for I and Q

# Same normalization applied to validation and test sets
valid_dataset = SingleStreamImageDataset(..., normalization_stats=norm_stats)
```

### 2. Training Loop

For each epoch:

1. **Training Phase**
   - Model in training mode (`model.train()`)
   - Forward pass: `outputs = model(images)`
   - Loss calculation: CrossEntropyLoss with label smoothing (0.1)
   - Backward pass with gradient clipping (max_norm=1.0)
   - Optimizer step (AdamW with weight decay 1e-3)

2. **Validation Phase**
   - Model in evaluation mode (`model.eval()`)
   - No gradient computation
   - Calculate validation loss and accuracy

3. **Learning Rate Scheduling**
   - ReduceLROnPlateau scheduler
   - Monitors validation loss
   - Reduces LR by factor of 0.5 when loss plateaus for 5 epochs

4. **Checkpointing**
   - Save checkpoint every 10 epochs
   - Save best model based on validation loss
   - Checkpoint includes:
     - Model state dict
     - Optimizer state dict
     - Scheduler state dict
     - Training history
     - Configuration

5. **Early Stopping**
   - Patience: 10 epochs
   - Monitors validation loss
   - Stops training if no improvement

### 3. Training Configuration

```python
# Optimizer
AdamW(lr=1e-4, weight_decay=1e-3, betas=(0.9, 0.99))

# Loss Function
CrossEntropyLoss(label_smoothing=0.1)

# Learning Rate Scheduler
ReduceLROnPlateau(mode='min', factor=0.5, patience=5)

# Data Loading
DataLoader(
    batch_size=256,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=3
)
```

### 4. Training Output

During training, you'll see:

```
======================================================================
AMC TRANSFORMER TRAINING
======================================================================
Experiment: production_v2
Device: cuda
Batch size: 256
Num workers: 6
Learning rate: 0.0001
======================================================================

📂 Loading data...
✅ Data loaded:
   Train: 1,234 batches (316,160 samples)
   Valid: 265 batches (67,840 samples)

🤖 Initializing model...
✅ Model created:
   Total parameters: 1,234,567
   Trainable parameters: 1,234,567

======================================================================
STARTING TRAINING
======================================================================

Epoch 1 [Train]: 100%|████████| 1234/1234 [02:15<00:00, 9.11it/s, loss=2.1234, acc=45.67%]
Epoch 1 [Valid]: 100%|████████| 265/265 [00:20<00:00, 13.01it/s, loss=1.9876, acc=48.23%]

Epoch 1/100 Summary:
   Train Loss: 2.1234 | Train Acc: 45.67%
   Val Loss:   1.9876 | Val Acc:   48.23%
   Time: 155.3s | LR: 1.00e-04
   💾 Checkpoint saved: checkpoint_epoch_10.pth
```

## Evaluation

### Automatic Evaluation

After training completes, the model is automatically evaluated on the test set:

```python
# Creates evaluation metrics and visualizations
evaluate_model_with_confusion(
    model=model,
    dataloader=test_loader,
    device=device,
    class_names=TARGET_MODULATIONS,
    save_dir="result/checkpoints/{experiment}/evaluation/"
)
```

### Evaluation Outputs

The evaluation generates:

1. **Confusion Matrix** (`test_confusion_matrix.pdf`)
   - Normalized confusion matrix heatmap
   - Shows per-class classification accuracy
   - Identifies common misclassifications

2. **Classification Report** (`test_classification_report.txt`)
   ```
   Classification Report:
                 precision    recall  f1-score   support

        OOK       0.95      0.93      0.94      1234
       4ASK       0.89      0.91      0.90      1156
       8ASK       0.87      0.85      0.86      1198
       ...

   accuracy                           0.88     23456
   macro avg      0.89      0.88      0.88     23456
   weighted avg   0.88      0.88      0.88     23456
   ```

3. **Training History Plot** (`training_history.png`)
   - Training and validation loss curves
   - Training and validation accuracy curves
   - Learning rate schedule

### Manual Evaluation

```python
from training.evaluate import evaluate_model_with_confusion
from models.amc_transformer import AMCTransformer
from training.utils import load_checkpoint

# Load trained model
model = AMCTransformer(**model_params)
checkpoint = load_checkpoint("result/checkpoints/production_v2/model_final.pth", model)

# Evaluate
results = evaluate_model_with_confusion(
    model=model,
    dataloader=test_loader,
    device='cuda',
    class_names=TARGET_MODULATIONS,
    save_dir="custom_evaluation/"
)

print(f"Test Accuracy: {results['accuracy']:.2f}%")
```

## Results

### Model Performance

Based on the training history, the model achieves:

- **Training Accuracy**: ~92% (after convergence)
- **Validation Accuracy**: ~83% (after convergence)
- **Test Accuracy**: Available in evaluation reports

### Performance Characteristics

- **Training Time**: ~2-3 minutes per epoch on GPU (depends on hardware)
- **Convergence**: Typically reaches plateau around 60-80 epochs
- **Early Stopping**: Usually triggers around epoch 70-90

### Checkpoints

Available model checkpoints:
- `result/checkpoints/production_v1/`: First production model
- `result/checkpoints/production_v2/`: Improved model with 6 layers
- `result/checkpoints/quick_test/`: Quick test run

### Visualizations

Example outputs available:
- `example_history_plot.pdf`: Training history visualization
- `example_cm_plot.pdf`: Confusion matrix example

## Configuration

### Key Hyperparameters

#### Model Architecture
```python
PATCH_SIZE = 4           # Size of each patch
D_MODEL = 128            # Model dimension
N_HEAD = 8               # Number of attention heads
N_LAYERS = 6             # Number of transformer layers
FFN_HIDDEN = 512         # Feedforward hidden dimension (4 × D_MODEL)
DROP_PROB = 0.1          # Dropout probability
```

#### Training
```python
BATCH_SIZE = 256         # Batch size
NUM_EPOCHS = 100         # Maximum epochs
LEARNING_RATE = 1e-4     # Initial learning rate
WEIGHT_DECAY = 1e-3      # AdamW weight decay
LABEL_SMOOTHING = 0.1    # Label smoothing factor
PATIENCE = 10            # Early stopping patience
```

#### Data Loading
```python
NUM_WORKERS = 6          # Number of data loading workers
PREFETCH_FACTOR = 3      # Number of batches to prefetch
PIN_MEMORY = True        # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = True # Keep workers alive between epochs
```

### Modifying Configuration

To change hyperparameters:

1. **Via Command Line** (recommended for experiments):
```bash
python train.py --d_model 256 --n_layers 8 --batch_size 128
```

2. **Edit Config Class** in `training/train.py`:
```python
class Config:
    D_MODEL = 256        # Change from default 128
    N_LAYERS = 8         # Change from default 6
    BATCH_SIZE = 128     # Change from default 256
```

## Development History

Key milestones based on git history:

1. **Initial Development**: Base transformer architecture implementation
2. **Architecture Refinement**: Added Vision Transformer components
3. **Training Optimization**: Implemented efficient HDF5 data loading
4. **Production Models**:
   - `production_v1`: Initial production model
   - `production_v2`: Improved with 6 layers and weight decay tuning
5. **Performance Plateau**: Model reached convergence (~83% validation accuracy)

## Troubleshooting

### Common Issues

1. **HDF5 File Locking Error**
   ```python
   # Set before running
   import os
   os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
   ```

2. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE`
   - Reduce `NUM_WORKERS`
   - Reduce model size (`D_MODEL`, `N_LAYERS`)

3. **Slow Data Loading**
   - Increase `NUM_WORKERS` (but not too high)
   - Enable `PIN_MEMORY = True`
   - Use `PERSISTENT_WORKERS = True`

4. **Model Not Learning**
   - Check learning rate (may need adjustment)
   - Verify data normalization
   - Check for gradient flow issues

## Citation

If you use this code, please cite:

```bibtex
@misc{amc_vit,
  title={Vision Transformer for Automatic Modulation Classification},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/amc-vit}}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your contact info].

## Acknowledgments

- Vision Transformer (ViT) paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- RadioML dataset for automatic modulation classification research
