# Transformer Architectures for Automatic Modulation Classification (AMC)

A comprehensive deep learning framework implementing two distinct Transformer-based architectures for classifying radio signal modulation schemes from I/Q data.

## Table of Contents
- [Overview](#overview)
- [Architectures](#architectures)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements two state-of-the-art Transformer architectures for Automatic Modulation Classification:

1. **Vision Transformer (ViT)**: Treats I/Q signals as 2D images (32×64) and processes them using patch embeddings
2. **Pure Transformer**: Directly processes raw 1D I/Q sequences using temporal embeddings

Both models classify **19 different modulation schemes** across various SNR conditions.

### Supported Modulation Types

- **Amplitude Shift Keying (ASK)**: OOK, 4ASK, 8ASK
- **Phase Shift Keying (PSK)**: BPSK, QPSK, 8PSK, 16PSK, 32PSK
- **Amplitude-Phase Shift Keying (APSK)**: 16APSK, 32APSK, 64APSK, 128APSK
- **Quadrature Amplitude Modulation (QAM)**: 16QAM, 32QAM, 64QAM, 128QAM, 256QAM
- **Others**: GMSK, OQPSK

### Key Features

- **Two Transformer Architectures**: ViT for image-based processing and Pure Transformer for sequential data
- **Flexible Configuration**: Easily adjustable hyperparameters for experimentation
- **Production-Ready**: Comprehensive training pipeline with checkpointing, early stopping, and logging
- **Efficient Data Loading**: Multiprocessing-safe HDF5 data loading with on-the-fly normalization
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and training visualizations

---

## Architectures

### 1. Vision Transformer (ViT)

```
Input: Raw I/Q [1024, 2] → Reshape to [1, 32, 64] (image format)
   ↓
Patch Embedding (Conv2d)
   ↓ Split into patches (e.g., 8×16 = 128 patches)
[Batch, num_patches, d_model]
   ↓
Add CLS Token
   ↓
[Batch, num_patches+1, d_model]
   ↓
Positional Encoding (Sinusoidal)
   ↓
Transformer Encoder Layers × N
   ├── Multi-Head Self-Attention
   ├── Layer Normalization
   ├── Feed-Forward Network
   └── Residual Connections
   ↓
Extract CLS Token
   ↓
Classification Head (Linear)
   ↓
Output: [Batch, num_classes]
```

**Key Characteristics:**
- Treats concatenated I/Q as 2D "image" (32×64 pixels)
- Uses 2D patch embedding via Conv2d
- Number of patches: `(32/patch_size) × (64/patch_size)`
- Typical configuration: patch_size=4 → 128 patches (8×16)

### 2. Pure Transformer (Raw I/Q)

```
Input: Raw I/Q [2, 1024] (2 channels: I and Q)
   ↓
Sequence Embedding (Conv1d or Segment-based)
   ↓
[Batch, num_tokens, d_model]
   ↓
Add CLS Token (optional)
   ↓
Positional Encoding (Sinusoidal)
   ↓
Transformer Encoder Layers × N
   ├── Multi-Head Self-Attention
   ├── Layer Normalization
   ├── Feed-Forward Network
   └── Residual Connections
   ↓
CLS Token or Global Average Pooling
   ↓
Classification Head (LayerNorm + Linear)
   ↓
Output: [Batch, num_classes]
```

**Key Characteristics:**
- Processes raw 1D I/Q sequences directly
- Two embedding options:
  - **Conv1D**: Full sequence (1024 tokens)
  - **Segment**: Groups timesteps (e.g., 16 tokens with segment_size=64)
- Flexible aggregation: CLS token or Global Average Pooling

---

## Project Structure

```
Transformer_Thesis/
├── ViT/                              # Vision Transformer implementation
│   ├── models/
│   │   ├── amc_transformer.py       # Main ViT model
│   │   ├── encoder.py               # Transformer encoder
│   │   ├── blocks/
│   │   │   └── encoder_layer.py     # Single encoder layer
│   │   ├── layers/
│   │   │   ├── multi_head_attention.py
│   │   │   ├── scale_dot_product_attention.py
│   │   │   ├── position_wise_feed_forward.py
│   │   │   └── layers_norm.py
│   │   └── embedding/
│   │       ├── patch_embedding.py   # 2D patch embedding
│   │       └── positional_encoding.py
│   ├── dataloader/
│   │   ├── dataset.py               # Dataset for image format
│   │   └── utils.py
│   ├── training/
│   │   ├── train.py                 # Training script
│   │   ├── evaluate.py              # Evaluation utilities
│   │   └── utils.py
│   ├── result/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── ARCHITECTURE_VIT.md          # ViT architecture documentation
│   └── README.md
│
├── transformer_rawIQ/                # Pure Transformer implementation
│   ├── models/
│   │   ├── transformer_rawIQ.py     # Main Transformer model
│   │   ├── encoder.py
│   │   ├── blocks/
│   │   │   └── encoder_layer.py
│   │   ├── layers/
│   │   │   ├── multi_head_attention.py
│   │   │   ├── scale_dot_product_attention.py
│   │   │   ├── position_wise_feed_forward.py
│   │   │   └── layers_norm.py
│   │   └── embedding/
│   │       ├── patch_embedding.py   # 1D sequence embedding
│   │       └── positional_encoding.py
│   ├── dataloader/
│   │   ├── dataset.py               # Dataset for raw I/Q
│   │   └── utils.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── utils.py
│   ├── result/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── ARCHITECTURE.md              # Pure Transformer documentation
│   └── README.md
│
└── README.md                         # This file
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.3+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Transformer_Thesis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy scikit-learn matplotlib seaborn tqdm
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
h5py>=3.8.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## Dataset

### Data Format

The project uses **RadioML 2018.01A** dataset in HDF5 format:

```
HDF5 File Structure:
├── X: (N, 1024, 2) - I/Q signal data
│   ├── [:, :, 0] - In-phase (I) component
│   └── [:, :, 1] - Quadrature (Q) component
├── Y: (N, num_classes) - One-hot encoded labels
└── Z: (N, 1) - SNR values (in dB)
```

- **N**: Total number of samples
- **1024**: Sequence length (time samples)
- **2**: I and Q channels

### Download

1. Download RadioML 2018.01A dataset from [DeepSig](https://www.deepsig.ai/datasets)
2. Place the HDF5 file in your data directory
3. Update `FILE_PATH` in training configuration

### Data Processing

Both architectures use the same preprocessing pipeline:

```python
# 1. Load raw I/Q from HDF5
raw_iq = [1024, 2]

# 2. Normalize per channel (statistics from training set)
I_normalized = (I - mean_I) / std_I
Q_normalized = (Q - mean_Q) / std_Q

# 3. Transform to model input
# ViT: Concatenate and reshape to [1, 32, 64]
# Pure Transformer: Transpose to [2, 1024]

# 4. Batch and feed to model
```

**Data Splits:**
- Training: 70%
- Validation: 15%
- Test: 15%
- Stratified by modulation class for balanced representation

---

## Usage

### Quick Start

#### 1. Test Model Architecture

**Vision Transformer:**
```bash
cd ViT
python test_model.py
```

**Pure Transformer:**
```bash
cd transformer_rawIQ
python test_model.py
```

#### 2. Train a Model

**Vision Transformer:**
```bash
cd ViT/training
python train.py --experiment_name vit_exp1 --batch_size 256 --num_epochs 100
```

**Pure Transformer:**
```bash
cd transformer_rawIQ/training
python train.py --experiment_name raw_exp1 --batch_size 256 --num_epochs 100
```

#### 3. Evaluate Trained Model

```bash
python evaluate.py \
    --checkpoint ../result/checkpoints/[experiment_name]/model_best.pth \
    --dataset test \
    --batch_size 256
```

---

## Model Configurations

### Vision Transformer (ViT) Configuration

```python
# Model Architecture
PATCH_SIZE = 4           # Patch size (4×4)
D_MODEL = 128            # Embedding dimension
N_HEAD = 8               # Number of attention heads
N_LAYERS = 6             # Number of transformer layers
FFN_HIDDEN = 512         # Feedforward hidden dimension
DROP_PROB = 0.1          # Dropout probability

# Input Configuration
IMG_SIZE_H = 32          # Image height
IMG_SIZE_W = 64          # Image width
IN_CHANNELS = 1          # Number of input channels
NUM_CLASSES = 19         # Number of modulation classes

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
```

**Number of patches:** `(32/4) × (64/4) = 8 × 16 = 128 patches`

### Pure Transformer Configuration

```python
# Model Architecture
D_MODEL = 128            # Embedding dimension
N_HEAD = 8               # Number of attention heads
N_LAYERS = 6             # Number of transformer layers
FFN_HIDDEN = 1024        # Feedforward hidden dimension (4×D_MODEL)
DROP_PROB = 0.2          # Dropout probability

# Embedding Configuration
EMBEDDING_TYPE = 'segment'   # 'segment' or 'conv1d'
SEGMENT_SIZE = 64            # Segment size (for segment mode)
USE_CLS_TOKEN = True         # Use CLS token

# Input Configuration
SEQUENCE_LENGTH = 1024   # Input sequence length
INPUT_CHANNELS = 2       # I and Q channels
NUM_CLASSES = 19         # Number of modulation classes

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
```

**Number of tokens (segment mode):** `1024/64 = 16 tokens`

### Architecture Comparison

| Feature | Vision Transformer | Pure Transformer |
|---------|-------------------|------------------|
| **Input Format** | 2D Image [1, 32, 64] | 1D Sequence [2, 1024] |
| **Embedding** | 2D Patch (Conv2d) | 1D Sequence (Conv1d/Segment) |
| **Number of Tokens** | 128 (8×16 patches) | 16-1024 (configurable) |
| **Spatial Inductive Bias** | Yes (2D structure) | No (pure sequence) |
| **Memory Usage** | Moderate | Low (segment) / High (conv1d) |
| **Best For** | Treating signals as images | Processing raw temporal data |

---

## Training

### Training Command-Line Arguments

Both models support similar CLI arguments:

```bash
# Data Configuration
--file_path PATH          # Path to HDF5 data file
--json_path PATH          # Path to classes JSON file

# Training Configuration
--batch_size INT          # Batch size (default: 256)
--num_epochs INT          # Number of epochs (default: 100)
--learning_rate FLOAT     # Learning rate (default: 1e-4)
--num_workers INT         # Data loading workers (default: 6-8)
--experiment_name STR     # Experiment name for logging

# Model Architecture (ViT)
--patch_size INT          # Patch size (default: 4)
--d_model INT             # Model dimension (default: 128)
--n_head INT              # Number of attention heads (default: 8)
--n_layers INT            # Number of transformer layers (default: 6)

# Model Architecture (Pure Transformer)
--embedding_type STR      # 'segment' or 'conv1d' (default: segment)
--segment_size INT        # Segment size (default: 64)
--use_cls_token BOOL      # Use CLS token (default: True)

# Other
--resume PATH             # Path to checkpoint to resume from
```

### Training Process

The training loop includes:

1. **Forward Pass**: Model prediction on batch
2. **Loss Calculation**: CrossEntropyLoss with label smoothing (0.1)
3. **Backward Pass**: Gradient computation with clipping (max_norm=1.0)
4. **Optimization**: AdamW optimizer with weight decay
5. **Learning Rate Scheduling**: ReduceLROnPlateau (reduces LR on plateau)
6. **Validation**: Evaluate on validation set each epoch
7. **Checkpointing**: Save model every N epochs and best model
8. **Early Stopping**: Stop if validation loss doesn't improve (patience=10)

### Training Output Example

```
======================================================================
AMC TRANSFORMER TRAINING
======================================================================
Experiment: production_v1
Architecture: Vision Transformer / Pure Transformer
Device: cuda
Batch size: 256
Num workers: 8
Learning rate: 0.0001
======================================================================

📂 Loading data...
✅ Data loaded:
   Train: 1,234 batches (316,160 samples)
   Valid: 265 batches (67,840 samples)
   Test: 265 batches (67,840 samples)

🤖 Initializing model...
✅ Model created:
   Total parameters: 1,234,567
   Trainable parameters: 1,234,567

======================================================================
STARTING TRAINING
======================================================================

Epoch 1/100:
Train: 100%|████████| 1234/1234 [02:15<00:00, 9.11it/s, loss=2.123, acc=45.6%]
Valid: 100%|████████| 265/265 [00:20<00:00, 13.0it/s, loss=1.987, acc=48.2%]

Epoch 1 Summary:
   Train Loss: 2.1234 | Train Acc: 45.67%
   Val Loss:   1.9876 | Val Acc:   48.23%
   Time: 155.3s | LR: 1.00e-04
   
Epoch 50/100:
Train: 100%|████████| 1234/1234 [02:10<00:00, 9.45it/s, loss=0.234, acc=92.1%]
Valid: 100%|████████| 265/265 [00:19<00:00, 13.5it/s, loss=0.512, acc=83.4%]

Epoch 50 Summary:
   Train Loss: 0.2345 | Train Acc: 92.12%
   Val Loss:   0.5123 | Val Acc:   83.45%
   Time: 149.8s | LR: 5.00e-05
   💾 Checkpoint saved: checkpoint_epoch_50.pth
   ⭐ New best model! (val_loss: 0.5123)
```

### Optimization Configuration

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-3,      # ViT: 1e-3, Pure: 1e-4
    betas=(0.9, 0.99)
)

# Loss Function
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Evaluation

### Automatic Evaluation

After training completes, models are automatically evaluated on the test set:

```python
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

#### 1. Confusion Matrix
- **Normalized** (`test_confusion_matrix.pdf`): Percentages (0-1 scale)
- **Raw Counts** (`test_confusion_matrix_counts.pdf`): Actual sample counts
- Heatmap visualization showing per-class accuracy
- Identifies common misclassification patterns

#### 2. Classification Report
(`test_classification_report.txt`)

```
Classification Report:
              precision    recall  f1-score   support

         OOK       0.95      0.93      0.94      1234
        4ASK       0.89      0.91      0.90      1156
        8ASK       0.87      0.85      0.86      1198
        BPSK       0.92      0.94      0.93      1267
        QPSK       0.88      0.87      0.87      1223
       ...

    accuracy                           0.88     23456
   macro avg       0.89      0.88      0.88     23456
weighted avg       0.88      0.88      0.88     23456
```

#### 3. Training History Plot
(`training_history.png`)
- Training and validation loss curves
- Training and validation accuracy curves
- Learning rate schedule over epochs

### Manual Evaluation

```bash
# Evaluate on test set
python training/evaluate.py \
    --checkpoint result/checkpoints/production_v1/model_best.pth \
    --dataset test \
    --batch_size 256

# Evaluate on validation set
python training/evaluate.py \
    --checkpoint result/checkpoints/production_v1/model_best.pth \
    --dataset valid \
    --batch_size 256
```

---

## Results

### Typical Performance

Based on production models:

**Vision Transformer (ViT):**


**Pure Transformer (Raw I/Q):**


### Performance Characteristics

- **Best Performance**: High SNR conditions (>10 dB)
- **Challenging**: Low SNR conditions (<0 dB), similar modulations
- **Common Confusions**: Within same modulation families (e.g., 16QAM vs 64QAM)

### Model Size Comparison

| Model | Parameters | Memory (approx) |
|-------|------------|-----------------|
| ViT (d_model=128, n_layers=6) | ~1.2M | ~500 MB |
| ViT (d_model=256, n_layers=8) | ~4.5M | ~1.5 GB |
| Pure Transformer (segment, d_model=128) | ~1.0M | ~400 MB |
| Pure Transformer (conv1d, d_model=128) | ~1.2M | ~800 MB |

---

## Troubleshooting

### Common Issues

#### 1. HDF5 File Locking Error

```bash
# Solution 1: Set environment variable
export HDF5_USE_FILE_LOCKING=FALSE

# Solution 2: In Python script (already handled)
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
```

#### 2. CUDA Out of Memory

**Solutions:**
- Reduce batch size: `--batch_size 128` or `--batch_size 64`
- Use segment embedding (Pure Transformer): `--embedding_type segment`
- Reduce model size: `--d_model 64 --n_layers 4`
- Enable gradient checkpointing (advanced)

#### 3. DataLoader Hangs or Crashes

**Solutions:**
- **CRITICAL**: Use `num_workers > 0` for HDF5 datasets
- Recommended: 4-8 workers per GPU
- Ensure `worker_init_fn` is passed to DataLoader (already implemented)
- Check available RAM if using many workers

#### 4. Training Not Converging

**Possible causes and solutions:**
- **Learning rate too high**: Reduce to 1e-5 or 1e-6
- **Data not normalized**: Verify normalization statistics
- **Gradient explosion**: Check gradient clipping (max_norm=1.0)
- **Insufficient model capacity**: Increase d_model or n_layers
- **Overfitting**: Increase dropout, add weight decay

#### 5. Model Not Learning (Loss Stays High)

**Solutions:**
- Verify data loading: Check shapes and ranges
- Check label mapping: Ensure correct class indices
- Reduce learning rate: Try 1e-5
- Check for NaN/Inf: Monitor gradients
- Verify loss function: Ensure correct for classification

#### 6. Slow Training

**Optimizations:**
- Increase batch size (if memory allows)
- Use more workers: `--num_workers 8`
- Enable pin_memory (already enabled)
- Use persistent_workers (already enabled)
- Ensure GPU utilization (check with `nvidia-smi`)

---

## Advanced Usage

### Custom Model Configuration

Edit the `Config` class in `training/train.py`:

```python
class Config:
    # Experiment
    EXPERIMENT_NAME = "my_experiment"
    
    # Data
    FILE_PATH = "path/to/data.hdf5"
    JSON_PATH = "path/to/classes.json"
    
    # Model Architecture
    D_MODEL = 256           # Increase for larger model
    N_HEAD = 16             # More attention heads
    N_LAYERS = 12           # Deeper network
    FFN_HIDDEN = 1024       # Larger FFN
    DROP_PROB = 0.3         # More regularization
    
    # Training
    BATCH_SIZE = 128        # Smaller for memory
    LEARNING_RATE = 5e-5    # Lower LR for stability
    NUM_EPOCHS = 200        # More epochs
```

### Hyperparameter Tuning

**Start with baseline:**
- d_model: 128, n_head: 8, n_layers: 6
- batch_size: 256, learning_rate: 1e-4

**If underfitting (low training accuracy):**
- Increase model capacity: d_model → 256, n_layers → 8-12
- Decrease regularization: drop_prob → 0.1
- Increase training time: num_epochs → 150-200

**If overfitting (gap between train and val accuracy):**
- Increase regularization: drop_prob → 0.3-0.4
- Increase weight_decay → 1e-3
- Use label smoothing: 0.1 → 0.2
- Reduce model size: d_model → 64-128

**For faster training:**
- Use segment embedding (Pure Transformer)
- Reduce batch size but use gradient accumulation
- Use mixed precision training (TF32 on Ampere+)

---

## Architecture Details

### Multi-Head Attention Mechanism

```python
# Attention formula
Attention(Q, K, V) = softmax(QK^T / √d_k) V

# Multi-head
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Intuition:** Each head learns to attend to different aspects of the signal (e.g., amplitude patterns, phase patterns, temporal dependencies).

### Positional Encoding

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Purpose:** Injects position information since self-attention is permutation-invariant.

### Feed-Forward Network

```python
FFN(x) = Linear2(GELU(Linear1(x)))
# d_model → ffn_hidden → d_model
# 128 → 512/1024 → 128
```

**Purpose:** Adds non-linearity and increases model expressiveness.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{transformer_amc_2024,
  title={Transformer Architectures for Automatic Modulation Classification},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-repo}
}
```

### Related Papers

- **Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- **RadioML Dataset**: O'Shea et al., "Over-the-Air Deep Learning Based Radio Signal Classification", IEEE 2018

---


---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Project Link: [https://github.com/your-username/transformer-amc](https://github.com/your-username/transformer-amc)

---

## Acknowledgments

- **DeepSig Inc.** for the RadioML dataset
- **PyTorch Team** for the excellent deep learning framework
- **Vaswani et al.** for the original Transformer architecture
- **Dosovitskiy et al.** for the Vision Transformer adaptation
- All contributors to this project

---

## Project Status

**Current Version**: 1.0.0

**Active Development**: ✅ Yes

**Production Models Available**: 
- ViT: `production_v1`, `production_v2`
- Pure Transformer: `production_rawIQv1`

---

## Roadmap

- [ ] Add pre-trained model weights
- [ ] Implement attention visualization tools
- [ ] Add support for more datasets (RadioML 2016.10A, etc.)
- [ ] Integrate Weights & Biases for experiment tracking
- [ ] Add model quantization for deployment
- [ ] Create Docker container for easy setup
- [ ] Add real-time inference script

---

*Last Updated: 2024*