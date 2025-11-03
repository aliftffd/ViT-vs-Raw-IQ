# AMC Transformer Architecture

A comprehensive visualization of the data flow from input preprocessing to model classification.

## Table of Contents
- [Overview](#overview)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Component Details](#component-details)

---

## Overview

This architecture implements a Vision Transformer (ViT) for Automatic Modulation Classification (AMC) using I/Q signal data.

**Input**: Raw I/Q signals (1024 samples × 2 channels)
**Output**: Modulation class predictions (11 classes)

---

## Data Pipeline

### 1. Data Loading & Preprocessing (`dataloader/`)

#### 1.1 Dataset Metadata Loading (`utils.py`)

```
load_dataset_metadata()
├── Load HDF5 metadata (Y, Z)
├── Convert labels: Y_int → Y_strings
├── Extract SNR values (Z)
└── Return: (Y_strings, Z_data, available_modulations, total_samples)
```

**File**: `dataloader/utils.py:12-55`

#### 1.2 Data Splitting (`utils.py`)

```
split_data()
├── Stratify by: Modulation × SNR
├── Split ratios: train/valid/test
├── Create label_map: {modulation_string: int}
└── Return: (train_indices, valid_indices, test_indices, label_map)
```

**File**: `dataloader/utils.py:58-148`

#### 1.3 Dataset Class (`dataset.py`)

**Class**: `SingleStreamImageDataset`
**File**: `dataloader/dataset.py:41-241`

##### Initialization Flow:
```
__init__()
├── Store paths and indices (don't load data yet)
├── Load metadata (Y, Z) from HDF5
├── Calculate normalization stats (if mode='train')
│   └── _calculate_normalization_stats()
│       ├── Sample 5000 random training samples
│       ├── Compute: i_mean, i_std, q_mean, q_std
│       └── Store in norm_stats dict
└── Set image dimensions: H=32, W=64
```

##### Data Loading Flow (`__getitem__`):
```
Raw I/Q Data: (1024, 2)
      ↓
1. Read from HDF5
   X_h5[index] → x_raw (1024, 2)

2. Normalize I and Q channels separately
   I_channel: (I - i_mean) / i_std
   Q_channel: (Q - q_mean) / q_std
   → iq_sequence (1024, 2)

3. Separate and Concatenate
   i_signal = iq_sequence[:, 0]  # [1024]
   q_signal = iq_sequence[:, 1]  # [1024]
   iq_concat = cat(i_signal, q_signal)  # [2048]

4. Reshape to Image Format
   iq_concat.view(1, 32, 64)
   → iq_image (1, 32, 64)

5. Return
   (iq_image, y_label, z_snr)
```

**Key Files**:
- Normalization stats calculation: `dataloader/dataset.py:116-158`
- Data retrieval: `dataloader/dataset.py:181-226`

---

## Model Architecture

### 2. Model Pipeline (`models/`)

```
                    INPUT: (B, 1, 32, 64)
                            ↓
        ╔═══════════════════════════════════════╗
        ║         AMCTransformer                ║
        ║   (models/amc_transformer.py)         ║
        ╚═══════════════════════════════════════╝
                            ↓
        ┌───────────────────────────────────────┐
        │           ENCODER                     │
        │      (models/encoder.py)              │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │      1. PATCH EMBEDDING               │
        │   (embedding/patch_embedding.py)      │
        │                                       │
        │   Conv2d(in_channels=1,               │
        │          out_channels=d_model,        │
        │          kernel_size=patch_size,      │
        │          stride=patch_size)           │
        │                                       │
        │   Input:  (B, 1, 32, 64)              │
        │   Output: (B, E, P_h, P_w)            │
        │   Flatten & Transpose                 │
        │   Output: (B, N, E)                   │
        │                                       │
        │   where N = num_patches               │
        │         = (32/patch_size) ×           │
        │           (64/patch_size)             │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │      2. ADD CLS TOKEN                 │
        │                                       │
        │   cls_token: (1, 1, d_model)          │
        │   Repeat for batch                    │
        │   Concatenate with patches            │
        │                                       │
        │   Output: (B, N+1, d_model)           │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   3. POSITIONAL ENCODING              │
        │   (embedding/positional_encoding.py)  │
        │                                       │
        │   PE(pos, 2i)   = sin(pos/10000^(2i/d))│
        │   PE(pos, 2i+1) = cos(pos/10000^(2i/d))│
        │                                       │
        │   x = x + PE[:seq_len, :]             │
        │                                       │
        │   Output: (B, N+1, d_model)           │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │      4. DROPOUT                       │
        │                                       │
        │   Dropout(p=drop_prob)                │
        └───────────────────────────────────────┘
                            ↓
        ╔═══════════════════════════════════════╗
        ║   5. ENCODER LAYERS (×n_layers)       ║
        ║   (blocks/encoder_layer.py)           ║
        ╠═══════════════════════════════════════╣
        ║                                       ║
        ║   ┌─────────────────────────────────┐ ║
        ║   │  MULTI-HEAD ATTENTION           │ ║
        ║   │  (layers/multi_head_attention)  │ ║
        ║   │                                 │ ║
        ║   │  1. Linear projections:         │ ║
        ║   │     Q = W_q(x)                  │ ║
        ║   │     K = W_k(x)                  │ ║
        ║   │     V = W_v(x)                  │ ║
        ║   │                                 │ ║
        ║   │  2. Split into n_head heads:    │ ║
        ║   │     (B, L, D) →                 │ ║
        ║   │     (B, n_head, L, D/n_head)    │ ║
        ║   │                                 │ ║
        ║   │  3. Scaled Dot-Product Attn:    │ ║
        ║   │     Attention(Q,K,V) =          │ ║
        ║   │     softmax(QK^T/√d_k)V         │ ║
        ║   │                                 │ ║
        ║   │  4. Concat & Linear projection  │ ║
        ║   └─────────────────────────────────┘ ║
        ║                ↓                      ║
        ║   ┌─────────────────────────────────┐ ║
        ║   │  RESIDUAL + LAYER NORM          │ ║
        ║   │                                 │ ║
        ║   │  x = LayerNorm(x + Dropout(attn))│║
        ║   └─────────────────────────────────┘ ║
        ║                ↓                      ║
        ║   ┌─────────────────────────────────┐ ║
        ║   │  FEED FORWARD NETWORK           │ ║
        ║   │  (layers/position_wise_ffn)     │ ║
        ║   │                                 │ ║
        ║   │  FFN(x) = Linear(GELU(          │ ║
        ║   │            Linear(x)))          │ ║
        ║   │                                 │ ║
        ║   │  d_model → ffn_hidden → d_model │ ║
        ║   └─────────────────────────────────┘ ║
        ║                ↓                      ║
        ║   ┌─────────────────────────────────┐ ║
        ║   │  RESIDUAL + LAYER NORM          │ ║
        ║   │                                 │ ║
        ║   │  x = LayerNorm(x + Dropout(ffn))│ ║
        ║   └─────────────────────────────────┘ ║
        ║                                       ║
        ╚═══════════════════════════════════════╝
                            ↓
                  (B, N+1, d_model)
                            ↓
        ┌───────────────────────────────────────┐
        │   6. EXTRACT CLS TOKEN                │
        │                                       │
        │   cls_output = output[:, 0, :]        │
        │                                       │
        │   Output: (B, d_model)                │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   7. MLP HEAD (Classification)        │
        │   (models/amc_transformer.py)         │
        │                                       │
        │   Linear(d_model → num_classes)       │
        │                                       │
        │   Output: (B, num_classes)            │
        └───────────────────────────────────────┘
                            ↓
              CLASS PREDICTIONS (B, 11)
```

---

## Component Details

### Patch Embedding
**File**: `models/embedding/patch_embedding.py`

Converts the input image into a sequence of patch embeddings using a convolutional layer.

```python
Conv2d(in_channels=1,
       out_channels=d_model,
       kernel_size=patch_size,
       stride=patch_size)
```

**Transform**:
- Input: `(B, 1, 32, 64)`
- After Conv2d: `(B, d_model, H', W')` where `H'=32/patch_size`, `W'=64/patch_size`
- After Flatten: `(B, d_model, N)` where `N=H'×W'`
- After Transpose: `(B, N, d_model)`

### Positional Encoding
**File**: `models/embedding/positional_encoding.py`

Adds position information to patch embeddings using sinusoidal functions.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Encoder Layer
**File**: `models/blocks/encoder_layer.py`

Each encoder layer consists of:
1. **Multi-Head Self-Attention** with residual connection and layer normalization
2. **Feed-Forward Network** with residual connection and layer normalization

### Multi-Head Attention
**File**: `models/layers/multi_head_attention.py`

Process:
1. Linear projections: Q, K, V
2. Split into multiple heads
3. Scaled dot-product attention per head
4. Concatenate heads
5. Final linear projection

### MLP Head
**File**: `models/amc_transformer.py:24`

Simple linear layer for final classification:
```python
Linear(d_model → num_classes)
```

---

## Shape Transformation Summary

| Stage | Shape | Description |
|-------|-------|-------------|
| Raw I/Q Data | `(1024, 2)` | 1024 samples, 2 channels (I, Q) |
| After Normalization | `(1024, 2)` | Z-score normalized |
| After Concatenation | `(2048,)` | Flattened I+Q |
| After Reshape | `(1, 32, 64)` | Image format |
| Batch Input | `(B, 1, 32, 64)` | Batched images |
| After Patch Embedding | `(B, N, E)` | N patches, E=d_model |
| After CLS Token | `(B, N+1, E)` | Added class token |
| After Encoder | `(B, N+1, E)` | Transformed features |
| CLS Token Output | `(B, E)` | Extracted class token |
| Final Output | `(B, 11)` | Class logits |

Where:
- `B` = batch size
- `N` = number of patches = `(32/patch_size) × (64/patch_size)`
- `E` = embedding dimension = `d_model`

---

## Key Features

### 1. Multiprocessing-Safe HDF5 Loading
**File**: `dataloader/dataset.py:160-171`

- HDF5 files opened in worker processes (not main process)
- Uses `worker_init_fn` for DataLoader
- Prevents file handle conflicts

### 2. Stratified Data Splitting
**File**: `dataloader/utils.py:102-132`

- Stratifies by both modulation type AND SNR
- Ensures balanced representation across splits

### 3. Vision Transformer Architecture
**File**: `models/encoder.py`

- Patch-based processing of I/Q "images"
- Self-attention mechanism for feature learning
- Position-aware embeddings

### 4. Normalization Strategy
**File**: `dataloader/dataset.py:210-213`

- Separate normalization for I and Q channels
- Statistics calculated from training set
- Applied consistently to validation and test sets

---

## Parameter Flow

### AMCTransformer Initialization
**File**: `models/amc_transformer.py:9`

```python
AMCTransformer(
    in_channels=1,        # Number of input channels
    img_size_h=32,        # Image height
    img_size_w=64,        # Image width
    patch_size=8,         # Patch size (typical)
    num_classes=11,       # Number of modulation classes
    d_model=256,          # Embedding dimension
    n_head=8,             # Number of attention heads
    n_layers=6,           # Number of encoder layers
    ffn_hidden=1024,      # FFN hidden dimension
    drop_prob=0.1,        # Dropout probability
    device='cuda'         # Device
)
```

### Encoder Initialization
**File**: `models/encoder.py:11`

The encoder automatically calculates:
```python
num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
```

For example, with `img_size_h=32`, `img_size_w=64`, `patch_size=8`:
```
num_patches = (32 // 8) × (64 // 8) = 4 × 8 = 32 patches
```

---

## Files Reference

### DataLoader Module
- `dataloader/__init__.py` - Module initialization
- `dataloader/dataset.py` - Dataset class and data loading
- `dataloader/utils.py` - Utility functions for data splitting

### Model Module
- `models/amc_transformer.py` - Main model class
- `models/encoder.py` - Encoder module

### Embedding Components
- `models/embedding/patch_embedding.py` - Patch embedding layer
- `models/embedding/positional_encoding.py` - Positional encoding

### Blocks
- `models/blocks/encoder_layer.py` - Transformer encoder layer

### Layers
- `models/layers/multi_head_attention.py` - Multi-head attention
- `models/layers/position_wise_feed_forward.py` - FFN
- `models/layers/layers_norm.py` - Layer normalization
- `models/layers/scale_dot_product_attention.py` - Scaled dot-product attention

---

## Usage Example

```python
from dataloader.dataset import SingleStreamImageDataset, worker_init_fn
from dataloader.utils import split_data
from models.amc_transformer import AMCTransformer
from torch.utils.data import DataLoader

# 1. Split data
train_idx, valid_idx, test_idx, label_map = split_data(
    file_path='data.h5',
    json_path='classes.json',
    target_mods=['BPSK', 'QPSK', ...],
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=49
)

# 2. Create datasets
train_dataset = SingleStreamImageDataset(
    file_path='data.h5',
    json_path='classes.json',
    target_modulations=['BPSK', 'QPSK', ...],
    mode='train',
    indices=train_idx,
    label_map=label_map,
    normalization_stats=None  # Will be calculated
)

# 3. Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,  # Critical!
    shuffle=True
)

# 4. Create model
model = AMCTransformer(
    in_channels=1,
    img_size_h=32,
    img_size_w=64,
    patch_size=8,
    num_classes=11,
    d_model=256,
    n_head=8,
    n_layers=6,
    ffn_hidden=1024,
    drop_prob=0.1,
    device='cuda'
)

# 5. Training loop
for batch_idx, (images, labels, snrs) in enumerate(train_loader):
    # images: (B, 1, 32, 64)
    # labels: (B,)
    # snrs: (B,)

    outputs = model(images)  # (B, 11)
    # ... compute loss and backprop
```
