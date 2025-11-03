# Transformer for Raw I/Q Signal Classification

A **Pure Transformer** architecture for Automatic Modulation Classification (AMC) that directly processes raw I/Q signal data. This implementation uses Transformer encoder layers with multi-head self-attention to classify radio frequency modulation schemes.

## Overview

This project implements a state-of-the-art deep learning model for classifying different types of radio signal modulations using raw In-phase/Quadrature (I/Q) data. Unlike traditional CNN-based approaches, this model leverages the Transformer architecture to capture long-range dependencies in signal sequences.

### Key Features

- **Pure Transformer Architecture**: Not a Vision Transformer (ViT), but a custom Transformer for 1D signal sequences
- **Raw I/Q Processing**: Directly processes raw I/Q data without feature extraction or preprocessing
- **Flexible Embedding**: Supports both segment-based (1D ViT-style) and full-sequence (Pure Transformer) embeddings
- **CLS Token Classification**: Uses learnable CLS token or global average pooling for classification
- **Multi-Head Self-Attention**: Captures complex temporal patterns in signal data
- **Production-Ready**: Includes training, evaluation, and testing scripts with comprehensive error handling

### Supported Modulations

The model classifies 19 different modulation types:
- **Amplitude Shift Keying (ASK)**: OOK, 4ASK, 8ASK
- **Phase Shift Keying (PSK)**: BPSK, QPSK, 8PSK, 16PSK, 32PSK, OQPSK
- **Amplitude-Phase Shift Keying (APSK)**: 16APSK, 32APSK, 64APSK, 128APSK
- **Quadrature Amplitude Modulation (QAM)**: 16QAM, 32QAM, 64QAM, 128QAM, 256QAM
- **Others**: GMSK

---

## Architecture

### Model Overview

```
Input: [Batch, 2, 1024]  (2 channels: I and Q, 1024 time steps)
   ↓
Sequence Embedding (Conv1D or Segment-based)
   ↓
[Batch, num_tokens, d_model]
   ↓
Add CLS Token (optional)
   ↓
[Batch, num_tokens+1, d_model]
   ↓
Positional Encoding (Sinusoidal)
   ↓
Transformer Encoder Layers × N
   ├── Multi-Head Self-Attention
   ├── Layer Normalization
   ├── Feed-Forward Network
   └── Residual Connections
   ↓
Extract CLS Token or Global Average Pooling
   ↓
Classification Head (LayerNorm + Linear)
   ↓
Output: [Batch, num_classes]
```

### Components Breakdown

#### 1. **Sequence Embedding** (`models/embedding/patch_embedding.py`)

Converts raw I/Q data into token embeddings:

- **Conv1D Method**: Each time step becomes a token
  - Input: `[B, 2, 1024]` → Output: `[B, 1024, d_model]`
  - Creates 1024 tokens for fine-grained temporal modeling

- **Segment Method**: Groups of time steps become tokens
  - Input: `[B, 2, 1024]` → Output: `[B, 16, d_model]` (with segment_size=64)
  - Creates 16 tokens (1024/64) for efficient processing
  - Similar to patch embedding in Vision Transformers

#### 2. **Positional Encoding** (`models/embedding/positional_encoding.py`)

Adds positional information using sinusoidal encoding:
- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
- Allows model to understand temporal order of tokens

#### 3. **Transformer Encoder** (`models/encoder.py`)

Stacks N encoder layers, each containing:

**a) Multi-Head Attention** (`models/layers/multi_head_attention.py`)
- Splits d_model into n_head parallel attention heads
- Each head learns different aspects of signal patterns
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

**b) Feed-Forward Network** (`models/layers/position_wise_feed_forward.py`)
- Two linear transformations with GELU activation
- `FFN(x) = Linear2(GELU(Linear1(x)))`
- Typical hidden dimension: 4 × d_model

**c) Residual Connections & Layer Normalization**
- Helps training stability and gradient flow
- Applied after both attention and FFN

#### 4. **Classification Head** (`models/transformer_rawIQ.py`)

Final classification layer:
- Uses CLS token output (if enabled) or global average pooling
- Layer normalization for stable predictions
- Linear layer to output logits for each modulation class

---

## Directory Structure

```
transformer_rawIQ/
├── models/
│   ├── transformer_rawIQ.py          # Main AMC Transformer model
│   ├── encoder.py                     # Transformer encoder
│   ├── embedding/
│   │   ├── patch_embedding.py        # Sequence embedding (Conv1D/Segment)
│   │   └── positional_encoding.py    # Sinusoidal positional encoding
│   ├── blocks/
│   │   └── encoder_layer.py          # Single encoder layer
│   └── layers/
│       ├── multi_head_attention.py   # Multi-head self-attention
│       ├── scale_dot_product_attention.py
│       ├── position_wise_feed_forward.py
│       └── layers_norm.py            # Layer normalization
│
├── dataloader/
│   ├── dataset.py                    # HDF5 dataset for raw I/Q data
│   ├── utils.py                      # Data splitting utilities
│   └── __init__.py
│
├── training/
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Model evaluation script
│   └── utils.py                      # Training utilities (checkpoints, etc.)
│
├── result/
│   ├── checkpoints/                  # Saved model checkpoints
│   │   └── [experiment_name]/
│   │       ├── config.json
│   │       ├── checkpoint_epoch_X.pth
│   │       └── evaluation/
│   └── logs/                         # Training logs and plots
│
├── test_model.py                     # Quick model testing script
├── main.ipynb                        # Jupyter notebook for experiments
└── README.md                         # This file
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

### Dependencies

```bash
pip install torch torchvision
pip install h5py numpy
pip install tqdm matplotlib seaborn
pip install scikit-learn
```

### Dataset

This project uses the **RadioML 2018.01A** dataset:
- Download from: [RadioML Datasets](https://www.deepsig.ai/datasets)
- Expected format: HDF5 file with keys `X` (signals), `Y` (labels), `Z` (SNR)
- Expected shape: `X`: [N, 1024, 2] where N = number of samples

---

## Usage

### 1. Quick Model Test

Test if the model works correctly:

```bash
python test_model.py
```

This will:
- Create a model with default parameters
- Run a forward pass with dummy data
- Verify output shapes and dimensions
- Test different batch sizes

**Expected Output:**
```
Testing Pure Transformer for Raw I/Q Signal Classification
Device: cuda
Input Data:
   Shape: torch.Size([4, 2, 1024])
   - Batch size: 4
   - Channels: 2 (I and Q)
   - Sequence length: 1024

Model initialized successfully!
Total parameters: XXX,XXX (X.XX M)
Forward pass successful!
Output:
   Shape: torch.Size([4, 11])
   Expected: (4, 11)
```

### 2. Training

Train the model from scratch:

```bash
python training/train.py \
    --experiment_name my_experiment \
    --batch_size 256 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --num_workers 8
```

**Key Training Arguments:**
- `--experiment_name`: Name for saving checkpoints and logs
- `--batch_size`: Batch size (default: 256)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--num_workers`: Number of data loading workers (use >0 for HDF5)
- `--resume`: Path to checkpoint to resume training

**Training Configuration** (in `training/train.py`):
```python
# Model Architecture
D_MODEL = 128           # Embedding dimension
N_HEAD = 8              # Number of attention heads
N_LAYERS = 6            # Number of encoder layers
FFN_HIDDEN = 1024       # Feed-forward hidden dimension
DROP_PROB = 0.2         # Dropout probability

# Embedding Configuration
EMBEDDING_TYPE = 'segment'  # 'segment' or 'conv1d'
SEGMENT_SIZE = 16           # Segment size (for segment mode)
USE_CLS_TOKEN = True        # Use CLS token

# Training Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP_MAX_NORM = 1.0
```

### 3. Evaluation

Evaluate a trained model on test set:

```bash
python training/evaluate.py \
    --checkpoint result/checkpoints/my_experiment/model_best.pth \
    --dataset test \
    --batch_size 256
```

This generates:
- Confusion matrix (normalized and raw counts)
- Classification report (precision, recall, F1-score)
- Per-class accuracy metrics
- Results saved to `result/checkpoints/[experiment]/evaluation/`

### 4. Resume Training

Resume from a saved checkpoint:

```bash
python training/train.py \
    --resume result/checkpoints/my_experiment/checkpoint_epoch_50.pth \
    --experiment_name my_experiment_continued
```

---

## Model Configuration

### Architecture Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `d_model` | Embedding dimension | 128 | 64-512 |
| `n_head` | Number of attention heads | 8 | 4-16 |
| `n_layers` | Number of encoder layers | 6 | 2-12 |
| `ffn_hidden` | FFN hidden dimension | 1024 | d_model×2 to d_model×8 |
| `drop_prob` | Dropout probability | 0.2 | 0.0-0.5 |

### Embedding Configuration

| Parameter | Description | Options |
|-----------|-------------|---------|
| `embedding_type` | Type of sequence embedding | `'segment'`, `'conv1d'` |
| `segment_size` | Size of each segment (for segment mode) | 16, 32, 64 |
| `use_cls_token` | Use CLS token for classification | `True`, `False` |

**Embedding Comparison:**

- **Segment (1D ViT-style)**:
  - Pros: Fewer tokens, faster training, lower memory
  - Cons: Coarser temporal resolution
  - Best for: Limited GPU memory, large models

- **Conv1D (Pure Transformer)**:
  - Pros: Fine-grained temporal modeling, captures all details
  - Cons: More tokens, slower, higher memory
  - Best for: Maximum performance, sufficient GPU memory

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `BATCH_SIZE` | Batch size | 256 |
| `NUM_EPOCHS` | Maximum epochs | 100 |
| `LEARNING_RATE` | Initial learning rate | 1e-4 |
| `WEIGHT_DECAY` | L2 regularization | 1e-4 |
| `LABEL_SMOOTHING` | Label smoothing factor | 0.1 |
| `GRAD_CLIP_MAX_NORM` | Gradient clipping threshold | 1.0 |
| `PATIENCE` | Early stopping patience | 10 |

---

## Data Processing Pipeline

### 1. **Data Loading** (`dataloader/dataset.py`)

The `SingleStreamImageDataset` class handles:
- **Multiprocessing-safe HDF5 loading**: Each worker opens its own file handle
- **Normalization**: Z-score normalization per channel (I and Q separately)
- **On-the-fly processing**: Minimal memory footprint
- **Returns**: `(iq_data, label, snr)` tuples
  - `iq_data`: Tensor of shape `[2, 1024]`
  - `label`: Integer class label
  - `snr`: Signal-to-noise ratio (float)

### 2. **Data Splitting** (`dataloader/utils.py`)

- Train/Validation/Test split: 70%/15%/15% (configurable)
- Stratified by modulation class
- Reproducible with fixed seed

### 3. **Normalization**

Computed on training set and applied to all splits:
```python
I_normalized = (I - mean_I) / std_I
Q_normalized = (Q - mean_Q) / std_Q
```

### 4. **Data Shape Flow**

```
HDF5 file: [N, 1024, 2]
   ↓ (transpose)
Dataset output: [2, 1024]
   ↓ (batch)
DataLoader: [Batch, 2, 1024]
   ↓ (embedding)
Tokens: [Batch, num_tokens, d_model]
   ↓ (encoder)
Output: [Batch, d_model]
   ↓ (classification head)
Logits: [Batch, num_classes]
```

---

## Model Checkpoints

Checkpoints are saved in `result/checkpoints/[experiment_name]/` and include:

```python
{
    'epoch': int,                    # Last completed epoch
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'val_loss': float,               # Validation loss
    'history': {                     # Training history
        'train_loss': [...],
        'train_acc': [...],
        'val_loss': [...],
        'val_acc': [...]
    },
    'config': {...}                  # Full configuration
}
```

### Checkpoint Types

- `checkpoint_epoch_X.pth`: Regular checkpoints (every SAVE_FREQ epochs)
- `model_best.pth`: Best model (lowest validation loss)
- `model_final.pth`: Final model after all epochs
- `checkpoint_interrupted.pth`: Saved on keyboard interrupt

---

## Training Process

### 1. **Optimization**

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Initial LR with ReduceLROnPlateau scheduler
  - Reduces LR by 0.5 when validation loss plateaus (patience=5)
- **Gradient Clipping**: Max norm = 1.0 to prevent exploding gradients

### 2. **Loss Function**

CrossEntropyLoss with label smoothing (0.1):
- Prevents overconfident predictions
- Improves generalization

### 3. **Early Stopping**

- Monitors validation loss
- Patience: 10 epochs
- Saves best model automatically

### 4. **Performance Optimizations**

- **Mixed Precision**: Enabled via TF32 on Ampere+ GPUs
- **Pin Memory**: For faster CPU-to-GPU transfers
- **Persistent Workers**: Keeps data loading workers alive between epochs
- **Prefetch Factor**: Pre-loads batches for minimal GPU idle time

---

## Evaluation Metrics

The evaluation script (`training/evaluate.py`) generates:

### 1. **Confusion Matrix**

Two versions:
- **Normalized**: Shows percentages (0-1 scale)
- **Raw Counts**: Shows actual sample counts

Saved as: `[dataset]_confusion_matrix_[normalized/counts].png`

### 2. **Classification Report**

Per-class metrics:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class

Saved as: `[dataset]_classification_report.txt`

### 3. **Overall Metrics**

- **Accuracy**: Overall classification accuracy
- **Macro Average**: Unweighted mean of per-class metrics
- **Weighted Average**: Weighted by class support

---

## Typical Results

Based on the production model (`production_rawIQv1`):

**Configuration:**
- Embedding: Segment-based (segment_size=64)
- Architecture: 6 layers, 8 heads, d_model=128
- Training: 300 epochs, batch_size=256

**Performance:**
- Test accuracy: ~XX% (check `result/checkpoints/production_rawIQv1/evaluation/`)
- Training converges after ~50-100 epochs
- Best performance at SNR > 0 dB

---

## Tips and Best Practices

### 1. **Choosing Embedding Type**

- Start with `segment` for faster experimentation
- Use `segment_size=64` for balanced performance/speed
- Try `conv1d` for maximum accuracy with sufficient GPU memory

### 2. **Model Size**

- Smaller models (d_model=64-128, n_layers=4-6): Faster, less overfitting
- Larger models (d_model=256-512, n_layers=8-12): Better capacity, needs more data

### 3. **Regularization**

If overfitting:
- Increase `drop_prob` (0.1 → 0.3)
- Increase `weight_decay` (1e-4 → 1e-3)
- Increase `label_smoothing` (0.1 → 0.2)

### 4. **Learning Rate**

- Start with 1e-4 for Adam/AdamW
- Use ReduceLROnPlateau for automatic adjustment
- If loss explodes: reduce LR by 10x

### 5. **Batch Size**

- Larger batch sizes (256-512): More stable gradients, faster training
- Smaller batch sizes (32-128): Better generalization, more noise in gradients
- Adjust based on GPU memory

### 6. **Number of Workers**

- **IMPORTANT**: Must use `num_workers > 0` for HDF5 dataset
- Recommended: 4-8 workers per GPU
- Too many workers: Overhead, slower loading

---

## Troubleshooting

### Common Issues

**1. HDF5 file locking error**
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```
Or set in training script (already handled).

**2. CUDA out of memory**
- Reduce batch size
- Use segment embedding instead of conv1d
- Reduce d_model or n_layers

**3. Training not converging**
- Check learning rate (may be too high/low)
- Verify data normalization
- Ensure sufficient data diversity

**4. DataLoader hangs**
- Set `num_workers > 0` (required for HDF5)
- Check worker_init_fn is passed to DataLoader

---

## File Descriptions

### Model Files

- **`models/transformer_rawIQ.py`**: Main model class `AMCTransformer`
- **`models/encoder.py`**: Transformer encoder with embedding and positional encoding
- **`models/blocks/encoder_layer.py`**: Single encoder layer (attention + FFN)
- **`models/layers/multi_head_attention.py`**: Multi-head self-attention mechanism
- **`models/layers/position_wise_feed_forward.py`**: Position-wise FFN
- **`models/embedding/patch_embedding.py`**: Sequence embedding (Conv1D/Segment)
- **`models/embedding/positional_encoding.py`**: Sinusoidal positional encoding

### Data Files

- **`dataloader/dataset.py`**: `SingleStreamImageDataset` class for HDF5 loading
- **`dataloader/utils.py`**: Data splitting and preprocessing utilities

### Training Files

- **`training/train.py`**: Main training script with CLI arguments
- **`training/evaluate.py`**: Evaluation script for generating metrics
- **`training/utils.py`**: Helper functions (checkpointing, plotting, early stopping)

### Other Files

- **`test_model.py`**: Quick test script to verify model architecture
- **`main.ipynb`**: Jupyter notebook for interactive experiments

---

## Key Implementation Details

### 1. **Why Not ViT?**

This is NOT a Vision Transformer because:
- Input is 1D signal sequence, not 2D image
- No patch grid or 2D spatial relationships
- Sequence embedding (Conv1D), not patch embedding
- Optimized for temporal patterns in signals

### 2. **CLS Token**

- Learnable token prepended to sequence
- Aggregates information from all tokens via self-attention
- Alternative: Global average pooling (set `use_cls_token=False`)

### 3. **Segment-based Embedding**

Similar to ViT patches but for 1D:
- Groups consecutive time steps into segments
- Reduces sequence length: 1024 → 16 tokens (segment_size=64)
- More efficient than processing every time step

### 4. **Multi-Head Attention**

- Splits d_model into n_head independent attention heads
- Each head: d_k = d_model / n_head
- Parallel processing of different signal aspects
- Concatenated and projected back to d_model

---

## References

- **Transformer**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- **Vision Transformer**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- **RadioML Dataset**: O'Shea et al., "Over-the-Air Deep Learning Based Radio Signal Classification", IEEE 2018

---

## Citation

If you use this code, please cite:

```bibtex
@misc{transformer_rawiq,
  author = {Your Name},
  title = {Transformer for Raw I/Q Signal Classification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-repo}
}
```

---

## License

[Your chosen license]

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

## Acknowledgments

- RadioML dataset by DeepSig Inc.
- PyTorch team for the excellent deep learning framework
- Transformer architecture by Vaswani et al.
