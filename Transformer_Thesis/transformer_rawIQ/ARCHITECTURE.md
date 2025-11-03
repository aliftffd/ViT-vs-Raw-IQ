# AMC Transformer Architecture for Raw I/Q Signals

## Overview
This document visualizes the complete data flow from raw I/Q signal preprocessing in the `dataloader/` module to classification output through the Transformer model in `models/`.

---

## Table of Contents
1. [Data Loading Pipeline](#1-data-loading-pipeline)
2. [Input Shape Transformations](#2-input-shape-transformations)
3. [Model Architecture](#3-model-architecture)
4. [Component Details](#4-component-details)
5. [Complete Data Flow](#5-complete-data-flow)

---

## 1. Data Loading Pipeline

### 1.1 Data Source
- **Input Format**: HDF5 file containing raw I/Q signal data
- **Location**: `dataloader/dataset.py` - `SingleStreamImageDataset`

### 1.2 Data Preprocessing Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                    HDF5 FILE STRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  'X': Signal data [N, 1024, 2]                                  │
│       - N: Number of samples                                    │
│       - 1024: Sequence length (time samples)                    │
│       - 2: I and Q channels                                     │
│                                                                 │
│  'Y': One-hot encoded labels [N, num_classes]                  │
│  'Z': SNR values [N, 1]                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Splitting
**File**: `dataloader/utils.py` - `split_data()`

```
Raw Dataset
     │
     ├─► Stratified by: Modulation Type + SNR
     │
     ├──► Training Set (default: 70%)
     ├──► Validation Set (default: 15%)
     └──► Test Set (default: 15%)
```

### 1.4 Normalization
**Calculated during training** (`dataset.py:115-157`):

```python
# Statistics calculated from training data
normalization_stats = {
    'i_mean': mean(I_channel),
    'i_std': std(I_channel),
    'q_mean': mean(Q_channel),
    'q_std': std(Q_channel)
}

# Applied per sample
I_normalized = (I - i_mean) / i_std
Q_normalized = (Q - q_mean) / q_std
```

### 1.5 DataLoader Output
**From** `dataset.py:180-224` - `__getitem__()`

```
Input:  [1024, 2]  (sequence_length, channels)
        ↓ transpose
Output: [2, 1024]  (channels, sequence_length)

Returns: (iq_data, label, snr)
    - iq_data: torch.Tensor [2, 1024]
    - label: int (class index)
    - snr: float (SNR in dB)
```

---

## 2. Input Shape Transformations

### Complete Batch Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    SHAPE TRANSFORMATION FLOW                     │
└──────────────────────────────────────────────────────────────────┘

1. HDF5 Storage:         [1024, 2]

2. After Normalization:  [1024, 2]

3. After Transpose:      [2, 1024]
                          ↓ I channel (index 0)
                          ↓ Q channel (index 1)

4. Batched:             [B, 2, 1024]
                         │  │  └─ sequence length
                         │  └──── channels (I/Q)
                         └─────── batch size

5. Model Input:         [B, 2, 1024]
```

---

## 3. Model Architecture

### 3.1 High-Level Architecture
**File**: `models/transformer_rawIQ.py` - `AMCTransformer`

```
┌─────────────────────────────────────────────────────────────────┐
│                     AMCTransformer                              │
│                                                                 │
│  Input: [B, 2, 1024]                                           │
│    │                                                            │
│    ├──► Encoder                                                │
│    │      │                                                     │
│    │      ├─► Sequence Embedding                               │
│    │      ├─► CLS Token (optional)                             │
│    │      ├─► Positional Encoding                              │
│    │      └─► N × Encoder Layers                               │
│    │                                                            │
│    │    Output: [B, num_tokens, d_model]                       │
│    │                                                            │
│    ├──► Aggregation (CLS token or GAP)                         │
│    │      │                                                     │
│    │      Output: [B, d_model]                                 │
│    │                                                            │
│    └──► Classification Head                                    │
│           │                                                     │
│           ├─► LayerNorm                                        │
│           └─► Linear(d_model → num_classes)                    │
│                                                                 │
│  Output: [B, num_classes]                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Encoder Architecture
**File**: `models/encoder.py` - `Encoder`

```
Input: [B, 2, 1024]
  │
  ├─► SequenceEmbedding
  │    │
  │    ├─► Method 1: Conv1D (kernel_size=1)
  │    │    Input:  [B, 2, 1024]
  │    │    Output: [B, 1024, d_model]
  │    │    (Full sequence, each timestep → embedding)
  │    │
  │    └─► Method 2: Segment (kernel_size=segment_size, stride=segment_size)
  │         Input:  [B, 2, 1024]
  │         Output: [B, num_segments, d_model]
  │         (e.g., 1024/64 = 16 segments)
  │
  ├─► CLS Token Addition (if enabled)
  │    Input:  [B, num_tokens, d_model]
  │    Output: [B, num_tokens+1, d_model]
  │    (CLS token prepended to sequence)
  │
  ├─► Positional Encoding
  │    (Add sinusoidal position information)
  │    Output: [B, num_tokens, d_model]
  │
  ├─► Dropout
  │
  └─► N × Encoder Layers
       │
       ├─► Multi-Head Self-Attention
       ├─► Add & Norm
       ├─► Position-wise FFN
       └─► Add & Norm

  Output: [B, num_tokens, d_model]
```

---

## 4. Component Details

### 4.1 Sequence Embedding
**File**: `models/embedding/patch_embedding.py` - `SequenceEmbedding`

#### Method 1: Conv1D (Full Sequence)
```
Purpose: Project each timestep independently to embedding dimension

Input:  [B, 2, 1024]
  │
  ├─► Conv1d(in_channels=2, out_channels=d_model, kernel_size=1)
  │
Output: [B, d_model, 1024]
  │
  ├─► Transpose(1, 2)
  │
Final:  [B, 1024, d_model]

Result: 1024 tokens, each representing one timestep
```

#### Method 2: Segment-Based
```
Purpose: Group consecutive timesteps into segments

Example: segment_size=64

Input:  [B, 2, 1024]
  │
  ├─► Conv1d(in_channels=2, out_channels=d_model,
  │          kernel_size=64, stride=64)
  │
Output: [B, d_model, 16]  (1024/64 = 16 segments)
  │
  ├─► Transpose(1, 2)
  │
Final:  [B, 16, d_model]

Result: 16 tokens, each representing 64 timesteps
```

### 4.2 Positional Encoding
**File**: `models/embedding/positional_encoding.py` - `PositionalEncoding`

```
Purpose: Add position information using sinusoidal functions

Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Input:  [B, num_tokens, d_model]
  │
  ├─► Get PE[:num_tokens, :] from precomputed buffer
  │
  ├─► Add PE to input (element-wise)
  │
Output: [B, num_tokens, d_model]
```

### 4.3 Encoder Layer
**File**: `models/blocks/encoder_layer.py` - `EncoderLayer`

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENCODER LAYER                              │
└─────────────────────────────────────────────────────────────────┘

Input: [B, num_tokens, d_model]
  │
  ├─► Multi-Head Self-Attention
  │    │
  │    ├─► Q, K, V = Linear(input)
  │    ├─► Split into n_head heads
  │    ├─► Scaled Dot-Product Attention per head
  │    ├─► Concat heads
  │    └─► Linear projection
  │
  ├─► Add & Norm (Residual Connection + Layer Norm)
  │
  ├─► Position-wise Feed-Forward Network
  │    │
  │    ├─► Linear(d_model → ffn_hidden)
  │    ├─► ReLU
  │    ├─► Dropout
  │    └─► Linear(ffn_hidden → d_model)
  │
  └─► Add & Norm (Residual Connection + Layer Norm)

Output: [B, num_tokens, d_model]
```

### 4.4 Multi-Head Attention
**File**: `models/layers/multi_head_attention.py` - `MultiHeadAttention`

```
Input: Q, K, V = [B, num_tokens, d_model]
  │
  ├─► Linear Projections
  │    Q' = W_q @ Q  [B, num_tokens, d_model]
  │    K' = W_k @ K  [B, num_tokens, d_model]
  │    V' = W_v @ V  [B, num_tokens, d_model]
  │
  ├─► Split into n_head heads
  │    Shape: [B, n_head, num_tokens, d_model/n_head]
  │
  ├─► Scaled Dot-Product Attention (per head)
  │    Attention(Q,K,V) = softmax(Q@K^T / sqrt(d_k)) @ V
  │    │
  │    Output: [B, n_head, num_tokens, d_model/n_head]
  │
  ├─► Concatenate heads
  │    Shape: [B, num_tokens, d_model]
  │
  └─► Output projection
       W_o @ concat

Output: [B, num_tokens, d_model]
```

### 4.5 Position-wise Feed-Forward Network
**File**: `models/layers/position_wise_feed_forward.py` - `PositionwiseFeedForward`

```
Input: [B, num_tokens, d_model]
  │
  ├─► Linear(d_model → ffn_hidden)
  │
  ├─► ReLU
  │
  ├─► Dropout
  │
  ├─► Linear(ffn_hidden → d_model)
  │
Output: [B, num_tokens, d_model]

Note: Applied independently to each token (position-wise)
```

### 4.6 Classification Head
**File**: `models/transformer_rawIQ.py:67-70`

```
Input: [B, d_model]  (aggregated representation)
  │
  ├─► LayerNorm(d_model)
  │
  ├─► Linear(d_model → num_classes)
  │
Output: [B, num_classes]  (logits)
```

---

## 5. Complete Data Flow

### 5.1 End-to-End Example

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPLETE DATA FLOW                            │
└─────────────────────────────────────────────────────────────────┘

Example Configuration:
- Batch size (B) = 32
- Sequence length = 1024
- Input channels = 2 (I/Q)
- d_model = 256
- n_head = 8
- n_layers = 6
- ffn_hidden = 1024
- num_classes = 11
- embedding_type = 'segment'
- segment_size = 64

Step 1: Data Loading (dataloader/dataset.py)
───────────────────────────────────────────────
HDF5: [1024, 2]
  ↓ normalize I and Q channels
  ↓ transpose to [2, 1024]
  ↓ batch
DataLoader: [32, 2, 1024]


Step 2: Sequence Embedding (models/embedding/patch_embedding.py)
─────────────────────────────────────────────────────────────────
Input:  [32, 2, 1024]
  ↓ Conv1d(2, 256, kernel_size=64, stride=64)
  ↓ output: [32, 256, 16]
  ↓ transpose(1, 2)
Output: [32, 16, 256]
        (16 segments, each segment = 64 timesteps)


Step 3: CLS Token Addition (models/encoder.py)
───────────────────────────────────────────────
Input:  [32, 16, 256]
  ↓ prepend learnable CLS token
Output: [32, 17, 256]


Step 4: Positional Encoding (models/embedding/positional_encoding.py)
──────────────────────────────────────────────────────────────────────
Input:  [32, 17, 256]
  ↓ add sinusoidal position embeddings
Output: [32, 17, 256]


Step 5: Transformer Encoder (models/blocks/encoder_layer.py)
─────────────────────────────────────────────────────────────
Input:  [32, 17, 256]
  ↓
  ├─► Layer 1:
  │    ├─ Multi-Head Attention (8 heads)
  │    ├─ Add & Norm
  │    ├─ FFN (256 → 1024 → 256)
  │    └─ Add & Norm
  │
  ├─► Layer 2: (same structure)
  ├─► Layer 3: (same structure)
  ├─► Layer 4: (same structure)
  ├─► Layer 5: (same structure)
  └─► Layer 6: (same structure)
  ↓
Output: [32, 17, 256]


Step 6: Aggregation (models/transformer_rawIQ.py)
──────────────────────────────────────────────────
Option A - CLS Token (use_cls_token=True):
  Input:  [32, 17, 256]
    ↓ select first token (CLS)
  Output: [32, 256]

Option B - Global Average Pooling (use_cls_token=False):
  Input:  [32, 17, 256]
    ↓ mean along token dimension
  Output: [32, 256]


Step 7: Classification Head (models/transformer_rawIQ.py)
──────────────────────────────────────────────────────────
Input:  [32, 256]
  ↓ LayerNorm
  ↓ Linear(256 → 11)
Output: [32, 11] (logits for 11 modulation classes)


Step 8: Loss Computation & Training
────────────────────────────────────
Logits: [32, 11]
Labels: [32] (class indices)
  ↓ CrossEntropyLoss
Loss: scalar
  ↓ backward
  ↓ optimizer.step()
```

### 5.2 Shape Summary Table

| Stage | Location | Input Shape | Output Shape | Description |
|-------|----------|-------------|--------------|-------------|
| **HDF5 Storage** | `dataloader/dataset.py` | - | `[1024, 2]` | Raw I/Q from file |
| **Normalize** | `dataset.py:216-217` | `[1024, 2]` | `[1024, 2]` | Z-score normalization |
| **Transpose** | `dataset.py:222` | `[1024, 2]` | `[2, 1024]` | Channels-first format |
| **Batch** | DataLoader | `[2, 1024]` | `[B, 2, 1024]` | Collate into batch |
| **Embedding** | `encoder.py:101` | `[B, 2, 1024]` | `[B, 16, 256]` | Conv1d projection |
| **Add CLS** | `encoder.py:107` | `[B, 16, 256]` | `[B, 17, 256]` | Prepend CLS token |
| **Pos Encode** | `encoder.py:110` | `[B, 17, 256]` | `[B, 17, 256]` | Add positional info |
| **Encoder** | `encoder.py:114-115` | `[B, 17, 256]` | `[B, 17, 256]` | 6 encoder layers |
| **CLS Extract** | `transformer_rawIQ.py:90` | `[B, 17, 256]` | `[B, 256]` | Select CLS token |
| **Classifier** | `transformer_rawIQ.py:96` | `[B, 256]` | `[B, 11]` | Linear + LayerNorm |

---

## 6. Key Design Choices

### 6.1 Raw I/Q Processing
- **Input**: Raw I/Q signals `[2, 1024]` instead of spectrograms
- **Advantage**: Preserves all signal information without hand-crafted features
- **Challenge**: Requires model to learn time-domain patterns

### 6.2 Embedding Strategy
Two options available:

1. **Full Sequence** (`conv1d`):
   - Creates 1024 tokens (one per timestep)
   - More fine-grained but higher memory usage
   - Better for capturing detailed temporal patterns

2. **Segment-based** (`segment`):
   - Groups timesteps into segments (e.g., 16 segments of 64 timesteps)
   - Lower memory, faster training
   - Captures local patterns within segments

### 6.3 CLS Token vs Global Average Pooling
- **CLS Token**: Learnable aggregation, standard in ViT/BERT
- **GAP**: Parameter-free, more robust to sequence length changes

### 6.4 Architecture Hyperparameters
From `encoder.py` and `transformer_rawIQ.py`:
- `d_model`: Embedding dimension (e.g., 256, 512)
- `n_head`: Number of attention heads (typically 8)
- `n_layers`: Number of encoder layers (e.g., 6, 12)
- `ffn_hidden`: FFN hidden size (typically 4 × d_model)
- `drop_prob`: Dropout rate for regularization

---

## 7. File Reference Map

### Data Loading
- `dataloader/dataset.py` - Main dataset class, normalization, HDF5 handling
- `dataloader/utils.py` - Data splitting, metadata loading
- `dataloader/__init__.py` - Module initialization

### Model Components
- `models/transformer_rawIQ.py` - Top-level model class
- `models/encoder.py` - Transformer encoder
- `models/embedding/patch_embedding.py` - Sequence embedding (Conv1d)
- `models/embedding/positional_encoding.py` - Positional encoding
- `models/blocks/encoder_layer.py` - Single encoder layer
- `models/layers/multi_head_attention.py` - Multi-head attention
- `models/layers/position_wise_feed_forward.py` - Feed-forward network
- `models/layers/scale_dot_product_attention.py` - Attention mechanism
- `models/layers/layers_norm.py` - Layer normalization

---

## 8. Quick Reference: Key Line Numbers

### Data Pipeline
- Normalization calculation: `dataset.py:115-157`
- Data loading: `dataset.py:180-224`
- Data splitting: `utils.py:58-148`

### Model Architecture
- AMCTransformer forward: `transformer_rawIQ.py:72-97`
- Encoder forward: `encoder.py:86-117`
- Embedding: `patch_embedding.py:47-60`
- Positional encoding: `positional_encoding.py:52-82`
- Encoder layer: `encoder_layer.py:18-35`
- Multi-head attention: `multi_head_attention.py:16-32`

---

## 9. Mathematical Formulations

### 9.1 Normalization
```
I_norm = (I - μ_I) / σ_I
Q_norm = (Q - μ_Q) / σ_Q
```

### 9.2 Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 9.3 Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 9.4 Position-wise Feed-Forward
```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### 9.5 Layer Normalization
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```

### 9.6 Residual Connection
```
Output = LayerNorm(x + Sublayer(x))
```

---

## 10. Performance Considerations

### Memory Usage
- **Full Sequence** (1024 tokens): ~4× memory vs 16 segments
- **Batch Size**: Critical trade-off with sequence length
- **Multi-worker DataLoader**: Ensures HDF5 thread safety

### Training Tips
- Use `worker_init_fn` for HDF5 file handling (`dataset.py:20-38`)
- Calculate normalization stats once on training data
- Stratify splits by both modulation type and SNR
- Monitor attention patterns for interpretability

---

## Conclusion

This architecture implements a pure Transformer model for automatic modulation classification from raw I/Q signals. The design emphasizes:

1. **End-to-end learning**: No manual feature engineering
2. **Flexibility**: Multiple embedding strategies (full vs segment)
3. **Efficiency**: Multiprocessing-safe HDF5 loading
4. **Standard architecture**: Based on "Attention is All You Need" (Vaswani et al., 2017)

The data flows from normalized raw I/Q signals through learned embeddings, positional encoding, and multiple Transformer encoder layers to produce modulation class predictions.
