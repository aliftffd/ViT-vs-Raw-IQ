# VRAM Optimization Guide: Maximizing GPU Utilization
## From 5GB → 14GB+ Usage for Faster Training

**Current Status:**
- VRAM Usage: ~5GB / 16GB (31% utilization)
- Training Speed: 35 it/s
- **Problem:** Underutilizing GPU resources

**Target:**
- VRAM Usage: 12-14GB / 16GB (80-85% utilization)
- Training Speed: 60-100+ it/s
- **Goal:** 2-3× speedup through better resource utilization

---

## Table of Contents
1. [Quick Diagnosis](#quick-diagnosis)
2. [Immediate Fixes (Easy)](#immediate-fixes-easy)
3. [Advanced Optimizations](#advanced-optimizations)
4. [Configuration Changes](#configuration-changes)
5. [Verification & Monitoring](#verification--monitoring)
6. [Expected Results](#expected-results)

---

## Quick Diagnosis

### Why Only 5GB VRAM Usage?

Your current configuration likely has **conservative settings**:

```python
# Current (CONSERVATIVE) - train_optimized.py:84-92
BATCH_SIZE = 128          # ← TOO SMALL for 16GB VRAM
NUM_WORKERS = 8           # ← Good
PREFETCH_FACTOR = 3       # ← Good
USE_AMP = True            # ✓ Good
D_MODEL = 256             # ← Can increase
N_LAYERS = 9              # ← Can increase
```

**The bottleneck:** Small batch size + moderate model size = GPU underutilized

---

## Immediate Fixes (Easy)

### Fix #1: Increase Batch Size (Biggest Impact)

**Edit `train_optimized.py` Line 84:**

```python
# OLD:
BATCH_SIZE = 128

# NEW (for 16GB VRAM):
BATCH_SIZE = 256  # 2× increase → ~2× speed improvement
```

**Why this works:**
- Batch size has **linear** impact on VRAM usage
- 128→256 doubles GPU utilization
- More parallelism = better GPU efficiency
- Expected: 5GB → 9-10GB usage

**Verification:**
```bash
python train_optimized.py --tune --n_trials 5 --study_name test_batch256
```

Monitor VRAM with:
```bash
watch -n 1 nvidia-smi
```

---

### Fix #2: Increase Model Capacity

**Edit `train_optimized.py` Lines 77-80:**

```python
# OLD:
D_MODEL = 256
N_HEAD = 16
N_LAYERS = 9
FFN_HIDDEN = 512

# NEW (for 16GB VRAM):
D_MODEL = 384        # +50% capacity
N_HEAD = 16          # Keep (must divide 384)
N_LAYERS = 12        # +33% depth
FFN_HIDDEN = 768     # 2× d_model (standard)
```

**Impact:**
- More parameters = more computation = better GPU utilization
- Expected additional VRAM: +2-3GB
- May improve model accuracy too!

---

### Fix #3: Aggressive Optuna Search Space

**Edit `train_optimized.py` Lines 427-443:**

```python
# OLD (Conservative):
d_model = trial.suggest_categorical("d_model", [128, 192, 256, 320])
n_layers = trial.suggest_int("n_layers", 6, 12)

# NEW (Aggressive):
d_model = trial.suggest_categorical("d_model", [256, 320, 384, 448])
n_layers = trial.suggest_int("n_layers", 9, 15)
```

**Edit Lines 454-462 (Batch Size Logic):**

```python
# OLD:
if d_model <= 192:
    batch_size = trial.suggest_categorical("batch_size", [96, 128, 160])
elif d_model == 256:
    batch_size = trial.suggest_categorical("batch_size", [96, 128, 144])
else:  # d_model >= 320
    batch_size = trial.suggest_categorical("batch_size", [64, 96, 128])

# NEW (More aggressive):
if d_model <= 256:
    batch_size = trial.suggest_categorical("batch_size", [192, 224, 256])
elif d_model <= 384:
    batch_size = trial.suggest_categorical("batch_size", [128, 160, 192])
else:  # d_model >= 448
    batch_size = trial.suggest_categorical("batch_size", [96, 128, 160])
```

---

## Advanced Optimizations

### Optimization A: Increase Prefetch Factor

**Edit `train_optimized.py` Line 93:**

```python
# OLD:
PREFETCH_FACTOR = 3

# NEW:
PREFETCH_FACTOR = 5  # More aggressive data pipeline
```

**Why:**
- Keeps GPU fed with data
- Reduces GPU idle time waiting for batches
- Uses more RAM but speeds up training

---

### Optimization B: Enable Gradient Accumulation (If needed)

**For even larger effective batch sizes without VRAM increase:**

Add to Config class (after line 89):

```python
# Gradient accumulation
ACCUMULATION_STEPS = 2  # Effective batch = BATCH_SIZE × 2
```

**Modify `train_epoch()` function (around line 240):**

```python
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, scaler=None, trial=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    accumulation_steps = getattr(config, 'ACCUMULATION_STEPS', 1)

    desc = f'Epoch {epoch+1} [Train]'
    if trial:
        desc = f'Trial {trial.number} Epoch {epoch+1} [Train]'

    pbar = tqdm(train_loader, desc=desc, leave=False)
    use_amp = config.USE_AMP and scaler is not None

    for batch_idx, (images, labels, snrs) in enumerate(pbar):
        if batch_idx % config.CACHE_CLEAR_FREQUENCY == 0 and batch_idx > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # ← Scale loss

            scaler.scale(loss).backward()

            # Only step optimizer every N batches
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc
```

**Benefit:**
- Effective batch size = 256 × 2 = 512
- Better gradient estimates
- No extra VRAM needed

---

### Optimization C: Reduce Cache Clearing Frequency

**Edit `train_optimized.py` Line 101:**

```python
# OLD:
CACHE_CLEAR_FREQUENCY = 50

# NEW:
CACHE_CLEAR_FREQUENCY = 100  # Clear less often
```

**Why:**
- Cache clearing has overhead
- Less frequent = faster training
- Still safe for 16GB VRAM

---

### Optimization D: Increase Worker Count (If CPU/RAM allows)

**Edit `train_optimized.py` Line 92:**

```python
# OLD:
NUM_WORKERS = 8

# NEW (if you have 16+ CPU cores):
NUM_WORKERS = 12
```

**Check your CPU cores:**
```bash
nproc  # Shows available cores
```

**RAM consideration:**
- Each worker loads batches into RAM
- With 32GB RAM, 12 workers is safe
- Faster data loading = less GPU idle time

---

## Configuration Changes

### Complete Optimized Config Block

Replace lines 72-102 in `train_optimized.py`:

```python
    # --- Model architecture (OPTIMIZED FOR 16GB VRAM) ---
    SEQ_LENGTH = 1024
    EMBEDDING_TYPE = 'segment'
    SEGMENT_SIZE = 16
    USE_CLS_TOKEN = True
    D_MODEL = 384        # ← INCREASED from 256
    N_HEAD = 16          # ← Unchanged (divides 384)
    N_LAYERS = 12        # ← INCREASED from 9
    FFN_HIDDEN = 768     # ← INCREASED (2× d_model)
    DROP_PROB = 0.1

    # Training hyperparameters (OPTIMIZED)
    BATCH_SIZE = 256     # ← DOUBLED from 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_MAX_NORM = 1.0

    # DataLoader settings (OPTIMIZED)
    NUM_WORKERS = 12     # ← INCREASED from 8
    PREFETCH_FACTOR = 5  # ← INCREASED from 3
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    # Mixed precision training
    USE_AMP = True

    # Memory management (OPTIMIZED)
    CACHE_CLEAR_FREQUENCY = 100  # ← INCREASED from 50
    EMPTY_CACHE_BETWEEN_TRIALS = True

    # Gradient accumulation (NEW)
    ACCUMULATION_STEPS = 1  # Set to 2 for effective batch=512
```

---

## Verification & Monitoring

### Real-time VRAM Monitoring

**Terminal 1 - Run Training:**
```bash
python train_optimized.py --tune --n_trials 5 --study_name optimized_test
```

**Terminal 2 - Monitor GPU:**
```bash
watch -n 0.5 nvidia-smi
```

**What to look for:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx      Driver Version: 535.xx       CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Utilization |         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  RTX 3090        On       |  12500MiB / 16384MiB |    95%      Default  |
+-----------------------------------------------------------------------------+
```

**Target metrics:**
- **Memory Usage:** 12-14GB (75-85%)
- **GPU Utilization:** 90-100%
- **Training Speed:** 60-100 it/s (up from 35 it/s)

---

### Performance Benchmarking

**Test different batch sizes:**

```bash
# Baseline
python train_optimized.py --batch_size 128 --num_epochs 1 --experiment_name bench_128

# Test 192
python train_optimized.py --batch_size 192 --num_epochs 1 --experiment_name bench_192

# Test 256
python train_optimized.py --batch_size 256 --num_epochs 1 --experiment_name bench_256

# Test 320 (if VRAM allows)
python train_optimized.py --batch_size 320 --num_epochs 1 --experiment_name bench_320
```

**Compare speeds:**
- Check `it/s` in progress bar
- Monitor VRAM in `nvidia-smi`
- Find sweet spot before OOM

---

### Find Maximum Batch Size

**Binary search script:**

Create `find_max_batch.sh`:

```bash
#!/bin/bash

for batch in 128 160 192 224 256 288 320 352 384; do
    echo "Testing batch size: $batch"
    timeout 60 python train_optimized.py \
        --batch_size $batch \
        --num_epochs 1 \
        --experiment_name batch_test_$batch

    if [ $? -ne 0 ]; then
        echo "OOM at batch size $batch"
        echo "Maximum safe batch: $((batch - 32))"
        break
    fi

    echo "✓ Batch $batch successful"
done
```

Run:
```bash
chmod +x find_max_batch.sh
./find_max_batch.sh
```

---

## Expected Results

### Performance Comparison Table

| Configuration | Batch | d_model | Layers | VRAM | Speed | Speedup |
|---------------|-------|---------|--------|------|-------|---------|
| **Current (Conservative)** | 128 | 256 | 9 | 5 GB | 35 it/s | 1.0× |
| **Optimized Batch** | 256 | 256 | 9 | 9 GB | 60-70 it/s | 1.7-2.0× |
| **Optimized Model** | 256 | 384 | 12 | 12 GB | 55-65 it/s | 1.6-1.8× |
| **Fully Optimized** | 256 | 384 | 12 | 13 GB | 65-80 it/s | 1.8-2.3× |
| **Maximum (w/ Accum)** | 256×2 | 384 | 12 | 13 GB | 60-75 it/s | 1.7-2.1× |

**Expected improvements:**
- **VRAM Usage:** 5GB → 12-13GB (2.4-2.6× increase)
- **Training Speed:** 35 it/s → 65-80 it/s (1.8-2.3× faster)
- **Epoch Time:** ~40 min → ~18-22 min (2× faster)
- **Total Training:** 67 hours → 30-37 hours (massive savings!)

---

## Safety Mechanisms

### Prevent OOM Crashes

**Edit `train_optimized.py` - Add OOM handler around line 669:**

```python
def objective(trial: optuna.trial.Trial) -> float:
    """Optimized Optuna objective function for 16GB VRAM"""

    global g_train_dataset, g_valid_dataset, g_config, g_device, g_train_loader, g_valid_loader
    if g_train_dataset is None or g_valid_dataset is None:
        raise ValueError("Global datasets not set. Run data loading first.")

    try:
        # Existing code...

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number}: OOM Error - pruning trial")
            # Clean up
            if 'model' in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()
        else:
            raise
```

---

## Troubleshooting

### Issue: Still Low VRAM After Changes

**Check:**
1. Did you restart Python? (Old config cached)
2. Is AMP enabled? `USE_AMP = True`
3. Check actual batch size in logs
4. Verify model size: `print(num_params)`

**Debug command:**
```bash
python -c "from train_optimized import Config; c=Config(); print(f'Batch: {c.BATCH_SIZE}, d_model: {c.D_MODEL}')"
```

---

### Issue: OOM Errors

**Solutions:**
1. Reduce batch size by 32
2. Reduce `d_model` by 64
3. Reduce `n_layers` by 2
4. Enable gradient accumulation
5. Reduce `PREFETCH_FACTOR` to 2

**Emergency fallback config:**
```python
BATCH_SIZE = 192
D_MODEL = 320
N_LAYERS = 10
PREFETCH_FACTOR = 2
```

---

### Issue: Slower Speed Despite Higher VRAM

**Possible causes:**
1. **Data bottleneck:** Increase `NUM_WORKERS`
2. **CPU bottleneck:** Check `htop`, reduce workers if CPU maxed
3. **RAM bottleneck:** Check `free -h`, reduce `PREFETCH_FACTOR`
4. **Disk I/O:** Move HDF5 file to SSD if on HDD

**Check data pipeline:**
```python
import time
start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 100:
        break
elapsed = time.time() - start
print(f"Data loading speed: {100/elapsed:.1f} batches/sec")
```

If < 40 batches/sec, data is bottleneck.

---

## Quick Start: Copy-Paste Solution

### Step 1: Backup Original

```bash
cp train_optimized.py train_optimized_backup.py
```

### Step 2: Apply Changes

**Option A: Conservative (Safe)**
```python
# Line 77-84
D_MODEL = 320
N_HEAD = 16
N_LAYERS = 10
FFN_HIDDEN = 640
BATCH_SIZE = 224

# Expected: 9-11GB VRAM, 55-65 it/s
```

**Option B: Aggressive (Recommended)**
```python
# Line 77-84
D_MODEL = 384
N_HEAD = 16
N_LAYERS = 12
FFN_HIDDEN = 768
BATCH_SIZE = 256

# Expected: 12-13GB VRAM, 65-80 it/s
```

**Option C: Maximum (Experimental)**
```python
# Line 77-84
D_MODEL = 448
N_HEAD = 16
N_LAYERS = 12
FFN_HIDDEN = 896
BATCH_SIZE = 224

# Expected: 13-15GB VRAM, 60-75 it/s
```

### Step 3: Test

```bash
# Test for 1 epoch to verify no OOM
python train_optimized.py --num_epochs 1 --experiment_name vram_test

# If successful, run full tuning
python train_optimized.py --tune --n_trials 10 --study_name optimized_run
```

---

## Advanced: Dynamic Batch Size Finder

**Add to `train_optimized.py` after line 161:**

```python
    @classmethod
    def find_optimal_batch_size(cls):
        """Automatically find maximum safe batch size"""
        import torch
        from models.transformer_rawIQ import AMCTransformer

        device = cls.DEVICE
        test_batch = 64
        max_batch = 512

        print("🔍 Finding optimal batch size...")

        while test_batch <= max_batch:
            try:
                # Create dummy model
                model = AMCTransformer(
                    in_channels=2,
                    seq_length=cls.SEQ_LENGTH,
                    num_classes=len(cls.TARGET_MODULATIONS),
                    d_model=cls.D_MODEL,
                    n_head=cls.N_HEAD,
                    n_layers=cls.N_LAYERS,
                    ffn_hidden=cls.FFN_HIDDEN,
                    drop_prob=cls.DROP_PROB,
                    device=device,
                    use_cls_token=cls.USE_CLS_TOKEN,
                    embedding_type=cls.EMBEDDING_TYPE,
                    segment_size=cls.SEGMENT_SIZE
                ).to(device)

                # Create dummy batch
                dummy_input = torch.randn(test_batch, 2, cls.SEQ_LENGTH).to(device)
                dummy_target = torch.randint(0, len(cls.TARGET_MODULATIONS), (test_batch,)).to(device)

                # Test forward + backward
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                criterion = torch.nn.CrossEntropyLoss()

                output = model(dummy_input)
                loss = criterion(output, dummy_target)
                loss.backward()
                optimizer.step()

                print(f"  ✓ Batch {test_batch} works")

                # Cleanup
                del model, optimizer, dummy_input, dummy_target, output, loss
                torch.cuda.empty_cache()

                # Try next size
                test_batch += 32

            except RuntimeError as e:
                if "out of memory" in str(e):
                    optimal = test_batch - 32
                    print(f"  ✗ Batch {test_batch} OOM")
                    print(f"\n✅ Optimal batch size: {optimal}")
                    torch.cuda.empty_cache()
                    return optimal
                else:
                    raise

        return max_batch
```

**Use it:**
```python
# In main(), before training:
optimal_batch = Config.find_optimal_batch_size()
Config.BATCH_SIZE = optimal_batch
```

---

## Summary Checklist

- [ ] Increase `BATCH_SIZE` to 256+
- [ ] Increase `D_MODEL` to 384+
- [ ] Increase `N_LAYERS` to 12+
- [ ] Increase `NUM_WORKERS` to 12 (if CPU allows)
- [ ] Increase `PREFETCH_FACTOR` to 5
- [ ] Reduce `CACHE_CLEAR_FREQUENCY` to 100
- [ ] Update Optuna search space for larger models
- [ ] Test with 1 epoch first
- [ ] Monitor VRAM with `nvidia-smi`
- [ ] Verify speed improvement (target: 65+ it/s)
- [ ] Add OOM error handling
- [ ] Backup original config

**Expected Final Result:**
- **VRAM:** 12-14GB (80-85% utilization) ✓
- **Speed:** 65-80 it/s (2× faster) ✓
- **Training Time:** Cut in half ✓

---

## References

**Memory estimation formula** (line 345-405):
```
VRAM = (params × 4) + (params × 8) + (batch × seq × d_model × layers × 12 × 4)
     = params × 12 + batch × seq × d_model × layers × 48 bytes
```

For d_model=384, layers=12, batch=256:
```
VRAM ≈ 45M × 12 + 256 × 1024 × 384 × 12 × 48
     ≈ 0.5GB + 11.8GB
     ≈ 12.3GB (+ 30% overhead = 16GB total)
```

**Perfect fit for 16GB GPU!**
