
import torch
from models.transformer_rawIQ import AMCTransformer

print("=" * 70)
print("Testing Pure Transformer for Raw I/Q Signal Classification")
print("=" * 70)

# --- 1. Define Hyperparameters for Raw I/Q Data ---
batch_size = 4          
num_classes = 11        # Number of modulation classes
in_channels = 2         # I and Q channels (NOT 1!)
seq_length = 1024       # Raw I/Q sequence length (NOT img_height/width!)

# Transformer Architecture Parameters
d_model = 128           
n_head = 8              
n_layers = 2            
ffn_hidden = 128 * 4    # 512
drop_prob = 0.1         

# Embedding Configuration
use_cls_token = True
embedding_type = 'segment'  # Options: 'conv1d' or 'segment'
segment_size = 64           # Only used if embedding_type='segment'

# --- 2. Setup Device and Create Dummy Input ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n📱 Device: {device}")

# ✅ CORRECT: Raw I/Q input shape
# Shape: (batch_size, in_channels, seq_length)
# NOT (batch_size, in_channels, img_height, img_width)
dummy_input = torch.randn(batch_size, in_channels, seq_length, device=device)
print(f"\n📊 Input Data:")
print(f"   Shape: {dummy_input.shape}")
print(f"   - Batch size: {batch_size}")
print(f"   - Channels: {in_channels} (I and Q)")
print(f"   - Sequence length: {seq_length}")

# --- 3. Initialize Model ---
print(f"\n🏗️  Initializing Model...")
print(f"   - Embedding type: {embedding_type}")
if embedding_type == 'segment':
    num_tokens = seq_length // segment_size
    print(f"   - Segment size: {segment_size}")
    print(f"   - Number of tokens: {num_tokens}")
else:
    num_tokens = seq_length
    print(f"   - Number of tokens: {num_tokens} (full sequence)")

try:
    model = AMCTransformer(
        in_channels=in_channels,        # ✅ NEW: I/Q channels
        seq_length=seq_length,          # ✅ NEW: Sequence length
        num_classes=num_classes,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        ffn_hidden=ffn_hidden,
        drop_prob=drop_prob,
        device=device,
        use_cls_token=use_cls_token,    # ✅ NEW: CLS token option
        embedding_type=embedding_type,  # ✅ NEW: Embedding type
        segment_size=segment_size       # ✅ NEW: Segment size
    ).to(device)
    
    print("\n✅ Model initialized successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 Model Statistics:")
    print(f"   - Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   - Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # --- 4. Perform Forward Pass ---
    print(f"\n🔄 Performing forward pass...")
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():  # No gradients needed for testing
        output = model(dummy_input)
    
    # --- 5. Check Output Shape ---
    print("\n✅ Forward pass successful!")
    print(f"\n📤 Output:")
    print(f"   Shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {num_classes})")
    
    # Verify output shape
    assert output.shape == (batch_size, num_classes), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {num_classes})"
    
    # Check output properties
    print(f"\n🔍 Output Properties:")
    print(f"   - Min value: {output.min().item():.4f}")
    print(f"   - Max value: {output.max().item():.4f}")
    print(f"   - Mean: {output.mean().item():.4f}")
    print(f"   - Std: {output.std().item():.4f}")
    
    # Apply softmax to get probabilities
    probs = torch.softmax(output, dim=1)
    print(f"\n🎯 Predictions (after softmax):")
    print(f"   Sample 1 probabilities: {probs[0].cpu().numpy()}")
    print(f"   Predicted class: {probs[0].argmax().item()}")
    print(f"   Confidence: {probs[0].max().item():.4f}")
    
    # --- 6. Test with different batch sizes ---
    print(f"\n🧪 Testing with different batch sizes...")
    for test_batch_size in [1, 8, 16]:
        test_input = torch.randn(test_batch_size, in_channels, seq_length, device=device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"   Batch size {test_batch_size:2d}: {test_input.shape} -> {test_output.shape} ✓")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED! Your Pure Transformer is working correctly!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\n🔍 Debugging information:")
    print(f"   - Check if AMCTransformer accepts these parameters:")
    print(f"     * in_channels, seq_length (NOT img_height, img_width)")
    print(f"     * use_cls_token, embedding_type, segment_size")
    print(f"   - Ensure you're using SequenceEmbedding (NOT PatchEmbedding)")
    print(f"   - Input should be 3D: [batch, channels, seq_len]")
    print(f"     NOT 4D: [batch, channels, height, width]")
    
    import traceback
    print("\n📋 Full traceback:")
    traceback.print_exc()