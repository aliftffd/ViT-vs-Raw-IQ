import torch
from models.amc_transformer import AMCTransformer

# --- 1. Definisikan Hyperparameter untuk Uji Coba ---
batch_size = 4          
num_classes = 11        
in_channels = 1         # Jumlah channel (I/Q untuk sinyal RF biasanya 2)
img_height = 32
img_width = 32
patch_size = 16
num_patches = (img_height // patch_size) * (img_width // patch_size)  # 4 patches

# Parameter Arsitektur Transformer
d_model = 128           
n_head = 8              
n_layers = 2            
ffn_hidden = 128 * 4    
drop_prob = 0.1         

# --- 2. Siapkan Device dan Data Input Tiruan ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {device}")

# PERBAIKAN: Buat tensor dengan shape gambar 2D
# Shape: (batch_size, in_channels, img_height, img_width)
dummy_input = torch.randn(batch_size, in_channels, img_height, img_width, device=device)
print(f"Shape input tiruan: {dummy_input.shape}")

# --- 3. Inisialisasi Model ---
try:
    model = AMCTransformer(
        num_patches=num_patches,
        patch_size=patch_size,
        num_classes=num_classes,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        ffn_hidden=ffn_hidden,
        drop_prob=drop_prob,
        device=device
    ).to(device)
    
    print("\nModel berhasil diinisialisasi.")
    print(f"Jumlah parameter: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    # --- 4. Lakukan Forward Pass ---
    print("\nMencoba melakukan forward pass...")
    output = model(dummy_input)
    
    # --- 5. Periksa Shape Output ---
    print("Forward pass berhasil!")
    print(f"Shape output: {output.shape}")
    print(f"Shape output yang diharapkan: ({batch_size}, {num_classes})")
    
    assert output.shape == (batch_size, num_classes)
    print("\n✅ Sanity check berhasil! Arsitektur Anda tampaknya terhubung dengan benar.")
    
except Exception as e:
    print(f"\n❌ Terjadi Error: {e}")
    import traceback
    traceback.print_exc()
