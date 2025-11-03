import torch
from models.amc_transformer import AMCTransformer

# --- 1. Definisikan Hyperparameter untuk Uji Coba ---
batch_size = 4          # Jumlah sampel dalam satu batch
num_classes = 11        # Contoh jumlah kelas modulasi (misalnya di RML2016.10a)
num_patches = 64        # Contoh jumlah patch dari satu sinyal
patch_size = 16         # Contoh ukuran patch yang sudah diratakan (misal 4x4)

# Parameter Arsitektur Transformer
d_model = 128           # Dimensi embedding
n_head = 8              # Jumlah attention heads
n_layers = 2            # Jumlah lapisan encoder
ffn_hidden = 128 * 4    # Ukuran hidden layer di feed-forward network
drop_prob = 0.1         # Dropout probability

# --- 2. Siapkan Device dan Data Input Tiruan ---
# Tentukan device secara dinamis
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {device}")

# Buat tensor acak sebagai input tiruan
# Shape: (batch_size, num_patches, patch_size)
dummy_input = torch.randn(batch_size, num_patches, patch_size, device=device)
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
