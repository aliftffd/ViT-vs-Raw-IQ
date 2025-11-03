import torch 
from torch import nn 

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_len=5000, device='cpu'): 
        super(PositionalEncoding, self).__init__()
        
        # Buat encoding di CPU
        encoding = torch.zeros(max_len, d_model)
        
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()
        denominator = torch.pow(10000.0, _2i / d_model)
        
        encoding[:, 0::2] = torch.sin(pos / denominator)
        encoding[:, 1::2] = torch.cos(pos / denominator)
        
        # Register sebagai buffer dengan shape (max_len, d_model)
        self.register_buffer('encoding', encoding)
    
    def forward(self, x): 
        batch_size, seq_len, d_model = x.shape
        
        # PERBAIKAN: Ambil encoding dengan 2 indeks, lalu unsqueeze untuk batch
        # Shape: (seq_len, d_model) -> (1, seq_len, d_model)
        pos_encoding = self.encoding[:seq_len, :].unsqueeze(0)
        
        # Broadcast ke semua batch
        return x + pos_encoding



