import torch 
from torch import nn 
import math


class PositionalEncoding(nn.Module): 
    """
    Sinusoidal positional encoding for Transformer models.
    
    This implementation:
    - Works with any sequence length up to max_len
    - Automatically handles device placement
    - Uses stable sinusoidal encoding (Vaswani et al., 2017)
    - Compatible with both 1D sequences (pure Transformer) and 2D patches (ViT)
    """
    
    def __init__(self, d_model, max_len=5000, device='cpu', dropout=0.0): 
        """
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length supported
            device: Device to place encoding on ('cpu' or 'cuda')
            dropout: Optional dropout rate (default: 0.0)
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        encoding = torch.zeros(max_len, d_model, device=device)
        
        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Division term: 10000^(2i/d_model) for i in [0, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device) 
            * -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        encoding[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        # Shape: (max_len, d_model)
        self.register_buffer('encoding', encoding)
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
    
    def forward(self, x): 
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Ensure sequence length doesn't exceed max_len
        if seq_len > self.encoding.size(0):
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.encoding.size(0)}. "
                f"Increase max_len parameter."
            )
        
        # Get positional encoding for current sequence length
        # Shape: (seq_len, d_model) -> (1, seq_len, d_model)
        pos_encoding = self.encoding[:seq_len, :].unsqueeze(0)
        
        # Add to input (broadcasting over batch dimension)
        x = x + pos_encoding
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x