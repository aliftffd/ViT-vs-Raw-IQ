import math
from typing import Literal

import torch
import torch.nn as nn  

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :]
        return self.dropout(x) 

class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.2):
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.positional_embedding = nn.Parameter(torch.rand(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_embedding[:, : x.size(1), :]
        return self.dropout(x)
    
class TransformerAggregator(nn.Module):

    def __init__(self, d_model: int = 64, seq_length: int = 256, n_head: int = 4, n_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1, pos_encoding: Literal["sinusoidal", "learnable"] = "learnable"):

        super().__init__()

        self.d_model = d_model
        self.seq_length = seq_length 

        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model = d_model, max_len = seq_length, dropout = dropout
            )       
        else:
            self.pos_encoder = LearnablePositionalEncoding(
                d_model = d_model,max_len = seq_length, dropout=dropout
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = n_head,
            dim_feedforward= dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer, num_layers= n_layers, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.permute(0,2,1)  # (B, d_model, seq_length) -> (B, seq_length, d_model
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Transformer Encoder
        x = self.norm(x)  # Layer Normalization

        return x  # (B, seq_length, d_model) 