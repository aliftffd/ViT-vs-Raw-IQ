import torch
from torch import nn
from .blocks.encoder_layer import EncoderLayer
from .embedding.patch_embedding import SequenceEmbedding
from .embedding.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """
    Pure Transformer Encoder for raw I/Q signal data.
    
    This encoder processes 1D sequences, not 2D images.
    Compatible with raw I/Q data of shape [batch, 2, seq_length].
    """
    
    def __init__(self, 
                 in_channels,           # Number of input channels (2 for I/Q)
                 seq_length,            # Sequence length (e.g., 1024)
                 d_model,               # Embedding dimension
                 ffn_hidden,            # FFN hidden dimension
                 n_head,                # Number of attention heads
                 n_layers,              # Number of encoder layers
                 drop_prob,             # Dropout probability
                 device,                # Device (cuda/cpu)
                 use_cls_token=True,    # Whether to use CLS token
                 embedding_type='conv1d',  # 'conv1d' or 'segment'
                 segment_size=64):      # Segment size (only used if embedding_type='segment')
        super().__init__()
        self.device = device
        self.use_cls_token = use_cls_token
        self.embedding_type = embedding_type
        
        # ✅ 1D SEQUENCE EMBEDDING (replaces PatchEmbedding)
        if embedding_type == 'conv1d':
            # Full sequence: [B, 2, 1024] -> [B, 1024, d_model]
            self.sequence_embedding = SequenceEmbedding(
                in_channels=in_channels,
                embedding_dim=d_model,
                method='conv1d'
            )
            num_tokens = seq_length
            
        elif embedding_type == 'segment':
            # Segment-based: [B, 2, 1024] -> [B, num_segments, d_model]
            if seq_length % segment_size != 0:
                raise ValueError(
                    f"seq_length ({seq_length}) must be divisible by segment_size ({segment_size})"
                )
            self.sequence_embedding = SequenceEmbedding(
                in_channels=in_channels,
                embedding_dim=d_model,
                segment_size=segment_size,
                method='segment'
            )
            num_tokens = seq_length // segment_size
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        
        # ✅ POSITIONAL ENCODING for 1D sequences
        # Calculate max_len: num_tokens + 1 (for CLS token if used)
        max_len = num_tokens + (1 if use_cls_token else 0)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            device=device,
            dropout=0.0  # We apply dropout separately
        )
        
        # ✅ CLS TOKEN (optional, for classification tasks)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # ✅ TRANSFORMER ENCODER LAYERS
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                ffn_hidden=ffn_hidden,
                n_head=n_head,
                drop_prob=drop_prob
            )
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, src, src_mask=None):
        """
        Forward pass through the encoder.
        
        Args:
            src: Input tensor of shape [batch_size, in_channels, seq_length]
                 For I/Q data: [batch_size, 2, 1024]
            src_mask: Optional attention mask
        
        Returns:
            Encoded sequence of shape [batch_size, num_tokens, d_model]
            If use_cls_token=True: [batch_size, num_tokens+1, d_model]
        """
        # ✅ 1. Apply 1D sequence embedding
        # Input: [B, 2, seq_length] -> Output: [B, num_tokens, d_model]
        x = self.sequence_embedding(src)
        
        # ✅ 2. Add CLS token (if enabled)
        if self.use_cls_token:
            batch_size = src.shape[0]
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_tokens+1, d_model]
        
        # ✅ 3. Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # ✅ 4. Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x
    
    def get_cls_token_output(self, src, src_mask=None):
        """
        Convenience method to get only the CLS token output.
        Useful for classification tasks.
        
        Args:
            src: Input tensor [batch_size, in_channels, seq_length]
            src_mask: Optional attention mask
        
        Returns:
            CLS token embeddings [batch_size, d_model]
        """
        if not self.use_cls_token:
            raise ValueError("CLS token is not enabled. Set use_cls_token=True")
        
        x = self.forward(src, src_mask)
        return x[:, 0, :]  # Return only the first token (CLS)
    
    def get_sequence_output(self, src, src_mask=None):
        """
        Get the encoded sequence without the CLS token.
        Useful for sequence-to-sequence tasks.
        
        Args:
            src: Input tensor [batch_size, in_channels, seq_length]
            src_mask: Optional attention mask
        
        Returns:
            Sequence embeddings [batch_size, num_tokens, d_model]
        """
        x = self.forward(src, src_mask)
        
        if self.use_cls_token:
            return x[:, 1:, :]  # Skip the first token (CLS)
        else:
            return x