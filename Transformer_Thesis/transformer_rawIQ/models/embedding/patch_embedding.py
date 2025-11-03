import torch
import torch.nn as nn


class SequenceEmbedding(nn.Module):
    """
    Projects raw I/Q sequences to embedding dimension.
    For raw I/Q data with shape [B, 2, seq_len] where 2 = I and Q channels.
    
    This is a pure Transformer embedding without patch-based vision operations.
    """
    def __init__(self, in_channels=2, embedding_dim=256, method='conv1d', 
                 segment_size=None):
        """
        Args:
            in_channels: Number of input channels (2 for I/Q)
            embedding_dim: Dimension to project to (e.g., 256, 512)
            method: 'conv1d' or 'segment'
            segment_size: Size of segments (required if method='segment')
        """
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.method = method
        self.segment_size = segment_size
        
        if method == 'conv1d':
            # Full sequence: 1D Convolution with kernel_size=1
            self.projection = nn.Conv1d(
                in_channels, 
                embedding_dim, 
                kernel_size=1
            )
        elif method == 'segment':
            # Segment-based: 1D Convolution with stride
            if segment_size is None:
                raise ValueError("segment_size is required for 'segment' method")
            self.projection = nn.Conv1d(
                in_channels,
                embedding_dim,
                kernel_size=segment_size,
                stride=segment_size
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'conv1d' or 'segment'")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, 2, seq_len]
        
        Returns:
            Embedded sequence of shape [B, seq_len, embedding_dim] (for conv1d)
            or [B, num_segments, embedding_dim] (for segment)
        """
        # Input: [B, 2, seq_len] -> Output: [B, embedding_dim, seq_len or num_segments]
        x = self.projection(x)
        # Transpose to [B, seq_len or num_segments, embedding_dim] for Transformer
        x = x.transpose(1, 2)
        return x

