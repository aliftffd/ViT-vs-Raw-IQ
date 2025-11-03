
import torch
from torch import nn
from .encoder import Encoder 


class AMCTransformer(nn.Module):
    """
    Automatic Modulation Classification Transformer for Raw I/Q Signals.
    
    Pure Transformer architecture (not ViT) for 1D signal processing.
    """

    def __init__(self, 
                 in_channels,           # Number of input channels (2 for I/Q)
                 seq_length,            # Sequence length (e.g., 1024)
                 num_classes,           # Number of modulation classes
                 d_model,               # Embedding dimension
                 n_head,                # Number of attention heads
                 n_layers,              # Number of encoder layers
                 ffn_hidden,            # FFN hidden dimension
                 drop_prob,             # Dropout probability
                 device,                # Device (cuda/cpu)
                 use_cls_token=True,    # Whether to use CLS token
                 embedding_type='segment',  # 'conv1d' or 'segment'
                 segment_size=64):      # Segment size (for 'segment' type)
        """
        Initialize AMC Transformer for raw I/Q signal classification.
        
        Args:
            in_channels: Number of input channels (2 for I and Q)
            seq_length: Length of input sequence (e.g., 1024)
            num_classes: Number of modulation classes to classify
            d_model: Transformer embedding dimension
            n_head: Number of attention heads
            n_layers: Number of Transformer encoder layers
            ffn_hidden: Hidden dimension in feed-forward network
            drop_prob: Dropout probability
            device: Device to run on ('cuda' or 'cpu')
            use_cls_token: If True, use CLS token for classification.
                          If False, use global average pooling.
            embedding_type: Type of embedding ('conv1d' for full sequence,
                           'segment' for segment-based)
            segment_size: Size of segments (only used if embedding_type='segment')
        """
        super().__init__()
        
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        # Pure Transformer Encoder for raw I/Q sequences
        self.encoder = Encoder(
            in_channels=in_channels,
            seq_length=seq_length,
            d_model=d_model,
            n_head=n_head,
            ffn_hidden=ffn_hidden,
            drop_prob=drop_prob,
            n_layers=n_layers,
            device=device,
            use_cls_token=use_cls_token,
            embedding_type=embedding_type,
            segment_size=segment_size
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, src): 
        """
        Forward pass through the model.
        
        Args:
            src: Input tensor of shape [batch_size, in_channels, seq_length]
                 For I/Q data: [batch_size, 2, 1024]
        
        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # src shape: (batch_size, in_channels, seq_length)
        # Encode the sequence
        enc_output = self.encoder(src)  # -> (batch_size, num_tokens, d_model)
        
        # Extract features for classification
        if self.use_cls_token:
            # Use the CLS token (first token) for classification
            cls_output = enc_output[:, 0]  # (batch_size, d_model)
        else:
            # Use global average pooling over all tokens
            cls_output = enc_output.mean(dim=1)  # (batch_size, d_model)
        
        # Classify
        output = self.mlp_head(cls_output)  # (batch_size, num_classes)
        
        return output