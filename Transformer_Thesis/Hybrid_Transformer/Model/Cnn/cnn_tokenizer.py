from typing import Literal

import torch 
import torch.nn as nn

class ConvBlock1D(nn.Module):

    """
    Architecturre: Conv1D -> BatchNorm1D -> ReLU -> Activation
    """

    def __init__(self,in_channels: int, out_channels: int, kernel_size: int, padding: Literal["same","valid"] = "same", activation: Literal["relu","selu"] = "relu"):
        
        super().__init__()

        # Define padding value
        if padding == "same":
            pad = kernel_size // 2
        else: # "Valid"
            pad = 0

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        #Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        # Activation Function
        self.activation = (
            nn.ReLU(inplace=True) if activation == "relu" else nn.SELU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))
    
class ShallowCNNTokenizer(nn.Module):

    def __init__(self, in_channels: int = 2, d_model: int = 64, seq_length: int = 1024, kernel_size1: int = 7, kernel_size2: int = 5,pool_size: int = 4,padding: Literal["same","valid"] = "same", activation: Literal["relu","selu"] = "relu"):

        super().__init__()

        # First Conv Block (B,2,1024) -> (B,64,1024) 
        # Large Kenrle (7)
        self.conv_block1 = ConvBlock1D(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=kernel_size1,
            padding=padding,
            activation=activation,
        )

        # 2nd Conv Block (B,64,1024) -> (B,64,1024)
        # Maintains d_model for Transformer compatibility
        self.conv_block2 = ConvBlock1D(
            in_channels = d_model,
            out_channels = d_model,
            kernel_size = kernel_size2,
            padding = padding,
            activation = activation,
        )

        # Tokeziation max Pooling : (B,64,1024) -> (B,64,256)
        #Reduce sequence length to mitigate O(N^2) attention in transformer 
        self.pool = nn.MaxPool1d(kernel_size=pool_size,stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pool(x)
        return x 