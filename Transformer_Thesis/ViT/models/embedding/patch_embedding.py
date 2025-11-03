import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Projects the flattened patches to a specified dimension (D).
    """
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (B, E, P_h, P_w)
        x = x.flatten(2)       # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x
