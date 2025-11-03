import torch
from torch import nn
from .encoder import Encoder 

class AMCTransformer(nn.Module): 

    # 1. UBAH PARAMETER __init__:
    # Hapus 'num_patches', tambahkan 'img_size_h', 'img_size_w'
    def __init__(self, in_channels, img_size_h, img_size_w, patch_size, num_classes, d_model, n_head, n_layers, ffn_hidden, drop_prob, device):
        super().__init__()

        # 2. TERUSKAN PARAMETER BARU ke Encoder
        self.encoder = Encoder(in_channels=in_channels,
                               img_size_h=img_size_h, # Tambahkan ini
                               img_size_w=img_size_w, # Tambahkan ini
                               patch_size=patch_size,
                               d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, src): 
        # src shape: (batch_size, in_channels, img_size_h, img_size_w)
        enc_output = self.encoder(src) #-> (batch_size, num_patches + 1, d_model)
        cls_output = enc_output[:, 0]
        output = self.mlp_head(cls_output)
        return output
