from matplotlib.pyplot import xcorr
import numpy as np
from sqlalchemy.sql.operators import op
import pyswarms.single.global_best import GlobalBestPSO
import torch 
import torch.utils.data import DataLoader

def build_models(params,rawiq_cfg,vit_cfg):

    model_type = int(params[0])

    d_model = int(params[1])
    n_head = int(params[2])
    n_layers = int(params[3])
    ffn_hidden = int(params[4])
    drop_prob = float(params[5])

    if model_type = 0:
        from .ViT.models.amc_transformer import AMCTransformer # ViT 
        patch_size = int(params[8])

        return AMCTransformer(
            in_channels     = vit_cfg["in_channels"],
            img_size_h      = vit_cfg["img_h"],
            img_size_w      = vit_cfg["img_w"],
            patch_size      = patch_size,
            num_classes     = vit_cfg["num_classes"],
            d_model         = d_model,
            n_head          = n_head,
            n_layers        = n_layers,
            ffn_hidden      = ffn_hidden,
            drop_prob       = drop_prob,
            device          = vit_cfg["device"]
        )

    else:
        from .transformer_rawIQ.models.transformer_rawIQ import AMCTransformer
        segment_size = int(params[8])
        embedding_type = "segment"

        return AMCTransformer(
            in_channels     = rawiq_cfg["in_channels"],
            seq_length      = rawiq_cfg["seq_length"],
            num_classes     = rawiq_cfg["num_classes"],
            d_model         = d_model,
            n_head          = n_head,
            n_layers        = n_layers,
            ffn_hidden      = ffn_hidden,
            drop_prob       = drop_prob,
            device          = rawiq_cfg["device"],
            use_cls_token   = True,
            embedding_type  = embedding_type,
            segment_size    = segment_size
        )

def fast_train(model,train_ds, val_ds,lr,batch_size,device):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle = True)
    val_loader = DataLoader(val_ds,batch_size)
    
    model.train()
    for batch_x,batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device),batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_size)
        loss = loss_fn(pred,batch_y)
        loss.backward()
        optimizer.step()
        break
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device),y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

"""
Add PSO Fittness function
"""

def fitness_function(X,train_ds,val_ds,rawiq_cfg,vit_cfg,device):
    scores = []
    for p in x:
        model = build_models(p,rawiq_cfg,vit_cfg)
        
        lr = float(p[6])
        batch_size = int(p[7])

        acc = fast_train(model,train_ds, val_ds, lr, batch_size,device)

        scores.append(-acc)
    return np.array(scores)

def run_pso(train_ds, val_ds, rawiq_cfg, vit_cfg, device):

    dim = 9  # params

    # bounds for params
        0,      # model_type
    min_bounds = np.array([
        32,     # d_model
        2,      # n_head
        1,      # n_layers
        64,     # ffn_hidden
        0.0,    # drop_prob
        1e-5,   # lr
        16,     # batch
        4       # patch_size or segment_size
    ])

    max_bounds = np.array([
        1,
        512,
        16,
        8,
        2048,
        0.4,
        5e-3,
        128,
        64
    ])

    bounds = (min_bounds, max_bounds)

    pso = GlobalBestPSO(
        n_particles=18,
        dimensions=dim,
        options={"c1": 1.5, "c2": 1.5, "w": 0.6},
        bounds=bounds
    )

    best_cost, best_params = pso.optimize(
        lambda x: fitness_function(x, train_ds, val_ds, rawiq_cfg, vit_cfg, device),
        iters=25
    )

    return best_params





