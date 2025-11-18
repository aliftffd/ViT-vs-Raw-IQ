# Hyperparameter Tuning Plan for ViT and transformer_rawIQ

This document outlines the plan for hyperparameter tuning of the `ViT` and `transformer_rawIQ` models using Optuna in a Jupyter Notebook environment.

## 1. General Setup

### 1.1. Dependencies

First, ensure you have the necessary libraries installed:

```bash
pip install optuna jupyterlab pandas numpy torch scikit-learn matplotlib seaborn
```

### 1.2. Jupyter Notebook

All tuning will be performed in a Jupyter Notebook for interactive exploration and visualization.

## 2. Hyperparameter Tuning for `ViT`

### 2.1. Objective

The goal is to find the optimal set of hyperparameters that maximizes the validation accuracy of the `AMCTransformer` model in the `ViT` directory.

### 2.2. Hyperparameter Space

We will explore the following hyperparameters:

-   **`d_model`**: The embedding dimension.
-   **`n_head`**: The number of attention heads. Must be a divisor of `d_model`.
-   **`n_layers`**: The number of transformer encoder layers.
-   **`drop_prob`**: The dropout probability.
-   **`learning_rate`**: The initial learning rate for the optimizer.
-   **`optimizer`**: The choice of optimizer (e.g., AdamW, Adam).
-   **`weight_decay`**: The weight decay for regularization.

### 2.3. Optuna Objective Function

We will define an `objective` function for Optuna that encapsulates the training and validation process.

```python
import optuna
import torch
# Import necessary modules from your project
# from ViT.models.amc_transformer import AMCTransformer
# from ViT.dataloader.dataset import SingleStreamImageDataset
# from ViT.training.train import train_epoch, validate_epoch

def objective_vit(trial):
    # 1. Suggest Hyperparameters
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_head = trial.suggest_categorical("n_head", [4, 8, 16])
    n_layers = trial.suggest_int("n_layers", 2, 8)
    drop_prob = trial.suggest_uniform("drop_prob", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)

    # Ensure n_head is a divisor of d_model
    if d_model % n_head != 0:
        # Prune this trial if the condition is not met
        raise optuna.exceptions.TrialPruned()

    # 2. Setup Model, Dataloaders, etc.
    # (This part assumes you have your data loading and splitting logic ready)
    # train_loader, valid_loader = setup_dataloaders()

    model = AMCTransformer(
        # ... other fixed params
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        drop_prob=drop_prob,
        # ...
    ).to(device)

    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Training and Validation Loop
    for epoch in range(NUM_EPOCHS): # Use a smaller number of epochs for tuning
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device)

        # Report intermediate results to Optuna
        trial.report(val_acc, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc
```

### 2.4. Running the Study

```python
study_vit = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study_vit.optimize(objective_vit, n_trials=100)

print("Best trial for ViT:")
trial = study_vit.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

## 3. Hyperparameter Tuning for `transformer_rawIQ`

### 3.1. Objective

Similar to the ViT model, the objective is to maximize the validation accuracy for the `AMCTransformer` in the `transformer_rawIQ` directory.

### 3.2. Hyperparameter Space

The hyperparameter space is similar, with the addition of `embedding_type` and `segment_size`.

-   **`embedding_type`**: `'segment'` or `'conv1d'`.
-   **`segment_size`**: The size of segments for the `'segment'` embedding type.
-   `d_model`, `n_head`, `n_layers`, `drop_prob`, `learning_rate`, `optimizer`, `weight_decay` (same as ViT).

### 3.3. Optuna Objective Function

```python
import optuna
import torch
# from transformer_rawIQ.models.transformer_rawIQ import AMCTransformer
# ... other imports

def objective_rawiq(trial):
    # 1. Suggest Hyperparameters
    embedding_type = trial.suggest_categorical("embedding_type", ["segment", "conv1d"])
    
    segment_size = None
    if embedding_type == "segment":
        segment_size = trial.suggest_categorical("segment_size", [16, 32, 64])

    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    n_head = trial.suggest_categorical("n_head", [4, 8, 16])
    n_layers = trial.suggest_int("n_layers", 2, 8)
    drop_prob = trial.suggest_uniform("drop_prob", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam"])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)

    if d_model % n_head != 0:
        raise optuna.exceptions.TrialPruned()

    # 2. Setup Model
    model = AMCTransformer(
        # ... other fixed params
        embedding_type=embedding_type,
        segment_size=segment_size,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        drop_prob=drop_prob,
        # ...
    ).to(device)

    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Training and Validation Loop (similar to ViT)
    # ...
    
    return val_acc
```

### 3.4. Running the Study

```python
study_rawiq = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study_rawiq.optimize(objective_rawiq, n_trials=100)

print("Best trial for transformer_rawIQ:")
trial = study_rawiq.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

```

## 4. Next Steps

1.  **Implement the Notebooks**: Create two separate Jupyter Notebooks, one for each model.
2.  **Refine the Search Space**: Based on initial results, the hyperparameter search space can be narrowed down.
3.  **Visualization**: Use Optuna's visualization tools (`plot_optimization_history`, `plot_param_importances`) to analyze the results.
4.  **Final Training**: Once the best hyperparameters are found, train the models with the full dataset for a larger number of epochs.
