import torch
import torch.nn as nn

from .Aggregator.transformer_aggregator import TransformerAggregator 
from .Classifier_Head.classifier_head import ClassifierHead
from .Cnn.cnn_tokenizer import ShallowCNNTokenizer

from typing import Dict, Literal

class HybridTransformer(nn.Module):

    def __init__(self, in_channels: int = 2, seq_length: int = 1024, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 256, n_classes: int = 24, pool_size: int = 4, dropout: float = 0.1, classifier_dropout: float = 0.3, pooling: Literal["gap","attention"] = "gap"):

        super().__init__()

        # Store Parameters
        self.config = {
            "in_channels": in_channels,
            "seq_length": seq_length,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_feedforward": dim_feedforward,
            "n_classes": n_classes,
            "pool_size": pool_size,
            "dropout": dropout,
            "classifier_dropout": classifier_dropout,
            "pooling": pooling,
        }

        # Reudce sequence length after pooling 
        self.seq_length_reduced = seq_length // pool_size 

        # Block 1 CNN Tokenizer
        self.tokenizer = ShallowCNNTokenizer(
            in_channels=in_channels,
            d_model=d_model,
            seq_length=seq_length,
            kernel_size1 = 7,
            kernel_size2 = 5,
            pool_size=pool_size,
            padding="same", 
            activation="relu",
        )
        # Block 2 Transformer Aggregator
        self.aggregator = TransformerAggregator(
            d_model=d_model,
            seq_length=self.seq_length_reduced,
            n_head=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pos_encoding="learnable",
        )

        # Block 3 Classifier Head
        self.classifier = ClassifierHead(
            d_model=d_model,
            hidden_dim=dim_feedforward,
            n_classes=n_classes,
            dropout=classifier_dropout,
            activation="relu",
            pooling=pooling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1: CNN Tokenizer
        x = self.tokenizer(x)  # (B, in_channels, seq_length) -> (B, d_model, seq_length_reduced)
        # x = x.permute(0, 2, 1)  # (B, d_model, seq_length_reduced) -> (B, seq_length_reduced, d_model)

        # Block 2: Transformer Aggregator
        x = self.aggregator(x)  # (B, seq_length_reduced, d_model) -> (B, seq_length_reduced, d_model)

        # Block 3: Classifier Head
        x = self.classifier(x)  # (B, seq_length_reduced, d_model) -> (B, n_classes)

        return x  # (B, n_classes) 

    def get_token_statistics(self, x: torch.Tensor) -> torch.Tensor:
        
        tokens = self.tokenizer(x)  # (B, in_channels, seq_length) -> (B, d_model, seq_length_reduced)
        amplitudes = tokens.norm(dim=1)

        return {
            "amplitudes": amplitudes,
            "mean": amplitudes.mean(dim=1),
            "std": amplitudes.std(dim=1),
            "min": amplitudes.min(dim=1)[0],
            "max": amplitudes.max(dim=1)[0],
        }
    
    def get_features(self, x: torch.Tensor, layer: str = "aggregator") -> torch.Tensor:
        
        # Block 1
        x = self.tokenizer(x)  # (B, in_channels, seq_length) -> (B, d_model, seq_length_reduced)
        if layer == "tokenizer":
            return x  # (B, d_model, seq_length_reduced) 
        # Block 2 
        x  = self.aggregator(x)  # (B, seq_length_reduced, d_model)
        if layer == "aggregator":
            return x  # (B, seq_length_reduced, d_model)    
        
        if layer == "pooled":
            return x.mean(dim=1)  # (B, d_model)
        raise ValueError(f"Invalid layer: {layer}. Choose from 'tokenizer', 'aggregator', or 'pooled'.") 
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters per block.

        Returns:
            Dict with parameter counts per block and total
        """
        tokenizer_params = sum(
            p.numel() for p in self.tokenizer.parameters() if p.requires_grad
        )
        aggregator_params = sum(
            p.numel() for p in self.aggregator.parameters() if p.requires_grad
        )
        classifier_params = sum(
            p.numel() for p in self.classifier.parameters() if p.requires_grad
        )
        total = tokenizer_params + aggregator_params + classifier_params

        return {
            "tokenizer": tokenizer_params,
            "aggregator": aggregator_params,
            "classifier": classifier_params,
            "total": total,
        }

    def get_config(self) -> Dict:
        """Return model configuration."""
        return self.config.copy()


def create_model(n_classes: int = 24, **kwargs) -> HybridTransformer:
    """
    Factory function to create HybridTransformer.

    Args:
        n_classes: Number of output classes
        **kwargs: Override default config

    Returns:
        HybridTransformer model
    """
    return HybridTransformer(n_classes=n_classes, **kwargs)


# =============================================================================
# Testing and Verification
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HybridTransformer for AMC")
    print("=" * 60)

    # Create model with default config (RadioML 2018)
    model = HybridTransformer(n_classes=24)

    # Print configuration
    print("\n[Configuration]")
    for key, value in model.get_config().items():
        print(f"  {key}: {value}")

    # Parameter count
    print("\n[Parameters]")
    params = model.count_parameters()
    for block, count in params.items():
        print(f"  {block}: {count:,}")

    # Test forward pass
    print("\n[Forward Pass Test]")
    batch_size = 32
    x = torch.randn(batch_size, 2, 1024)
    print(f"  Input shape:  {x.shape}")

    with torch.no_grad():
        logits = model(x)
    print(f"  Output shape: {logits.shape}")

    # Test token statistics extraction
    print("\n[Token Statistics Test]")
    with torch.no_grad():
        stats = model.get_token_statistics(x)

    print(f"  Amplitudes shape: {stats['amplitudes'].shape}")
    print(f"  Mean shape:       {stats['mean'].shape}")
    print(f"  Mean value:       {stats['mean'][0]:.4f}")
    print(f"  Std value:        {stats['std'][0]:.4f}")

    assert stats["amplitudes"].shape == (
        batch_size,
        256,
    ), f"Expected (32, 256), got {stats['amplitudes'].shape}"
    print("  ✓ Token statistics verified")

    # Verify output
    assert logits.shape == (batch_size, 24), f"Expected (32, 24), got {logits.shape}"
    print("  ✓ Forward pass verified")

    # Test feature extraction
    print("\n[Feature Extraction Test]")
    with torch.no_grad():
        cnn_features = model.get_features(x, layer="tokenizer")
        trans_features = model.get_features(x, layer="aggregator")
        pooled_features = model.get_features(x, layer="pooled")

    print(f"  CNN features:    {cnn_features.shape}")
    print(f"  Trans features:  {trans_features.shape}")
    print(f"  Pooled features: {pooled_features.shape}")

    # Test prediction
    print("\n[Prediction Test]")
    with torch.no_grad():
        predictions = logits.argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Probabilities sum: {probabilities[0].sum():.4f} (should be 1.0)")

    # Test attention pooling variant
    print("\n[Attention Pooling Test]")
    model_attn = HybridTransformer(n_classes=24, pooling="attention")

    with torch.no_grad():
        logits_gap = model(x)
        logits_attn = model_attn(x)

    print(f"  GAP output shape:       {logits_gap.shape}")
    print(f"  Attention output shape: {logits_attn.shape}")
    print(f"  ✓ Both pooling strategies work")

    print("\n" + "=" * 60)
    print("Model ready for training!")
    print("Usage: from model.HybridTransformer import HybridTransformer")
    print("=" * 60)