import torch 
import torch.nn as nn

from typing import Literal

class ClassifierHead(nn.Module):

    def __init__(self, d_model: int = 64,hidden_dim: int = 128, n_classes: int = 24, dropout: float = 0.3, activation: Literal["relu","selu"] = "relu", pooling: Literal["gap","attention"] = "gap",):
        
        super().__init__()

        self.d_model = d_model
        self.n_classses = n_classes
        self.pooling = pooling

        if pooling == "attention":
            self.attention_query = nn.Linear(d_model, 1, bias = False)
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        else:
            act_fn = nn.SELU(inplace=True)
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim), # Dense Layer: (B, d_model) -> (B, hidden_dim)
            act_fn,  # Activation
            nn.Dropout(p=dropout),  # Dropout
            nn.Linear(hidden_dim, n_classes)  # Output Layer: (B, hidden_dim) -> (B, n_classes)
            #No activation here, as we'll use CrossEntropyLoss which applies Softmax internally 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pooling: (B, seq_length, d_model) -> (B, d_model)
        if self.pooling == "gap":
            x = x.mean(dim=1) # Global Average Pooling over sequence length
        else:  # Attention Pooling
            attn_scores = self.attention_query(x) # compute attention scores: (B, seq_length, d_model) -> (B, seq_length, 1) 

            attn_weights = torch.softmax(attn_scores, dim=1)  # Softmax over sequence length: (B, seq_length, 1)

            # Weighted sum
            x = (x * attn_weights).sum(dim=1)  # (B, seq_length, d_model) * (B, seq_length, 1) -> (B, d_model)
        
        x = self.classifier(x)  # MLP Classifier: (B, d_model) -> (B, n_classes)
        return x  # (B, n_classes) 
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)  # Forward pass to get logits
        return logits.argmax(dim=1)  # Predicted class labels 
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)  # Forward pass to get logits
        return torch.softmax(logits, dim=1)  # Class probabilities
    
    def get_pooling_weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling != "attention":
            raise ValueError("Pooling weights are only available for attention pooling.")
        
        attn_scores = self.attention_query(x)  # (B, seq_length, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, seq_length, 1)
        return attn_weights.squeeze(-1) # (B, seq_length, 1) 