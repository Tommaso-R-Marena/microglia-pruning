"""MicrogliaAgent: Learnable pruning decision network."""

import torch
import torch.nn as nn


class MicrogliaAgent(nn.Module):
    """Small MLP that learns which attention heads to prune.
    
    Takes activation statistics as input and outputs soft masks (0-1 values)
    for each attention head. The agent learns to identify unimportant heads
    that can be pruned without hurting task performance.
    
    Args:
        hidden_dim: Hidden dimension of the MLP
        num_heads: Number of attention heads in the layer
        temperature: Temperature for sigmoid activation (lower = more binary masks)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, temperature: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Input: 2 statistics per head (activation norm + attention entropy)
        input_dim = 2 * num_heads
        
        # Simple 2-layer MLP
        self.monitor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        
    def forward(self, activation_stats: torch.Tensor) -> torch.Tensor:
        """Predict pruning masks from activation statistics.
        
        Args:
            activation_stats: Tensor of shape (batch, 2*num_heads) containing
                            activation norms and attention entropy for each head
                            
        Returns:
            masks: Tensor of shape (batch, num_heads) with values in [0, 1].
                  Values close to 1 mean "keep this head", close to 0 mean "prune it".
        """
        logits = self.monitor(activation_stats)
        
        # Soft gating: sigmoid with temperature scaling
        masks = torch.sigmoid(logits / self.temperature)
        
        return masks
    
    def set_temperature(self, temperature: float):
        """Update temperature parameter.
        
        Lower temperatures produce more binary masks (closer to 0 or 1).
        Higher temperatures produce softer masks (closer to 0.5).
        """
        self.temperature = temperature
