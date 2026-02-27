"""MicrogliaAgent: Learnable pruning decision network."""

import torch
import torch.nn as nn


class MicrogliaAgent(nn.Module):
    """Small MLP that learns which attention heads to prune.
    
    Takes activation statistics as input and outputs soft masks (0-1 values)
    for each attention head. The agent learns to identify unimportant heads
    that can be pruned without hurting task performance.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, temperature: float = 1.0):
        """Initializes the MicrogliaAgent.

        Args:
            hidden_dim (int): Hidden dimension of the MLP.
            num_heads (int): Number of attention heads in the layer.
            temperature (float): Temperature for sigmoid activation (lower = more binary masks).
        """
        super().__init__()
        self.num_heads: int = num_heads
        self.temperature: float = temperature
        
        # Input: 4 statistics per head (mean_norm, std_norm, entropy, max_attn)
        input_dim: int = 4 * num_heads
        
        # Enhanced MLP with better capacity and residual-like structure
        self.fc1: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.act: nn.GELU = nn.GELU()
        self.norm: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.fc2: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.fc3: nn.Linear = nn.Linear(hidden_dim, num_heads)
        
    def forward(self, activation_stats: torch.Tensor) -> torch.Tensor:
        """Predicts pruning masks from activation statistics.
        
        Args:
            activation_stats (torch.Tensor): Tensor of shape (batch, 4*num_heads) containing
                activation norms and attention entropy for each head.
                            
        Returns:
            torch.Tensor: Masks of shape (batch, num_heads) with values in [0, 1].
                Values close to 1 mean "keep this head", close to 0 mean "prune it".
        """
        x = self.fc1(activation_stats)
        x = self.act(x)
        x = self.norm(x)

        residual = x
        x = self.fc2(x)
        x = self.act(x)
        x = x + residual

        logits = self.fc3(x)
        
        # Soft gating: sigmoid with temperature scaling
        masks = torch.sigmoid(logits / self.temperature)
        
        return masks
    
    def set_temperature(self, temperature: float) -> None:
        """Updates the temperature parameter.
        
        Lower temperatures produce more binary masks (closer to 0 or 1).
        Higher temperatures produce softer masks (closer to 0.5).

        Args:
            temperature (float): The new temperature value.
        """
        self.temperature = temperature
