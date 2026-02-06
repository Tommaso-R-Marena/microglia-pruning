"""Pruned attention wrapper with dynamic masking."""

import torch
import torch.nn as nn
from .statistics import compute_layer_stats
from .agent import MicrogliaAgent


class PrunedAttention(nn.Module):
    """Wrapper around standard attention that applies learned pruning masks.
    
    This module wraps an existing attention layer and dynamically prunes
    attention heads based on activation statistics. The pruning is "soft"
    during training (masks are continuous 0-1 values) and can be made "hard"
    during inference (masks are binary 0/1).
    
    Args:
        original_attn: The original attention module to wrap
        agent: MicrogliaAgent that predicts pruning masks
        hard_prune: If True, use binary masks during inference (deterministic)
                   If False, use soft masks (differentiable)
    """
    
    def __init__(self, original_attn: nn.Module, agent: MicrogliaAgent, hard_prune: bool = False):
        super().__init__()
        self.attn = original_attn
        self.agent = agent
        self.hard_prune = hard_prune
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                **kwargs) -> tuple:
        """Forward pass with dynamic pruning.
        
        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            **kwargs: Additional arguments passed to original attention
            
        Returns:
            attn_output: Attention output with pruning applied
            attn_weights: Attention weights (for monitoring)
        """
        # Run original attention with attention weights output enabled
        attn_output, attn_weights = self.attn(
            hidden_states,
            attention_mask,
            output_attentions=True,
            **kwargs
        )
        
        # Compute statistics and get pruning masks
        stats = compute_layer_stats(hidden_states, attn_weights)
        masks = self.agent(stats)  # Shape: (batch, num_heads)
        
        # Apply hard thresholding if in inference mode
        if self.hard_prune and not self.training:
            masks = (masks > 0.5).float()
        
        # Reshape masks to broadcast across sequence and feature dimensions
        # From (batch, num_heads) to (batch, num_heads, 1, 1)
        masks = masks.unsqueeze(-1).unsqueeze(-1)
        
        # Apply masks to attention output
        # Assuming attn_output has shape (batch, seq_len, hidden_dim)
        # We need to reshape to apply per-head masks
        batch_size, seq_len, hidden_dim = attn_output.shape
        num_heads = masks.shape[1]
        head_dim = hidden_dim // num_heads
        
        # Reshape: (batch, seq_len, hidden_dim) -> (batch, seq_len, num_heads, head_dim)
        attn_output_heads = attn_output.view(batch_size, seq_len, num_heads, head_dim)
        
        # Apply masks: (batch, seq_len, num_heads, head_dim) * (batch, num_heads, 1, 1)
        # Broadcasting handles the rest
        attn_output_heads = attn_output_heads * masks
        
        # Reshape back: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_dim)
        attn_output = attn_output_heads.view(batch_size, seq_len, hidden_dim)
        
        return attn_output, attn_weights
    
    def get_mask_stats(self) -> dict:
        """Get current pruning statistics.
        
        Returns:
            dict with keys:
                - 'sparsity': Fraction of heads currently pruned (0 to 1)
                - 'active_heads': Number of heads with mask > 0.5
                - 'mean_mask': Average mask value
        """
        # This requires storing the last masks - would need to be added in forward
        # For now, return placeholder
        return {
            'sparsity': 0.0,
            'active_heads': self.agent.num_heads,
            'mean_mask': 1.0
        }
