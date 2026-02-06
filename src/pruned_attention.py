"""Pruned attention wrapper with dynamic masking."""

import torch
import torch.nn as nn
from .statistics import compute_layer_stats
from .agent import MicrogliaAgent


class PrunedAttention(nn.Module):
    """Wraps attention layer with learned dynamic pruning.
    
    During forward pass:
    1. Run standard attention
    2. Compute activation statistics
    3. Use agent to predict which heads to keep
    4. Apply masks to attention output
    
    The masks are stored so we can compute sparsity metrics later.
    """
    
    def __init__(self, original_attn: nn.Module, agent: MicrogliaAgent, hard_prune: bool = False):
        super().__init__()
        self.attn = original_attn
        self.agent = agent
        self.hard_prune = hard_prune
        self.last_masks = None  # Store masks for monitoring
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_value = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs) -> tuple:
        """Forward with dynamic head pruning."""
        
        # Run original attention - need to handle different model architectures
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # We need attention weights for statistics
            use_cache=use_cache,
            **kwargs
        )
        
        # Unpack outputs (format varies by model)
        if isinstance(attn_outputs, tuple):
            attn_output = attn_outputs[0]
            attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
        else:
            attn_output = attn_outputs
            attn_weights = None
        
        # If we have attention weights, compute stats and apply pruning
        if attn_weights is not None and self.training:
            # Compute per-head statistics
            stats = compute_layer_stats(hidden_states, attn_weights)
            
            # Get pruning masks from agent
            masks = self.agent(stats)  # (batch, num_heads)
            
            # Apply hard threshold in eval mode
            if self.hard_prune and not self.training:
                masks = (masks > 0.5).float()
            
            # Store for monitoring
            self.last_masks = masks.detach()
            
            # Apply masks to attention output
            # Output shape: (batch, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = attn_output.shape
            num_heads = masks.shape[1]
            head_dim = hidden_dim // num_heads
            
            # Reshape to separate heads
            attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
            
            # Broadcast masks: (batch, num_heads) -> (batch, 1, num_heads, 1)
            masks_expanded = masks.unsqueeze(1).unsqueeze(-1)
            
            # Apply masking
            attn_output = attn_output * masks_expanded
            
            # Reshape back
            attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
        
        # Return in original format
        if isinstance(attn_outputs, tuple):
            return (attn_output,) + attn_outputs[1:]
        else:
            return attn_output
