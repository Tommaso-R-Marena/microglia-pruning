"""Compute activation statistics for pruning decisions."""

import torch
import torch.nn.functional as F


def compute_layer_stats(hidden_states: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute per-head statistics from layer activations.
    
    We compute two key statistics for each attention head:
    1. Activation Norm: Magnitude of hidden state activity (high = active, low = dormant)
    2. Attention Entropy: Spread of attention distribution (low = focused, high = scattered)
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
        
    Returns:
        stats: Tensor of shape (batch, 2*num_heads) containing concatenated
               activation norms and attention entropy for each head
    """
    batch_size = hidden_states.shape[0]
    num_heads = attn_weights.shape[1]
    
    # Reshape hidden states to separate heads
    # Assuming hidden_dim = num_heads * head_dim
    hidden_dim = hidden_states.shape[-1]
    head_dim = hidden_dim // num_heads
    
    # Reshape: (batch, seq_len, hidden_dim) -> (batch, seq_len, num_heads, head_dim)
    hidden_states_heads = hidden_states.view(batch_size, -1, num_heads, head_dim)
    
    # Statistic 1: Per-head activation norms
    # Shape: (batch, num_heads)
    act_norms = hidden_states_heads.norm(dim=-1).mean(dim=1)
    
    # Statistic 2: Attention entropy per head
    # Add small epsilon to avoid log(0)
    attn_probs = attn_weights + 1e-10
    
    # Compute entropy: -sum(p * log(p))
    # Shape: (batch, num_heads, seq_len, seq_len) -> (batch, num_heads)
    entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean(dim=-1)
    
    # Concatenate statistics
    # Shape: (batch, 2*num_heads)
    stats = torch.cat([act_norms, entropy], dim=-1)
    
    return stats


def compute_head_importance(hidden_states: torch.Tensor, 
                           attn_weights: torch.Tensor,
                           task_loss: torch.Tensor) -> torch.Tensor:
    """Compute head importance scores based on gradient magnitudes.
    
    This is used during training to identify which heads contribute most
    to task performance. Higher importance = more critical for accuracy.
    
    Args:
        hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
        task_loss: Scalar loss tensor (must have gradients)
        
    Returns:
        importance: Tensor of shape (batch, num_heads) with importance scores
    """
    # Compute gradients of loss w.r.t. attention weights
    if attn_weights.requires_grad:
        grads = torch.autograd.grad(
            outputs=task_loss,
            inputs=attn_weights,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Importance = magnitude of gradients
        importance = grads.abs().mean(dim=(-2, -1))
    else:
        # Fallback: use activation norms as proxy
        batch_size = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        num_heads = attn_weights.shape[1]
        head_dim = hidden_dim // num_heads
        
        hidden_states_heads = hidden_states.view(batch_size, -1, num_heads, head_dim)
        importance = hidden_states_heads.norm(dim=-1).mean(dim=1)
    
    return importance
