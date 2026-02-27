"""Loss functions for training pruning agents."""

import torch
import torch.nn.functional as F


from typing import Dict, Tuple

def compute_pruning_loss(task_loss: torch.Tensor,
                        masks: torch.Tensor,
                        alpha: float = 0.1,
                        beta: float = 0.01) -> Dict[str, any]:
    """Computes the combined loss for training pruning agents.
    
    The total loss is formulated as:
    L_total = L_task + α * L_sparsity + β * L_entropy
    
    Args:
        task_loss (torch.Tensor): Cross-entropy loss on answer tokens (scalar).
        masks (torch.Tensor): Pruning masks of shape (batch, num_heads) with values in [0, 1].
        alpha (float): Weight for sparsity loss (higher = more aggressive pruning).
        beta (float): Weight for entropy regularization (minimizing entropy pushes masks to 0 or 1).
        
    Returns:
        Dict[str, any]: A dictionary containing the total loss and its components.
    """
    # Component 1: Task loss (from language modeling objective)
    # Already computed - just use it
    
    # Component 2: Sparsity loss
    # Mean of masks - lower is better (more pruning)
    sparsity_loss = masks.mean()
    
    # Component 3: Entropy regularization
    # Prevent masks from getting stuck at 0.5 by encouraging binary decisions
    # Entropy = -p*log(p) - (1-p)*log(1-p)
    # High entropy = mask near 0.5, low entropy = mask near 0 or 1
    eps = 1e-10
    mask_entropy = -(masks * (masks + eps).log() + 
                     (1 - masks) * (1 - masks + eps).log()).mean()
    
    # Combine losses
    # Use + beta to minimize entropy (encourage binary decisions 0 or 1)
    # The original - beta was encouraging max entropy (pushing toward 0.5)
    total_loss = task_loss + alpha * sparsity_loss + beta * mask_entropy
    
    return {
        'total_loss': total_loss,
        'task_loss': task_loss.item(),
        'sparsity_loss': sparsity_loss.item(),
        'entropy_loss': mask_entropy.item(),
    }


def get_alpha_schedule(epoch: int, max_epochs: int, 
                      alpha_min: float = 0.01, 
                      alpha_max: float = 0.3) -> float:
    """Calculates the sparsity weight (alpha) for curriculum learning.
    
    The pruning pressure (alpha) increases linearly over training to allow the
    agents to first identify head importance before enforcing sparsity.
    
    Args:
        epoch (int): Current epoch (0-indexed).
        max_epochs (int): Total number of epochs.
        alpha_min (float): Starting value of alpha.
        alpha_max (float): Final value of alpha.
        
    Returns:
        float: The sparsity weight for the current epoch.
    """
    # Linear schedule
    progress = epoch / max(max_epochs - 1, 1)
    alpha = alpha_min + (alpha_max - alpha_min) * progress
    
    return alpha


def compute_efficiency_metrics(masks: torch.Tensor) -> Dict[str, float]:
    """Computes pruning efficiency metrics.
    
    Args:
        masks (torch.Tensor): Pruning masks of shape (batch, num_heads).
        
    Returns:
        Dict[str, float]: A dictionary containing sparsity, mean mask value,
            and average number of active heads.
    """
    with torch.no_grad():
        # Binary masks for counting
        binary_masks = (masks > 0.5).float()
        
        active_heads = binary_masks.sum(dim=1).mean().item()
        total_heads = masks.shape[1]
        sparsity = 1.0 - (active_heads / total_heads)
        mean_mask = masks.mean().item()
        
    return {
        'sparsity': sparsity,
        'mean_mask': mean_mask,
        'active_heads': active_heads,
    }
