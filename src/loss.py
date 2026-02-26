"""Loss functions for training pruning agents."""

import torch
import torch.nn.functional as F


def compute_pruning_loss(task_loss: torch.Tensor,
                        masks: torch.Tensor,
                        alpha: float = 0.1,
                        beta: float = 0.01) -> dict:
    """Compute combined loss for training pruning agents.
    
    The total loss has three components:
    1. Task loss: Preserve model accuracy on the target task
    2. Sparsity loss: Encourage pruning (push masks toward 0)
    3. Entropy loss: Prevent masks from getting stuck at 0.5
    
    Args:
        task_loss: Cross-entropy loss on answer tokens (scalar)
        masks: Pruning masks of shape (batch, num_heads) with values in [0, 1]
        alpha: Weight for sparsity loss (higher = more aggressive pruning)
        beta: Weight for entropy regularization
        
    Returns:
        dict containing:
            - 'total_loss': Combined loss for backprop
            - 'task_loss': Task accuracy component
            - 'sparsity_loss': Pruning encouragement component
            - 'entropy_loss': Mask stability component
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
    """Get sparsity weight for curriculum learning.
    
    The pruning pressure (alpha) increases linearly over training:
    - Early epochs: Low alpha -> learn which heads are important
    - Mid epochs: Medium alpha -> gradually prune less important heads
    - Late epochs: High alpha -> stabilize pruning pattern
    
    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total number of epochs
        alpha_min: Starting value of alpha
        alpha_max: Final value of alpha
        
    Returns:
        alpha: Sparsity weight for current epoch
    """
    # Linear schedule
    progress = epoch / max(max_epochs - 1, 1)
    alpha = alpha_min + (alpha_max - alpha_min) * progress
    
    return alpha


def compute_efficiency_metrics(masks: torch.Tensor) -> dict:
    """Compute pruning efficiency metrics.
    
    Args:
        masks: Pruning masks of shape (batch, num_heads)
        
    Returns:
        dict containing:
            - 'sparsity': Fraction of heads pruned (using 0.5 threshold)
            - 'mean_mask': Average mask value
            - 'active_heads': Average number of active heads per sample
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
