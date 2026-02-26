"""Tests for PrunedAttention."""

import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pruned_attention import PrunedAttention
from src.agent import MicrogliaAgent


class DummyAttention(nn.Module):
    """Dummy attention for testing."""
    
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=True, **kwargs):
        hidden = self.proj(hidden_states)
        attn = torch.ones(hidden_states.shape[0], self.num_heads, 
                         hidden_states.shape[1], hidden_states.shape[1])
        return hidden, attn


class TestPrunedAttention:
    """Test suite for PrunedAttention."""
    
    def test_initialization(self):
        """Test pruned attention initialization."""
        attn = DummyAttention()
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        
        pruned_attn = PrunedAttention(attn, agent)
        
        assert pruned_attn.attn is attn
        assert pruned_attn.agent is agent
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent)
        
        # Input
        batch_size, seq_len, hidden_dim = 2, 10, 64
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward
        output, weights = pruned_attn(x)
        
        # Check shapes
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert weights.shape == (batch_size, 4, seq_len, seq_len)
    
    def test_masking_applied(self):
        """Test that masking is actually applied."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent)
        pruned_attn.enable_pruning = True
        
        x = torch.randn(2, 10, 64)
        
        # Forward pass
        output_pruned, _ = pruned_attn(x)
        
        # Without pruning
        output_original, _ = attn(x, output_attentions=True)
        
        # Outputs should be different (masks applied)
        assert not torch.allclose(output_pruned, output_original)
    
    def test_hard_pruning(self):
        """Test hard pruning mode."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent, hard_prune=True)
        pruned_attn.enable_pruning = True
        
        # Set to eval mode
        pruned_attn.eval()
        
        x = torch.randn(2, 10, 64)
        
        # In eval mode with hard_prune=True, masks should be binary
        # This is hard to test directly without modifying the module
        # to expose masks, but we can at least check it runs
        output, _ = pruned_attn(x)
        assert output.shape == (2, 10, 64)
    
    def test_gradient_flow(self):
        """Test that gradients flow through pruned attention."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent)
        pruned_attn.enable_pruning = True
        
        x = torch.randn(2, 10, 64, requires_grad=True)
        output, _ = pruned_attn(x)
        
        # Compute loss and backprop
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in agent.parameters():
            assert param.grad is not None

    def test_eval_mode_pruning(self):
        """Test that pruning still works in eval mode."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent)
        pruned_attn.enable_pruning = True

        # Set to eval mode
        pruned_attn.eval()

        x = torch.randn(2, 10, 64)

        # Forward pass in eval mode
        output_eval, _ = pruned_attn(x)

        # Without pruning
        output_original, _ = attn(x, output_attentions=True)

        # Pruning should still be applied even in eval mode
        assert not torch.allclose(output_eval, output_original)

    def test_hard_pruning_inference(self):
        """Test that hard pruning produces binary masks during inference."""
        attn = DummyAttention(hidden_dim=64, num_heads=4)
        agent = MicrogliaAgent(hidden_dim=128, num_heads=4)
        pruned_attn = PrunedAttention(attn, agent, hard_prune=True)
        pruned_attn.enable_pruning = True

        pruned_attn.eval()

        x = torch.randn(2, 10, 64)
        _ = pruned_attn(x)

        # Check that last_masks are binary (0.0 or 1.0)
        masks = pruned_attn.last_masks
        assert torch.all((masks == 0.0) | (masks == 1.0))


if __name__ == "__main__":
    pytest.main([__file__])
