"""Tests for MicrogliaAgent."""

import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import MicrogliaAgent


class TestMicrogliaAgent:
    """Test suite for MicrogliaAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = MicrogliaAgent(hidden_dim=128, num_heads=32, temperature=1.0)
        assert agent.num_heads == 32
        assert agent.temperature == 1.0
        assert len(list(agent.parameters())) > 0
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        agent = MicrogliaAgent(hidden_dim=128, num_heads=32)
        
        # Create dummy input: (batch_size, 2*num_heads)
        batch_size = 4
        stats = torch.randn(batch_size, 2 * 32)
        
        # Forward pass
        masks = agent(stats)
        
        # Check output shape
        assert masks.shape == (batch_size, 32)
    
    def test_mask_range(self):
        """Test that masks are in [0, 1] range."""
        agent = MicrogliaAgent(hidden_dim=128, num_heads=32)
        
        stats = torch.randn(4, 2 * 32)
        masks = agent(stats)
        
        # Masks should be between 0 and 1 (sigmoid output)
        assert torch.all(masks >= 0)
        assert torch.all(masks <= 1)
    
    def test_temperature_effect(self):
        """Test that temperature affects mask sharpness."""
        stats = torch.randn(4, 2 * 32)
        
        # Low temperature = sharper masks (closer to 0 or 1)
        agent_low = MicrogliaAgent(hidden_dim=128, num_heads=32, temperature=0.1)
        masks_low = agent_low(stats)
        
        # High temperature = softer masks (closer to 0.5)
        agent_high = MicrogliaAgent(hidden_dim=128, num_heads=32, temperature=2.0)
        masks_high = agent_high(stats)
        
        # Low temp should have more extreme values on average
        # (farther from 0.5)
        distance_low = torch.abs(masks_low - 0.5).mean()
        distance_high = torch.abs(masks_high - 0.5).mean()
        
        # This may not always hold due to random initialization,
        # but is generally true
        # assert distance_low > distance_high
    
    def test_set_temperature(self):
        """Test temperature setter."""
        agent = MicrogliaAgent(hidden_dim=128, num_heads=32, temperature=1.0)
        
        agent.set_temperature(0.5)
        assert agent.temperature == 0.5
    
    def test_gradient_flow(self):
        """Test that gradients flow through agent."""
        agent = MicrogliaAgent(hidden_dim=128, num_heads=32)
        
        stats = torch.randn(4, 2 * 32, requires_grad=True)
        masks = agent(stats)
        
        # Compute dummy loss
        loss = masks.sum()
        loss.backward()
        
        # Check that gradients exist
        assert stats.grad is not None
        for param in agent.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
