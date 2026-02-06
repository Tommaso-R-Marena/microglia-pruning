"""Tests for activation hooks."""

import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hooks import create_activation_hook, remove_hooks


class DummyAttention(nn.Module):
    """Dummy attention module for testing."""
    
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # Return (hidden_states, attention_weights)
        hidden = self.proj(x)
        attn = torch.ones(x.shape[0], self.num_heads, x.shape[1], x.shape[1])
        return hidden, attn


class TestHooks:
    """Test suite for activation hooks."""
    
    def test_create_hook(self):
        """Test hook creation."""
        cache = {}
        hook = create_activation_hook(layer_idx=0, activation_cache=cache)
        assert callable(hook)
    
    def test_hook_captures_activations(self):
        """Test that hook captures activations."""
        # Create dummy module
        module = DummyAttention()
        
        # Create cache and hook
        cache = {}
        hook = create_activation_hook(layer_idx=0, activation_cache=cache)
        
        # Register hook
        handle = module.register_forward_hook(hook)
        
        # Forward pass
        x = torch.randn(2, 10, 64)
        _ = module(x)
        
        # Check cache is populated
        assert "layer_0" in cache
        assert "hidden_states" in cache["layer_0"]
        assert "attn_weights" in cache["layer_0"]
        
        # Clean up
        handle.remove()
    
    def test_hook_detaches_tensors(self):
        """Test that hook detaches tensors (no gradients)."""
        module = DummyAttention()
        cache = {}
        hook = create_activation_hook(layer_idx=0, activation_cache=cache)
        handle = module.register_forward_hook(hook)
        
        x = torch.randn(2, 10, 64, requires_grad=True)
        _ = module(x)
        
        # Cached tensors should not require grad
        assert not cache["layer_0"]["hidden_states"].requires_grad
        assert not cache["layer_0"]["attn_weights"].requires_grad
        
        handle.remove()
    
    def test_multiple_hooks(self):
        """Test multiple hooks on different layers."""
        modules = [DummyAttention() for _ in range(3)]
        cache = {}
        handles = []
        
        # Register hooks on all modules
        for idx, module in enumerate(modules):
            hook = create_activation_hook(layer_idx=idx, activation_cache=cache)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        
        # Forward pass on all modules
        x = torch.randn(2, 10, 64)
        for module in modules:
            x, _ = module(x)
        
        # Check all layers captured
        assert len(cache) == 3
        assert "layer_0" in cache
        assert "layer_1" in cache
        assert "layer_2" in cache
        
        # Clean up
        remove_hooks(handles)
    
    def test_remove_hooks(self):
        """Test hook removal."""
        module = DummyAttention()
        cache = {}
        hook = create_activation_hook(layer_idx=0, activation_cache=cache)
        handle = module.register_forward_hook(hook)
        
        # Remove hook
        remove_hooks([handle])
        
        # Forward pass should not populate cache
        cache.clear()
        x = torch.randn(2, 10, 64)
        _ = module(x)
        
        # Cache should be empty
        assert len(cache) == 0


if __name__ == "__main__":
    pytest.main([__file__])
