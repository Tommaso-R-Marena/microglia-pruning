"""Integration tests for the complete pruning system."""

import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system import MicrogliaPruningSystem

class FakeAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        batch, seq, dim = hidden_states.shape
        # Fake attention weights
        attn_weights = torch.ones(batch, self.num_heads, seq, seq, device=hidden_states.device)
        # Fake output
        attn_output = self.o_proj(hidden_states)
        if output_attentions:
            return attn_output, attn_weights
        return attn_output

class FakeLayer(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.self_attn = FakeAttention(hidden_dim, num_heads)

class FakeModelInternal(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden_dim, num_heads) for _ in range(num_layers)])

class FakeModel(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=64, num_heads=4):
        super().__init__()
        self.model = FakeModelInternal(num_layers, hidden_dim, num_heads)
        self.config = nn.Module()
        self.config.hidden_size = hidden_dim
        self.config.num_attention_heads = num_heads
        self.config.pad_token_id = 0
        self.config.eos_token_id = 2

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Fake forward pass
        # Just use some weights to have gradients
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 64, device=input_ids.device, requires_grad=True)
        for layer in self.model.layers:
            x = layer.self_attn(x, attention_mask=attention_mask)

        loss = x.sum()
        class Output:
            def __init__(self, loss):
                self.loss = loss
        return Output(loss)

class TestSystemIntegration:
    """Test suite for MicrogliaPruningSystem integration."""

    def test_system_initialization(self):
        """Test that the system initializes with a model and wraps it."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')

        # Check agents were initialized
        assert len(system.agents) == 2

        # Wrap layers
        system._wrap_attention_layers()
        assert system.wrapped

        # Check that layers were replaced
        layers = system.get_layers()
        for layer in layers:
            from src.pruned_attention import PrunedAttention
            assert isinstance(layer.self_attn, PrunedAttention)

    def test_generate_toggles_pruning(self):
        """Test that generate method correctly toggles pruning state."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')
        system._wrap_attention_layers()

        # Mock tokenizer
        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                class BatchEncoding(dict):
                    def to(self, device):
                        return self
                return BatchEncoding({'input_ids': torch.zeros(1, 10, dtype=torch.long),
                                     'attention_mask': torch.ones(1, 10, dtype=torch.long)})
            def decode(self, tokens, **kwargs):
                return "Fake response"

        system.tokenizer = FakeTokenizer()

        # Mock model.generate
        def fake_generate(**kwargs):
            return torch.zeros(1, 20, dtype=torch.long)
        system.model.generate = fake_generate

        # Test with pruning enabled
        _ = system.generate("Test", use_pruning=True)
        assert system.pruning_enabled == True

        # Test with pruning disabled
        _ = system.generate("Test", use_pruning=False)
        assert system.pruning_enabled == False

    def test_hard_prune_toggling(self):
        """Test that set_hard_prune toggles hard_prune on all layers."""
        fake_model = FakeModel(num_layers=2, hidden_dim=64, num_heads=4)
        system = MicrogliaPruningSystem(model=fake_model, num_heads=4, hidden_dim=32, device='cpu')
        system._wrap_attention_layers()

        system.set_hard_prune(True)
        layers = system.get_layers()
        for layer in layers:
            assert layer.self_attn.hard_prune == True

        system.set_hard_prune(False)
        layers = system.get_layers()
        for layer in layers:
            assert layer.self_attn.hard_prune == False
