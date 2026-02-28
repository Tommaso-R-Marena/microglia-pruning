"""Tests for statistics module."""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.statistics import NUM_STATS_PER_HEAD, compute_layer_stats


def test_compute_layer_stats_shape_and_components() -> None:
    batch_size, seq_len, num_heads, head_dim = 2, 5, 4, 8
    hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim)
    attn_weights = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)

    stats = compute_layer_stats(hidden_states, attn_weights)

    assert stats.shape == (batch_size, NUM_STATS_PER_HEAD * num_heads)


def test_compute_layer_stats_with_gradients() -> None:
    batch_size, seq_len, num_heads, head_dim = 2, 4, 2, 6
    hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim)
    attn_weights = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1).requires_grad_(True)
    task_loss = attn_weights.square().sum()

    stats = compute_layer_stats(hidden_states, attn_weights, task_loss=task_loss)

    grad_slice = stats[:, 4 * num_heads:5 * num_heads]
    assert torch.all(grad_slice > 0)


def test_vectorized_stats_matches_batch_dimension() -> None:
    batch_size, seq_len, num_heads, head_dim = 8, 16, 6, 8
    hidden_states = torch.randn(batch_size, seq_len, num_heads * head_dim)
    attn_weights = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)

    stats = compute_layer_stats(hidden_states, attn_weights)

    assert stats.ndim == 2
    assert stats.shape[0] == batch_size
