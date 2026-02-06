"""Microglia-Inspired Dynamic Pruning for Reasoning Models."""

from .agent import MicrogliaAgent
from .hooks import create_activation_hook, register_hooks
from .pruned_attention import PrunedAttention
from .statistics import compute_layer_stats
from .loss import compute_pruning_loss
from .system import MicrogliaPruningSystem

__version__ = "0.1.0"

__all__ = [
    "MicrogliaAgent",
    "create_activation_hook",
    "register_hooks",
    "PrunedAttention",
    "compute_layer_stats",
    "compute_pruning_loss",
    "MicrogliaPruningSystem",
]
