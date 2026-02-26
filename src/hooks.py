"""PyTorch hooks for capturing layer activations."""

from typing import Dict, Any
import torch
import torch.nn as nn


def create_activation_hook(layer_idx: int, activation_cache: Dict[str, Any]):
    """Create a forward hook to capture activations from a layer.
    
    Args:
        layer_idx: Index of the layer to monitor
        activation_cache: Dictionary to store captured activations
        
    Returns:
        hook: Function that will be called during forward pass
    """
    
    def hook(module: nn.Module, input: tuple, output: tuple):
        """Hook function that captures hidden states and attention weights."""
        # Extract hidden states and attention weights from output
        # Output format depends on model, but typically:
        # output[0] = hidden_states, output[1] = attention_weights (if output_attentions=True)
        
        hidden_states = output[0].detach()
        attn_weights = output[1].detach() if len(output) > 1 and output[1] is not None else None
        
        activation_cache[f"layer_{layer_idx}"] = {
            "hidden_states": hidden_states,
            "attn_weights": attn_weights,
        }
    
    return hook


def register_hooks(model: nn.Module, activation_cache: Dict[str, Any]) -> list:
    """Register forward hooks on all attention layers.
    
    Args:
        model: Transformer model to monitor
        activation_cache: Dictionary to store activations
        
    Returns:
        handles: List of hook handles (for later removal if needed)
    """
    handles = []
    
    # Get layers, handling PEFT wrapping and different model architectures
    if hasattr(model, "base_model"):
        model = model.base_model.model

    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h if hasattr(model.transformer, "h") else model.transformer.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError(f"Could not find layers for model type {type(model)}")

    # Register hooks on all attention layers
    for idx, layer in enumerate(layers):
        hook = create_activation_hook(idx, activation_cache)
        handle = layer.self_attn.register_forward_hook(hook)
        handles.append(handle)
    
    return handles


def remove_hooks(handles: list):
    """Remove all registered hooks.
    
    Args:
        handles: List of hook handles returned by register_hooks
    """
    for handle in handles:
        handle.remove()
