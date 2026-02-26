"""Utility functions for model architecture handling."""

import torch.nn as nn

def get_model_layers(model: nn.Module):
    """Get the layers of a transformer model, handling PEFT wrapping and different architectures.

    Args:
        model: The model to extract layers from.

    Returns:
        layers: The ModuleList containing the model layers.

    Raises:
        AttributeError: If layers cannot be found for the model type.
    """
    # Handle PEFT wrapping
    if hasattr(model, "base_model"):
        model = model.base_model.model

    # Standard Transformers architectures
    if hasattr(model, "model"): # Llama, Phi-3, etc.
        if hasattr(model.model, "layers"):
            return model.model.layers

    if hasattr(model, "transformer"): # GPT-2, etc.
        if hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model.transformer, "layers"):
            return model.transformer.layers

    # Direct access fallbacks
    if hasattr(model, "layers"):
        return model.layers

    if hasattr(model, "h"):
        return model.h

    raise AttributeError(f"Could not find layers for model type {type(model)}")
