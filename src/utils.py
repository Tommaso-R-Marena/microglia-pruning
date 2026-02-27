"""Utility functions for model architecture handling."""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
import sys

def set_seed(seed: int = 42):
    """Set seeds for reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in some operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(name: str = "microglia", level: int = logging.INFO) -> logging.Logger:
    """Set up structured logging.

    Args:
        name: Name of the logger.
        level: Logging level.

    Returns:
        logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

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
        if hasattr(model.base_model, "model"):
            model = model.base_model.model
        else:
            # For some models, base_model is the internal model itself
            model = model.base_model

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
