"""Utilities for mixed precision training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch

Precision = Literal["fp32", "fp16", "bf16"]


@dataclass
class PrecisionConfig:
    precision: Precision = "fp32"

    @property
    def dtype(self):
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return torch.float32

    @property
    def amp_enabled(self) -> bool:
        return self.precision in {"fp16", "bf16"}


class MixedPrecisionTrainer:
    """Performs a single optimization step with AMP where available."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: PrecisionConfig):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        use_scaler = config.precision == "fp16" and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def train_step(self, loss_fn, *args, **kwargs) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        if self.config.amp_enabled and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self.config.dtype):
                loss = loss_fn(*args, **kwargs)
        else:
            ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16) if self.config.precision == "bf16" else nullcontext()
            with ctx:
                loss = loss_fn(*args, **kwargs)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return float(loss.detach().cpu().item())
