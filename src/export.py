"""
Microglia-Inspired Dynamic Pruning for Reasoning Models

Authors: Tommaso R. Marena (The Catholic University of America)
         Panos Ketonis (Yale University)

Copyright (c) 2026
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset: int = 17,
    seq_len: int = 32,
    vocab_size: int = 1000,
    device: Optional[str] = None,
) -> Path:
    """Export a torch module to ONNX with dynamic batch and sequence dimensions."""
    model.eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)

    dummy_input = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long, device=target_device)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
    except Exception as exc:
        raise RuntimeError("ONNX export failed. Ensure `onnx` is installed and model is export-compatible.") from exc
    return path
