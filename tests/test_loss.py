"""Tests for loss module."""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loss import compute_pruning_loss


def test_compute_pruning_loss_with_distillation_and_layer_targets() -> None:
    task_loss = torch.tensor(1.2)
    masks = torch.tensor(
        [[0.9, 0.7, 0.2, 0.1], [0.8, 0.5, 0.3, 0.2], [0.7, 0.6, 0.4, 0.2]],
        dtype=torch.float32,
    )
    student_logits = torch.randn(3, 10)
    teacher_logits = torch.randn(3, 10)
    layer_targets = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)

    out = compute_pruning_loss(
        task_loss=task_loss,
        masks=masks,
        alpha=0.1,
        beta=0.01,
        distillation_weight=0.2,
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        kd_temperature=2.0,
        layer_sparsity_targets=layer_targets,
        layer_target_weight=0.3,
    )

    assert "distillation_loss" in out
    assert "layer_sparsity_target_loss" in out
    assert out["total_loss"].item() > 0


def test_compute_pruning_loss_without_optional_terms() -> None:
    task_loss = torch.tensor(0.8)
    masks = torch.rand(2, 4)

    out = compute_pruning_loss(task_loss=task_loss, masks=masks)

    assert out["distillation_loss"] == 0.0
    assert out["layer_sparsity_target_loss"] == 0.0
