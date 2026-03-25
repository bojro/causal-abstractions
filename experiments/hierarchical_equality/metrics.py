"""Hierarchical-equality evaluation metrics."""

from __future__ import annotations

import torch


def exact_match_accuracy(
    predictions: torch.Tensor | list[int],
    targets: torch.Tensor | list[int],
) -> float:
    """Compute exact binary class-match accuracy."""
    preds = torch.as_tensor(predictions, dtype=torch.long).view(-1)
    gold = torch.as_tensor(targets, dtype=torch.long).view(-1)
    return float((preds == gold).to(torch.float32).mean().item())


def mean_true_class_probability(
    logits: torch.Tensor,
    targets: torch.Tensor | list[int],
) -> float:
    """Compute mean probability assigned to the gold class."""
    gold = torch.as_tensor(targets, dtype=torch.long, device=logits.device).view(-1)
    probabilities = torch.softmax(logits, dim=1)
    return float(probabilities[torch.arange(probabilities.shape[0], device=logits.device), gold].mean().item())


def metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor | list[int],
) -> dict[str, float]:
    """Convert logits to standard hierarchical-equality metrics."""
    predictions = torch.argmax(logits, dim=1)
    return {
        "exact_acc": exact_match_accuracy(predictions, targets),
        "mean_true_class_prob": mean_true_class_probability(logits, targets),
    }
