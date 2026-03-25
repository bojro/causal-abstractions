"""Hierarchical-equality selection policy for OT and DAS candidate search."""

from __future__ import annotations


def summarize_hierarchical_equality_selection_records(
    records: list[dict[str, object]],
) -> dict[str, float]:
    """Average the calibration metrics used to rank candidates."""
    if not records:
        return {"exact_acc": 0.0, "mean_true_class_prob": 0.0}
    exact_acc = sum(float(record["exact_acc"]) for record in records) / len(records)
    mean_true_class_prob = sum(float(record["mean_true_class_prob"]) for record in records) / len(records)
    return {
        "exact_acc": exact_acc,
        "mean_true_class_prob": mean_true_class_prob,
    }


def choose_better_hierarchical_equality_candidate(
    candidate: dict[str, object],
    incumbent: dict[str, object] | None,
) -> bool:
    """Prefer higher exact accuracy, then higher gold-class confidence."""
    if incumbent is None:
        return True
    candidate_metrics = dict(candidate.get("selection_metrics", {}))
    incumbent_metrics = dict(incumbent.get("selection_metrics", {}))
    candidate_key = (
        float(candidate_metrics.get("exact_acc", 0.0)),
        float(candidate_metrics.get("mean_true_class_prob", 0.0)),
    )
    incumbent_key = (
        float(incumbent_metrics.get("exact_acc", 0.0)),
        float(incumbent_metrics.get("mean_true_class_prob", 0.0)),
    )
    return candidate_key > incumbent_key
