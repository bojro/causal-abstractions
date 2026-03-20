"""Addition-specific selection policy for OT and DAS candidate search."""

from __future__ import annotations


def summarize_addition_selection_records(records: list[dict[str, object]]) -> dict[str, float]:
    """Summarize addition candidate records using the current calibration metrics."""
    if not records:
        return {"exact_acc": 0.0, "mean_shared_digits": 0.0}
    exact_acc = sum(float(record["exact_acc"]) for record in records) / len(records)
    mean_shared_digits = sum(float(record["mean_shared_digits"]) for record in records) / len(records)
    return {
        "exact_acc": exact_acc,
        "mean_shared_digits": mean_shared_digits,
    }


def choose_better_addition_candidate(
    candidate: dict[str, object],
    incumbent: dict[str, object] | None,
) -> bool:
    """Preserve the current addition lexicographic selection rule."""
    if incumbent is None:
        return True
    candidate_metrics = dict(candidate.get("selection_metrics", {}))
    incumbent_metrics = dict(incumbent.get("selection_metrics", {}))
    candidate_key = (
        float(candidate_metrics.get("exact_acc", 0.0)),
        float(candidate_metrics.get("mean_shared_digits", 0.0)),
    )
    incumbent_key = (
        float(incumbent_metrics.get("exact_acc", 0.0)),
        float(incumbent_metrics.get("mean_shared_digits", 0.0)),
    )
    return candidate_key > incumbent_key
