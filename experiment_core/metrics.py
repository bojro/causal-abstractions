"""Core metric-contract helpers."""

from __future__ import annotations

from .experiment_spec import ExperimentSpec


def validate_core_metrics(metrics: dict[str, float], experiment_spec: ExperimentSpec) -> None:
    """Assert that a metric payload satisfies the experiment's comparable core metrics."""
    missing = [metric_name for metric_name in experiment_spec.core_metrics if metric_name not in metrics]
    if missing:
        raise AssertionError(f"Missing required core metrics: {missing}")


def validate_core_metric_records(
    records: list[dict[str, object]],
    experiment_spec: ExperimentSpec,
) -> None:
    """Assert that every result record exposes the experiment's required core metrics."""
    for index, record in enumerate(records):
        try:
            validate_core_metrics(record, experiment_spec)
        except AssertionError as exc:
            raise AssertionError(f"Record {index} violates core metric contract: {exc}") from exc
