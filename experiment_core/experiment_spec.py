"""Experiment metadata contracts for comparable experiment runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    """Minimal contract describing one experiment family's comparable outputs."""

    experiment_id: str
    local_target_vars: tuple[str, ...]
    canonical_variable_mapping: dict[str, str]
    core_metrics: tuple[str, ...] = ("exact_acc", "mean_shared_digits")

    @property
    def canonical_target_vars(self) -> tuple[str, ...]:
        """Return canonical variable names in the same order as local target vars."""
        return tuple(self.canonical_variable_mapping[variable] for variable in self.local_target_vars)
