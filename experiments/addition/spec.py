"""Addition experiment specification and adapter hooks."""

from __future__ import annotations

from experiment_core.adapter import ExperimentAdapter
from experiment_core.experiment_spec import ExperimentSpec

from .constants import DEFAULT_TARGET_VARS, NUM_CLASSES
from .metrics import metrics_from_logits
from .selection import (
    choose_better_addition_candidate,
    summarize_addition_selection_records,
)


def build_addition_experiment_spec(
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    canonical_variable_mapping: dict[str, str] | None = None,
) -> ExperimentSpec:
    """Build the addition experiment spec with canonical variable metadata."""
    if canonical_variable_mapping is None:
        canonical_variable_mapping = {variable: variable for variable in target_vars}
    missing = [variable for variable in target_vars if variable not in canonical_variable_mapping]
    if missing:
        raise ValueError(
            "canonical_variable_mapping must cover all target_vars; "
            f"missing mappings for {missing}"
        )
    return ExperimentSpec(
        experiment_id="addition",
        local_target_vars=tuple(target_vars),
        canonical_variable_mapping={
            variable: str(canonical_variable_mapping[variable]) for variable in target_vars
        },
    )


def build_addition_checkpoint_metadata(train_config) -> dict[str, object]:
    """Build minimal checkpoint metadata for the addition experiment."""
    return {
        "seed": int(train_config.seed),
        "task": "two_digit_base10_addition_onehot",
        "abstract_variables": list(train_config.abstract_variables),
        "scm_variables": ["A1", "B1", "A2", "B2", "S1", "C1", "S2", "C2"],
        "target": "O",
        "output_classes": list(range(NUM_CLASSES)),
    }


def build_addition_adapter(
    *,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    canonical_variable_mapping: dict[str, str] | None = None,
) -> ExperimentAdapter:
    """Build the addition experiment adapter."""
    from .backbone import build_factual_tensors
    from .pair_bank import build_pair_bank

    return ExperimentAdapter(
        experiment_spec=build_addition_experiment_spec(
            target_vars=target_vars,
            canonical_variable_mapping=canonical_variable_mapping,
        ),
        build_factual_tensors=build_factual_tensors,
        build_pair_bank=build_pair_bank,
        metrics_from_logits=metrics_from_logits,
        build_checkpoint_metadata=build_addition_checkpoint_metadata,
        summarize_selection_records=summarize_addition_selection_records,
        choose_better_selection_candidate=choose_better_addition_candidate,
    )
