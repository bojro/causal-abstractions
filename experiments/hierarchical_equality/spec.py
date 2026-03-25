"""Experiment spec and adapter wiring for hierarchical equality."""

from __future__ import annotations

from experiment_core.adapter import ExperimentAdapter
from experiment_core.experiment_spec import ExperimentSpec

from .constants import (
    CANONICAL_VARIABLE_MAPPING,
    DEFAULT_CORE_METRICS,
    DEFAULT_TARGET_VARS,
    EXPERIMENT_ID,
    NUM_CLASSES,
)
from .metrics import metrics_from_logits
from .pair_bank import build_pair_bank
from .selection import (
    choose_better_hierarchical_equality_candidate,
    summarize_hierarchical_equality_selection_records,
)


def build_hierarchical_equality_experiment_spec(
    *,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    canonical_variable_mapping: dict[str, str] | None = None,
) -> ExperimentSpec:
    """Build the experiment contract for hierarchical equality."""
    if canonical_variable_mapping is None:
        canonical_variable_mapping = {
            variable: CANONICAL_VARIABLE_MAPPING[variable] for variable in target_vars
        }
    missing = [variable for variable in target_vars if variable not in canonical_variable_mapping]
    if missing:
        raise ValueError(f"Missing canonical mapping for target vars: {missing}")
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        local_target_vars=tuple(target_vars),
        canonical_variable_mapping={variable: str(canonical_variable_mapping[variable]) for variable in target_vars},
        core_metrics=DEFAULT_CORE_METRICS,
    )


def build_hierarchical_equality_checkpoint_metadata(train_config) -> dict[str, object]:
    """Describe the trained checkpoint in shared experiment metadata."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "task_name": "hierarchical_equality",
        "task_type": "binary_classification",
        "abstract_variables": list(train_config.abstract_variables),
        "output_variable": "O",
        "num_classes": int(NUM_CLASSES),
        "input_dim": int(train_config.input_dim),
        "hidden_dims": list(train_config.hidden_dims),
        "embedding_dim": int(train_config.input_dim // 4),
    }


def build_hierarchical_equality_adapter(
    *,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    canonical_variable_mapping: dict[str, str] | None = None,
) -> ExperimentAdapter:
    """Assemble the experiment-local hooks used by the shared runner."""
    from .backbone import build_factual_tensors

    return ExperimentAdapter(
        experiment_spec=build_hierarchical_equality_experiment_spec(
            target_vars=tuple(target_vars),
            canonical_variable_mapping=canonical_variable_mapping,
        ),
        build_factual_tensors=build_factual_tensors,
        build_pair_bank=build_pair_bank,
        metrics_from_logits=metrics_from_logits,
        build_checkpoint_metadata=build_hierarchical_equality_checkpoint_metadata,
        summarize_selection_records=summarize_hierarchical_equality_selection_records,
        choose_better_selection_candidate=choose_better_hierarchical_equality_candidate,
    )
