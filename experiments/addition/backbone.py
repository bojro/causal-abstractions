"""Addition-specific backbone wrappers built on top of the shared core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from experiment_core.backbone import (
    ClassifierTrainConfig,
    load_classifier_backbone,
    train_classifier_backbone,
)

from .constants import (
    DEFAULT_ACTIVATION,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_TARGET_VARS,
    INPUT_DIM,
    NUM_CLASSES,
)
from .scm import AdditionProblem, compute_states_for_digits, digits_to_inputs_embeds, sample_digit_rows
from .spec import build_addition_adapter


@dataclass(frozen=True)
class AdditionTrainConfig(ClassifierTrainConfig):
    """Configuration for supervised training and loading of the addition backbone."""

    seed: int = 42
    n_train: int = DEFAULT_FACTUAL_TRAIN_SIZE
    n_validation: int = DEFAULT_FACTUAL_VALIDATION_SIZE
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    input_dim: int = INPUT_DIM
    num_classes: int = NUM_CLASSES
    dropout: float = DEFAULT_DROPOUT
    activation: str = DEFAULT_ACTIVATION
    abstract_variables: tuple[str, ...] = DEFAULT_TARGET_VARS


def build_factual_tensors(
    problem: AdditionProblem,
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample supervised addition examples and convert them to tensors."""
    digits = sample_digit_rows(size, seed)
    states = compute_states_for_digits(digits)
    inputs = digits_to_inputs_embeds(digits, problem.input_var_order)
    labels = torch.tensor(states["O"], dtype=torch.long)
    return inputs, labels


def train_backbone(
    *,
    problem: AdditionProblem,
    train_config: AdditionTrainConfig,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
) -> tuple[object, object, dict[str, object]]:
    """Train the addition backbone MLP and return the model, config, and metrics."""
    adapter = build_addition_adapter(
        target_vars=tuple(train_config.abstract_variables),
        canonical_variable_mapping={
            variable: problem.experiment_spec.canonical_variable_mapping[variable]
            for variable in train_config.abstract_variables
        },
    )
    return train_classifier_backbone(
        problem=problem,
        adapter=adapter,
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def load_backbone(
    *,
    problem: AdditionProblem,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
    train_config: AdditionTrainConfig | None = None,
) -> tuple[object, object, dict[str, object]]:
    """Load an existing addition checkpoint and fail if it is incompatible."""
    eval_config = train_config or AdditionTrainConfig()
    adapter = build_addition_adapter(
        target_vars=tuple(eval_config.abstract_variables),
        canonical_variable_mapping={
            variable: problem.experiment_spec.canonical_variable_mapping[variable]
            for variable in eval_config.abstract_variables
        },
    )
    return load_classifier_backbone(
        problem=problem,
        adapter=adapter,
        checkpoint_path=checkpoint_path,
        device=device,
        train_config=eval_config,
    )
