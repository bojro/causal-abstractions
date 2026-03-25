"""Hierarchical-equality backbone wrappers built on top of the shared core."""

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
from .scm import HierarchicalEqualityProblem, sample_structured_examples
from .spec import build_hierarchical_equality_adapter


@dataclass(frozen=True)
class HierarchicalEqualityTrainConfig(ClassifierTrainConfig):
    """Configuration for supervised training and loading of the equality backbone."""

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
    problem: HierarchicalEqualityProblem,
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample supervised equality examples and convert them to tensors."""
    inputs, labels, _ = sample_structured_examples(size, seed, embedding_dim=problem.embedding_dim)
    return inputs, labels


def train_backbone(
    *,
    problem: HierarchicalEqualityProblem,
    train_config: HierarchicalEqualityTrainConfig,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
) -> tuple[object, object, dict[str, object]]:
    """Train the hierarchical-equality backbone MLP."""
    adapter = build_hierarchical_equality_adapter(
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
    problem: HierarchicalEqualityProblem,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | str = "cpu",
    train_config: HierarchicalEqualityTrainConfig | None = None,
) -> tuple[object, object, dict[str, object]]:
    """Load an existing equality checkpoint and fail if it is incompatible."""
    eval_config = train_config or HierarchicalEqualityTrainConfig()
    adapter = build_hierarchical_equality_adapter(
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
