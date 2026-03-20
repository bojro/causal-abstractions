"""Addition-specific constants and defaults."""

from pathlib import Path

from experiment_core.defaults import (
    DEFAULT_ACTIVATION,
    DEFAULT_DROPOUT,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
)


DEFAULT_CHECKPOINT_PATH = Path("models/addition_mlp_seed42.pt")

DEFAULT_TARGET_VARS = ("S1", "C1", "S2", "C2")
CANONICAL_INPUT_VARS = ("A1", "B1", "A2", "B2")
VAR_VALUE_SPACE = {
    "S1": tuple(range(10)),
    "C1": (0, 1),
    "S2": tuple(range(10)),
    "C2": (0, 1),
}

INPUT_DIM = 40
NUM_CLASSES = 200
OUTPUT_DIGIT_COUNT = 3

__all__ = [
    "CANONICAL_INPUT_VARS",
    "DEFAULT_ACTIVATION",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DROPOUT",
    "DEFAULT_FACTUAL_TRAIN_SIZE",
    "DEFAULT_FACTUAL_VALIDATION_SIZE",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_TARGET_VARS",
    "INPUT_DIM",
    "NUM_CLASSES",
    "OUTPUT_DIGIT_COUNT",
    "VAR_VALUE_SPACE",
]
