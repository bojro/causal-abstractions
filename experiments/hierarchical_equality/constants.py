"""Hierarchical-equality-specific constants and defaults."""

from pathlib import Path

from experiment_core.defaults import (
    DEFAULT_ACTIVATION,
    DEFAULT_DROPOUT,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
)

EXPERIMENT_ID = "hierarchical_equality"

# Match `train.py` / `compare.py` entrypoints (single canonical seed for shipped checkpoints).
DEFAULT_TRAIN_SEED = 44
DEFAULT_CHECKPOINT_PATH = Path(f"models/hierarchical_equality_mlp_seed{DEFAULT_TRAIN_SEED}.pt")

CANONICAL_INPUT_VARS = ("W", "X", "Y", "Z")
DEFAULT_TARGET_VARS = ("WX", "YZ")
CANONICAL_VARIABLE_MAPPING = {
    "WX": "WX",
    "YZ": "YZ",
}
VAR_VALUE_SPACE = {
    "WX": (0, 1),
    "YZ": (0, 1),
}

EMBEDDING_DIM = 4
INPUT_DIM = EMBEDDING_DIM * len(CANONICAL_INPUT_VARS)
NUM_CLASSES = 2

# Four-layer MLP: 64-wide matches the equality tutorial scale; wider 128 hid did not improve this seed.
DEFAULT_HIDDEN_DIMS = (64, 64, 64, 64)

# Slightly above the shared 30k default stabilizes validation on continuous draws (≈0.999+ exact_acc).
DEFAULT_FACTUAL_TRAIN_SIZE = 40_000

DEFAULT_CORE_METRICS = ("exact_acc", "mean_true_class_prob")

# Pair banks for OT/GW/FGW/DAS: between generic defaults (48/96) and addition (1k/5k) for two targets.
DEFAULT_COUNTERFACTUAL_TRAIN_SIZE = 512
DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE = 512
DEFAULT_COUNTERFACTUAL_TEST_SIZE = 2048

# Backbone usually plateaus well before 100 epochs; shared early-stop triggers only at val exact_acc == 1.0.
DEFAULT_TRAIN_EPOCHS = 60

__all__ = [
    "CANONICAL_INPUT_VARS",
    "CANONICAL_VARIABLE_MAPPING",
    "DEFAULT_ACTIVATION",
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_CORE_METRICS",
    "DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE",
    "DEFAULT_COUNTERFACTUAL_TEST_SIZE",
    "DEFAULT_COUNTERFACTUAL_TRAIN_SIZE",
    "DEFAULT_DROPOUT",
    "DEFAULT_FACTUAL_TRAIN_SIZE",
    "DEFAULT_FACTUAL_VALIDATION_SIZE",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_TARGET_VARS",
    "DEFAULT_TRAIN_EPOCHS",
    "DEFAULT_TRAIN_SEED",
    "EMBEDDING_DIM",
    "EXPERIMENT_ID",
    "INPUT_DIM",
    "NUM_CLASSES",
    "VAR_VALUE_SPACE",
]
