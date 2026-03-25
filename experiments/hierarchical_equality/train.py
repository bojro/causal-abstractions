from experiment_core.runtime import resolve_device

from .backbone import HierarchicalEqualityTrainConfig, train_backbone
from .constants import (
    CANONICAL_VARIABLE_MAPPING,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_TARGET_VARS,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_TRAIN_SEED,
)
from .scm import load_hierarchical_equality_problem

SEED = DEFAULT_TRAIN_SEED
DEVICE = None
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH

TRAIN_SIZE = DEFAULT_FACTUAL_TRAIN_SIZE
VALIDATION_SIZE = DEFAULT_FACTUAL_VALIDATION_SIZE
HIDDEN_DIMS = DEFAULT_HIDDEN_DIMS
TARGET_VARS = DEFAULT_TARGET_VARS
LEARNING_RATE = 1e-3
EPOCHS = DEFAULT_TRAIN_EPOCHS
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256


def main() -> None:
    problem = load_hierarchical_equality_problem(
        run_checks=True,
        target_vars=tuple(TARGET_VARS),
        canonical_variable_mapping=CANONICAL_VARIABLE_MAPPING,
    )
    device = resolve_device(DEVICE)
    train_config = HierarchicalEqualityTrainConfig(
        seed=SEED,
        n_train=TRAIN_SIZE,
        n_validation=VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )
    train_backbone(
        problem=problem,
        train_config=train_config,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )
    print(f"Wrote checkpoint to {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()
