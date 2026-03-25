import os
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment_core.compare_runner import CompareExperimentConfig, run_comparison_from_checkpoint
from experiment_core.runtime import resolve_device

from .backbone import HierarchicalEqualityTrainConfig
from .constants import (
    CANONICAL_VARIABLE_MAPPING,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE,
    DEFAULT_COUNTERFACTUAL_TEST_SIZE,
    DEFAULT_COUNTERFACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_TARGET_VARS,
    DEFAULT_TRAIN_SEED,
)
from .scm import load_hierarchical_equality_problem
from .spec import build_hierarchical_equality_adapter

SEED = DEFAULT_TRAIN_SEED
DEVICE = None
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / RUN_TIMESTAMP
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
OUTPUT_PATH = RUN_DIR / "hierarchical_equality_compare_results.json"
SUMMARY_PATH = RUN_DIR / "hierarchical_equality_compare_summary.txt"

METHODS = ("gw", "ot", "fgw", "das")
FACTUAL_VALIDATION_SIZE = DEFAULT_FACTUAL_VALIDATION_SIZE
TRAIN_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_TRAIN_SIZE
CALIBRATION_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE
TEST_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_TEST_SIZE
TARGET_VARS = DEFAULT_TARGET_VARS

BATCH_SIZE = 128
RESOLUTION = 1
FGW_ALPHA = 0.5
OT_SITE_POLICY = "current"
OT_PCA_COMPONENTS = 8
OT_PCA_CANDIDATE_COUNT = 128
OT_TOP_K_VALUES = None
OT_LAMBDAS = tuple(np.linspace(0.25, 4.0, 16))

DAS_MAX_EPOCHS = 50
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 2
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 2, 4, 8, 16, 32, 64)
DAS_LAYERS = None


def main() -> None:
    problem = load_hierarchical_equality_problem(
        run_checks=True,
        target_vars=tuple(TARGET_VARS),
        canonical_variable_mapping=CANONICAL_VARIABLE_MAPPING,
    )
    adapter = build_hierarchical_equality_adapter(
        target_vars=tuple(TARGET_VARS),
        canonical_variable_mapping=CANONICAL_VARIABLE_MAPPING,
    )
    device = resolve_device(DEVICE)
    backbone_train_config = HierarchicalEqualityTrainConfig(
        seed=SEED,
        n_validation=FACTUAL_VALIDATION_SIZE,
        abstract_variables=tuple(TARGET_VARS),
    )
    compare_config = CompareExperimentConfig(
        seed=SEED,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
        summary_path=SUMMARY_PATH,
        requested_device=DEVICE,
        methods=tuple(METHODS),
        train_pair_size=TRAIN_PAIR_SIZE,
        calibration_pair_size=CALIBRATION_PAIR_SIZE,
        test_pair_size=TEST_PAIR_SIZE,
        target_vars=tuple(TARGET_VARS),
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        fgw_alpha=FGW_ALPHA,
        ot_top_k_values=OT_TOP_K_VALUES,
        ot_lambdas=tuple(OT_LAMBDAS),
        ot_site_policy=OT_SITE_POLICY,
        ot_pca_components=OT_PCA_COMPONENTS,
        ot_pca_candidate_count=OT_PCA_CANDIDATE_COUNT,
        das_max_epochs=DAS_MAX_EPOCHS,
        das_min_epochs=DAS_MIN_EPOCHS,
        das_plateau_patience=DAS_PLATEAU_PATIENCE,
        das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
        das_learning_rate=DAS_LEARNING_RATE,
        das_subspace_dims=DAS_SUBSPACE_DIMS,
        das_layers=DAS_LAYERS,
    )
    run_comparison_from_checkpoint(
        problem=problem,
        adapter=adapter,
        device=device,
        backbone_train_config=backbone_train_config,
        config=compare_config,
    )


if __name__ == "__main__":
    main()
