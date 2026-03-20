import os
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment_core.compare_runner import CompareExperimentConfig, run_comparison_from_checkpoint
from experiment_core.runtime import resolve_device

from .backbone import AdditionTrainConfig
from .scm import load_addition_problem
from .spec import build_addition_adapter


SEED = 44
DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / RUN_TIMESTAMP
CHECKPOINT_PATH = Path(f"models/addition_mlp_seed{SEED}.pt")
OUTPUT_PATH = RUN_DIR / "addition_compare_results.json"
SUMMARY_PATH = RUN_DIR / "addition_compare_summary.txt"

METHODS = ("gw", "ot", "fgw", "das")
FACTUAL_VALIDATION_SIZE = 4000
TRAIN_PAIR_SIZE = 1000
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 5000
TARGET_VARS = ("S1", "C1", "S2", "C2")
CANONICAL_VARIABLE_MAPPING = {
    "S1": "S1",
    "C1": "C1",
    "S2": "S2",
    "C2": "C2",
}

BATCH_SIZE = 128
RESOLUTION = 1
FGW_ALPHA = 0.5
OT_SITE_POLICY = "current"
OT_PCA_COMPONENTS = 8
OT_PCA_CANDIDATE_COUNT = 128
OT_TOP_K_VALUES = None
OT_LAMBDAS = tuple(np.linspace(0.25, 4.0, 16))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 10
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192)
DAS_LAYERS = None


def main() -> None:
    problem = load_addition_problem(
        run_checks=True,
        target_vars=tuple(TARGET_VARS),
        canonical_variable_mapping=CANONICAL_VARIABLE_MAPPING,
    )
    adapter = build_addition_adapter(
        target_vars=tuple(TARGET_VARS),
        canonical_variable_mapping=CANONICAL_VARIABLE_MAPPING,
    )
    device = resolve_device(DEVICE)
    backbone_train_config = AdditionTrainConfig(
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
