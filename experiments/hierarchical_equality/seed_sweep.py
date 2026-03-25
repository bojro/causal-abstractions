import os
from datetime import datetime
from pathlib import Path

import numpy as np

from experiment_core.compare_runner import CompareExperimentConfig, run_comparison_with_model
from experiment_core.reporting import write_text_report
from experiment_core.runtime import resolve_device, write_json
from experiment_core.seed_sweep import (
    build_seed_sweep_payload,
    format_seed_sweep_summary,
    save_seed_sweep_plots,
)

from .backbone import HierarchicalEqualityTrainConfig, load_backbone, train_backbone
from .constants import (
    CANONICAL_VARIABLE_MAPPING,
    DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE,
    DEFAULT_COUNTERFACTUAL_TEST_SIZE,
    DEFAULT_COUNTERFACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_TARGET_VARS,
    DEFAULT_TRAIN_EPOCHS,
)
from .scm import load_hierarchical_equality_problem
from .spec import build_hierarchical_equality_adapter


SEEDS = (41, 42, 43, 44, 45)
DEVICE = None
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / RUN_TIMESTAMP / "hierarchical_equality" / "seed_sweep"
CHECKPOINT_PATH_TEMPLATE = "models/hierarchical_equality_mlp_seed{seed}.pt"
OUTPUT_PATH = RUN_DIR / "hierarchical_equality_seed_sweep_results.json"
SUMMARY_PATH = RUN_DIR / "hierarchical_equality_seed_sweep_summary.txt"
RETRAIN_BACKBONES = False

FACTUAL_TRAIN_SIZE = DEFAULT_FACTUAL_TRAIN_SIZE
FACTUAL_VALIDATION_SIZE = DEFAULT_FACTUAL_VALIDATION_SIZE
HIDDEN_DIMS = DEFAULT_HIDDEN_DIMS
TARGET_VARS = DEFAULT_TARGET_VARS
LEARNING_RATE = 1e-3
EPOCHS = DEFAULT_TRAIN_EPOCHS
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256

METHODS = ("gw", "ot", "fgw", "das")
TRAIN_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_TRAIN_SIZE
CALIBRATION_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_CALIBRATION_SIZE
TEST_PAIR_SIZE = DEFAULT_COUNTERFACTUAL_TEST_SIZE

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


def build_train_config(seed: int) -> HierarchicalEqualityTrainConfig:
    return HierarchicalEqualityTrainConfig(
        seed=seed,
        n_train=FACTUAL_TRAIN_SIZE,
        n_validation=FACTUAL_VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )


def build_compare_config(seed: int, checkpoint_path: Path, run_dir: Path) -> CompareExperimentConfig:
    return CompareExperimentConfig(
        seed=seed,
        checkpoint_path=checkpoint_path,
        output_path=run_dir / "hierarchical_equality_compare_results.json",
        summary_path=run_dir / "hierarchical_equality_compare_summary.txt",
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


def load_or_train_backbone(problem, device, seed: int, checkpoint_path: Path):
    train_config = build_train_config(seed)
    if RETRAIN_BACKBONES or not checkpoint_path.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        return model, backbone_meta, "trained"

    model, _, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=checkpoint_path,
        device=device,
        train_config=train_config,
    )
    return model, backbone_meta, "loaded"


def print_loaded_backbone_validation(backbone_meta: dict[str, object]) -> None:
    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    exact_acc = float(factual_metrics.get("exact_acc", 0.0))
    num_examples = int(factual_metrics.get("num_examples", 0))
    print(
        "Loaded backbone factual validation "
        f"| exact_acc={exact_acc:.4f} "
        f"| num_examples={num_examples}"
    )


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
    seed_runs = []

    for index, seed in enumerate(SEEDS, start=1):
        checkpoint_path = Path(CHECKPOINT_PATH_TEMPLATE.format(seed=seed))
        seed_run_dir = RUN_DIR / f"seed_{seed}"
        print(f"[{index}/{len(SEEDS)}] seed={seed} | checkpoint={checkpoint_path}")
        model, backbone_meta, backbone_source = load_or_train_backbone(
            problem=problem,
            device=device,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        if backbone_source == "loaded":
            print_loaded_backbone_validation(backbone_meta)
        comparison = run_comparison_with_model(
            problem=problem,
            adapter=adapter,
            model=model,
            backbone_meta=backbone_meta,
            device=device,
            config=build_compare_config(seed, checkpoint_path, seed_run_dir),
        )
        seed_runs.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "backbone_source": backbone_source,
                "comparison": comparison,
            }
        )
        print()

    payload = build_seed_sweep_payload(seed_runs)
    payload["experiment_id"] = "hierarchical_equality"
    payload["requested_device"] = DEVICE
    payload["device"] = str(device)
    payload["resolved_device"] = str(device)
    payload["checkpoint_path_template"] = CHECKPOINT_PATH_TEMPLATE
    payload["retrain_backbones"] = RETRAIN_BACKBONES
    payload["training_config"] = {
        "train_size": FACTUAL_TRAIN_SIZE,
        "validation_size": FACTUAL_VALIDATION_SIZE,
        "hidden_dims": list(HIDDEN_DIMS),
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
    }
    payload["comparison_config"] = {
        "methods": list(METHODS),
        "train_pair_size": TRAIN_PAIR_SIZE,
        "calibration_pair_size": CALIBRATION_PAIR_SIZE,
        "test_pair_size": TEST_PAIR_SIZE,
        "batch_size": BATCH_SIZE,
        "resolution": RESOLUTION,
        "fgw_alpha": FGW_ALPHA,
        "ot_site_policy": OT_SITE_POLICY,
        "ot_pca_components": OT_PCA_COMPONENTS,
        "ot_pca_candidate_count": OT_PCA_CANDIDATE_COUNT,
        "ot_top_k_values": OT_TOP_K_VALUES,
        "ot_lambdas": list(OT_LAMBDAS),
        "das_max_epochs": DAS_MAX_EPOCHS,
        "das_min_epochs": DAS_MIN_EPOCHS,
        "das_plateau_patience": DAS_PLATEAU_PATIENCE,
        "das_plateau_rel_delta": DAS_PLATEAU_REL_DELTA,
        "das_learning_rate": DAS_LEARNING_RATE,
        "das_subspace_dims": None if DAS_SUBSPACE_DIMS is None else list(DAS_SUBSPACE_DIMS),
        "das_layers": DAS_LAYERS,
    }
    payload["summary_path"] = str(SUMMARY_PATH)
    payload["plots"] = save_seed_sweep_plots(payload, OUTPUT_PATH)
    write_json(OUTPUT_PATH, payload)
    write_text_report(SUMMARY_PATH, format_seed_sweep_summary(payload))

    print(f"Wrote seed sweep results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote seed sweep summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
