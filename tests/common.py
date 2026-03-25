from __future__ import annotations

import tempfile
from pathlib import Path

from experiment_core.compare_runner import CompareExperimentConfig
from experiments.addition.backbone import AdditionTrainConfig, train_backbone
from experiments.addition.scm import load_addition_problem
from experiments.hierarchical_equality.backbone import (
    HierarchicalEqualityTrainConfig,
    train_backbone as train_hierarchical_equality_backbone,
)
from experiments.hierarchical_equality.scm import load_hierarchical_equality_problem


def small_train_config(seed: int = 42) -> AdditionTrainConfig:
    return AdditionTrainConfig(
        seed=seed,
        n_train=64,
        n_validation=32,
        hidden_dims=(16, 16),
        train_epochs=1,
        train_batch_size=16,
        eval_batch_size=16,
    )


def load_small_model(checkpoint_path: str, seed: int = 42):
    problem = load_addition_problem(run_checks=False)
    model, _, meta = train_backbone(
        problem=problem,
        train_config=small_train_config(seed=seed),
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    return problem, model, meta


def small_hierarchical_equality_train_config(seed: int = 42) -> HierarchicalEqualityTrainConfig:
    return HierarchicalEqualityTrainConfig(
        seed=seed,
        n_train=64,
        n_validation=32,
        hidden_dims=(16, 16),
        train_epochs=1,
        train_batch_size=16,
        eval_batch_size=16,
    )


def load_small_hierarchical_equality_model(checkpoint_path: str, seed: int = 42):
    problem = load_hierarchical_equality_problem(run_checks=False)
    model, _, meta = train_hierarchical_equality_backbone(
        problem=problem,
        train_config=small_hierarchical_equality_train_config(seed=seed),
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    return problem, model, meta


def small_compare_config(
    run_dir: Path,
    checkpoint_path: Path,
    *,
    seed: int = 42,
    methods: tuple[str, ...] = ("gw",),
    target_vars: tuple[str, ...] = ("S1", "C1"),
    ot_site_policy: str = "current",
    ot_pca_components: int | None = 2,
    ot_pca_candidate_count: int | None = 4,
) -> CompareExperimentConfig:
    return CompareExperimentConfig(
        seed=seed,
        checkpoint_path=checkpoint_path,
        output_path=run_dir / "compare.json",
        summary_path=run_dir / "compare.txt",
        methods=methods,
        train_pair_size=8,
        calibration_pair_size=8,
        test_pair_size=8,
        target_vars=target_vars,
        batch_size=4,
        resolution=2,
        fgw_alpha=0.5,
        ot_top_k_values=(1, 2),
        ot_lambdas=(0.5,),
        ot_site_policy=ot_site_policy,
        ot_pca_components=ot_pca_components,
        ot_pca_candidate_count=ot_pca_candidate_count,
        das_max_epochs=1,
        das_min_epochs=1,
        das_plateau_patience=1,
        das_plateau_rel_delta=1e-2,
        das_learning_rate=1e-3,
        das_subspace_dims=(1,),
        das_layers=(0,),
    )


def temporary_checkpoint(seed: int = 42) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temp_dir = tempfile.TemporaryDirectory()
    checkpoint_path = Path(temp_dir.name) / f"mini_model_seed{seed}.pt"
    return temp_dir, checkpoint_path
