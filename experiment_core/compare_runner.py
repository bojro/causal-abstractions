"""Reusable comparison runner for single-seed and multi-seed experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .adapter import ExperimentAdapter
from .backbone import ClassifierTrainConfig, load_classifier_backbone
from .contracts import annotate_result_records, build_das_method_id
from .das import DASConfig, run_das_pipeline
from .defaults import (
    DEFAULT_ALIGNMENT_RESOLUTION,
    DEFAULT_PAIR_TEST_SIZE,
    DEFAULT_PAIR_TRAIN_SIZE,
)
from .metrics import validate_core_metric_records
from .ot import OTConfig, run_alignment_pipeline
from .plots import save_comparison_plots
from .reporting import (
    build_method_selection_summary,
    format_method_candidate_sweep,
    format_method_selection_summary,
    print_results_table,
    summarize_method_records,
    write_text_report,
)
from .runtime import collect_environment_metadata, set_seed, write_json


@dataclass(frozen=True)
class CompareExperimentConfig:
    """Config controlling one end-to-end comparison run for a fixed seed."""

    seed: int
    checkpoint_path: Path
    output_path: Path
    summary_path: Path
    requested_device: str | None = None
    methods: tuple[str, ...] = ("gw", "ot", "fgw", "das")
    train_pair_size: int = DEFAULT_PAIR_TRAIN_SIZE
    calibration_pair_size: int = DEFAULT_PAIR_TRAIN_SIZE
    test_pair_size: int = DEFAULT_PAIR_TEST_SIZE
    target_vars: tuple[str, ...] = ()
    batch_size: int = 128
    resolution: int = DEFAULT_ALIGNMENT_RESOLUTION
    fgw_alpha: float = 0.5
    ot_top_k_values: tuple[int, ...] | None = None
    ot_lambdas: tuple[float, ...] = (1.0,)
    ot_site_policy: str = "current"
    ot_pca_components: int | None = 8
    ot_pca_candidate_count: int | None = 128
    das_max_epochs: int = 1
    das_min_epochs: int = 1
    das_plateau_patience: int = 2
    das_plateau_rel_delta: float = 5e-3
    das_learning_rate: float = 1e-3
    das_subspace_dims: tuple[int, ...] | None = None
    das_layers: tuple[int, ...] | None = None


def _stable_text_seed(value: str) -> int:
    """Derive a deterministic integer offset from a short text token."""
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value)))


def build_seed_trace(config: CompareExperimentConfig) -> dict[str, object]:
    """Record all deterministic seeds derived from the run config."""
    compare_seed = int(config.seed + 101)
    method_execution_seeds = {
        str(method): int(config.seed + 1000 + _stable_text_seed(str(method)))
        for method in config.methods
    }
    return {
        "main_seed": int(config.seed),
        "compare_seed": compare_seed,
        "factual_validation_seed": int(config.seed + 2),
        "pair_bank_seeds": {
            "train": int(config.seed + 201),
            "calibration": int(config.seed + 301),
            "test": int(config.seed + 401),
        },
        "method_execution_seeds": method_execution_seeds,
        "das_seed_base": int(config.seed + 2000),
    }


def _build_summary_lines(
    experiment_id: str,
    config: CompareExperimentConfig,
    core_metrics: tuple[str, ...],
    device,
    backbone_meta: dict[str, object],
    method_payloads: dict[str, dict[str, object]],
    method_runtime_seconds: dict[str, float],
    summary_records: list[dict[str, object]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    method_selections = {
        method: build_method_selection_summary(method, method_payloads[method], core_metrics=core_metrics)
        for method in config.methods
    }
    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    summary_lines = [
        f"{experiment_id.capitalize()} Compare Summary",
        f"checkpoint: {config.checkpoint_path}",
        f"seed: {config.seed}",
        f"device: {device}",
        (
            f"requested_device: {config.requested_device}"
            if config.requested_device is not None
            else "requested_device: <auto>"
        ),
        f"target_vars: {', '.join(config.target_vars)}",
        (
            "pair_sizes: "
            f"train={config.train_pair_size}, "
            f"calibration={config.calibration_pair_size}, "
            f"test={config.test_pair_size}"
        ),
        f"factual_validation_exact_acc: {float(factual_metrics.get('exact_acc', 0.0)):.4f}",
        "",
    ]
    for method in config.methods:
        summary_lines.append(format_method_selection_summary(method_selections[method]))
        summary_lines.append("")
    summary_lines.append("Average Summary")
    for record in summary_records:
        metric_bits = ", ".join(
            f"{metric_name}={float(record[metric_name]):.4f}" for metric_name in core_metrics
        )
        summary_lines.append(
            f"{str(record['method']).upper()}: "
            f"{metric_bits}, "
            f"runtime_s={float(method_runtime_seconds.get(str(record['method']), 0.0)):.2f}"
        )
    candidate_sections = []
    for method in config.methods:
        candidate_section = format_method_candidate_sweep(
            method,
            method_payloads[method],
            core_metrics=core_metrics,
        )
        if candidate_section:
            candidate_sections.append(candidate_section)
    if candidate_sections:
        summary_lines.extend(["", "-" * 72, "", "Candidate Sweeps", "", "\n\n".join(candidate_sections)])
    return summary_lines, method_selections


def run_comparison_with_model(
    *,
    problem,
    adapter: ExperimentAdapter,
    model,
    backbone_meta: dict[str, object],
    device,
    config: CompareExperimentConfig,
) -> dict[str, object]:
    """Run the shared alignment evaluation starting from an in-memory model."""
    set_seed(config.seed + 101)
    core_metrics = tuple(adapter.experiment_spec.core_metrics)
    target_vars = tuple(config.target_vars or adapter.experiment_spec.local_target_vars)
    canonical_variable_mapping = {
        variable: adapter.experiment_spec.canonical_variable_mapping[variable]
        for variable in target_vars
    }
    train_bank = adapter.build_pair_bank(problem, config.train_pair_size, config.seed + 201, "train", False)
    calibration_bank = adapter.build_pair_bank(
        problem,
        config.calibration_pair_size,
        config.seed + 301,
        "calibration",
        False,
    )
    test_bank = adapter.build_pair_bank(problem, config.test_pair_size, config.seed + 401, "test", False)

    method_payloads: dict[str, dict[str, object]] = {}
    method_runtime_seconds: dict[str, float] = {}
    all_records: list[dict[str, object]] = []
    for method_index, method in enumerate(config.methods, start=1):
        print(f"[{method_index}/{len(config.methods)}] Starting {method.upper()}")
        method_seed = int(config.seed + 1000 + _stable_text_seed(str(method)))
        set_seed(method_seed)
        method_start_time = perf_counter()
        if method in {"gw", "ot", "fgw"}:
            payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device=device,
                config=OTConfig(
                    method=method,
                    batch_size=config.batch_size,
                    resolution=config.resolution,
                    alpha=config.fgw_alpha,
                    target_vars=target_vars,
                    top_k_values=config.ot_top_k_values,
                    lambda_values=config.ot_lambdas,
                    site_policy=config.ot_site_policy,
                    pca_components=config.ot_pca_components,
                    pca_candidate_count=config.ot_pca_candidate_count,
                ),
                metrics_from_logits_fn=adapter.metrics_from_logits,
                summarize_selection_records_fn=adapter.summarize_selection_records,
                choose_better_selection_candidate_fn=adapter.choose_better_selection_candidate,
            )
        elif method == "das":
            payload = run_das_pipeline(
                model=model,
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device=device,
                config=DASConfig(
                    batch_size=config.batch_size,
                    max_epochs=config.das_max_epochs,
                    min_epochs=config.das_min_epochs,
                    plateau_patience=config.das_plateau_patience,
                    plateau_rel_delta=config.das_plateau_rel_delta,
                    learning_rate=config.das_learning_rate,
                    subspace_dims=config.das_subspace_dims,
                    search_layers=config.das_layers,
                    target_vars=target_vars,
                    random_seed_base=int(config.seed + 2000),
                ),
                metrics_from_logits_fn=adapter.metrics_from_logits,
                summarize_selection_records_fn=adapter.summarize_selection_records,
                choose_better_selection_candidate_fn=adapter.choose_better_selection_candidate,
            )
        else:
            raise ValueError(f"Unsupported method {method}")
        method_id = str(payload.get("method_id", build_das_method_id()))
        payload["results"] = annotate_result_records(
            list(payload["results"]),
            method_id=method_id,
            canonical_variable_mapping=canonical_variable_mapping,
        )
        validate_core_metric_records(payload["results"], adapter.experiment_spec)
        payload["method_id"] = method_id
        payload["method_execution_seed"] = method_seed
        runtime_seconds = perf_counter() - method_start_time
        payload["runtime_seconds"] = float(runtime_seconds)
        method_payloads[method] = payload
        method_runtime_seconds[method] = float(runtime_seconds)
        all_records.extend(payload["results"])
        print(f"{method.upper()} runtime: {float(runtime_seconds):.2f}s")
        print()

    summary_records = summarize_method_records(all_records, core_metrics=core_metrics)
    validate_core_metric_records(summary_records, adapter.experiment_spec)
    for record in summary_records:
        record["method_id"] = str(method_payloads[str(record["method"])].get("method_id", record["method"]))
        record["runtime_seconds"] = float(method_runtime_seconds.get(str(record["method"]), 0.0))
    summary_lines, method_selections = _build_summary_lines(
        experiment_id=adapter.experiment_spec.experiment_id,
        config=CompareExperimentConfig(**{**config.__dict__, "target_vars": target_vars}),
        core_metrics=core_metrics,
        device=device,
        backbone_meta=backbone_meta,
        method_payloads=method_payloads,
        method_runtime_seconds=method_runtime_seconds,
        summary_records=summary_records,
    )
    payload = {
        "contract_version": 1,
        "seed": config.seed,
        "experiment_id": adapter.experiment_spec.experiment_id,
        "methods": list(config.methods),
        "method_ids": [str(method_payloads[method]["method_id"]) for method in config.methods],
        "checkpoint_path": str(config.checkpoint_path),
        "requested_device": config.requested_device,
        "resolved_device": str(device),
        "target_vars": list(target_vars),
        "canonical_target_vars": [canonical_variable_mapping[variable] for variable in target_vars],
        "canonical_variable_mapping": canonical_variable_mapping,
        "core_metrics": list(core_metrics),
        "environment": collect_environment_metadata(device, requested_device=config.requested_device),
        "seed_trace": build_seed_trace(config),
        "backbone": backbone_meta,
        "banks": {
            "train": train_bank.metadata(),
            "calibration": calibration_bank.metadata(),
            "test": test_bank.metadata(),
        },
        "method_payloads": method_payloads,
        "results": all_records,
        "method_summary": summary_records,
        "method_selections": method_selections,
        "method_runtime_seconds": method_runtime_seconds,
    }

    plot_paths = save_comparison_plots(payload, config.output_path, method_payloads=method_payloads)
    payload["plots"] = plot_paths
    payload["summary_path"] = str(config.summary_path)
    write_json(config.output_path, payload)
    write_text_report(config.summary_path, "\n".join(summary_lines))

    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    print(f"Backbone factual validation accuracy: {float(factual_metrics.get('exact_acc', 0.0)):.4f}")
    print_results_table(all_records, "Counterfactual Test Results", core_metrics=core_metrics)
    print_results_table(summary_records, "Method Average Summary", core_metrics=core_metrics)
    print(f"Wrote comparison results to {Path(config.output_path).resolve()}")
    print(f"Wrote comparison summary to {Path(config.summary_path).resolve()}")
    return payload


def run_comparison_from_checkpoint(
    *,
    problem,
    adapter: ExperimentAdapter,
    device,
    backbone_train_config: ClassifierTrainConfig,
    config: CompareExperimentConfig,
) -> dict[str, object]:
    """Load a compatible checkpoint and run the comparison pipeline."""
    model, _, backbone_meta = load_classifier_backbone(
        problem=problem,
        adapter=adapter,
        checkpoint_path=config.checkpoint_path,
        device=device,
        train_config=backbone_train_config,
    )
    return run_comparison_with_model(
        problem=problem,
        adapter=adapter,
        model=model,
        backbone_meta=backbone_meta,
        device=device,
        config=config,
    )
