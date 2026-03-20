"""Aggregation and plotting helpers for multi-seed experiment sweeps."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import _env  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .runtime import ensure_parent_dir


def _resolve_core_metrics(core_metrics: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the payload's declared core metrics into a tuple."""
    if not core_metrics:
        return ("exact_acc", "mean_shared_digits")
    return tuple(str(metric_name) for metric_name in core_metrics)


def _metric_display_name(metric_name: str) -> str:
    """Build a human-readable metric label."""
    label_map = {
        "exact_acc": "Exact Accuracy",
        "mean_shared_digits": "Mean Shared Digits",
    }
    return label_map.get(str(metric_name), str(metric_name).replace("_", " ").title())


def _metric_short_label(metric_name: str) -> str:
    """Build a compact human-readable metric label."""
    label_map = {
        "exact_acc": "exact",
        "mean_shared_digits": "shared",
    }
    return label_map.get(str(metric_name), str(metric_name).replace("_", " "))


def _metric_summary_fields(values: list[float], metric_name: str) -> dict[str, float]:
    """Build the standard mean/std field pair for one metric."""
    metric_mean, metric_std = _mean_std(values)
    return {
        f"{metric_name}_mean": metric_mean,
        f"{metric_name}_std": metric_std,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    values_np = np.asarray(values, dtype=float)
    return float(values_np.mean()), float(values_np.std(ddof=0))


def build_seed_sweep_payload(seed_runs: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-seed comparison payloads into cross-seed summaries."""
    method_average_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    variable_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    backbone_factual = []
    per_seed_method_summary = []
    per_seed_variable_results = []
    per_seed_method_runtime = []
    methods_seen = set()
    target_vars = []
    seeds = []
    core_metrics: tuple[str, ...] = ("exact_acc", "mean_shared_digits")

    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        seeds.append(seed)
        comparison = dict(seed_run["comparison"])
        core_metrics = _resolve_core_metrics(comparison.get("core_metrics", core_metrics))
        target_vars = [str(variable) for variable in comparison.get("target_vars", target_vars)]
        method_runtime_seconds = {
            str(method): float(seconds)
            for method, seconds in dict(comparison.get("method_runtime_seconds", {})).items()
        }

        backbone = dict(comparison.get("backbone", {}))
        factual_metrics = dict(backbone.get("factual_validation_metrics", {}))
        backbone_factual.append(
            {
                "seed": seed,
                "exact_acc": float(factual_metrics.get("exact_acc", 0.0)),
                "num_examples": int(factual_metrics.get("num_examples", 0)),
            }
        )

        for record in comparison.get("method_summary", []):
            method = str(record.get("method_id", record["method"]))
            method_family = str(record["method"])
            methods_seen.add(method)
            average_record = {
                "seed": seed,
                "method": method,
                "method_family": method_family,
                "runtime_seconds": float(record.get("runtime_seconds", method_runtime_seconds.get(method_family, 0.0))),
            }
            for metric_name in core_metrics:
                average_record[metric_name] = float(record[metric_name])
            per_seed_method_summary.append(average_record)
            method_average_grouped[method].append(average_record)
            per_seed_method_runtime.append(
                {
                    "seed": seed,
                    "method": method,
                    "runtime_seconds": float(average_record["runtime_seconds"]),
                }
            )

        for record in comparison.get("results", []):
            method = str(record.get("method_id", record["method"]))
            variable = str(record["variable"])
            result_record = {
                "seed": seed,
                "method": method,
                "method_family": str(record["method"]),
                "variable": variable,
            }
            for metric_name in core_metrics:
                result_record[metric_name] = float(record[metric_name])
            per_seed_variable_results.append(result_record)
            variable_grouped[(method, variable)].append(result_record)

    seeds = sorted(set(seeds))
    methods = sorted(methods_seen)

    method_summary = []
    for method in methods:
        method_records = method_average_grouped[method]
        runtime_mean, runtime_std = _mean_std([record["runtime_seconds"] for record in method_records])
        summary_record: dict[str, object] = {
            "method": method,
            "num_seeds": len(method_records),
            "runtime_seconds_mean": runtime_mean,
            "runtime_seconds_std": runtime_std,
        }
        for metric_name in core_metrics:
            summary_record.update(
                _metric_summary_fields(
                    [float(record[metric_name]) for record in method_records],
                    metric_name,
                )
            )
        method_summary.append(summary_record)

    variable_summary = []
    for method in methods:
        for variable in target_vars:
            records = variable_grouped.get((method, variable), [])
            summary_record = {
                "method": method,
                "variable": variable,
                "num_seeds": len(records),
            }
            for metric_name in core_metrics:
                summary_record.update(
                    _metric_summary_fields(
                        [float(record[metric_name]) for record in records],
                        metric_name,
                    )
                )
            variable_summary.append(summary_record)

    backbone_exact_mean, backbone_exact_std = _mean_std([record["exact_acc"] for record in backbone_factual])
    return {
        "seeds": seeds,
        "methods": methods,
        "target_vars": target_vars,
        "core_metrics": list(core_metrics),
        "seed_runs": seed_runs,
        "backbone_factual_validation": backbone_factual,
        "backbone_factual_validation_summary": {
            "num_seeds": len(backbone_factual),
            "exact_acc_mean": backbone_exact_mean,
            "exact_acc_std": backbone_exact_std,
        },
        "per_seed_method_summary": per_seed_method_summary,
        "per_seed_method_runtime": per_seed_method_runtime,
        "per_seed_variable_results": per_seed_variable_results,
        "method_summary_across_seeds": method_summary,
        "variable_summary_across_seeds": variable_summary,
    }


def _plot_mean_std_bars(
    records: list[dict[str, object]],
    output_path: Path,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
) -> str:
    methods = [str(record["method"]).upper() for record in records]
    means = [float(record[mean_key]) for record in records]
    stds = [float(record[std_key]) for record in records]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(x, means, yerr=stds, capsize=6)
    ax.set_xticks(x, methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _plot_grouped_mean_std_bars(
    records: list[dict[str, object]],
    output_path: Path,
    group_key: str,
    series_key: str,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    group_order: list[str] | None = None,
    series_order: list[str] | None = None,
) -> str:
    if group_order is None:
        group_values = sorted({str(record[group_key]) for record in records})
    else:
        group_values = [str(value) for value in group_order]
    if series_order is None:
        series_values = sorted({str(record[series_key]) for record in records})
    else:
        series_values = [str(value) for value in series_order]

    record_map = {(str(record[group_key]), str(record[series_key])): record for record in records}
    x = np.arange(len(group_values), dtype=float)
    width = 0.8 / max(len(series_values), 1)
    offsets = (np.arange(len(series_values), dtype=float) - (len(series_values) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for index, series in enumerate(series_values):
        means = []
        stds = []
        for group in group_values:
            record = record_map.get((group, series))
            means.append(float(record[mean_key]) if record is not None else 0.0)
            stds.append(float(record[std_key]) if record is not None else 0.0)
        ax.bar(x + offsets[index], means, width=width, yerr=stds, capsize=5, label=str(series).upper())
    ax.set_xticks(x, group_values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def save_seed_sweep_plots(payload: dict[str, object], output_path: str | Path) -> dict[str, str]:
    """Write aggregate multi-seed plots next to the provided output path."""
    output_path = Path(output_path)
    plot_dir = output_path.parent
    ensure_parent_dir(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {"plot_dir": str(plot_dir)}
    core_metrics = _resolve_core_metrics(payload.get("core_metrics"))
    for metric_name in core_metrics:
        metric_slug = str(metric_name)
        average_plot_key = "average_exact_summary" if metric_slug == "exact_acc" else (
            "average_shared_summary" if metric_slug == "mean_shared_digits" else f"average_{metric_slug}_summary"
        )
        variable_plot_key = "variable_exact_summary" if metric_slug == "exact_acc" else (
            "variable_shared_summary" if metric_slug == "mean_shared_digits" else f"variable_{metric_slug}_summary"
        )
        average_filename = "average_exact_summary.png" if metric_slug == "exact_acc" else (
            "average_shared_summary.png" if metric_slug == "mean_shared_digits" else f"average_{metric_slug}_summary.png"
        )
        variable_filename = "variable_exact_summary.png" if metric_slug == "exact_acc" else (
            "variable_shared_summary.png" if metric_slug == "mean_shared_digits" else f"variable_{metric_slug}_summary.png"
        )
        plot_paths[average_plot_key] = _plot_mean_std_bars(
            records=list(payload.get("method_summary_across_seeds", [])),
            output_path=plot_dir / average_filename,
            mean_key=f"{metric_slug}_mean",
            std_key=f"{metric_slug}_std",
            ylabel=f"Average {_metric_display_name(metric_slug)}",
            title=f"Average {_metric_display_name(metric_slug).lower()} mean +/- std across seeds",
        )
        plot_paths[variable_plot_key] = _plot_grouped_mean_std_bars(
            records=list(payload.get("variable_summary_across_seeds", [])),
            output_path=plot_dir / variable_filename,
            group_key="variable",
            series_key="method",
            mean_key=f"{metric_slug}_mean",
            std_key=f"{metric_slug}_std",
            ylabel=f"Per-Variable {_metric_display_name(metric_slug)}",
            title=f"Per-variable {_metric_display_name(metric_slug).lower()} mean +/- std across seeds",
            group_order=[str(value) for value in payload.get("target_vars", [])],
            series_order=[str(value) for value in payload.get("methods", [])],
        )
    plot_paths["runtime_summary"] = _plot_mean_std_bars(
        records=list(payload.get("method_summary_across_seeds", [])),
        output_path=plot_dir / "runtime_summary.png",
        mean_key="runtime_seconds_mean",
        std_key="runtime_seconds_std",
        ylabel="Runtime (s)",
        title="Method runtime mean +/- std across seeds",
    )
    return plot_paths


def format_seed_sweep_summary(payload: dict[str, object]) -> str:
    """Format a compact text summary of the aggregated multi-seed results."""
    core_metrics = _resolve_core_metrics(payload.get("core_metrics"))
    lines = [
        "Experiment Seed Sweep Summary",
        f"seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
        "",
        "Backbone Factual Validation",
    ]
    backbone_summary = dict(payload.get("backbone_factual_validation_summary", {}))
    lines.append(
        "mean_exact="
        f"{float(backbone_summary.get('exact_acc_mean', 0.0)):.4f}, "
        "std_exact="
        f"{float(backbone_summary.get('exact_acc_std', 0.0)):.4f}"
    )
    lines.append("")
    lines.append("Method Average Across Seeds")
    for record in payload.get("method_summary_across_seeds", []):
        metric_bits = ", ".join(
            f"{_metric_short_label(metric_name)}={float(record[f'{metric_name}_mean']):.4f} +/- "
            f"{float(record[f'{metric_name}_std']):.4f}"
            for metric_name in core_metrics
        )
        lines.append(
            f"{str(record['method']).upper()}: "
            f"{metric_bits}, "
            f"runtime_s={float(record['runtime_seconds_mean']):.2f} +/- "
            f"{float(record['runtime_seconds_std']):.2f}"
        )
    variable_summary = list(payload.get("variable_summary_across_seeds", []))
    if variable_summary:
        lines.append("")
        lines.append("Per-Variable Summary Across Seeds")
        for record in variable_summary:
            metric_bits = ", ".join(
                f"{_metric_short_label(metric_name)}={float(record[f'{metric_name}_mean']):.4f} +/- "
                f"{float(record[f'{metric_name}_std']):.4f}"
                for metric_name in core_metrics
            )
            lines.append(
                f"{str(record['method']).upper()} [{str(record['variable'])}]: "
                f"{metric_bits}"
            )
    return "\n".join(lines)
