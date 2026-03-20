"""Plotting helpers for comparison outputs written by experiment scripts."""

from __future__ import annotations

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


def _metric_plot_key(metric_name: str) -> str:
    """Build a stable plot dictionary key for one metric."""
    key_map = {
        "exact_acc": "exact_accuracy",
        "mean_shared_digits": "shared_digits",
    }
    return key_map.get(str(metric_name), str(metric_name))


def _metric_plot_filename(metric_name: str) -> str:
    """Build a stable filename for one per-variable metric plot."""
    filename_map = {
        "exact_acc": "exact_accuracy.png",
        "mean_shared_digits": "shared_digits.png",
    }
    return filename_map.get(str(metric_name), f"{str(metric_name)}.png")


def _metric_ylim(values: list[float]) -> tuple[float, float] | None:
    """Infer a simple y-limit range from plotted values."""
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return None
    if all(0.0 <= value <= 1.0 for value in finite_values):
        return (0.0, 1.0)
    max_value = max(finite_values)
    if max_value <= 0.0:
        return (0.0, 1.0)
    return (0.0, max_value * 1.1)


def _group_records(records: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    """Group result values by method and abstract variable."""
    grouped = {}
    for record in records:
        method = str(record["method"])
        variable = str(record["variable"])
        grouped.setdefault(method, {})
        grouped[method][variable] = float(record[key])
    return grouped


def save_comparison_plots(
    payload: dict[str, object],
    output_path: str | Path,
    method_payloads: dict[str, dict[str, object]] | None = None,
) -> dict[str, str]:
    """Render the standard comparison plots and return their output paths."""
    output_path = Path(output_path)
    plot_dir = output_path.parent
    ensure_parent_dir(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = list(payload.get("results", []))
    summary = list(payload.get("method_summary", []))
    core_metrics = _resolve_core_metrics(payload.get("core_metrics"))

    grouped_metrics = {
        metric_name: _group_records(records, metric_name) for metric_name in core_metrics
    }
    methods = sorted({method for metric_records in grouped_metrics.values() for method in metric_records})
    variables = [str(variable) for variable in payload.get("target_vars", [])]

    x = np.arange(len(variables))
    width = 0.8 / max(len(methods), 1)

    plot_paths = {"plot_dir": str(plot_dir)}
    for metric_name in core_metrics:
        metric_records = grouped_metrics.get(metric_name, {})
        metric_path = plot_dir / _metric_plot_filename(metric_name)
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        all_values = []
        for idx, method in enumerate(methods):
            y = [metric_records.get(method, {}).get(variable, np.nan) for variable in variables]
            all_values.extend([float(value) for value in y if np.isfinite(value)])
            ax.bar(x + (idx - (len(methods) - 1) / 2.0) * width, y, width=width, label=method.upper())
        ax.set_xticks(x, variables)
        y_limits = _metric_ylim(all_values)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        metric_label = _metric_display_name(metric_name)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-variable {metric_label.lower()}")
        ax.legend(loc="best")
        fig.savefig(metric_path, dpi=200)
        plt.close(fig)
        plot_paths[_metric_plot_key(metric_name)] = str(metric_path)

    summary_path = plot_dir / "average_summary.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    summary_methods = [str(record["method"]).upper() for record in summary]
    summary_x = np.arange(len(summary_methods))
    summary_width = 0.8 / max(len(core_metrics), 1)
    summary_offsets = (
        np.arange(len(core_metrics), dtype=float) - (len(core_metrics) - 1) / 2.0
    ) * summary_width
    for metric_index, metric_name in enumerate(core_metrics):
        metric_values = [float(record.get(metric_name, 0.0)) for record in summary]
        ax.bar(
            summary_x + summary_offsets[metric_index],
            metric_values,
            width=summary_width,
            label=_metric_display_name(metric_name),
        )
    ax.set_xticks(summary_x, summary_methods)
    ax.set_title("Average summary across abstract variables")
    ax.legend(loc="best")
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    plot_paths["average_summary"] = str(summary_path)

    transport_methods = []
    if method_payloads is not None:
        for method in ("gw", "ot", "fgw"):
            method_payload = method_payloads.get(method)
            if method_payload is not None and "transport" in method_payload:
                transport_methods.append((method, method_payload))
    if transport_methods:
        vmax = max(
            float(np.asarray(method_payload["transport"], dtype=float).max())
            for _, method_payload in transport_methods
        )
        transport_path = plot_dir / "transport_plans.png"
        fig, axes = plt.subplots(
            nrows=len(transport_methods),
            ncols=1,
            figsize=(10, 2.8 * len(transport_methods)),
            constrained_layout=True,
            squeeze=False,
        )
        for axis, (method, method_payload) in zip(axes.flat, transport_methods):
            transport = np.asarray(method_payload["transport"], dtype=float)
            image = axis.imshow(
                transport,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0.0,
                vmax=vmax if vmax > 0.0 else None,
            )
            axis.set_title(f"{method.upper()} transport plan")
            axis.set_ylabel("Abstract variable")
            axis.set_yticks(np.arange(len(variables)), variables)
            axis.set_xlabel("Neural site")
            fig.colorbar(image, ax=axis, shrink=0.9)
        fig.savefig(transport_path, dpi=200)
        plt.close(fig)
        plot_paths["transport_plans"] = str(transport_path)

    return plot_paths
