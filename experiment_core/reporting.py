"""Console reporting helpers for per-variable and average experiment summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .runtime import ensure_parent_dir


def _resolve_core_metrics(core_metrics: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Normalize the declared core metrics into a stable tuple."""
    if not core_metrics:
        return ("exact_acc", "mean_shared_digits")
    return tuple(str(metric_name) for metric_name in core_metrics)


def _metric_short_label(metric_name: str) -> str:
    """Build a compact human-readable label for one metric name."""
    label_map = {
        "exact_acc": "exact",
        "mean_shared_digits": "shared",
    }
    return label_map.get(str(metric_name), str(metric_name).replace("_", " "))


def _metric_field(record: dict[str, object], metric_name: str, default: float = 0.0) -> float:
    """Extract one metric from a record and coerce it to float."""
    return float(record.get(str(metric_name), default))


def _prefixed_metric_field(record: dict[str, object], prefix: str, metric_name: str, default: float = 0.0) -> float:
    """Extract one prefixed metric from a record and coerce it to float."""
    return float(record.get(f"{prefix}{metric_name}", default))


def _format_metric_bits(metrics: dict[str, float]) -> str:
    """Format a small metric dictionary into `key=value` fragments."""
    return ", ".join(f"{_metric_short_label(name)}={float(value):.4f}" for name, value in metrics.items())


def _format_site_config(record: dict[str, object]) -> str:
    site_label = str(record.get("site_label", "n/a"))
    if str(record.get("method", "")).lower() != "das":
        return site_label
    epochs_ran = record.get("train_epochs_ran")
    if epochs_ran is None:
        return site_label
    return f"{site_label},e{int(epochs_ran)}"


def print_results_table(
    records: list[dict[str, object]],
    title: str,
    core_metrics: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Print a compact table of per-variable experiment results."""
    print(title)
    if not records:
        print("(no records)")
        return

    metric_names = _resolve_core_metrics(core_metrics)
    primary_metric = metric_names[0]
    secondary_metric = metric_names[1] if len(metric_names) > 1 else None
    primary_label = _metric_short_label(primary_metric)[:8]
    secondary_label = _metric_short_label(secondary_metric)[:8] if secondary_metric is not None else "metric2"
    header = f"{'method':<8} {'variable':<8} {primary_label:>8} {secondary_label:>8} {'select/cal':>10} {'site/config':<24}"
    print(header)
    print("-" * len(header))
    for record in records:
        selection_metric = _prefixed_metric_field(record, "selection_", primary_metric, 0.0)
        secondary_value = _metric_field(record, secondary_metric, 0.0) if secondary_metric is not None else 0.0
        print(
            f"{str(record['method']):<8} "
            f"{str(record.get('variable', 'average')):<8} "
            f"{_metric_field(record, primary_metric):>8.4f} "
            f"{secondary_value:>8.4f} "
            f"{selection_metric:>10.4f} "
            f"{_format_site_config(record):<24}"
        )


def summarize_method_records(
    records: list[dict[str, object]],
    core_metrics: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, object]]:
    """Average per-variable metrics into one average summary per method."""
    metric_names = _resolve_core_metrics(core_metrics)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    summaries = []
    for method, method_records in sorted(grouped.items()):
        summary_record: dict[str, object] = {"method": method}
        for metric_name in metric_names:
            summary_record[str(metric_name)] = sum(
                _metric_field(record, str(metric_name)) for record in method_records
            ) / len(method_records)
        summaries.append(summary_record)
    return summaries


def build_method_selection_summary(
    method: str,
    payload: dict[str, object],
    core_metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, object]:
    """Build a compact per-method summary for saved artifacts."""
    metric_names = _resolve_core_metrics(core_metrics)
    if method in {"gw", "ot", "fgw"}:
        return {
            "method": method,
            "core_metrics": list(metric_names),
            "transport_meta": dict(payload.get("transport_meta", {})),
            "selected_hyperparameters": dict(payload.get("selected_hyperparameters", {})),
            "results": [
                {
                    "variable": record["variable"],
                    "site_label": record["site_label"],
                    "top_site_label": record.get("top_site_label"),
                    "top_k": record.get("top_k"),
                    "lambda": record.get("lambda"),
                    "selection_metrics": {
                        metric_name: _prefixed_metric_field(record, "selection_", metric_name, 0.0)
                        for metric_name in metric_names
                    },
                    "calibration_metrics": {
                        metric_name: _prefixed_metric_field(record, "calibration_", metric_name, 0.0)
                        for metric_name in metric_names
                    },
                    "test_metrics": {
                        metric_name: _metric_field(record, metric_name, 0.0) for metric_name in metric_names
                    },
                }
                for record in payload.get("results", [])
            ],
        }

    if method == "das":
        return {
            "method": method,
            "core_metrics": list(metric_names),
            "training_stopping_rule": dict(payload.get("training_stopping_rule", {})),
            "results": [
                {
                    "variable": record["variable"],
                    "site_label": record["site_label"],
                    "layer": record.get("layer"),
                    "subspace_dim": record.get("subspace_dim"),
                    "train_epochs_ran": record.get("train_epochs_ran"),
                    "selection_metrics": {
                        metric_name: _prefixed_metric_field(record, "selection_", metric_name, 0.0)
                        for metric_name in metric_names
                    },
                    "calibration_metrics": {
                        metric_name: _prefixed_metric_field(record, "calibration_", metric_name, 0.0)
                        for metric_name in metric_names
                    },
                    "test_metrics": {
                        metric_name: _metric_field(record, metric_name, 0.0) for metric_name in metric_names
                    },
                }
                for record in payload.get("results", [])
            ],
        }

    raise ValueError(f"Unsupported method summary type: {method}")


def format_method_selection_summary(summary: dict[str, object]) -> str:
    """Format one compact method summary as human-readable text."""
    method = str(summary["method"]).upper()
    metric_names = _resolve_core_metrics(summary.get("core_metrics"))
    lines = [method]

    if method in {"GW", "OT", "FGW"}:
        transport_meta = dict(summary.get("transport_meta", {}))
        selected_hyperparameters = dict(summary.get("selected_hyperparameters", {}))
        if transport_meta:
            solver_bits = ", ".join(f"{key}={value}" for key, value in sorted(transport_meta.items()))
            lines.append(f"solver: {solver_bits}")
        lines.append("selected soft matches:")
        for record in summary.get("results", []):
            variable = str(record["variable"])
            top_k = record.get("top_k")
            lambda_value = record.get("lambda")
            if isinstance(selected_hyperparameters.get("top_k_by_variable"), dict):
                top_k = selected_hyperparameters["top_k_by_variable"].get(variable, top_k)
            if isinstance(selected_hyperparameters.get("lambda_by_variable"), dict):
                lambda_value = selected_hyperparameters["lambda_by_variable"].get(variable, lambda_value)
            calibration_metrics = {
                metric_name: float(dict(record.get("calibration_metrics", {})).get(metric_name, 0.0))
                for metric_name in metric_names
            }
            selection_metrics = {
                metric_name: float(dict(record.get("selection_metrics", {})).get(metric_name, 0.0))
                for metric_name in metric_names
            }
            test_metrics = {
                metric_name: float(dict(record.get("test_metrics", {})).get(metric_name, 0.0))
                for metric_name in metric_names
            }
            lines.append(
                f"{variable}: top_k={top_k}, lambda={lambda_value}, top_site={record.get('top_site_label')}, "
                f"selection=({_format_metric_bits(selection_metrics)}), "
                f"calibration=({_format_metric_bits(calibration_metrics)}), "
                f"test=({_format_metric_bits(test_metrics)})"
            )
        return "\n".join(lines)

    training_stopping_rule = dict(summary.get("training_stopping_rule", {}))
    if training_stopping_rule:
        display_rule = {key: value for key, value in training_stopping_rule.items() if key != "max_epochs"}
        stopping_bits = ", ".join(f"{key}={value}" for key, value in sorted(display_rule.items()))
        lines.append(f"training: {stopping_bits}")
    lines.append("selected sites:")
    for record in summary.get("results", []):
        calibration_metrics = {
            metric_name: float(dict(record.get("calibration_metrics", {})).get(metric_name, 0.0))
            for metric_name in metric_names
        }
        selection_metrics = {
            metric_name: float(dict(record.get("selection_metrics", {})).get(metric_name, 0.0))
            for metric_name in metric_names
        }
        test_metrics = {
            metric_name: float(dict(record.get("test_metrics", {})).get(metric_name, 0.0))
            for metric_name in metric_names
        }
        lines.append(
            f"{record['variable']}: site={record['site_label']}, layer={record.get('layer')}, "
            f"subspace_dim={record.get('subspace_dim')}, epochs_ran={record.get('train_epochs_ran')}, "
            f"selection=({_format_metric_bits(selection_metrics)}), "
            f"calibration=({_format_metric_bits(calibration_metrics)}), "
            f"test=({_format_metric_bits(test_metrics)})"
        )
    return "\n".join(lines)


def format_method_candidate_sweep(
    method: str,
    payload: dict[str, object],
    core_metrics: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Format only the candidate-sweep section for one method."""
    metric_names = _resolve_core_metrics(core_metrics)
    lines = [str(method).upper()]
    if method in {"gw", "ot", "fgw"}:
        calibration_sweep = dict(payload.get("calibration_sweep", {}))
        if calibration_sweep:
            lines.append("calibration sweep:")
            summary = build_method_selection_summary(method, payload, core_metrics=metric_names)
            selected_hyperparameters = dict(summary.get("selected_hyperparameters", {}))
            selected_top_k_by_variable = dict(selected_hyperparameters.get("top_k_by_variable", {}))
            selected_lambda_by_variable = dict(selected_hyperparameters.get("lambda_by_variable", {}))
            for variable in payload.get("target_vars", []):
                variable_key = str(variable)
                lines.append(f"{variable_key}:")
                candidates = list(calibration_sweep.get(variable_key, []))
                selected_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.get("top_k", -1)) == int(selected_top_k_by_variable.get(variable_key, -1))
                    and float(candidate.get("lambda", float("nan")))
                    == float(selected_lambda_by_variable.get(variable_key, float("nan")))
                ]
                remaining_candidates = [candidate for candidate in candidates if candidate not in selected_candidates]
                for candidate in [*selected_candidates, *remaining_candidates]:
                    result = dict(candidate.get("result", {}))
                    top_site_label = result.get("top_site_label", "n/a")
                    selected_marker = " [selected]" if candidate in selected_candidates else ""
                    metric_bits = ", ".join(
                        f"{_metric_short_label(metric_name)}={float(candidate.get(metric_name, 0.0)):.4f}"
                        for metric_name in metric_names
                    )
                    lines.append(
                        f"{variable_key}{selected_marker}: top_k={int(candidate['top_k'])}, "
                        f"lambda={float(candidate['lambda']):g}, top_site={top_site_label}, {metric_bits}"
                    )
        return "\n".join(lines) if len(lines) > 1 else ""

    search_records = dict(payload.get("search_records", {}))
    if search_records:
        lines.append("candidate sweep:")
        selected_records = {str(record["variable"]): record for record in payload.get("results", [])}
        for variable in payload.get("target_vars", []):
            variable_key = str(variable)
            lines.append(f"{variable_key}:")
            selected_record = selected_records.get(variable_key, {})
            candidates = list(search_records.get(variable_key, []))

            def is_selected(candidate: dict[str, object]) -> bool:
                return (
                    str(candidate.get("site_label")) == str(selected_record.get("site_label"))
                    and int(candidate.get("layer", -1)) == int(selected_record.get("layer", -1))
                    and int(candidate.get("subspace_dim", -1)) == int(selected_record.get("subspace_dim", -1))
                )

            selected_candidates = [candidate for candidate in candidates if is_selected(candidate)]
            remaining_candidates = [candidate for candidate in candidates if not is_selected(candidate)]
            for candidate in [*selected_candidates, *remaining_candidates]:
                loss_history = list(candidate.get("train_loss_history", []))
                final_train_loss = float(loss_history[-1]) if loss_history else 0.0
                selected_marker = " [selected]" if candidate in selected_candidates else ""
                calibration_bits = ", ".join(
                    f"{_metric_short_label(metric_name)}={_prefixed_metric_field(candidate, 'calibration_', metric_name, 0.0):.4f}"
                    for metric_name in metric_names
                )
                lines.append(
                    f"{variable_key}{selected_marker}: site={candidate['site_label']}, "
                    f"layer={candidate.get('layer')}, subspace_dim={candidate.get('subspace_dim')}, "
                    f"epochs_ran={candidate.get('train_epochs_ran')}, "
                    f"train_loss={final_train_loss:.4f}, {calibration_bits}"
                )
    return "\n".join(lines) if len(lines) > 1 else ""


def write_text_report(path: str | Path, text: str) -> None:
    """Write a plain-text report to disk."""
    ensure_parent_dir(path)
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")
