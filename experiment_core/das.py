"""DAS search and evaluation on pair-bank counterfactual interventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pyvene import RotatedSpaceIntervention
from torch.utils.data import DataLoader

from . import _env  # noqa: F401
from .contracts import build_das_method_id
from .pair_bank import PairBank, PairBankVariableDataset
from .pyvene_utils import DASSearchSpec, build_intervenable, run_intervenable_logits
from variable_width_mlp import VariableWidthMLPForClassification, logits_from_output


@dataclass(frozen=True)
class DASConfig:
    """Hyperparameters controlling DAS training, selection, and evaluation."""

    batch_size: int = 128
    max_epochs: int = 1
    learning_rate: float = 1e-3
    subspace_dims: tuple[int, ...] | None = None
    search_layers: tuple[int, ...] | None = None
    target_vars: tuple[str, ...] = ()
    plateau_patience: int = 2
    plateau_rel_delta: float = 5e-3
    min_epochs: int = 1
    verbose: bool = True
    progress_interval: int = 25
    random_seed_base: int = 0


def _stable_text_seed(value: str) -> int:
    """Derive a deterministic integer offset from a short text token."""
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value)))


def iter_search_specs(
    model: VariableWidthMLPForClassification,
    config: DASConfig,
) -> list[DASSearchSpec]:
    """Enumerate DAS layer and subspace-size candidates to test."""
    layers = list(range(model.config.n_layer)) if config.search_layers is None else [int(layer) for layer in config.search_layers]
    specs = []
    for layer in layers:
        width = int(model.config.hidden_dims[layer])
        subspace_dims = config.subspace_dims
        if subspace_dims is None:
            subspace_dims = tuple(range(1, width + 1))
        for subspace_dim in subspace_dims:
            if int(subspace_dim) <= width:
                specs.append(DASSearchSpec(layer=layer, subspace_dim=int(subspace_dim), component=f"h[{layer}].output"))
    return specs


def evaluate_rotated_intervention(
    intervenable,
    dataset: PairBankVariableDataset,
    spec: DASSearchSpec,
    batch_size: int,
    device: torch.device,
    *,
    metrics_from_logits_fn,
) -> dict[str, float]:
    """Evaluate one trained DAS intervention on a dataset split."""
    logits_all = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in loader:
        logits = run_intervenable_logits(
            intervenable=intervenable,
            base_inputs=batch["input_ids"],
            source_inputs=batch["source_input_ids"],
            subspace_dims=spec.subspace_dims,
            position=spec.position,
            batch_size=batch_size,
            device=device,
        )
        logits_all.append(logits)
        labels_all.append(batch["labels"].to(torch.long).view(-1))
    return metrics_from_logits_fn(torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0))


def train_rotated_intervention(
    intervenable,
    dataset: PairBankVariableDataset,
    spec: DASSearchSpec,
    max_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    plateau_patience: int,
    plateau_rel_delta: float,
    min_epochs: int,
    shuffle_seed: int,
) -> list[float]:
    """Train the DAS rotation parameters for one candidate intervention."""
    optimizer_parameters = []
    for intervention in intervenable.interventions.values():
        if hasattr(intervention, "rotate_layer"):
            optimizer_parameters.append({"params": intervention.rotate_layer.parameters()})
    if not optimizer_parameters:
        raise RuntimeError("No rotate_layer parameters found for DAS intervention")
    optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    losses = []
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(shuffle_seed))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    best_loss = None
    plateau_steps = 0
    for epoch_index in range(max_epochs):
        epoch_losses = []
        for batch in loader:
            device_base = batch["input_ids"].to(device)
            device_source = batch["source_input_ids"].to(device)
            labels = batch["labels"].to(device).view(-1)
            batch_size_now = device_base.shape[0]
            positions = [[spec.position]] * batch_size_now
            subspaces = [spec.subspace_dims] * batch_size_now
            base_batch = device_base.unsqueeze(1) if device_base.ndim == 2 else device_base
            source_batch = device_source.unsqueeze(1) if device_source.ndim == 2 else device_source[:, :1, :]
            _, cf_output = intervenable(
                {"inputs_embeds": base_batch},
                [{"inputs_embeds": source_batch}],
                {"sources->base": ([positions], [positions])},
                subspaces=[subspaces],
            )
            logits = logits_from_output(cf_output)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(epoch_loss)
        if best_loss is None:
            improved = True
        else:
            relative_threshold = float(best_loss) * (1.0 - float(plateau_rel_delta))
            improved = epoch_loss < relative_threshold
        if improved:
            best_loss = epoch_loss
            plateau_steps = 0
        else:
            plateau_steps += 1
        if plateau_patience > 0 and epoch_index + 1 >= int(min_epochs) and plateau_steps >= int(plateau_patience):
            break
    return losses


def _format_selection_metrics(selection_metrics: dict[str, object]) -> str:
    """Format selection metrics as `key=value` fragments for logs."""
    if not selection_metrics:
        return "n/a"
    return ", ".join(
        f"{str(metric_name)}={float(metric_value):.4f}"
        for metric_name, metric_value in selection_metrics.items()
    )


def _metric_subset(record: dict[str, object], metric_names: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Extract only the named metrics from a search or result record."""
    return {
        str(metric_name): float(record.get(str(metric_name), 0.0))
        for metric_name in metric_names
    }


def run_das_search_for_variable(
    model: VariableWidthMLPForClassification,
    variable: str,
    train_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device,
    config: DASConfig,
    *,
    metrics_from_logits_fn,
    summarize_selection_records_fn,
    choose_better_selection_candidate_fn,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Search DAS candidates for one abstract variable and keep the best one."""
    specs = iter_search_specs(model, config)
    train_dataset = PairBankVariableDataset(train_bank, variable)
    calibration_dataset = PairBankVariableDataset(calibration_bank, variable)
    best_record = None
    best_intervenable = None
    best_spec = None
    all_records = []
    if config.verbose:
        print(
            f"DAS [{variable}] "
            f"| candidates={len(specs)} "
            f"| train_examples={len(train_dataset)} "
            f"| calibration_examples={len(calibration_dataset)} "
            f"| holdout_examples={holdout_bank.size}"
        )
    for index, spec in enumerate(specs, start=1):
        shuffle_seed = int(
            config.random_seed_base
            + _stable_text_seed(variable)
            + _stable_text_seed(spec.label)
            + index
        )
        intervention = RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[spec.layer]))
        intervenable = build_intervenable(
            model=model,
            layer=spec.layer,
            component=spec.component,
            intervention=intervention,
            device=device,
            unit=spec.unit,
            max_units=spec.max_units,
            freeze_model=True,
            freeze_intervention=False,
            use_fast=False,
        )
        loss_history = train_rotated_intervention(
            intervenable=intervenable,
            dataset=train_dataset,
            spec=spec,
            max_epochs=config.max_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            device=device,
            plateau_patience=config.plateau_patience,
            plateau_rel_delta=config.plateau_rel_delta,
            min_epochs=config.min_epochs,
            shuffle_seed=shuffle_seed,
        )
        calibration_metrics = evaluate_rotated_intervention(
            intervenable=intervenable,
            dataset=calibration_dataset,
            spec=spec,
            batch_size=config.batch_size,
            device=device,
            metrics_from_logits_fn=metrics_from_logits_fn,
        )
        selection_metrics = dict(summarize_selection_records_fn([calibration_metrics]))
        record = {
            "method": "das",
            "variable": variable,
            "split": calibration_bank.split,
            "seed": calibration_bank.seed,
            "site_label": spec.label,
            "layer": spec.layer,
            "subspace_dim": spec.subspace_dim,
            "selection_metrics": selection_metrics,
            **{f"selection_{metric_name}": metric_value for metric_name, metric_value in selection_metrics.items()},
            **{f"calibration_{metric_name}": metric_value for metric_name, metric_value in calibration_metrics.items()},
            "train_epochs_ran": len(loss_history),
            "train_loss_history": loss_history,
            "shuffle_seed": shuffle_seed,
        }
        all_records.append(record)
        is_better = choose_better_selection_candidate_fn(record, best_record)
        if config.verbose:
            status = "new best" if is_better else "candidate"
            print(
                f"DAS [{variable}] {status} {index}/{len(specs)} "
                f"| site={spec.label} "
                f"| epochs={len(loss_history)} "
                f"| train_loss={loss_history[-1]:.4f} "
                f"| selection={_format_selection_metrics(selection_metrics)} "
                f"| calibration={_format_selection_metrics(calibration_metrics)}"
            )
        if is_better:
            best_record = record
            best_intervenable = intervenable
            best_spec = spec
    if best_record is None or best_intervenable is None or best_spec is None:
        raise RuntimeError(f"Failed to select a DAS candidate for {variable}")
    holdout_dataset = PairBankVariableDataset(holdout_bank, variable)
    holdout_metrics = evaluate_rotated_intervention(
        intervenable=best_intervenable,
        dataset=holdout_dataset,
        spec=best_spec,
        batch_size=config.batch_size,
        device=device,
        metrics_from_logits_fn=metrics_from_logits_fn,
    )
    result_record = {**best_record, "split": holdout_bank.split, "seed": holdout_bank.seed, **holdout_metrics}
    if config.verbose:
        metric_names = tuple(str(name) for name in best_record.get("selection_metrics", {}).keys())
        calibration_metrics = {
            metric_name: float(result_record.get(f"calibration_{metric_name}", 0.0))
            for metric_name in metric_names
        }
        print(
            f"DAS [{variable}] selected {result_record['site_label']} "
            f"| epochs={int(result_record['train_epochs_ran'])} "
            f"| calibration={_format_selection_metrics(calibration_metrics)} "
            f"| holdout={_format_selection_metrics(_metric_subset(result_record, metric_names))}"
        )
    return result_record, all_records


def run_das_pipeline(
    model: VariableWidthMLPForClassification,
    train_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device | str,
    config: DASConfig,
    *,
    metrics_from_logits_fn,
    summarize_selection_records_fn,
    choose_better_selection_candidate_fn,
) -> dict[str, object]:
    """Run DAS for every abstract variable on shared pair-bank splits."""
    if not config.target_vars:
        raise ValueError("DASConfig.target_vars must be provided by the experiment")
    device = torch.device(device)
    results = []
    search_records = {}
    for variable in config.target_vars:
        best_record, all_records = run_das_search_for_variable(
            model=model,
            variable=variable,
            train_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=holdout_bank,
            device=device,
            config=config,
            metrics_from_logits_fn=metrics_from_logits_fn,
            summarize_selection_records_fn=summarize_selection_records_fn,
            choose_better_selection_candidate_fn=choose_better_selection_candidate_fn,
        )
        results.append(best_record)
        search_records[variable] = all_records
    return {
        "method_id": build_das_method_id(),
        "train_bank": train_bank.metadata(),
        "calibration_bank": calibration_bank.metadata(),
        "holdout_bank": holdout_bank.metadata(),
        "target_vars": list(config.target_vars),
        "training_stopping_rule": {
            "type": "plateau_on_train_loss",
            "max_epochs": config.max_epochs,
            "min_epochs": config.min_epochs,
            "plateau_rel_delta": config.plateau_rel_delta,
            "plateau_patience": config.plateau_patience,
        },
        "random_seed_base": int(config.random_seed_base),
        "search_records": search_records,
        "results": results,
    }
