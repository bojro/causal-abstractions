"""Generic classifier backbone training, loading, and factual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from variable_width_mlp import (
    VariableWidthMLPConfig,
    VariableWidthMLPForClassification,
    load_variable_width_mlp_checkpoint,
)

from .adapter import ExperimentAdapter
from .defaults import (
    DEFAULT_ACTIVATION,
    DEFAULT_DROPOUT,
    DEFAULT_FACTUAL_TRAIN_SIZE,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_HIDDEN_DIMS,
)
from .runtime import ensure_parent_dir, set_seed


@dataclass(frozen=True)
class ClassifierTrainConfig:
    """Configuration for supervised training and loading of a classifier backbone."""

    seed: int = 42
    n_train: int = DEFAULT_FACTUAL_TRAIN_SIZE
    n_validation: int = DEFAULT_FACTUAL_VALIDATION_SIZE
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    input_dim: int = 0
    num_classes: int = 0
    dropout: float = DEFAULT_DROPOUT
    activation: str = DEFAULT_ACTIVATION
    abstract_variables: tuple[str, ...] = ()
    learning_rate: float = 2e-3
    train_epochs: int = 20
    train_batch_size: int = 256
    eval_batch_size: int = 256
    verbose: bool = True


def evaluate_classifier_backbone(
    model: VariableWidthMLPForClassification,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Compute factual exact-match accuracy for a trained model."""
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            end = min(start + batch_size, inputs.shape[0])
            logits = model(inputs_embeds=inputs[start:end].to(device).unsqueeze(1))[0]
            predictions.append(torch.argmax(logits, dim=1).detach().cpu())
    preds = torch.cat(predictions, dim=0)
    acc = float((preds == labels.view(-1)).to(torch.float32).mean().item())
    return {
        "exact_acc": acc,
        "num_examples": int(labels.numel()),
    }


def save_backbone_checkpoint(
    model: VariableWidthMLPForClassification,
    config: VariableWidthMLPConfig,
    checkpoint_path: str | Path,
    metadata: dict[str, object],
) -> None:
    """Write the trained classifier checkpoint and experiment metadata to disk."""
    ensure_parent_dir(checkpoint_path)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": config.to_dict(),
        "metadata": metadata,
    }
    torch.save(payload, checkpoint_path)


def checkpoint_matches_train_config(
    checkpoint: dict[str, object],
    train_config: ClassifierTrainConfig,
) -> bool:
    """Check whether an existing checkpoint matches the requested model spec."""
    model_config = checkpoint.get("model_config", {})
    if not isinstance(model_config, dict):
        return False
    return (
        int(model_config.get("input_dim", -1)) == int(train_config.input_dim)
        and tuple(int(dim) for dim in model_config.get("hidden_dims", [])) == tuple(train_config.hidden_dims)
        and int(model_config.get("num_classes", -1)) == int(train_config.num_classes)
        and str(model_config.get("activation", "")) == str(train_config.activation)
    )


def train_classifier_backbone(
    *,
    problem,
    adapter: ExperimentAdapter,
    train_config: ClassifierTrainConfig,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[VariableWidthMLPForClassification, VariableWidthMLPConfig, dict[str, object]]:
    """Train the classifier backbone and return the model, config, and metrics."""
    device = torch.device(device)
    set_seed(train_config.seed)

    x_train, y_train = adapter.build_factual_tensors(problem, train_config.n_train, train_config.seed + 1)
    x_validation, y_validation = adapter.build_factual_tensors(
        problem,
        train_config.n_validation,
        train_config.seed + 2,
    )

    config = VariableWidthMLPConfig(
        input_dim=train_config.input_dim,
        hidden_dims=list(train_config.hidden_dims),
        num_classes=train_config.num_classes,
        dropout=train_config.dropout,
        activation=train_config.activation,
    )
    model = VariableWidthMLPForClassification(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_config.train_batch_size,
        shuffle=True,
    )

    loss_history = []
    validation_history = []
    perfect_validation_streak = 0
    stopping_reason = None
    if train_config.verbose:
        print(
            "Backbone training "
            f"| device={device} "
            f"| train_examples={train_config.n_train} "
            f"| validation_examples={train_config.n_validation} "
            f"| hidden_dims={tuple(train_config.hidden_dims)} "
            f"| epochs={train_config.train_epochs}"
        )
    for epoch in range(train_config.train_epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_labels in loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs_embeds=batch_inputs.unsqueeze(1))[0]
            loss = F.cross_entropy(logits, batch_labels.view(-1))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * batch_inputs.shape[0]
        epoch_loss = running_loss / len(loader.dataset)
        loss_history.append(epoch_loss)
        epoch_validation_metrics = evaluate_classifier_backbone(
            model=model,
            inputs=x_validation,
            labels=y_validation,
            batch_size=train_config.eval_batch_size,
            device=device,
        )
        validation_history.append(epoch_validation_metrics)
        if float(epoch_validation_metrics["exact_acc"]) >= 1.0:
            perfect_validation_streak += 1
        else:
            perfect_validation_streak = 0
        if train_config.verbose:
            print(
                f"Epoch {epoch + 1}/{train_config.train_epochs} "
                f"| train_loss={epoch_loss:.4f} "
                f"| val_exact_acc={epoch_validation_metrics['exact_acc']:.4f}"
            )
        if perfect_validation_streak >= 5:
            stopping_reason = "perfect_validation_exact_acc >= 1.0000 for 5 epochs"
            if train_config.verbose:
                print(f"Stopping early | reason={stopping_reason}")
            break

    factual_metrics = validation_history[-1] if validation_history else evaluate_classifier_backbone(
        model=model,
        inputs=x_validation,
        labels=y_validation,
        batch_size=train_config.eval_batch_size,
        device=device,
    )
    save_backbone_checkpoint(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        metadata=adapter.build_checkpoint_metadata(train_config),
    )
    return model, config, {
        "train_loss_history": loss_history,
        "validation_history": validation_history,
        "factual_validation_metrics": factual_metrics,
        "epochs_ran": len(loss_history),
        "stopped_early": stopping_reason is not None,
        "stopping_reason": stopping_reason,
        "checkpoint_path": str(checkpoint_path),
    }


def load_classifier_backbone(
    *,
    problem,
    adapter: ExperimentAdapter,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    train_config: ClassifierTrainConfig | None = None,
) -> tuple[VariableWidthMLPForClassification, VariableWidthMLPConfig, dict[str, object]]:
    """Load an existing checkpoint and fail if it is missing or incompatible."""
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train a backbone first.")

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    eval_config = train_config or ClassifierTrainConfig()
    if not checkpoint_matches_train_config(checkpoint, eval_config):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not match the requested model spec. "
            "Regenerate the checkpoint for the current config."
        )

    model, config, checkpoint = load_variable_width_mlp_checkpoint(str(checkpoint_path), device)
    x_validation, y_validation = adapter.build_factual_tensors(
        problem,
        eval_config.n_validation,
        eval_config.seed + 2,
    )
    factual_metrics = evaluate_classifier_backbone(
        model=model,
        inputs=x_validation,
        labels=y_validation,
        batch_size=eval_config.eval_batch_size,
        device=device,
    )
    return model, config, {
        "checkpoint_path": str(checkpoint_path),
        "loaded_existing_checkpoint": True,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
        "factual_validation_metrics": factual_metrics,
    }
