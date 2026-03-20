"""Experiment adapter contract used by the shared framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .experiment_spec import ExperimentSpec
from .pair_bank import PairBank


@dataclass(frozen=True)
class ExperimentAdapter:
    """Experiment-provided hooks used by the shared runner and backbone code."""

    experiment_spec: ExperimentSpec
    build_factual_tensors: Callable[[object, int, int], tuple[torch.Tensor, torch.Tensor]]
    build_pair_bank: Callable[[object, int, int, str, bool], PairBank]
    metrics_from_logits: Callable[[torch.Tensor, torch.Tensor | list[int]], dict[str, float]]
    build_checkpoint_metadata: Callable[[object], dict[str, object]]
    summarize_selection_records: Callable[[list[dict[str, object]]], dict[str, float]]
    choose_better_selection_candidate: Callable[[dict[str, object], dict[str, object] | None], bool]
