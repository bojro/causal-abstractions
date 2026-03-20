"""Addition-specific pair-bank construction."""

from __future__ import annotations

import numpy as np
import torch

from experiment_core.pair_bank import PairBank

from .scm import (
    AdditionProblem,
    compute_counterfactual_labels,
    compute_states_for_digits,
    digits_to_inputs_embeds,
    verify_counterfactual_labels_with_scm,
)


def build_pair_bank(
    problem: AdditionProblem,
    size: int,
    seed: int,
    split: str,
    verify_with_scm: bool = False,
) -> PairBank:
    """Create a deterministic base/source pair bank and counterfactual labels."""
    rng = np.random.default_rng(seed)
    base_digits = rng.integers(0, 10, size=(size, 4), dtype=np.int64)
    source_digits = rng.integers(0, 10, size=(size, 4), dtype=np.int64)

    base_states = compute_states_for_digits(base_digits)
    source_states = compute_states_for_digits(source_digits)
    cf_labels_np = compute_counterfactual_labels(base_states, source_states)
    target_vars = problem.experiment_spec.local_target_vars

    if verify_with_scm:
        verify_counterfactual_labels_with_scm(problem, base_digits, source_digits, cf_labels_np)

    return PairBank(
        split=split,
        seed=seed,
        base_digits=torch.tensor(base_digits, dtype=torch.long),
        source_digits=torch.tensor(source_digits, dtype=torch.long),
        base_inputs=digits_to_inputs_embeds(base_digits, problem.input_var_order),
        source_inputs=digits_to_inputs_embeds(source_digits, problem.input_var_order),
        base_labels=torch.tensor(base_states["O"], dtype=torch.long),
        cf_labels_by_var={
            var: torch.tensor(cf_labels_np[var], dtype=torch.long) for var in target_vars
        },
        target_vars=tuple(target_vars),
    )
