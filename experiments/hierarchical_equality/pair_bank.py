"""Hierarchical-equality pair-bank construction."""

from __future__ import annotations

import torch

from experiment_core.pair_bank import PairBank

from .scm import (
    HierarchicalEqualityProblem,
    compute_counterfactual_labels,
    compute_states_for_inputs,
    sample_structured_examples,
    verify_counterfactual_labels_with_scm,
)


def build_pair_bank(
    problem: HierarchicalEqualityProblem,
    size: int,
    seed: int,
    split: str,
    verify_with_scm: bool = False,
) -> PairBank:
    """Create a deterministic base/source pair bank and counterfactual labels."""
    base_inputs, base_labels, base_structures = sample_structured_examples(
        size,
        seed,
        embedding_dim=problem.embedding_dim,
    )
    source_inputs, _, source_structures = sample_structured_examples(
        size,
        seed + 1,
        embedding_dim=problem.embedding_dim,
    )
    base_states = compute_states_for_inputs(base_inputs, embedding_dim=problem.embedding_dim)
    source_states = compute_states_for_inputs(source_inputs, embedding_dim=problem.embedding_dim)
    cf_labels_np = compute_counterfactual_labels(base_states, source_states)
    target_vars = tuple(problem.experiment_spec.local_target_vars)

    if verify_with_scm:
        verify_counterfactual_labels_with_scm(problem, base_inputs, source_inputs, cf_labels_np)

    return PairBank(
        split=split,
        seed=seed,
        base_digits=base_structures,
        source_digits=source_structures,
        base_inputs=base_inputs,
        source_inputs=source_inputs,
        base_labels=base_labels,
        cf_labels_by_var={
            variable: torch.tensor(cf_labels_np[variable], dtype=torch.long)
            for variable in target_vars
        },
        target_vars=target_vars,
    )
