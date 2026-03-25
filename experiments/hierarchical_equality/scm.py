"""Hierarchical equality SCM and continuous-input data generation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pyvene import CausalModel

from experiment_core.experiment_spec import ExperimentSpec

from .constants import (
    CANONICAL_INPUT_VARS,
    CANONICAL_VARIABLE_MAPPING,
    DEFAULT_TARGET_VARS,
    EMBEDDING_DIM,
)

_STRUCTURE_ROWS = np.asarray(
    [
        [0, 1, 2, 3],  # both different -> WX=False, YZ=False
        [0, 0, 1, 1],  # both equal -> WX=True, YZ=True
        [0, 0, 1, 2],  # left equal, right different
        [0, 1, 2, 2],  # left different, right equal
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class HierarchicalEqualityProblem:
    """Bundle of the symbolic equality SCM and its canonical input ordering."""

    causal_model: CausalModel
    input_var_order: tuple[str, ...]
    experiment_spec: ExperimentSpec
    embedding_dim: int


def build_hierarchical_equality_causal_model(
    embedding_dim: int = EMBEDDING_DIM,
) -> CausalModel:
    """Construct the symbolic SCM for hierarchical equality."""
    if int(embedding_dim) <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
    zero_vector = np.zeros(int(embedding_dim), dtype=np.float32)
    variables = ["W", "X", "Y", "Z", "WX", "YZ", "O"]
    values = {
        "W": [zero_vector],
        "X": [zero_vector],
        "Y": [zero_vector],
        "Z": [zero_vector],
        "WX": [False, True],
        "YZ": [False, True],
        "O": [0, 1],
    }
    parents = {
        "W": [],
        "X": [],
        "Y": [],
        "Z": [],
        "WX": ["W", "X"],
        "YZ": ["Y", "Z"],
        "O": ["WX", "YZ"],
    }
    functions = {
        "W": lambda: zero_vector,
        "X": lambda: zero_vector,
        "Y": lambda: zero_vector,
        "Z": lambda: zero_vector,
        "WX": lambda w, x: bool(np.array_equal(np.asarray(w), np.asarray(x))),
        "YZ": lambda y, z: bool(np.array_equal(np.asarray(y), np.asarray(z))),
        "O": lambda wx, yz: int(bool(wx) == bool(yz)),
    }
    return CausalModel(variables, values, parents, functions)


def split_concatenated_inputs(
    inputs: np.ndarray | torch.Tensor | list[float] | list[list[float]],
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, np.ndarray]:
    """Split concatenated continuous inputs into named leaf vectors."""
    if isinstance(inputs, torch.Tensor):
        arr = inputs.detach().cpu().numpy()
    else:
        arr = np.asarray(inputs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    expected_width = len(CANONICAL_INPUT_VARS) * int(embedding_dim)
    if arr.shape[1] != expected_width:
        raise ValueError(f"Expected input rows shaped [N, {expected_width}], got {arr.shape}")
    return {
        variable: arr[:, index * embedding_dim : (index + 1) * embedding_dim]
        for index, variable in enumerate(CANONICAL_INPUT_VARS)
    }


def compute_states_for_inputs(
    inputs: np.ndarray | torch.Tensor | list[float] | list[list[float]],
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, np.ndarray]:
    """Compute the intermediate equality variables and final label."""
    pieces = split_concatenated_inputs(inputs, embedding_dim=embedding_dim)
    wx = np.all(pieces["W"] == pieces["X"], axis=1)
    yz = np.all(pieces["Y"] == pieces["Z"], axis=1)
    output = (wx == yz).astype(np.int64)
    return {
        "WX": wx.astype(np.int64),
        "YZ": yz.astype(np.int64),
        "O": output,
    }


def compute_counterfactual_labels(
    base_states: dict[str, np.ndarray],
    source_states: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute SCM counterfactual outputs for each abstract-variable swap."""
    base_wx = np.asarray(base_states["WX"], dtype=np.int64)
    base_yz = np.asarray(base_states["YZ"], dtype=np.int64)
    source_wx = np.asarray(source_states["WX"], dtype=np.int64)
    source_yz = np.asarray(source_states["YZ"], dtype=np.int64)
    return {
        "WX": (source_wx == base_yz).astype(np.int64),
        "YZ": (base_wx == source_yz).astype(np.int64),
    }


def structure_rows_from_patterns(
    pattern_ids: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    """Encode each structural pattern as four local identity ids."""
    indices = np.asarray(pattern_ids, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(_STRUCTURE_ROWS)):
        raise ValueError(f"Pattern ids must lie in [0, {len(_STRUCTURE_ROWS) - 1}]")
    return _STRUCTURE_ROWS[indices]


def _inputs_from_pattern_ids(
    pattern_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    embedding_dim: int,
) -> np.ndarray:
    """Materialize concatenated continuous inputs for the requested structural patterns."""
    size = int(pattern_ids.shape[0])
    dim = int(embedding_dim)
    a = rng.uniform(-1.0, 1.0, size=(size, dim)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, size=(size, dim)).astype(np.float32)
    c = rng.uniform(-1.0, 1.0, size=(size, dim)).astype(np.float32)
    d = rng.uniform(-1.0, 1.0, size=(size, dim)).astype(np.float32)

    w = np.empty_like(a)
    x = np.empty_like(a)
    y = np.empty_like(a)
    z = np.empty_like(a)

    both_different = pattern_ids == 0
    both_equal = pattern_ids == 1
    left_equal = pattern_ids == 2
    right_equal = pattern_ids == 3

    w[both_different] = a[both_different]
    x[both_different] = b[both_different]
    y[both_different] = c[both_different]
    z[both_different] = d[both_different]

    w[both_equal] = a[both_equal]
    x[both_equal] = a[both_equal]
    y[both_equal] = c[both_equal]
    z[both_equal] = c[both_equal]

    w[left_equal] = a[left_equal]
    x[left_equal] = a[left_equal]
    y[left_equal] = c[left_equal]
    z[left_equal] = d[left_equal]

    w[right_equal] = a[right_equal]
    x[right_equal] = b[right_equal]
    y[right_equal] = c[right_equal]
    z[right_equal] = c[right_equal]

    return np.concatenate([w, x, y, z], axis=1)


def sample_structured_examples(
    size: int,
    seed: int,
    *,
    embedding_dim: int = EMBEDDING_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample balanced continuous equality examples and return inputs, labels, and structures."""
    if int(size) <= 0:
        raise ValueError(f"size must be positive, got {size}")
    rng = np.random.default_rng(int(seed))
    dim = int(embedding_dim)
    pattern_ids = rng.integers(0, len(_STRUCTURE_ROWS), size=int(size), dtype=np.int64)
    packed = _inputs_from_pattern_ids(pattern_ids, rng, embedding_dim=dim)
    states = compute_states_for_inputs(packed, embedding_dim=dim)
    structures = structure_rows_from_patterns(pattern_ids)
    return (
        torch.tensor(packed, dtype=torch.float32),
        torch.tensor(states["O"], dtype=torch.long),
        torch.tensor(structures, dtype=torch.long),
    )


def assignment_from_input_row(
    row: np.ndarray | torch.Tensor | list[float],
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, np.ndarray]:
    """Convert one concatenated neural input back into an SCM leaf assignment."""
    pieces = split_concatenated_inputs(row, embedding_dim=embedding_dim)
    return {
        variable: pieces[variable][0].astype(np.float32, copy=True)
        for variable in CANONICAL_INPUT_VARS
    }


def verify_scm_pattern_table(
    causal_model: CausalModel,
    embedding_dim: int = EMBEDDING_DIM,
) -> None:
    """Check the SCM on one representative example for each structural pattern."""
    pattern_ids = np.arange(len(_STRUCTURE_ROWS), dtype=np.int64)
    sample_inputs = _inputs_from_pattern_ids(
        pattern_ids,
        np.random.default_rng(0),
        embedding_dim=int(embedding_dim),
    )
    states = compute_states_for_inputs(sample_inputs, embedding_dim=embedding_dim)
    for index in range(sample_inputs.shape[0]):
        assignment = assignment_from_input_row(sample_inputs[index], embedding_dim=embedding_dim)
        setting = causal_model.run_forward(assignment)
        for variable in ("WX", "YZ", "O"):
            expected = int(states[variable][index])
            actual = int(setting[variable])
            if expected != actual:
                raise AssertionError(
                    f"SCM mismatch at pattern_index={index}, var={variable}, expected={expected}, actual={actual}"
                )


def verify_counterfactual_labels_with_scm(
    problem: HierarchicalEqualityProblem,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    cf_labels_by_var: dict[str, np.ndarray | torch.Tensor],
) -> None:
    """Cross-check vectorized counterfactual labels against SCM interchange."""
    size = int(base_inputs.shape[0])
    for index in range(size):
        base_assignment = assignment_from_input_row(base_inputs[index], embedding_dim=problem.embedding_dim)
        source_assignment = assignment_from_input_row(source_inputs[index], embedding_dim=problem.embedding_dim)
        for variable in problem.experiment_spec.local_target_vars:
            expected = int(problem.causal_model.run_interchange(base_assignment, {variable: source_assignment})["O"])
            values = cf_labels_by_var[variable]
            actual = int(values[index].item() if isinstance(values, torch.Tensor) else values[index])
            if expected != actual:
                raise AssertionError(
                    f"Counterfactual mismatch at index={index}, var={variable}, expected={expected}, actual={actual}"
                )


def load_hierarchical_equality_problem(
    run_checks: bool = True,
    *,
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS,
    canonical_variable_mapping: dict[str, str] | None = None,
    embedding_dim: int = EMBEDDING_DIM,
) -> HierarchicalEqualityProblem:
    """Build the hierarchical-equality SCM bundle and optionally run consistency checks."""
    if canonical_variable_mapping is None:
        canonical_variable_mapping = dict(CANONICAL_VARIABLE_MAPPING)
    causal_model = build_hierarchical_equality_causal_model(embedding_dim=embedding_dim)
    from .spec import build_hierarchical_equality_experiment_spec

    experiment_spec = build_hierarchical_equality_experiment_spec(
        target_vars=tuple(target_vars),
        canonical_variable_mapping=canonical_variable_mapping,
    )
    if run_checks:
        verify_scm_pattern_table(causal_model, embedding_dim=embedding_dim)
    return HierarchicalEqualityProblem(
        causal_model=causal_model,
        input_var_order=CANONICAL_INPUT_VARS,
        experiment_spec=experiment_spec,
        embedding_dim=int(embedding_dim),
    )
