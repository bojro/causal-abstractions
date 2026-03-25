# Causal Abstractions OT

This repository is a script-first framework for running causal abstraction
experiments with a shared neural backbone/comparison pipeline and
experiment-specific SCM semantics.

At a high level, each experiment does the same thing:

1. Train a classifier backbone on factual data.
2. Build symbolic counterfactual pair banks from an SCM.
3. Compare `gw`, `ot`, `fgw`, and `das` on the same held-out
   counterfactual test split.
4. Save JSON results, text summaries, and plots under experiment-specific
   subdirectories such as `results/<timestamp>/<experiment>/...`.

The framework currently includes two concrete experiments:

- `addition`: two-digit addition with symbolic carry/sum variables.
- `hierarchical_equality`: a continuous-input equality task with abstract
  targets `WX` and `YZ`.

## Setup

Install the pinned environment:

```bash
python -m pip install -r requirements.txt
```

## Current Experiments

### `addition`

- Inputs: two two-digit numbers encoded as concatenated one-hot digit vectors.
- Input width: `40`.
- Default abstract variables: `S1`, `C1`, `S2`, `C2`.
- Output: `200`-class classification over sums `0..199`.
- Default backbone: four-layer ReLU MLP with hidden dims `(192, 192, 192, 192)`.
- Default core metrics: `exact_acc`, `mean_shared_digits`.

### `hierarchical_equality`

- Inputs: four continuous low-dimensional objects `W`, `X`, `Y`, `Z`.
- Input width: `16` with the current `embedding_dim=4`.
- Default abstract variables: `WX`, `YZ`.
- Output: binary classification.
- Default backbone: four-layer ReLU MLP with hidden dims `(64, 64, 64, 64)`.
- Default core metrics: `exact_acc`, `mean_true_class_prob`.

## Main Entry Points

### Addition

- `python -m experiments.addition.train`
  Train the addition backbone.
- `python -m experiments.addition.compare`
  Run one addition comparison.
- `python -m experiments.addition.seed_sweep`
  Run the addition multi-seed sweep.

### Hierarchical Equality

- `python -m experiments.hierarchical_equality.train`
  Train the hierarchical equality backbone.
- `python -m experiments.hierarchical_equality.compare`
  Run one hierarchical equality comparison.
- `python -m experiments.hierarchical_equality.seed_sweep`
  Run the hierarchical equality multi-seed sweep.

## Typical Workflow

Train a backbone:

```bash
python -m experiments.addition.train
```

Run a comparison:

```bash
python -m experiments.addition.compare
```

Use a fixed timestamp when you want stable artifact paths:

```bash
RESULTS_TIMESTAMP=paper_snapshot python -m experiments.addition.compare
```

With the current layout that produces experiment-specific folders, for example:

- `results/paper_snapshot/addition/...`
- `results/paper_snapshot/hierarchical_equality/...`

Example for the second experiment:

```bash
python -m experiments.hierarchical_equality.train
python -m experiments.hierarchical_equality.compare
python -m experiments.hierarchical_equality.seed_sweep
```

For transport-only experiments, you can narrow `METHODS` in an experiment's
`compare.py`. For PCA-guided site pre-selection, adjust:

- `OT_SITE_POLICY`
- `OT_PCA_COMPONENTS`
- `OT_PCA_CANDIDATE_COUNT`

## Shared Architecture

The repository is split into two layers.

- `experiment_core/`
  Shared framework code for backbones, comparison runners, contracts,
  transport methods, DAS, reporting, plotting, runtime metadata, and
  seed-sweep aggregation.
- `experiments/<name>/`
  Experiment-local semantics: SCM, factual data generation, pair-bank
  building, metrics, selection policy, experiment spec/adapter, and
  runnable entrypoints.

This separation is the intended pattern for all future experiments:

- Put reusable machinery in `experiment_core/`.
- Put domain semantics and experiment glue in `experiments/<name>/`.

## How A Comparison Run Works

The shared runner in `experiment_core/compare_runner.py` expects an
experiment-specific adapter and handles the rest.

An experiment provides:

- an `ExperimentSpec`
- a factual-data builder
- a pair-bank builder
- a metric function
- checkpoint metadata
- calibration summary logic
- incumbent-selection logic

Those hooks are packaged in `ExperimentAdapter` and used by the shared runner,
shared OT/GW/FGW pipeline, shared DAS pipeline, shared reporting, and shared
plotting.

## Output Layout

Single compare runs now write into experiment-specific directories and use
self-identifying plot names. For example:

- `results/<timestamp>/addition/addition_compare_results.json`
- `results/<timestamp>/addition/addition_compare_exact_accuracy.png`
- `results/<timestamp>/hierarchical_equality/hierarchical_equality_compare_method_runtime.png`

Seed sweeps write into a nested sweep directory, with per-seed outputs below it:

- `results/<timestamp>/addition/seed_sweep/addition_seed_sweep_results.json`
- `results/<timestamp>/addition/seed_sweep/seed_44/addition_compare_results.json`
- `results/<timestamp>/hierarchical_equality/seed_sweep/hierarchical_equality_seed_sweep_runtime_summary.png`

This avoids plot overwrites when multiple experiments share the same
`RESULTS_TIMESTAMP`.

## Result Contract

Every comparison run emits comparability metadata assembled by the shared
framework, not by individual experiments ad hoc.

Important fields include:

- `canonical_variable_mapping`
- `method_id`
- `core_metrics`
- `environment`
- `seed_trace`

`method_id` is a flat string. Examples:

- `gw_current_res1_cosine`
- `ot_pca_res2_cosine_pc8_keep128`
- `fgw_current_res1_cosine_a0p5`
- `das`

The point of the contract is that two experiments can differ in semantics and
metrics, while still producing a consistent result shape for downstream
reporting, plotting, and analysis.

## Methods Implemented

- `gw`
  Entropic Gromov-Wasserstein on relational effect geometry.
- `ot`
  Entropic optimal transport on direct abstract-to-neural signature costs.
- `fgw`
  Fused Gromov-Wasserstein hybrid transport.
- `das`
  Rotated-space intervention search with calibration-based model selection.

GW, OT, and FGW support two site policies:

- `current`
  Use the standard site enumeration behavior.
- `pca`
  Use PCA-guided pre-selection before transport ranking.

DAS intentionally does not use the PCA site-selection path.

## Adding A New Experiment

The easiest way to add a new experiment is to mirror the layout under
`experiments/addition/` or `experiments/hierarchical_equality/`.

### Minimum new package shape

Create `experiments/<new_name>/` with most or all of:

- `__init__.py`
- `constants.py`
- `scm.py`
- `metrics.py`
- `selection.py`
- `spec.py`
- `backbone.py`
- `pair_bank.py`
- `train.py`
- `compare.py`

Add a `seed_sweep.py` only if you want multi-seed aggregation for that
experiment. The repository currently has seed sweeps for both `addition` and
`hierarchical_equality`.

### Required shared hooks

Your `spec.py` should build an `ExperimentSpec` and an `ExperimentAdapter`.

`ExperimentSpec` must define:

- `experiment_id`
- `local_target_vars`
- `canonical_variable_mapping`
- `core_metrics`

`ExperimentAdapter` must provide:

- `build_factual_tensors(problem, size, seed)`
- `build_pair_bank(problem, size, seed, split, verify_with_scm)`
- `metrics_from_logits(logits, targets)`
- `build_checkpoint_metadata(train_config)`
- `summarize_selection_records(records)`
- `choose_better_selection_candidate(candidate, incumbent)`

### Implementation checklist

1. Define the SCM and factual sampling in `scm.py`.
2. Decide which abstract variables are local targets.
3. Build factual tensors for supervised backbone training.
4. Build counterfactual `PairBank`s for train, calibration, and test.
5. Define experiment-appropriate comparable metrics.
6. Wire the adapter/spec in `spec.py`.
7. Add `train.py` and `compare.py` entrypoints.
8. Reuse shared comparison/reporting code from `experiment_core/`.
9. Add unit, contract, and integration tests.
10. Run a real train/compare job and inspect the generated plots and summary.

### Design guidance

- Keep experiment semantics local; keep framework logic shared.
- Prefer experiment-defined `core_metrics` instead of hardcoding metric names
  in shared code.
- Use canonical variable mappings when different experiments use different
  local names for conceptually similar targets.
- Keep train/compare scripts script-first and explicit. This repo favors
  editable Python config blocks over a large CLI layer.

## Testing

The `unittest` suite is split into:

- `tests/unit`
- `tests/contract`
- `tests/integration`

Run everything:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run just unit tests:

```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

Run one contract module:

```bash
python -m unittest tests.contract.test_hierarchical_equality_contracts
```

Run one integration class:

```bash
python -m unittest tests.integration.test_addition_pipeline.AdditionPipelineIntegrationTests
```

Run one specific smoke test:

```bash
python -m unittest tests.integration.test_hierarchical_equality_pipeline.HierarchicalEqualityPipelineIntegrationTests.test_small_compare_pipeline
```

Minimal CI should run unit tests, contract tests, and a tiny integration smoke
on CPU.

## Publication / Snapshot Workflow

1. Install the pinned environment from `requirements.txt`.
2. Set the desired config in the relevant experiment scripts.
3. Set `RESULTS_TIMESTAMP` when you want stable output paths.
4. Run the needed train/compare or sweep scripts.
5. Keep the JSON artifacts, text summaries, and plots together under the
   matching `results/<timestamp>/<experiment>/` directory.

## Repository Layout

- `experiment_core/`
  Reusable experiment framework code.
- `experiments/addition/`
  Concrete addition experiment implementation.
- `experiments/hierarchical_equality/`
  Concrete hierarchical equality implementation.
- `tests/`
  `unit`, `contract`, and `integration` suites.
- `models/`
  Saved backbones such as `addition_mlp_seed44.pt` and
  `hierarchical_equality_mlp_seed44.pt`.
- `results/`
  Timestamped experiment outputs.
- `paper/`
  Draft paper materials, currently addition-focused.
