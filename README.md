# Causal Abstractions for Two-Digit Addition

This repo uses script-first experiment runs for a two-digit addition benchmark.
The main workflow trains a shared MLP backbone, builds symbolic counterfactual
pair banks from an SCM, and compares `gw`, `ot`, `fgw`, and `das` on the same
held-out counterfactual test split.

## Setup

Install the pinned environment:

```bash
python -m pip install -r requirements.txt
```

## Main Entry Points

- `python -m experiments.addition.train`
  - Train the addition backbone.
- `python -m experiments.addition.compare`
  - Run one addition comparison.
- `python -m experiments.addition.seed_sweep`
  - Run the addition multi-seed sweep.

## Current Default Experimental Spec

- Input: two 2-digit numbers encoded as concatenated one-hot digit vectors.
- Input width: `40`.
- Default abstract variables: `S1`, `C1`, `S2`, `C2`.
- Output: `200`-class classification over sums `0..199`.
- Backbone: four-hidden-layer ReLU MLP with hidden width `192`.
- Factual supervised data: `30,000` train, `4,000` validation.
- Counterfactual pair splits: `train=1000`, `calibration=1000`, `test=5000`.
- Core comparable metrics: `exact_acc`, `mean_shared_digits`.

## Result Contract

Each comparison run now emits required comparability metadata:

- `canonical_variable_mapping`
- `method_id`
- `core_metrics`
- `environment`
- `seed_trace`

These contract fields are assembled by the shared framework in `experiment_core/`,
not by the addition package directly.

For addition, the default canonical mapping is identity:

```python
CANONICAL_VARIABLE_MAPPING = {
    "S1": "S1",
    "C1": "C1",
    "S2": "S2",
    "C2": "C2",
}
```

`method_id` is a flat string. Examples:

- `gw_current_res1_cosine`
- `ot_pca_res2_cosine_pc8_keep128`
- `fgw_current_res1_cosine_a0p5`
- `das`

## Config Sections

The addition scripts expose these main knobs directly:

- canonical mapping via `CANONICAL_VARIABLE_MAPPING`
- method naming via `METHODS`
- optional PCA site selection via `OT_SITE_POLICY`, `OT_PCA_COMPONENTS`, `OT_PCA_CANDIDATE_COUNT`
- reproducibility via `SEED`, `SEEDS`, `RESULTS_TIMESTAMP`, and recorded environment metadata

## Architecture

The repo is now split into two layers:

- `experiment_core/`
  - reusable framework code for runners, contracts, runtime metadata, transport methods, DAS, reporting, plots, and sweep aggregation
- `experiments/addition/`
  - addition-specific SCM semantics, metrics, pair-bank building, experiment spec, backbone wiring, and experiment entrypoints

This separation is the intended pattern for future experiments:

- put reusable machinery in `experiment_core/`
- put domain semantics and experiment glue in `experiments/<name>/`

## Typical Workflow

Train a backbone:

```bash
python -m experiments.addition.train
```

Run the default baseline comparison:

```bash
python -m experiments.addition.compare
```

Run a PCA-enabled transport experiment:

```python
METHODS = ("ot", "fgw")
OT_SITE_POLICY = "pca"
OT_PCA_COMPONENTS = 8
OT_PCA_CANDIDATE_COUNT = 128
```

Then:

```bash
python -m experiments.addition.compare
```

Run a multi-seed sweep:

```bash
python -m experiments.addition.seed_sweep
```

To keep output folders stable for publication snapshots, set a fixed timestamp:

```bash
RESULTS_TIMESTAMP=paper_snapshot python -m experiments.addition.compare
```

## Methods Implemented

- `gw`: entropic Gromov-Wasserstein on relational effect geometry.
- `ot`: entropic optimal transport on direct abstract-to-neural signature costs.
- `fgw`: fused Gromov-Wasserstein hybrid transport.
- `das`: rotated-space intervention search with calibration-based model selection.

GW, OT, and FGW support two site policies:

- `current`: existing canonical site enumeration behavior.
- `pca`: optional PCA-guided pre-selection before transport ranking.

DAS intentionally does not use the PCA site-selection path.

## Tests

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
python -m unittest tests.contract.test_result_contracts
```

Run one integration class:

```bash
python -m unittest tests.integration.test_addition_pipeline.AdditionPipelineIntegrationTests
```

Run one specific smoke test:

```bash
python -m unittest tests.integration.test_addition_pipeline.AdditionPipelineIntegrationTests.test_small_compare_pipeline
```

Minimal CI runs unit tests, contract tests, and the tiny integration smoke on CPU.

## Adding A New Experiment

The intended workflow for a new experiment is:

1. Add `experiments/<new_name>/spec.py` with an `ExperimentSpec` and adapter hooks.
2. Add experiment-specific SCM/data generation, pair-bank building, and metrics under `experiments/<new_name>/`.
3. Reuse `experiment_core.compare_runner`, `experiment_core.ot`, `experiment_core.das`, and the shared reporting/runtime code.
4. Add experiment-local entrypoints under `experiments/<new_name>/` and run them with `python -m`.

## Publication Snapshot Procedure

1. Install the pinned environment from `requirements.txt`.
2. Set the final script config in `experiments/addition/train.py`, `experiments/addition/compare.py`, or `experiments/addition/seed_sweep.py`.
3. Use a fixed `RESULTS_TIMESTAMP` so artifact paths are stable.
4. Run the desired baseline and optional PCA-enabled comparisons.
5. Keep the generated JSON artifacts, text summaries, and plots together under the matching `results/<timestamp>/` directory.

## Repository Layout

- `experiment_core/`: reusable experiment framework code.
- `experiments/addition/`: the concrete addition experiment implementation.
- `tests/`: split `unit`, `contract`, and `integration` suites.
- `models/`: saved backbones such as `addition_mlp_seed44.pt`.
- `results/`: timestamped experiment outputs.
- `paper/`: draft paper materials, including `paper/addition_methodology.tex`.
