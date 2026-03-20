from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_core.das import DASConfig, run_das_search_for_variable
from experiment_core.ot import OTConfig, select_transport_hyperparameters
from experiment_core.pyvene_utils import DASSearchSpec
from experiments.addition.pair_bank import build_pair_bank
from experiments.addition.selection import (
    choose_better_addition_candidate,
    summarize_addition_selection_records,
)
from tests.common import load_small_model


class SelectionPolicyUnitTests(unittest.TestCase):
    def test_addition_selection_policy_preserves_exact_then_shared_ordering(self) -> None:
        summary = summarize_addition_selection_records(
            [
                {"exact_acc": 0.25, "mean_shared_digits": 1.0},
                {"exact_acc": 0.75, "mean_shared_digits": 2.0},
            ]
        )
        self.assertAlmostEqual(summary["exact_acc"], 0.5)
        self.assertAlmostEqual(summary["mean_shared_digits"], 1.5)

        incumbent = {"selection_metrics": {"exact_acc": 0.5, "mean_shared_digits": 1.0}}
        better = {"selection_metrics": {"exact_acc": 0.5, "mean_shared_digits": 1.2}}
        worse = {"selection_metrics": {"exact_acc": 0.4, "mean_shared_digits": 5.0}}
        self.assertTrue(choose_better_addition_candidate(better, incumbent))
        self.assertFalse(choose_better_addition_candidate(worse, incumbent))

    def test_transport_selection_supports_custom_experiment_policy(self) -> None:
        def summarize(records: list[dict[str, object]]) -> dict[str, float]:
            record = records[0]
            return {"score": float(record["score"]), "agreement": float(record["agreement"])}

        def choose(candidate: dict[str, object], incumbent: dict[str, object] | None) -> bool:
            if incumbent is None:
                return True
            return float(candidate["selection_metrics"]["score"]) > float(incumbent["selection_metrics"]["score"])

        def fake_eval(**kwargs):
            variable = kwargs["target_vars"][0]
            top_k = int(kwargs["top_k_by_variable"][variable])
            strength = float(kwargs["lambda_by_variable"][variable])
            score = 0.9 if (top_k, strength) == (2, 0.5) else 0.2
            return ([{"variable": variable, "score": score, "agreement": 1.0 - score}], {})

        config = OTConfig(
            method="ot",
            top_k_values=(1, 2),
            lambda_values=(0.5,),
            target_vars=("alpha",),
            selection_verbose=False,
        )
        with patch("experiment_core.ot.evaluate_soft_transport_interventions", side_effect=fake_eval):
            payload = select_transport_hyperparameters(
                method_name="ot",
                model=object(),
                calibration_bank=object(),
                sites=[object(), object()],
                normalized_transport=np.array([[0.6, 0.4]], dtype=np.float64),
                rankings={"alpha": [{"site_label": "site0"}]},
                target_vars=("alpha",),
                batch_size=1,
                device="cpu",
                config=config,
                metrics_from_logits_fn=lambda *_args, **_kwargs: {},
                summarize_selection_records_fn=summarize,
                choose_better_selection_candidate_fn=choose,
            )

        self.assertEqual(payload["selected_top_k_by_variable"]["alpha"], 2)
        self.assertEqual(payload["selected_lambda_by_variable"]["alpha"], 0.5)
        self.assertAlmostEqual(payload["selected_selection_metrics_by_variable"]["alpha"]["score"], 0.9)
        self.assertAlmostEqual(payload["average_selection_metrics"]["score"], 0.9)

    def test_das_selection_supports_custom_experiment_policy(self) -> None:
        def summarize(records: list[dict[str, object]]) -> dict[str, float]:
            record = records[0]
            return {"score": float(record["score"]), "agreement": float(record["agreement"])}

        def choose(candidate: dict[str, object], incumbent: dict[str, object] | None) -> bool:
            if incumbent is None:
                return True
            return float(candidate["selection_metrics"]["score"]) > float(incumbent["selection_metrics"]["score"])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "mini_model.pt")
            problem, model, _ = load_small_model(checkpoint_path)
            train_bank = build_pair_bank(problem, size=4, seed=101, split="train")
            calibration_bank = build_pair_bank(problem, size=4, seed=102, split="calibration")
            holdout_bank = build_pair_bank(problem, size=4, seed=103, split="test")

            specs = [
                DASSearchSpec(layer=0, subspace_dim=1, component="h[0].output"),
                DASSearchSpec(layer=0, subspace_dim=2, component="h[0].output"),
            ]
            with patch("experiment_core.das.iter_search_specs", return_value=specs), patch(
                "experiment_core.das.build_intervenable", return_value=object()
            ), patch("experiment_core.das.RotatedSpaceIntervention", return_value=object()), patch(
                "experiment_core.das.train_rotated_intervention", return_value=[1.0]
            ), patch(
                "experiment_core.das.evaluate_rotated_intervention",
                side_effect=[
                    {"score": 0.2, "agreement": 0.8},
                    {"score": 0.9, "agreement": 0.1},
                    {"score": 0.9, "agreement": 0.1},
                ],
            ):
                best_record, _ = run_das_search_for_variable(
                    model=model,
                    variable="S1",
                    train_bank=train_bank,
                    calibration_bank=calibration_bank,
                    holdout_bank=holdout_bank,
                    device=model.device,
                    config=DASConfig(
                        batch_size=2,
                        max_epochs=1,
                        learning_rate=1e-3,
                        target_vars=("S1",),
                        verbose=False,
                        random_seed_base=1000,
                    ),
                    metrics_from_logits_fn=lambda *_args, **_kwargs: {},
                    summarize_selection_records_fn=summarize,
                    choose_better_selection_candidate_fn=choose,
                )

        self.assertEqual(best_record["site_label"], "L0-k2")
        self.assertAlmostEqual(best_record["selection_metrics"]["score"], 0.9)
        self.assertAlmostEqual(best_record["calibration_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
