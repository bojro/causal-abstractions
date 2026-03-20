from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from experiment_core.metrics import validate_core_metric_records, validate_core_metrics
from experiments.addition.metrics import (
    labels_to_digits,
    shared_digit_counts,
)
from experiments.addition.scm import load_addition_problem


class MetricsUnitTests(unittest.TestCase):
    def test_metric_spot_checks(self) -> None:
        self.assertEqual(labels_to_digits(torch.tensor([197, 47])).tolist(), [[1, 9, 7], [0, 4, 7]])
        self.assertEqual(shared_digit_counts(torch.tensor([197]), torch.tensor([197])).item(), 3.0)
        self.assertEqual(shared_digit_counts(torch.tensor([197]), torch.tensor([187])).item(), 2.0)
        self.assertEqual(shared_digit_counts(torch.tensor([47]), torch.tensor([147])).item(), 2.0)

    def test_validate_core_metrics_accepts_required_metrics(self) -> None:
        problem = load_addition_problem(run_checks=False)
        validate_core_metrics(
            {"exact_acc": 0.5, "mean_shared_digits": 1.25},
            problem.experiment_spec,
        )

    def test_validate_core_metrics_rejects_missing_core_metrics(self) -> None:
        problem = load_addition_problem(run_checks=False)
        with self.assertRaises(AssertionError):
            validate_core_metrics({"exact_acc": 0.5}, problem.experiment_spec)

    def test_validate_core_metric_records_uses_experiment_spec_contract(self) -> None:
        problem = load_addition_problem(run_checks=False)
        custom_spec = replace(problem.experiment_spec, core_metrics=("score", "agreement"))

        validate_core_metric_records(
            [{"score": 0.8, "agreement": 0.6}],
            custom_spec,
        )

        with self.assertRaises(AssertionError):
            validate_core_metric_records(
                [{"score": 0.8}],
                custom_spec,
            )


if __name__ == "__main__":
    unittest.main()
