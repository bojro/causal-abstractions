from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_core.seed_sweep import (
    build_seed_sweep_payload,
    format_seed_sweep_summary,
    save_seed_sweep_plots,
)


class SeedSweepUnitTests(unittest.TestCase):
    def test_seed_sweep_aggregation_summarizes_average_metrics(self) -> None:
        payload = build_seed_sweep_payload(
            [
                {
                    "seed": 11,
                    "comparison": {
                        "target_vars": ["S1", "C1"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.91, "num_examples": 4000}
                        },
                        "method_runtime_seconds": {"gw": 12.0, "das": 30.0},
                        "method_summary": [
                            {"method": "gw", "exact_acc": 0.25, "mean_shared_digits": 1.0},
                            {"method": "das", "exact_acc": 0.50, "mean_shared_digits": 1.5},
                        ],
                        "results": [
                            {"method": "gw", "variable": "S1", "exact_acc": 0.20, "mean_shared_digits": 0.9},
                            {"method": "gw", "variable": "C1", "exact_acc": 0.30, "mean_shared_digits": 1.1},
                            {"method": "das", "variable": "S1", "exact_acc": 0.45, "mean_shared_digits": 1.4},
                            {"method": "das", "variable": "C1", "exact_acc": 0.55, "mean_shared_digits": 1.6},
                        ],
                    },
                },
                {
                    "seed": 12,
                    "comparison": {
                        "target_vars": ["S1", "C1"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.95, "num_examples": 4000}
                        },
                        "method_runtime_seconds": {"gw": 16.0, "das": 24.0},
                        "method_summary": [
                            {"method": "gw", "exact_acc": 0.35, "mean_shared_digits": 1.2},
                            {"method": "das", "exact_acc": 0.40, "mean_shared_digits": 1.4},
                        ],
                        "results": [
                            {"method": "gw", "variable": "S1", "exact_acc": 0.40, "mean_shared_digits": 1.3},
                            {"method": "gw", "variable": "C1", "exact_acc": 0.30, "mean_shared_digits": 1.1},
                            {"method": "das", "variable": "S1", "exact_acc": 0.35, "mean_shared_digits": 1.3},
                            {"method": "das", "variable": "C1", "exact_acc": 0.45, "mean_shared_digits": 1.5},
                        ],
                    },
                },
            ]
        )

        self.assertEqual(payload["seeds"], [11, 12])
        self.assertEqual(payload["methods"], ["das", "gw"])
        self.assertEqual(payload["target_vars"], ["S1", "C1"])

        backbone_summary = payload["backbone_factual_validation_summary"]
        self.assertAlmostEqual(backbone_summary["exact_acc_mean"], 0.93)
        self.assertAlmostEqual(backbone_summary["exact_acc_std"], 0.02)

        method_summary = {
            str(record["method"]): record for record in payload["method_summary_across_seeds"]
        }
        self.assertAlmostEqual(method_summary["gw"]["exact_acc_mean"], 0.30)
        self.assertAlmostEqual(method_summary["gw"]["exact_acc_std"], 0.05)
        self.assertAlmostEqual(method_summary["gw"]["runtime_seconds_mean"], 14.0)
        self.assertAlmostEqual(method_summary["gw"]["runtime_seconds_std"], 2.0)
        self.assertAlmostEqual(method_summary["das"]["exact_acc_mean"], 0.45)
        self.assertAlmostEqual(method_summary["das"]["exact_acc_std"], 0.05)
        self.assertAlmostEqual(method_summary["das"]["runtime_seconds_mean"], 27.0)
        self.assertAlmostEqual(method_summary["das"]["runtime_seconds_std"], 3.0)

        per_seed_runtime = {
            (int(record["seed"]), str(record["method"])): float(record["runtime_seconds"])
            for record in payload["per_seed_method_runtime"]
        }
        self.assertEqual(per_seed_runtime[(11, "gw")], 12.0)
        self.assertEqual(per_seed_runtime[(12, "das")], 24.0)

        variable_summary = {
            (str(record["method"]), str(record["variable"])): record
            for record in payload["variable_summary_across_seeds"]
        }
        self.assertAlmostEqual(variable_summary[("gw", "S1")]["exact_acc_mean"], 0.30)
        self.assertAlmostEqual(variable_summary[("gw", "S1")]["exact_acc_std"], 0.10)
        self.assertAlmostEqual(variable_summary[("das", "C1")]["mean_shared_digits_mean"], 1.55)

        summary_text = format_seed_sweep_summary(payload)
        self.assertIn("Per-Variable Summary Across Seeds", summary_text)
        self.assertIn("DAS [S1]: exact=0.4000 +/- 0.0500, shared=1.3500 +/- 0.0500", summary_text)
        self.assertIn("GW [C1]: exact=0.3000 +/- 0.0000, shared=1.1000 +/- 0.0000", summary_text)

    def test_seed_sweep_aggregation_respects_declared_core_metrics(self) -> None:
        payload = build_seed_sweep_payload(
            [
                {
                    "seed": 1,
                    "comparison": {
                        "core_metrics": ["score", "agreement"],
                        "target_vars": ["alpha"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.80, "num_examples": 100}
                        },
                        "method_runtime_seconds": {"demo": 5.0},
                        "method_summary": [
                            {"method": "demo", "score": 0.2, "agreement": 0.7},
                        ],
                        "results": [
                            {"method": "demo", "variable": "alpha", "score": 0.2, "agreement": 0.7},
                        ],
                    },
                },
                {
                    "seed": 2,
                    "comparison": {
                        "core_metrics": ["score", "agreement"],
                        "target_vars": ["alpha"],
                        "backbone": {
                            "factual_validation_metrics": {"exact_acc": 0.90, "num_examples": 100}
                        },
                        "method_runtime_seconds": {"demo": 7.0},
                        "method_summary": [
                            {"method": "demo", "score": 0.6, "agreement": 0.5},
                        ],
                        "results": [
                            {"method": "demo", "variable": "alpha", "score": 0.6, "agreement": 0.5},
                        ],
                    },
                },
            ]
        )

        self.assertEqual(payload["core_metrics"], ["score", "agreement"])
        method_summary = payload["method_summary_across_seeds"][0]
        self.assertAlmostEqual(method_summary["score_mean"], 0.4)
        self.assertAlmostEqual(method_summary["agreement_mean"], 0.6)
        variable_summary = payload["variable_summary_across_seeds"][0]
        self.assertAlmostEqual(variable_summary["score_std"], 0.2)
        self.assertAlmostEqual(variable_summary["agreement_std"], 0.1)

        summary_text = format_seed_sweep_summary(payload)
        self.assertIn("DEMO: score=0.4000 +/- 0.2000, agreement=0.6000 +/- 0.1000", summary_text)
        self.assertIn("DEMO [alpha]: score=0.4000 +/- 0.2000, agreement=0.6000 +/- 0.1000", summary_text)

    def test_seed_sweep_plot_paths_are_prefixed_and_files_exist(self) -> None:
        payload = {
            "experiment_id": "demo_experiment",
            "seeds": [1, 2],
            "methods": ["demo"],
            "target_vars": ["alpha"],
            "resolved_device": "cpu",
            "core_metrics": ["exact_acc"],
            "method_summary_across_seeds": [
                {
                    "method": "demo",
                    "exact_acc_mean": 0.5,
                    "exact_acc_std": 0.1,
                    "runtime_seconds_mean": 2.0,
                    "runtime_seconds_std": 0.5,
                }
            ],
            "variable_summary_across_seeds": [
                {
                    "method": "demo",
                    "variable": "alpha",
                    "exact_acc_mean": 0.5,
                    "exact_acc_std": 0.1,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "demo_seed_sweep_results.json"
            plot_paths = save_seed_sweep_plots(payload, output_path)

            self.assertEqual(Path(plot_paths["average_exact_summary"]).name, "demo_seed_sweep_average_exact_summary.png")
            self.assertEqual(Path(plot_paths["variable_exact_summary"]).name, "demo_seed_sweep_variable_exact_summary.png")
            self.assertEqual(Path(plot_paths["runtime_summary"]).name, "demo_seed_sweep_runtime_summary.png")
            self.assertTrue(Path(plot_paths["average_exact_summary"]).is_file())
            self.assertTrue(Path(plot_paths["variable_exact_summary"]).is_file())
            self.assertTrue(Path(plot_paths["runtime_summary"]).is_file())


if __name__ == "__main__":
    unittest.main()
