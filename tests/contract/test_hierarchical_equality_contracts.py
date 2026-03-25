from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_core.compare_runner import run_comparison_with_model
from experiments.hierarchical_equality.spec import build_hierarchical_equality_adapter

from tests.common import load_small_hierarchical_equality_model, small_compare_config


class HierarchicalEqualityContractTests(unittest.TestCase):
    def test_compare_runner_writes_required_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mini_model.pt"
            problem, model, backbone_meta = load_small_hierarchical_equality_model(str(checkpoint_path))
            adapter = build_hierarchical_equality_adapter(target_vars=("WX", "YZ"))
            config = small_compare_config(
                Path(temp_dir),
                checkpoint_path,
                methods=("gw", "das"),
                target_vars=("WX", "YZ"),
            )

            payload = run_comparison_with_model(
                problem=problem,
                adapter=adapter,
                model=model,
                backbone_meta=backbone_meta,
                device="cpu",
                config=config,
            )

            self.assertEqual(payload["contract_version"], 1)
            self.assertEqual(payload["experiment_id"], "hierarchical_equality")
            self.assertEqual(payload["core_metrics"], ["exact_acc", "mean_true_class_prob"])
            self.assertEqual(payload["canonical_variable_mapping"], {"WX": "WX", "YZ": "YZ"})
            self.assertEqual(payload["canonical_target_vars"], ["WX", "YZ"])
            self.assertEqual(payload["target_vars"], ["WX", "YZ"])
            self.assertIn("method_runtime", payload["plots"])
            self.assertTrue(Path(payload["plots"]["method_runtime"]).resolve().is_file())
            self.assertEqual(Path(payload["plots"]["method_runtime"]).name, "compare_method_runtime.png")
            self.assertEqual(
                Path(payload["plots"]["mean_gold_class_probability"]).name,
                "compare_mean_gold_class_probability.png",
            )

            for record in payload["results"]:
                self.assertIn("method_id", record)
                self.assertIn("local_variable", record)
                self.assertIn("canonical_variable", record)
                self.assertIn("exact_acc", record)
                self.assertIn("mean_true_class_prob", record)


if __name__ == "__main__":
    unittest.main()
