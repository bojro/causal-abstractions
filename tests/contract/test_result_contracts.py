from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment_core.compare_runner import run_comparison_with_model
from experiments.addition.spec import build_addition_adapter

from tests.common import load_small_model, small_compare_config


class ResultContractTests(unittest.TestCase):
    def test_compare_runner_writes_required_contract_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mini_model.pt"
            problem, model, backbone_meta = load_small_model(str(checkpoint_path))
            adapter = build_addition_adapter(target_vars=("S1", "C1"))
            config = small_compare_config(
                Path(temp_dir),
                checkpoint_path,
                methods=("gw", "das"),
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
            self.assertEqual(payload["experiment_id"], "addition")
            self.assertEqual(payload["core_metrics"], ["exact_acc", "mean_shared_digits"])
            self.assertEqual(payload["canonical_variable_mapping"], {"S1": "S1", "C1": "C1"})
            self.assertEqual(payload["canonical_target_vars"], ["S1", "C1"])
            self.assertEqual(payload["seed_trace"]["main_seed"], 42)
            self.assertEqual(payload["seed_trace"]["compare_seed"], 143)
            self.assertEqual(payload["seed_trace"]["pair_bank_seeds"]["train"], 243)
            self.assertEqual(payload["seed_trace"]["das_seed_base"], 2042)
            self.assertIn("gw", payload["seed_trace"]["method_execution_seeds"])
            self.assertIn("das", payload["seed_trace"]["method_execution_seeds"])
            self.assertEqual(payload["method_ids"], ["gw_current_res2_cosine", "das"])
            self.assertEqual(payload["requested_device"], None)
            self.assertEqual(payload["resolved_device"], "cpu")
            self.assertIn("packages", payload["environment"])
            self.assertEqual(payload["environment"]["device_resolution"]["resolved"], "cpu")
            self.assertFalse(payload["environment"]["device_resolution"]["used_fallback"])
            self.assertIn("cuda_available", payload["environment"])
            self.assertFalse(payload["environment"]["cuda_available"])
            self.assertIn("method_runtime", payload["plots"])
            self.assertTrue(Path(payload["plots"]["method_runtime"]).resolve().is_file())

            for record in payload["results"]:
                self.assertIn("method_id", record)
                self.assertIn("local_variable", record)
                self.assertIn("canonical_variable", record)
                self.assertIn("exact_acc", record)
                self.assertIn("mean_shared_digits", record)

    def test_ot_pca_policy_persists_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mini_model.pt"
            problem, model, backbone_meta = load_small_model(str(checkpoint_path))
            adapter = build_addition_adapter(target_vars=("S1", "C1"))
            config = small_compare_config(
                Path(temp_dir),
                checkpoint_path,
                methods=("ot",),
                ot_site_policy="pca",
                ot_pca_components=2,
                ot_pca_candidate_count=4,
            )

            payload = run_comparison_with_model(
                problem=problem,
                adapter=adapter,
                model=model,
                backbone_meta=backbone_meta,
                device="cpu",
                config=config,
            )

        ot_payload = payload["method_payloads"]["ot"]
        self.assertEqual(ot_payload["site_selection"]["site_policy"], "pca")
        self.assertEqual(ot_payload["site_selection"]["pca_components"], 2)
        self.assertEqual(ot_payload["site_selection"]["candidate_site_count"], 4)
        self.assertEqual(ot_payload["method_id"], "ot_pca_res2_cosine_pc2_keep4")
        self.assertIn("method_execution_seed", ot_payload)


if __name__ == "__main__":
    unittest.main()
