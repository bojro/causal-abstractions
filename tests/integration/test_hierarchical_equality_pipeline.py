from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from experiment_core import _env  # noqa: F401
from experiment_core.das import DASConfig, run_das_pipeline
from experiment_core.ot import OTConfig, run_alignment_pipeline
from experiment_core.pyvene_utils import (
    build_intervenable,
    enumerate_canonical_sites,
    run_intervenable_logits,
)
from pyvene import RotatedSpaceIntervention, VanillaIntervention

from experiments.hierarchical_equality.metrics import metrics_from_logits
from experiments.hierarchical_equality.pair_bank import build_pair_bank
from experiments.hierarchical_equality.scm import load_hierarchical_equality_problem
from experiments.hierarchical_equality.selection import (
    choose_better_hierarchical_equality_candidate,
    summarize_hierarchical_equality_selection_records,
)

from tests.common import load_small_hierarchical_equality_model


class HierarchicalEqualityPipelineIntegrationTests(unittest.TestCase):
    def test_problem_checks_and_pair_bank_determinism(self) -> None:
        problem = load_hierarchical_equality_problem(run_checks=True)
        bank_a = build_pair_bank(problem, size=16, seed=123, split="test", verify_with_scm=True)
        bank_b = build_pair_bank(problem, size=16, seed=123, split="test")

        self.assertTrue(torch.equal(bank_a.base_digits, bank_b.base_digits))
        self.assertTrue(torch.equal(bank_a.source_digits, bank_b.source_digits))
        self.assertTrue(torch.equal(bank_a.base_inputs, bank_b.base_inputs))
        self.assertTrue(torch.equal(bank_a.source_inputs, bank_b.source_inputs))
        self.assertTrue(torch.equal(bank_a.base_labels, bank_b.base_labels))
        self.assertEqual(bank_a.target_vars, ("WX", "YZ"))
        for variable in bank_a.target_vars:
            self.assertTrue(torch.equal(bank_a.cf_labels_by_var[variable], bank_b.cf_labels_by_var[variable]))

    def test_pyvene_intervention_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "mini_model.pt")
            problem, model, _ = load_small_hierarchical_equality_model(checkpoint_path)
            bank = build_pair_bank(problem, size=8, seed=99, split="smoke")
            site = enumerate_canonical_sites(model, resolution=1)[0]

            vanilla = build_intervenable(
                model=model,
                layer=site.layer,
                component=site.component,
                intervention=VanillaIntervention(),
                device="cpu",
                freeze_model=True,
                freeze_intervention=True,
            )
            vanilla_logits = run_intervenable_logits(
                intervenable=vanilla,
                base_inputs=bank.base_inputs,
                source_inputs=bank.source_inputs,
                subspace_dims=site.subspace_dims,
                position=site.position,
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(tuple(vanilla_logits.shape), (8, 2))
            self.assertTrue(torch.isfinite(vanilla_logits).all().item())

            rotated = build_intervenable(
                model=model,
                layer=0,
                component="h[0].output",
                intervention=RotatedSpaceIntervention(embed_dim=int(model.config.hidden_dims[0])),
                device="cpu",
                freeze_model=True,
                freeze_intervention=False,
            )
            rotated_logits = run_intervenable_logits(
                intervenable=rotated,
                base_inputs=bank.base_inputs,
                source_inputs=bank.source_inputs,
                subspace_dims=[0],
                position=0,
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(tuple(rotated_logits.shape), (8, 2))
            self.assertTrue(torch.isfinite(rotated_logits).all().item())

    def test_small_compare_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "mini_model.pt")
            problem, model, _ = load_small_hierarchical_equality_model(checkpoint_path)
            train_bank = build_pair_bank(problem, size=16, seed=1001, split="train")
            calibration_bank = build_pair_bank(problem, size=16, seed=1002, split="calibration")
            test_bank = build_pair_bank(problem, size=16, seed=1003, split="test")

            gw_payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=OTConfig(
                    method="gw",
                    batch_size=8,
                    ranking_k=2,
                    resolution=2,
                    target_vars=train_bank.target_vars,
                    top_k_values=(1, 2, 4),
                    lambda_values=(0.5, 1.0),
                    selection_verbose=False,
                ),
                metrics_from_logits_fn=metrics_from_logits,
                summarize_selection_records_fn=summarize_hierarchical_equality_selection_records,
                choose_better_selection_candidate_fn=choose_better_hierarchical_equality_candidate,
            )
            das_payload = run_das_pipeline(
                model=model,
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device="cpu",
                config=DASConfig(
                    batch_size=8,
                    max_epochs=1,
                    learning_rate=1e-3,
                    subspace_dims=(1,),
                    search_layers=(0,),
                    target_vars=train_bank.target_vars,
                    verbose=False,
                ),
                metrics_from_logits_fn=metrics_from_logits,
                summarize_selection_records_fn=summarize_hierarchical_equality_selection_records,
                choose_better_selection_candidate_fn=choose_better_hierarchical_equality_candidate,
            )

        self.assertEqual(len(gw_payload["results"]), 2)
        self.assertEqual(len(das_payload["results"]), 2)
        self.assertEqual(gw_payload["site_selection"]["site_policy"], "current")
        self.assertEqual(gw_payload["method_id"], "gw_current_res2_cosine")
        self.assertEqual(das_payload["method_id"], "das")
        for record in gw_payload["results"] + das_payload["results"]:
            self.assertIn("exact_acc", record)
            self.assertIn("mean_true_class_prob", record)


if __name__ == "__main__":
    unittest.main()
