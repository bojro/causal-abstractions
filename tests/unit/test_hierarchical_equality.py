from __future__ import annotations

import unittest

import torch

from experiment_core.metrics import validate_core_metric_records, validate_core_metrics
from experiments.hierarchical_equality.backbone import build_factual_tensors
from experiments.hierarchical_equality.metrics import metrics_from_logits
from experiments.hierarchical_equality.pair_bank import build_pair_bank
from experiments.hierarchical_equality.scm import (
    compute_counterfactual_labels,
    compute_states_for_inputs,
    load_hierarchical_equality_problem,
    sample_structured_examples,
)
from experiments.hierarchical_equality.selection import (
    choose_better_hierarchical_equality_candidate,
    summarize_hierarchical_equality_selection_records,
)


def _pack_rows(rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([torch.cat(parts, dim=0) for parts in rows], dim=0).to(torch.float32)


class HierarchicalEqualityUnitTests(unittest.TestCase):
    def test_factual_label_logic_on_representative_cases(self) -> None:
        a = torch.tensor([1.0, 0.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0, 0.0])
        c = torch.tensor([0.0, 0.0, 1.0, 0.0])
        d = torch.tensor([0.0, 0.0, 0.0, 1.0])
        inputs = _pack_rows(
            [
                (a, a, c, c),  # both equal -> 1
                (a, b, c, d),  # both different -> 1
                (a, a, c, d),  # mixed -> 0
                (a, b, c, c),  # mixed -> 0
            ]
        )
        states = compute_states_for_inputs(inputs)
        self.assertEqual(states["WX"].tolist(), [1, 0, 1, 0])
        self.assertEqual(states["YZ"].tolist(), [1, 0, 0, 1])
        self.assertEqual(states["O"].tolist(), [1, 1, 0, 0])

    def test_counterfactual_labels_follow_wx_yz_swap_semantics(self) -> None:
        base_states = {"WX": torch.tensor([1, 0]).numpy(), "YZ": torch.tensor([0, 1]).numpy()}
        source_states = {"WX": torch.tensor([0, 1]).numpy(), "YZ": torch.tensor([1, 0]).numpy()}
        labels = compute_counterfactual_labels(base_states, source_states)
        self.assertEqual(labels["WX"].tolist(), [1, 1])
        self.assertEqual(labels["YZ"].tolist(), [1, 1])

    def test_sampling_and_pair_bank_are_deterministic(self) -> None:
        inputs_a, labels_a, structures_a = sample_structured_examples(16, seed=123)
        inputs_b, labels_b, structures_b = sample_structured_examples(16, seed=123)
        self.assertTrue(torch.equal(inputs_a, inputs_b))
        self.assertTrue(torch.equal(labels_a, labels_b))
        self.assertTrue(torch.equal(structures_a, structures_b))

        problem = load_hierarchical_equality_problem(run_checks=False)
        bank_a = build_pair_bank(problem, size=16, seed=77, split="test", verify_with_scm=True)
        bank_b = build_pair_bank(problem, size=16, seed=77, split="test")
        self.assertTrue(torch.equal(bank_a.base_digits, bank_b.base_digits))
        self.assertTrue(torch.equal(bank_a.source_digits, bank_b.source_digits))
        self.assertTrue(torch.equal(bank_a.base_inputs, bank_b.base_inputs))
        self.assertTrue(torch.equal(bank_a.source_inputs, bank_b.source_inputs))
        self.assertTrue(torch.equal(bank_a.base_labels, bank_b.base_labels))
        self.assertEqual(bank_a.target_vars, ("WX", "YZ"))
        self.assertEqual(set(bank_a.cf_labels_by_var.keys()), {"WX", "YZ"})

    def test_build_factual_tensors_shapes_and_label_range(self) -> None:
        problem = load_hierarchical_equality_problem(run_checks=False)
        inputs, labels = build_factual_tensors(problem, size=12, seed=9)
        self.assertEqual(tuple(inputs.shape), (12, 16))
        self.assertEqual(tuple(labels.shape), (12,))
        self.assertTrue(set(labels.tolist()).issubset({0, 1}))

    def test_metrics_and_core_metric_contract_match_spec(self) -> None:
        problem = load_hierarchical_equality_problem(run_checks=False)
        logits = torch.tensor(
            [
                [4.0, 0.1],
                [0.2, 3.5],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 1], dtype=torch.long)
        metrics = metrics_from_logits(logits, labels)
        self.assertIn("exact_acc", metrics)
        self.assertIn("mean_true_class_prob", metrics)
        validate_core_metrics(metrics, problem.experiment_spec)
        validate_core_metric_records([metrics], problem.experiment_spec)

    def test_selection_policy_is_lexicographic(self) -> None:
        summary = summarize_hierarchical_equality_selection_records(
            [
                {"exact_acc": 0.25, "mean_true_class_prob": 0.55},
                {"exact_acc": 0.75, "mean_true_class_prob": 0.85},
            ]
        )
        self.assertAlmostEqual(summary["exact_acc"], 0.5)
        self.assertAlmostEqual(summary["mean_true_class_prob"], 0.7)

        incumbent = {"selection_metrics": {"exact_acc": 0.5, "mean_true_class_prob": 0.6}}
        better = {"selection_metrics": {"exact_acc": 0.5, "mean_true_class_prob": 0.7}}
        worse = {"selection_metrics": {"exact_acc": 0.4, "mean_true_class_prob": 0.99}}
        tied = {"selection_metrics": {"exact_acc": 0.5, "mean_true_class_prob": 0.6}}
        self.assertTrue(choose_better_hierarchical_equality_candidate(better, incumbent))
        self.assertFalse(choose_better_hierarchical_equality_candidate(worse, incumbent))
        self.assertFalse(choose_better_hierarchical_equality_candidate(tied, incumbent))


if __name__ == "__main__":
    unittest.main()
