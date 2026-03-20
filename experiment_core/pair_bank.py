"""Generic pair-bank containers shared across experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairBank:
    """Shared base/source pair split with factual and counterfactual labels."""

    split: str
    seed: int
    base_digits: torch.Tensor
    source_digits: torch.Tensor
    base_inputs: torch.Tensor
    source_inputs: torch.Tensor
    base_labels: torch.Tensor
    cf_labels_by_var: dict[str, torch.Tensor]
    target_vars: tuple[str, ...]

    @property
    def size(self) -> int:
        """Return the number of base/source pairs in the bank."""
        return int(self.base_inputs.shape[0])

    def metadata(self) -> dict[str, object]:
        """Return a compact summary of the pair-bank split."""
        return {
            "split": self.split,
            "seed": self.seed,
            "size": self.size,
            "target_vars": list(self.target_vars),
        }


class PairBankVariableDataset(Dataset):
    """Dataset view exposing one abstract variable's counterfactual labels."""

    def __init__(self, bank: PairBank, variable_name: str):
        """Wrap one abstract variable view of a shared pair bank for DAS."""
        if variable_name not in bank.cf_labels_by_var:
            raise KeyError(f"Unknown variable {variable_name}")
        self.bank = bank
        self.variable_name = variable_name
        self.variable_index = bank.target_vars.index(variable_name)

    def __len__(self) -> int:
        """Return the number of examples exposed by this dataset view."""
        return self.bank.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch one base/source intervention example for the chosen variable."""
        return {
            "input_ids": self.bank.base_inputs[index],
            "source_input_ids": self.bank.source_inputs[index],
            "labels": self.bank.cf_labels_by_var[self.variable_name][index],
            "base_labels": self.bank.base_labels[index],
            "intervention_id": torch.tensor(self.variable_index, dtype=torch.long),
        }
