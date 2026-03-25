from __future__ import annotations

import unittest
import warnings
from unittest.mock import patch

from experiment_core.runtime import collect_environment_metadata, resolve_device


class RuntimeUnitTests(unittest.TestCase):
    def test_resolve_device_warns_when_cuda_falls_back_to_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                resolved = resolve_device("cuda")

        self.assertEqual(str(resolved), "cpu")
        self.assertTrue(any("falling back to CPU" in str(warning.message) for warning in caught))

    def test_environment_metadata_records_requested_and_resolved_device(self) -> None:
        metadata = collect_environment_metadata("cpu", requested_device="cuda")

        self.assertEqual(metadata["device"], "cpu")
        self.assertEqual(metadata["requested_device"], "cuda")
        self.assertEqual(metadata["device_resolution"]["requested"], "cuda")
        self.assertEqual(metadata["device_resolution"]["resolved"], "cpu")
        self.assertTrue(metadata["device_resolution"]["used_fallback"])
        self.assertIn("cuda_available", metadata)
        self.assertIn("machine", metadata)
        self.assertFalse(metadata["cuda_available"])


if __name__ == "__main__":
    unittest.main()
