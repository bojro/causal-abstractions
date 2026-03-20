from __future__ import annotations

import unittest

import numpy as np

from experiment_core.contracts import build_transport_method_id
from experiment_core.ot import truncate_transport_rows, validate_transport_solution


class TransportUtilsUnitTests(unittest.TestCase):
    def test_transport_row_truncation_can_renormalize_per_row(self) -> None:
        transport = np.array(
            [
                [0.6, 0.3, 0.1, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        truncated = truncate_transport_rows(transport, [2, 3], renormalize=True)

        np.testing.assert_allclose(truncated[0], np.array([2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(truncated[1], np.array([4.0 / 9.0, 3.0 / 9.0, 2.0 / 9.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(truncated.sum(axis=1), np.ones(2), atol=1e-8)

    def test_transport_method_id_includes_pca_metadata_when_enabled(self) -> None:
        method_id = build_transport_method_id(
            "ot",
            site_policy="pca",
            resolution=2,
            geometry_metric="cosine",
            pca_components=4,
            pca_candidate_count=32,
        )
        self.assertEqual(method_id, "ot_pca_res2_cosine_pc4_keep32")

    def test_degenerate_transport_solution_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "transport solution unusable"):
            validate_transport_solution(
                "gw",
                np.zeros((2, 4), dtype=np.float64),
                {"method": "gw_degenerate"},
            )


if __name__ == "__main__":
    unittest.main()
