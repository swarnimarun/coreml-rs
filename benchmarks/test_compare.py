"""Tests for the benchmark's comparison and reporting logic."""
import math
import unittest

from compare import compare_outputs, quantile, stats


class ComparisonTests(unittest.TestCase):
    def test_checks_input_order_and_numeric_tolerance(self):
        first = [{"input_id": 0, "output": [0.0, 0.5]}]
        self.assertEqual(compare_outputs(first, [{"input_id": 0, "output": [0.0, 0.5]}]), 0.0)
        with self.assertRaises(RuntimeError):
            compare_outputs(first, [{"input_id": 1, "output": [0.0, 0.5]}])
        with self.assertRaises(RuntimeError):
            compare_outputs(first, [{"input_id": 0, "output": [0.0, 0.6]}])
        with self.assertRaises(RuntimeError):
            compare_outputs(first, [{"input_id": 0, "output": [0.0, math.nan]}])

    def test_percentile_and_units(self):
        self.assertEqual(quantile([1, 3], 0.5), 2)
        summary = stats([{"elapsed_ns": 1_000_000}, {"elapsed_ns": 3_000_000}])
        self.assertEqual(summary["median_ms"], 2)
        self.assertEqual(summary["count"], 2)


if __name__ == "__main__":
    unittest.main()
