import unittest

import numpy as np

from Training.metrics import classification_metrics, confusion_matrix


class MetricsTests(unittest.TestCase):
    def test_known_example(self):
        matrix = confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 1], 2)
        np.testing.assert_array_equal(matrix, [[1, 1], [0, 3]])
        metrics = classification_metrics(matrix)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["per_class"][0]["precision"], 1.0)
        self.assertAlmostEqual(metrics["per_class"][0]["recall"], 0.5)
        self.assertAlmostEqual(metrics["per_class"][1]["f1"], 6 / 7)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 6 / 7) / 2)
        self.assertEqual(metrics["per_class"][1]["support"], 3)

    def test_absent_class_and_no_correct_predictions(self):
        matrix = confusion_matrix([0, 1], [1, 0], 3)
        metrics = classification_metrics(matrix)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)
        self.assertEqual(metrics["per_class"][2]["support"], 0)

    def test_perfect_predictions(self):
        metrics = classification_metrics(confusion_matrix([0, 1, 2], [0, 1, 2], 3))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0], [], 2)
        with self.assertRaises(ValueError):
            confusion_matrix([0], [-1], 2)
        with self.assertRaises(ValueError):
            confusion_matrix([0], [2], 2)
        with self.assertRaises(ValueError):
            classification_metrics(confusion_matrix([], [], 2))


if __name__ == "__main__":
    unittest.main()
