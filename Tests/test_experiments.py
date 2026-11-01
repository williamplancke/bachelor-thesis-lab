import unittest

import pandas as pd

from Experiments.common import validate_synthetic_data


class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.real = pd.DataFrame({"text": ["a", "b", "c"], "intent": [0, 0, 1]})
        self.synthetic = pd.DataFrame({
            "text": ["first request", "second request", "third request"],
            "intent": [0, 0, 1],
        })

    def test_matching_counts(self):
        validate_synthetic_data(self.synthetic, self.real)

    def test_wrong_distribution(self):
        self.synthetic["intent"] = [0, 1, 1]
        with self.assertRaises(ValueError):
            validate_synthetic_data(self.synthetic, self.real)

    def test_duplicate_text(self):
        self.synthetic.loc[1, "text"] = " FIRST   REQUEST "
        with self.assertRaises(ValueError):
            validate_synthetic_data(self.synthetic, self.real)

    def test_missing_and_blank_text(self):
        for text in (None, "   "):
            self.synthetic.loc[0, "text"] = text
            with self.assertRaises(ValueError):
                validate_synthetic_data(self.synthetic, self.real)


if __name__ == "__main__":
    unittest.main()
