import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd


# Load the script by path because its parent folder contains a space.
script = Path(__file__).resolve().parents[1] / "Scripts" / "Data generation" / "intent_samples.py"
spec = importlib.util.spec_from_file_location("intent_samples", script)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class GenerationTests(unittest.TestCase):
    def test_main_saves_original_intent_id(self):
        params = {"example_num": 2, "intent_label": "accept_reservations", "temperature": 0.8, "seed": 42}
        content = json.dumps({"examples": [" Do you take reservations? ", "Is booking ahead possible?"]})
        response = SimpleNamespace(message=SimpleNamespace(content=content))
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", return_value=response):
            path = generator.main(params, output_dir=temp)
            df = pd.read_parquet(path)
            self.assertEqual(list(df.columns), ["text", "intent"])
            self.assertEqual(df["intent"].tolist(), [9, 9])
            self.assertEqual(df.iloc[0]["text"], "Do you take reservations?")

    def test_rejects_invalid_examples_without_writing(self):
        invalid = [
            [],
            {"examples": "not a list"},
            {"examples": ["only one"]},
            {"examples": ["valid", " "]},
            {"examples": ["valid", 123]},
            {"examples": ["Same message", " SAME   MESSAGE "]},
        ]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "batch.parquet"
            for value in invalid:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    generator.save_batch(json.dumps(value), 9, 2, path)
                self.assertFalse(path.exists())

    def test_existing_batch_is_not_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "batch.parquet"
            content = json.dumps({"examples": ["A request"]})
            generator.save_batch(content, 9, 1, path)
            with self.assertRaises(FileExistsError):
                generator.save_batch(content, 9, 1, path)


if __name__ == "__main__":
    unittest.main()
