from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from Training import train as training
from Training.runs import run_name, staged_run


class RunStorageTests(unittest.TestCase):
    def test_readable_deterministic_name(self):
        config = {
            "experiment": "synthetic_data_raw", "tokenizer_instance": "subword-bpe",
            "vocabulary_size": 500, "embedding_dim": 64, "seed": 42,
        }
        self.assertEqual(
            run_name(config, timestamp="20261106_120000_123456"),
            "synthetic-data-raw__subword-bpe-v500__emb64__seed42__20261106_120000_123456",
        )
        config["experiment"] = "../Experiment: invalid/name"
        name = run_name(config)
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn(":", name)

    def test_publish_only_after_success(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with staged_run(run) as staging:
                (staging / "config.json").write_text("{}", encoding="utf-8")
                self.assertFalse(run.exists())
            self.assertEqual((run / "config.json").read_text(), "{}")
            self.assertEqual(list(Path(directory).iterdir()), [run])

    def test_failed_and_interrupted_save_clean_up_partial_artifacts(self):
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error), TemporaryDirectory() as directory:
                run = Path(directory) / "run"
                with self.assertRaises(error):
                    with staged_run(run) as staging:
                        (staging / "partial.pt").write_bytes(b"partial")
                        raise error("save interrupted")
                self.assertFalse(run.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_run_is_never_overwritten(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            sentinel = run / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with staged_run(run):
                    self.fail("Existing runs must be rejected before saving.")
            self.assertEqual(sentinel.read_text(), "keep")

    def test_destination_created_during_save_is_preserved(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with self.assertRaises(FileExistsError):
                with staged_run(run):
                    run.mkdir()
                    (run / "keep.txt").write_text("keep", encoding="utf-8")
            self.assertEqual((run / "keep.txt").read_text(), "keep")
            self.assertEqual(list(Path(directory).iterdir()), [run])


class TrainingRunStorageTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"text": ["hello", "goodbye"], "label": [0, 1]})
        self.params = {
            "seed": 42, "sequence_length": 16, "embedding_dim": 4,
            "learning_rate": 0.01, "epochs": 1, "experiment": "tiny_baseline",
        }

    def test_automatic_name_and_complete_artifacts(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            # Keep automatic Runs/ placement within the temporary project root.
            with patch.object(training, "__file__", str(root / "Training" / "train.py")):
                run = training.main(self.data, self.data, self.params)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(run.parent, root / "Runs")
            self.assertTrue(run.name.startswith(
                f"tiny-baseline__word-v{config['vocabulary_size']}__emb4__seed42__"
            ))
            self.assertEqual(
                {path.name for path in run.iterdir()},
                {"config.json", "metrics.json", "model.pt", "tokenizer.json"},
            )
            self.assertEqual(list(run.parent.iterdir()), [run])

    def test_interrupted_training_does_not_create_run_folder(self):
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error), TemporaryDirectory() as directory:
                run = Path(directory) / "run"
                with patch.object(training.IntentClassifier, "forward", side_effect=error("interrupted")), \
                     redirect_stdout(StringIO()), self.assertRaises(error):
                    training.main(self.data, self.data, self.params, run_dir=run)
                self.assertFalse(run.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_failed_artifact_write_leaves_no_run_or_staging_folder(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            with patch.object(training.WordTokenizer, "save", side_effect=OSError("disk full")), \
                 redirect_stdout(StringIO()), self.assertRaises(OSError):
                training.main(self.data, self.data, self.params, run_dir=run)
            self.assertFalse(run.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
