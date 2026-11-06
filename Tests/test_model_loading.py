from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch

from Architecture.Classifier import IntentClassifier, architecture_config, build_model
from Training.load import load_run
from Training.train import main as train


class ArchitectureConfigurationTests(unittest.TestCase):
    def test_legacy_and_explicit_configuration(self):
        self.assertEqual(
            architecture_config({"embedding_dim": 8}),
            ("mean-pooling", {"embedding_dim": 8}),
        )
        self.assertEqual(
            architecture_config({"architecture": " Mean-Pooling ", "architecture_params": {"embedding_dim": 12}}),
            ("mean-pooling", {"embedding_dim": 12}),
        )
        model = build_model(
            "mean-pooling", vocabulary_size=10, number_of_classes=2,
            padding_id=0, architecture_params={"embedding_dim": 8},
        )
        self.assertIsInstance(model, IntentClassifier)
        self.assertEqual(model.embedding.embedding_dim, 8)

    def test_reject_unsupported_architectures_and_conflicting_settings(self):
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            architecture_config({"architecture": "transformer"})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            architecture_config({"embedding_dim": 8, "architecture_params": {"embedding_dim": 12}})
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            build_model(
                "unknown", vocabulary_size=10, number_of_classes=2,
                padding_id=0, architecture_params={"embedding_dim": 8},
            )


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name) / "run"
        frame = pd.DataFrame({"text": ["hello customer", "goodbye customer"], "label": [10, 30]})
        params = {
            "seed": 42, "sequence_length": 16, "learning_rate": 0.01, "epochs": 1,
            "architecture": "mean-pooling", "architecture_params": {"embedding_dim": 4},
        }
        with redirect_stdout(StringIO()):
            train(frame, frame, params, run_dir=self.run)
        self.config_path = self.run / "config.json"
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def write_config(self, config):
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def test_explicit_architecture_params_and_inference_without_metrics_file(self):
        (self.run / "metrics.json").unlink()
        loaded = load_run(self.run, device="cpu")
        self.assertEqual(loaded.config["architecture_params"], {"embedding_dim": 4})
        self.assertEqual(loaded.config["embedding_dim"], 4)
        self.assertEqual(loaded.labels, [10, 30])
        self.assertFalse(loaded.model.training)
        logits = loaded.logits("hello")
        self.assertEqual(logits.shape, (1, 2))
        self.assertFalse(logits.requires_grad)
        self.assertEqual(logits.device.type, "cpu")
        self.assertIn(loaded.predict("hello"), [10, 30])

    def test_legacy_run_loads_with_identical_predictions(self):
        expected = load_run(self.run).logits("hello")
        legacy = dict(self.config)
        for key in ("run_format_version", "architecture", "architecture_params", "labels",
                    "tokenizer_instance", "tokenizer_params", "total_parameters", "trainable_parameters"):
            legacy.pop(key)
        self.write_config(legacy)
        loaded = load_run(self.run)
        self.assertEqual(loaded.labels, [10, 30])
        torch.testing.assert_close(loaded.logits("hello"), expected, rtol=0, atol=0)

    def test_reject_incompatible_artifact_metadata(self):
        for changes, message in (
            ({"architecture": "unknown"}, "Unknown architecture"),
            ({"run_format_version": 99}, "format version"),
            ({"tokenizer_instance": "subword-bpe"}, "type mismatch"),
            ({"vocabulary_size": 999}, "vocabulary size"),
            ({"padding_id": 1}, "padding ID"),
            ({"labels": [10]}, "number of classes"),
            ({"labels": [30, 10]}, "label order"),
        ):
            with self.subTest(changes=changes):
                self.write_config({**self.config, **changes})
                with self.assertRaisesRegex(ValueError, message):
                    load_run(self.run)

    def test_missing_and_incompatible_checkpoints_fail(self):
        checkpoint = self.run / "model.pt"
        torch.save({}, checkpoint)
        with self.assertRaises(RuntimeError):
            load_run(self.run)
        checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            load_run(self.run)


if __name__ == "__main__":
    unittest.main()
