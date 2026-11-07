"""Tiny CPU tests; no full dataset training or test-split access."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from Experiments import Synthetic_data_finetuned as experiment
from Training.train import main as train
from Training.load import load_run


class FinetuningTests(unittest.TestCase):
    def setUp(self):
        self.original = pd.DataFrame({"text": ["hello customer", "goodbye customer"], "intent": [10, 30]})
        self.supplement = pd.DataFrame({"text": ["hi visitor", "bye visitor"], "intent": [10, 30]})
        self.params = {"seed": 42, "sequence_length": 32, "embedding_dim": 4,
                       "learning_rate": 0.01, "epochs": 1, "device": "cpu"}

    def test_replay_is_reproducible_and_keeps_every_original(self):
        mixture, info = experiment.replay_mixture(self.original, self.supplement, fraction=0.5)
        other, _ = experiment.replay_mixture(self.original, self.supplement, fraction=0.5)
        pd.testing.assert_frame_equal(mixture, other)
        pd.testing.assert_frame_equal(mixture.iloc[:2], self.original)
        self.assertEqual(info["actual_supplement_fraction"], 0.5)
        self.assertEqual(len(mixture), 4)

    def test_reject_bad_supplements(self):
        for supplement in (self.supplement.iloc[:0], self.supplement.assign(intent=99),
                           self.supplement.assign(text=" "), self.original,
                           pd.concat([self.supplement, self.supplement])):
            with self.subTest(supplement=supplement), self.assertRaises(ValueError):
                experiment.replay_mixture(self.original, supplement)
        for fraction in (0, 1, -1):
            with self.assertRaises(ValueError):
                experiment.replay_mixture(self.original, self.supplement, fraction)

    def test_checkpoint_preserved_then_all_weights_can_update(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            base = train(self.original, self.original, self.params, root / "base", "intent")
            before = load_run(base)
            # Zero updates proves initialization restores exactly the saved weights.
            restored = train(self.supplement, self.original, {**self.params, "epochs": 0},
                             root / "restored", "intent", checkpoint=base)
            restored_model = load_run(restored)
            for key, value in before.model.state_dict().items():
                torch.testing.assert_close(value, restored_model.model.state_dict()[key], rtol=0, atol=0)
            mixture, _ = experiment.replay_mixture(self.original, self.supplement, 0.5)
            final = train(mixture, self.original, {**self.params, "learning_rate": 0.001},
                          root / "final", "intent", checkpoint=base)
            after = load_run(final)
            self.assertEqual(before.labels, after.labels)
            self.assertEqual((base / "tokenizer.json").read_bytes(), (final / "tokenizer.json").read_bytes())
            self.assertNotIn("visitor", after.tokenizer.vocabulary)
            for key, value in before.model.state_dict().items():
                self.assertFalse(torch.equal(value, after.model.state_dict()[key]), key)
            self.assertEqual(after.config["trainable_parameters"], after.config["total_parameters"])
            self.assertEqual(after.config["parent_checkpoint"], str(base.resolve()))
            self.assertEqual(after.config["training_updates"], 4)
            self.assertEqual(after.config["learning_rate"], 0.001)
            with self.assertRaises(ValueError):
                train(self.supplement.assign(intent=99), self.original, self.params,
                      root / "bad", "intent", checkpoint=base)

    def test_two_stage_entrypoint(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            original_path, supplement_path = root / "original.parquet", root / "supplement.parquet"
            self.original.to_parquet(original_path)
            self.supplement.to_parquet(supplement_path)
            calls = []

            def train_tiny(frame, validation, params, **kwargs):
                calls.append(params)
                return train(frame, validation, params, run_dir=root / f"stage{len(calls)}", **kwargs)

            with patch.object(experiment, "load_training_data", return_value=(self.original, self.original)), \
                 patch.object(experiment, "SYNTHETIC_WITH_DEFINITION_PATH", original_path), \
                 patch.object(experiment, "SUPPLEMENT_PATH", supplement_path), \
                 patch.object(experiment, "PARAMS", {**self.params, "architecture": "transformer-encoder",
                                                       "architecture_params": {"dropout": 0.0}}), \
                 patch.object(experiment, "train", side_effect=train_tiny), \
                 self.assertWarns(UserWarning):
                final = experiment.main()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]["learning_rate"], calls[0]["learning_rate"] / 10)
            config = json.loads((final / "config.json").read_text())
            self.assertEqual(config["training_stage"], "full_finetuning")
            self.assertEqual(config["parent_checkpoint"], str((root / "stage1").resolve()))


if __name__ == "__main__":
    unittest.main()
