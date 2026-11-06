from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer
from Architecture.Tokenizers.WordTokenizer import Tokenizer as WordTokenizer
from Experiments import Real_data, Synthetic_data_raw, Synthetic_data_with_definition
from Experiments.common import PARAMS
from Training import train as training


class TokenizerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "text": ["happy customer", "sad customer"], "intent": [0, 1],
        })
        self.validation = pd.DataFrame({
            "text": ["happyhappy", "sad visitor"], "intent": [0, 1],
        })
        self.params = {
            "seed": 42, "sequence_length": 64, "embedding_dim": 4,
            "learning_rate": 0.01, "epochs": 1,
        }

    def test_bpe_preprocessing_uses_merges_without_mutating_training_state(self):
        tokenizer = SubwordBPETokenizer()
        for text in self.train["text"]:
            tokenizer.words_to_characters(tokenizer.prepare_input(text))
        tokenizer.fit()
        before = deepcopy(tokenizer.__dict__)
        for text in ["happy customer", *self.validation["text"]]:
            self.assertEqual(tokenizer.encode(text), tokenizer.encode(tokenizer.prepare_input(text)))
        self.assertEqual(tokenizer.__dict__, before)
        self.assertEqual(len(tokenizer.encode(tokenizer.prepare_input("happy customer"))), 2)

    def test_training_with_each_tokenizer_and_legacy_defaults(self):
        for name in (None, "word", "subword-bpe"):
            with self.subTest(tokenizer=name), TemporaryDirectory() as directory:
                params = dict(self.params)
                if name is not None:
                    params.update(tokenizer_instance=name, tokenizer_params={})
                tokenizer = training.get_tokenizer(name or "word")
                with patch.object(training, "get_tokenizer", return_value=tokenizer), redirect_stdout(StringIO()):
                    run = training.main(
                        self.train, self.validation, params,
                        run_dir=Path(directory) / "run", label_col_name="intent",
                    )
                config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(config["tokenizer_instance"], name or "word")
                self.assertTrue((run / "model.pt").is_file())
                self.assertTrue((run / "metrics.json").is_file())
                if name == "subword-bpe":
                    self.assertEqual(tokenizer.word_frequency_table, {
                        "_happy": 1, "_customer": 2, "_sad": 1,
                    })
                else:
                    self.assertTrue(tokenizer.frozen_vocabulary)
                    self.assertNotIn("visitor", tokenizer.vocabulary)

    def test_factory_accepts_names_and_classes_and_rejects_invalid_values(self):
        for name, cls in (("word", WordTokenizer), ("subword-bpe", SubwordBPETokenizer)):
            self.assertIsInstance(training.get_tokenizer(name), cls)
            self.assertIsInstance(training.get_tokenizer(cls), cls)
        for value in ("unknown", None, WordTokenizer()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training.get_tokenizer(value)

    def test_class_selection_can_be_saved_as_json(self):
        params = {**self.params, "tokenizer_instance": SubwordBPETokenizer}
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            run = training.main(
                self.train, self.validation, params,
                run_dir=Path(directory) / "run", label_col_name="intent",
            )
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["tokenizer_instance"], "subword-bpe")

    def test_experiment_entry_points_with_tiny_data(self):
        # Exercise each wrapper and the real trainer, without full CLINC150 runs.
        for module, path_name in (
            (Real_data, None),
            (Synthetic_data_raw, "SYNTHETIC_PATH"),
            (Synthetic_data_with_definition, "SYNTHETIC_WITH_DEFINITION_PATH"),
        ):
            with self.subTest(experiment=module.__name__), TemporaryDirectory() as directory:
                root = Path(directory)
                synthetic_path = root / "synthetic.parquet"
                self.train.to_parquet(synthetic_path, index=False)
                params = {**PARAMS, "epochs": 1, "embedding_dim": 4}

                def train_tiny(train, validation, experiment_params, label_col_name):
                    return training.main(
                        train, validation, experiment_params,
                        run_dir=root / "run", label_col_name=label_col_name,
                    )

                with patch.object(module, "load_training_data", return_value=(self.train, self.validation)), \
                     patch.object(module, "PARAMS", params), \
                     patch.object(module, "train", side_effect=train_tiny), \
                     redirect_stdout(StringIO()):
                    if path_name is None:
                        run = module.main()
                    else:
                        with patch.object(module, path_name, synthetic_path):
                            run = module.main()
                self.assertTrue((run / "config.json").is_file())


if __name__ == "__main__":
    unittest.main()
