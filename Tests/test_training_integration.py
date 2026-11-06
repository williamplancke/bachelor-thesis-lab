from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from Architecture.Classifier import IntentClassifier

from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer
from Architecture.Tokenizers.WordTokenizer import Tokenizer as WordTokenizer
from Experiments import Real_data, Synthetic_data_raw, Synthetic_data_with_definition
from Experiments.common import PARAMS
from Training import train as training
from Training.metrics import classification_metrics, confusion_matrix
from Training.data import prepare_input
from Training.load import load_run


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

    @staticmethod
    def prediction_inputs(tokenizer, texts, config):
        """Prepare a small padded batch using the saved run's settings."""
        examples = [
            prepare_input(tokenizer, text, config["sequence_length"], config["padding_id"])
            for text in texts
        ]
        return torch.stack([ids for ids, _ in examples]), torch.stack([mask for _, mask in examples])

    def test_saved_run_reproduces_logits_predictions_and_metrics(self):
        for name in ("word", "subword-bpe"):
            for labels in ((10, 30), ("positive", "negative")):
                with self.subTest(tokenizer=name, labels=labels), TemporaryDirectory() as directory:
                    train = self.train.copy()
                    validation = self.validation.copy()
                    mapping = dict(enumerate(labels))
                    train["intent"] = train["intent"].map(mapping)
                    validation["intent"] = validation["intent"].map(mapping)
                    tokenizer = training.get_tokenizer(name)
                    params = {**self.params, "tokenizer_instance": name, "tokenizer_params": {}}
                    captured = {}

                    def create_model(**kwargs):
                        model = IntentClassifier(**kwargs)
                        captured["model"] = model
                        captured["initial_weights"] = {
                            key: value.clone() for key, value in model.state_dict().items()
                        }
                        return model

                    # Keep the trained objects solely to compare them with fresh instances.
                    # Optimizer, training, and all file writes remain real.
                    with patch.object(training, "get_tokenizer", return_value=tokenizer), \
                         patch("Architecture.Classifier.IntentClassifier", side_effect=create_model), \
                         redirect_stdout(StringIO()):
                        run = training.main(
                            train, validation, params,
                            run_dir=Path(directory) / "run", label_col_name="intent",
                        )
                    original_model = captured["model"]
                    self.assertFalse(original_model.training)
                    self.assertTrue(any(
                        not torch.equal(value, captured["initial_weights"][key])
                        for key, value in original_model.state_dict().items()
                    ), "The smoke test must perform an actual optimization step.")

                    # Reconstruct from artifacts only, not from params or training data.
                    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                    saved_metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                    loaded = load_run(run)
                    loaded_tokenizer = loaded.tokenizer
                    loaded_model = loaded.model
                    self.assertFalse(loaded_model.training)
                    self.assertEqual(loaded.labels, sorted(labels))
                    self.assertEqual(config["architecture"], "mean-pooling")
                    self.assertEqual(config["architecture_params"], {"embedding_dim": 4})
                    parameter_count = sum(value.numel() for value in loaded_model.parameters())
                    self.assertEqual(config["total_parameters"], parameter_count)
                    self.assertEqual(config["trainable_parameters"], parameter_count)
                    if name == "subword-bpe":
                        self.assertEqual(loaded_tokenizer.word_symbols, {})
                    for key, value in original_model.state_dict().items():
                        self.assertTrue(torch.equal(value, loaded_model.state_dict()[key]))

                    # Include training text, unseen combinations/characters, and empty input.
                    texts = [*train["text"], *validation["text"], "happy🦆", ""]
                    original_state = deepcopy(tokenizer.__dict__)
                    loaded_state = deepcopy(loaded_tokenizer.__dict__)
                    for text in texts:
                        self.assertEqual(
                            tokenizer.encode(tokenizer.prepare_input(text)),
                            loaded_tokenizer.encode(loaded_tokenizer.prepare_input(text)),
                        )
                    original_inputs = self.prediction_inputs(tokenizer, texts, config)
                    loaded_inputs = self.prediction_inputs(loaded_tokenizer, texts, config)
                    with torch.no_grad():
                        expected_logits = original_model(*original_inputs)
                        actual_logits = loaded_model(*loaded_inputs)
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual_logits).all())
                    self.assertEqual(actual_logits.shape, (len(texts), len(labels)))
                    self.assertEqual(
                        actual_logits.argmax(dim=1).tolist(), expected_logits.argmax(dim=1).tolist(),
                    )
                    for index, text in enumerate(texts):
                        torch.testing.assert_close(loaded.logits(text), actual_logits[index:index + 1])
                        prediction = loaded.predict(text)
                        self.assertEqual(prediction, loaded.labels[actual_logits[index].argmax().item()])
                        self.assertIsInstance(prediction, type(labels[0]))

                    # JSON object keys become strings even for integer class labels.
                    label_to_id = config["label_to_id"]
                    self.assertEqual(set(label_to_id), {str(label) for label in labels})
                    id_to_label = {class_id: label for label, class_id in label_to_id.items()}
                    validation_inputs = self.prediction_inputs(loaded_tokenizer, validation["text"], config)
                    with torch.no_grad():
                        predictions = loaded_model(*validation_inputs).argmax(dim=1).tolist()
                    expected_predictions = expected_logits[len(train):len(train) + len(validation)].argmax(dim=1).tolist()
                    self.assertEqual(
                        [id_to_label[value] for value in predictions],
                        [id_to_label[value] for value in expected_predictions],
                    )
                    targets = [label_to_id[str(label)] for label in validation["intent"]]
                    matrix = confusion_matrix(targets, predictions, config["number_of_classes"])
                    metrics = classification_metrics(matrix)
                    self.assertEqual(matrix.tolist(), saved_metrics["confusion_matrix"])
                    self.assertEqual(metrics["accuracy"], saved_metrics["accuracy"])
                    self.assertEqual(metrics["macro_f1"], saved_metrics["macro_f1"])
                    for label, class_id in label_to_id.items():
                        self.assertEqual(metrics["per_class"][class_id], saved_metrics["per_class"][label])
                    self.assertEqual(tokenizer.__dict__, original_state)
                    self.assertEqual(loaded_tokenizer.__dict__, loaded_state)

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
