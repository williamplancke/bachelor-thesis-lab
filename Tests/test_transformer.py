from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch

from Architecture.Classifier import TransformerIntentClassifier, architecture_config
from Training.data import prepare_input
from Training.load import load_run
from Training.train import main as train


class TransformerClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TransformerIntentClassifier(
            vocabulary_size=20, embedding_dim=8, number_of_classes=3,
            padding_id=0, number_of_heads=2, feedforward_dim=16,
            dropout=0.0, max_sequence_length=16,
        )

    def test_shapes_gradients_and_empty_inputs(self):
        ids = torch.tensor([[2, 3, 0], [0, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
        logits = self.model(ids, mask)
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(self.model.embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.position_embedding.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 0], [0, 0, 0]])))

    def test_padding_and_masked_token_values_do_not_change_prediction(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(torch.tensor([[2, 3]]), torch.tensor([[1, 1]]))
            padded = self.model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))
            changed_padding = self.model(torch.tensor([[2, 3, 7, 8]]), torch.tensor([[1, 1, 0, 0]]))
        torch.testing.assert_close(expected, padded)
        torch.testing.assert_close(expected, changed_padding)

    def test_word_order_affects_output(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(torch.tensor([[2, 3, 4]]), torch.tensor([[1, 1, 1]]))
            reversed_order = self.model(torch.tensor([[4, 3, 2]]), torch.tensor([[1, 1, 1]]))
        self.assertFalse(torch.allclose(first, reversed_order))

    def test_invalid_configuration_and_shapes(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            TransformerIntentClassifier(10, 7, 2, 0, number_of_heads=2)
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 17), dtype=torch.long), torch.ones((1, 17)))
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 0), dtype=torch.long), torch.ones((1, 0)))
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 2), dtype=torch.long), torch.ones((1, 3)))

    def test_architecture_defaults_use_configured_sequence_length(self):
        name, params = architecture_config({
            "architecture": "transformer-encoder", "embedding_dim": 8, "sequence_length": 32,
        })
        self.assertEqual(name, "transformer-encoder")
        self.assertEqual(params["max_sequence_length"], 32)
        self.assertEqual(params["number_of_layers"], 1)

    def test_tiny_training_and_reload_with_both_tokenizers(self):
        data = pd.DataFrame({
            "text": ["happy customer", "sad customer", ""],
            "label": ["positive", "negative", "negative"],
        })
        for tokenizer in ("word", "subword-bpe"):
            with self.subTest(tokenizer=tokenizer), TemporaryDirectory() as directory:
                params = {
                    "seed": 42, "sequence_length": 16, "learning_rate": 0.01, "epochs": 1,
                    "architecture": "transformer-encoder",
                    "architecture_params": {
                        "embedding_dim": 8, "number_of_heads": 2, "feedforward_dim": 16,
                    },
                    "tokenizer_instance": tokenizer,
                }
                with redirect_stdout(StringIO()):
                    run_dir = train(data, data, params, run_dir=Path(directory) / "run")
                loaded = load_run(run_dir)
                again = load_run(run_dir)
                self.assertIsInstance(loaded.model, TransformerIntentClassifier)
                self.assertFalse(loaded.model.training)
                self.assertEqual(loaded.config["architecture_params"]["max_sequence_length"], 16)
                text = "happyhappy"
                ids, mask = prepare_input(loaded.tokenizer, text, 16, 0)
                with torch.no_grad():
                    direct_logits = loaded.model(ids.unsqueeze(0), mask.unsqueeze(0))
                torch.testing.assert_close(loaded.logits(text), direct_logits, rtol=0, atol=0)
                torch.testing.assert_close(loaded.logits(text), again.logits(text), rtol=0, atol=0)
                self.assertIn(loaded.predict(text), loaded.labels)
                self.assertTrue(torch.isfinite(loaded.logits("")).all())


if __name__ == "__main__":
    unittest.main()
