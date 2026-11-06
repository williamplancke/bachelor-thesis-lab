from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from Architecture.Classifier import IntentClassifier
from Training import train as training
from Training.data import encode_dataset, prepare_input


class InputPreparationTests(unittest.TestCase):
    def fitted_tokenizer(self, name):
        tokenizer = training.get_tokenizer(name)
        for text in ("happy customer", "sad customer"):
            words = tokenizer.prepare_input(text)
            if name == "subword-bpe":
                tokenizer.words_to_characters(words)
            else:
                tokenizer.encode(words)
        tokenizer.fit()
        tokenizer.freeze_vocabulary()
        return tokenizer

    def test_shared_inputs_match_original_padding_logic(self):
        for name in ("word", "subword-bpe"):
            tokenizer = self.fitted_tokenizer(name)
            before = deepcopy(tokenizer.__dict__)
            for text in ("happy customer", "happyhappy", "unseen🦆", ""):
                with self.subTest(tokenizer=name, text=text):
                    ids = tokenizer.encode(tokenizer.prepare_input(text))
                    actual_ids, mask = prepare_input(tokenizer, text, 64, 0)
                    self.assertEqual(actual_ids.tolist(), ids + [0] * (64 - len(ids)))
                    self.assertEqual(mask.tolist(), [1] * len(ids) + [0] * (64 - len(ids)))
                    self.assertEqual(actual_ids.dtype, torch.long)
                    self.assertEqual(mask.dtype, torch.long)
            self.assertEqual(tokenizer.__dict__, before)

    def test_exact_length_overflow_and_invalid_length(self):
        tokenizer = self.fitted_tokenizer("word")
        ids, mask = prepare_input(tokenizer, "happy customer", 2, 0)
        self.assertEqual(mask.tolist(), [1, 1])
        self.assertEqual(len(ids), 2)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prepare_input(tokenizer, "happy customer", 1, 0)
        for length in (0, -1, 2.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                prepare_input(tokenizer, "hello", length, 0)
        with self.assertRaises(TypeError):
            prepare_input(tokenizer, None, 4, 0)

    def test_dataset_preserves_row_order_and_label_mapping(self):
        tokenizer = self.fitted_tokenizer("word")
        frame = pd.DataFrame({
            "text": ["sad customer", "", "happy customer"],
            "intent": [30, 10, 30],
        }, index=[9, 2, 9])
        dataset = encode_dataset(frame, tokenizer, {10: 0, 30: 1}, 4, 0, "intent")
        self.assertEqual(dataset.tensors[0].shape, (3, 4))
        self.assertEqual(dataset.tensors[1].shape, (3, 4))
        self.assertEqual(dataset.tensors[2].tolist(), [1, 0, 1])
        for position, text in enumerate(frame["text"]):
            ids, mask = prepare_input(tokenizer, text, 4, 0)
            self.assertTrue(torch.equal(dataset[position][0], ids))
            self.assertTrue(torch.equal(dataset[position][1], mask))

    def test_bad_datasets_fail_clearly(self):
        tokenizer = self.fitted_tokenizer("word")
        for frame in (
            pd.DataFrame(columns=["text", "label"]),
            pd.DataFrame({"text": ["hello"]}),
            pd.DataFrame({"text": ["hello"], "label": [99]}),
        ):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                encode_dataset(frame, tokenizer, {0: 0}, 4, 0)


class CachedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "text": ["happy customer", "sad customer", "happy customer"],
            "label": [0, 1, 0],
        }, index=[8, 2, 8])
        self.validation = pd.DataFrame({"text": ["happyhappy"], "label": [0]})
        self.params = {
            "seed": 42, "sequence_length": 64, "embedding_dim": 4,
            "learning_rate": 0.01, "epochs": 3,
        }

    def test_encoding_count_does_not_grow_with_epochs(self):
        for name in ("word", "subword-bpe"):
            with self.subTest(tokenizer=name), TemporaryDirectory() as directory:
                tokenizer = training.get_tokenizer(name)
                with patch.object(training, "get_tokenizer", return_value=tokenizer), \
                     patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode, \
                     redirect_stdout(StringIO()):
                    training.main(
                        self.frame, self.validation, self.params,
                        run_dir=Path(directory) / "run",
                    )
                # Word fitting also encodes each training row to collect its vocabulary.
                fitting_calls = len(self.frame) if name == "word" else 0
                self.assertEqual(encode.call_count, fitting_calls + len(self.frame) + len(self.validation))

    def test_validation_errors_happen_before_model_creation(self):
        for validation in (
            self.validation.iloc[:0],
            pd.DataFrame({"text": ["hello"], "label": [99]}),
            pd.DataFrame({"text": ["hello " * 65], "label": [0]}),
        ):
            with self.subTest(validation=validation), TemporaryDirectory() as directory:
                with patch("Architecture.Classifier.IntentClassifier") as model, self.assertRaises(ValueError):
                    training.main(self.frame, validation, self.params, run_dir=Path(directory) / "run")
                model.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cached_training_matches_original_single_example_sgd(self):
        for name in ("word", "subword-bpe"):
            with self.subTest(tokenizer=name), TemporaryDirectory() as directory:
                tokenizer = training.get_tokenizer(name)
                models = {}

                def create_model(**kwargs):
                    model = IntentClassifier(**kwargs)
                    models["trained"] = model
                    models["reference"] = deepcopy(model)
                    return model

                with patch.object(training, "get_tokenizer", return_value=tokenizer), \
                     patch("Architecture.Classifier.IntentClassifier", side_effect=create_model), \
                     redirect_stdout(StringIO()):
                    training.main(
                        self.frame, self.validation, self.params,
                        run_dir=Path(directory) / "run",
                    )

                # Independent reference: the old tokenize/pad/update loop, including
                # its exact pandas shuffle and one optimizer step per sentence.
                reference = models["reference"]
                optimizer = torch.optim.SGD(reference.parameters(), lr=self.params["learning_rate"])
                loss_function = torch.nn.CrossEntropyLoss()
                length = self.params["sequence_length"]
                for epoch in range(self.params["epochs"]):
                    shuffled = self.frame.sample(frac=1, random_state=self.params["seed"] + epoch)
                    for _, row in shuffled.iterrows():
                        ids = tokenizer.encode(tokenizer.prepare_input(row["text"]))
                        mask = [1] * len(ids) + [0] * (length - len(ids))
                        ids = ids + [0] * (length - len(ids))
                        optimizer.zero_grad()
                        logits = reference(torch.tensor([ids]), torch.tensor([mask]))
                        loss = loss_function(logits, torch.tensor([int(row["label"])]))
                        loss.backward()
                        optimizer.step()
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(models["trained"].state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
