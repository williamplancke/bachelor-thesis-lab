from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd


from Scripts.DataGeneration import intent_samples as generator


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        data_dir = Path(temp.name)
        audit = data_dir / "audit"
        audit.mkdir()
        names = json.loads((generator.DATA_DIR / "audit" / "intent_names.json").read_text())
        (audit / "intent_names.json").write_text(json.dumps(names))
        self.definitions_path = audit / "label_definitions.parquet"
        pd.DataFrame({"label": names, "definition": [f"Requests concerning {name}." for name in names]}).to_parquet(
            self.definitions_path, index=False
        )
        patcher = patch.object(generator, "DATA_DIR", data_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_main_saves_original_intent_id(self):
        params = {"example_num": 2, "intent_label": "accept_reservations", "intent_definition": "Asking whether a restaurant accepts reservations, including advance bookings for larger groups.", "temperature": 0.8, "seed": 42}
        content = json.dumps({"examples": [" Do you take reservations? ", "Is booking ahead possible?"]})
        response = SimpleNamespace(message=SimpleNamespace(content=content))
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", return_value=response) as chat:
            path = generator.main(params, output_dir=temp)
            prompt = chat.call_args.kwargs["messages"][0]["content"]
            self.assertIn(f"Definition: {params['intent_definition']}", prompt)
            self.assertNotIn("dtype:", prompt)
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

    def test_main_accumulates_intents_in_one_file(self):
        params = {"example_num": 1, "intent_label": "accept_reservations", "intent_definition": "Asking whether a restaurant accepts reservations.", "temperature": 0.8, "seed": 42}
        responses = [
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": [text]})))
            for text in ("Do you accept bookings?", "What time is it?")
        ]
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", side_effect=responses):
            path = Path(temp) / "combined.parquet"
            generator.main(params, output_path=path)
            generator.main({**params, "intent_label": "time", "intent_definition": "Asking for the current time."}, output_path=path)
            df = pd.read_parquet(path)
            self.assertEqual(df["intent"].tolist(), [9, 4])
            self.assertEqual(len(list(Path(temp).glob("*.parquet"))), 1)

    def test_duplicate_batch_leaves_saved_data_unchanged(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "combined.parquet"
            content = json.dumps({"examples": ["A request"]})
            generator.save_batch(content, 9, 1, path, append=True)
            with self.assertRaises(ValueError):
                generator.save_batch(content, 4, 1, path, append=True)
            self.assertEqual(pd.read_parquet(path)["intent"].tolist(), [9])

    def test_full_dataset_matches_counts_including_oos(self):
        call_index = 0

        def fake_chat(**kwargs):
            nonlocal call_index
            prompt = kwargs["messages"][0]["content"]
            count = int(prompt.split()[1])
            texts = [f"batch {call_index} message {i}" for i in range(count)]
            call_index += 1
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": texts})))

        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", side_effect=fake_chat) as chat:
            real_path = Path(temp) / "real.parquet"
            real = pd.DataFrame({"text": ["unused"] * 6, "intent": [0, 0, 0, 9, 9, 42]})
            real.to_parquet(real_path, index=False)
            path = Path(temp) / "synthetic.parquet"
            generator.generate_dataset({"temperature": 0.8, "seed": 42}, path, batch_size=2, training_path=real_path)
            synthetic = pd.read_parquet(path)
            self.assertEqual(synthetic["intent"].value_counts().to_dict(), {0: 3, 9: 2, 42: 1})
            self.assertEqual(chat.call_count, 4)
            self.assertEqual([call.kwargs["options"]["seed"] for call in chat.call_args_list], [42, 43, 44, 45])
            self.assertIn("fit NONE", chat.call_args_list[-1].kwargs["messages"][0]["content"])
            self.assertIn("Definition: Requests concerning oos.", chat.call_args_list[-1].kwargs["messages"][0]["content"])
            with self.assertRaises(FileExistsError):
                generator.generate_dataset({"temperature": 0.8, "seed": 42}, path, training_path=real_path)

    def test_invalid_batch_is_retried_with_new_seed(self):
        params = {"example_num": 2, "intent_label": "accept_reservations", "intent_definition": "Asking whether reservations are accepted.", "temperature": 0.8, "seed": 42}
        responses = [
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": ["Only one"]}))),
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": ["First", "Second"]}))),
        ]
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", side_effect=responses) as chat:
            path = generator.main(params, output_dir=temp, max_retries=1)
            self.assertEqual(pd.read_parquet(path)["text"].tolist(), ["Only one", "First"])
            self.assertEqual([call.kwargs["options"]["seed"] for call in chat.call_args_list], [42, 43])
            self.assertEqual(chat.call_args.kwargs["format"]["properties"]["examples"]["minItems"], 3)
            self.assertIn("batch was invalid", chat.call_args.kwargs["messages"][-1]["content"])

    def test_invalid_batch_exhausts_retries_without_writing(self):
        params = {"example_num": 2, "intent_label": "accept_reservations", "intent_definition": "Asking whether reservations are accepted.", "temperature": 0.8, "seed": 42}
        response = SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": ["Only one"]})))
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", return_value=response) as chat:
            with self.assertRaisesRegex(ValueError, "after 2 attempts"):
                generator.main(params, output_dir=temp, max_retries=1)
            self.assertEqual(chat.call_count, 2)
            self.assertEqual(list(Path(temp).glob("*.parquet")), [])

    def test_duplicate_candidates_are_filtered_and_only_missing_are_requested(self):
        params = {"example_num": 3, "intent_label": "account_blocked", "intent_definition": "Asking about a blocked account.", "temperature": 0.8, "seed": 42}
        batches = [
            [" MY   ACCOUNT IS BLOCKED ", "I cannot access my account", " I CANNOT access my account "],
            ["Please help unlock my account", "My login has been suspended"],
        ]
        responses = [SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": batch})))
                     for batch in batches]
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", side_effect=responses) as chat:
            path = Path(temp) / "combined.parquet"
            generator.save_batch(json.dumps({"examples": ["My account is blocked"]}), 2, 1, path, append=True)
            generator.main(params, output_path=path, max_retries=1)
            self.assertEqual(len(pd.read_parquet(path)), 4)
            self.assertEqual(chat.call_args.kwargs["format"]["properties"]["examples"]["minItems"], 3)
            self.assertEqual(chat.call_args.kwargs["options"]["num_predict"], 256)
            prompt = chat.call_args.kwargs["messages"][0]["content"]
            self.assertIn("My account is blocked", prompt)
            self.assertIn("I cannot access my account", prompt)

    def test_logs_exclusion_counts_and_duplicate_sources(self):
        params = {"example_num": 3, "intent_label": "account_blocked", "intent_definition": "Asking about a blocked account.", "temperature": 0.8, "seed": 42}
        batches = [
            [" SAVED ", "new one", " NEW ONE "],
            ["new one", "new two"],
            ["new three"],
        ]
        responses = [SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": batch})))
                     for batch in batches]
        log = StringIO()
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", side_effect=responses):
            path = Path(temp) / "combined.parquet"
            generator.save_batch(json.dumps({"examples": ["saved"]}), 2, 1, path, append=True)
            with redirect_stdout(log):
                generator.main(params, output_path=path, max_retries=2)
            self.assertEqual(len(pd.read_parquet(path)), 4)
        output = log.getvalue()
        self.assertIn("sending 1 exclusions (1 saved examples checked globally), seed=42", output)
        self.assertIn("saved=1, within_attempt=1, earlier_retries=0", output)
        self.assertIn("saved=0, within_attempt=0, earlier_retries=1", output)
        self.assertIn("attempt 3/3, requesting 3 candidates for 1 remaining examples", output)

    def test_resume_fills_partial_and_absent_intents_preserving_saved_rows(self):
        with TemporaryDirectory() as temp, patch.object(generator, "main") as generate:
            real_path = Path(temp) / "real.parquet"
            pd.DataFrame({"text": ["unused"] * 10, "intent": [0] * 5 + [9] * 3 + [42] * 2}).to_parquet(real_path)
            path = Path(temp) / "synthetic.parquet"
            # Non-contiguous labels, out-of-order rows, and counts not aligned to batch size.
            original = pd.DataFrame({"text": ["saved booking", "saved review"], "intent": [9, 0]})
            original.to_parquet(path, index=False)

            def save_fake(params, output_path, max_retries):
                texts = [f"{params['intent_label']} seed {params['seed']} row {i}"
                         for i in range(params["example_num"])]
                names = json.loads((generator.DATA_DIR / "audit" / "intent_names.json").read_text())
                generator.save_batch(json.dumps({"examples": texts}), names.index(params["intent_label"]),
                                     params["example_num"], output_path, append=True)

            generate.side_effect = save_fake
            generator.generate_dataset({"temperature": 0.8, "seed": 42}, path, batch_size=3,
                                       training_path=real_path, resume=True)
            result = pd.read_parquet(path)
            pd.testing.assert_frame_equal(result.iloc[:2], original)
            self.assertEqual(result["intent"].value_counts().to_dict(), {0: 5, 9: 3, 42: 2})
            self.assertEqual([call.args[0]["example_num"] for call in generate.call_args_list], [3, 1, 2, 2])
            generate.reset_mock()
            before = path.read_bytes()
            generator.generate_dataset({"temperature": 0.8, "seed": 42}, path, training_path=real_path, resume=True)
            generate.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_resume_missing_file_starts_new_dataset(self):
        response = SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": ["new example"]})))
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat", return_value=response):
            real_path = Path(temp) / "real.parquet"
            pd.DataFrame({"text": ["unused"], "intent": [0]}).to_parquet(real_path)
            path = Path(temp) / "synthetic.parquet"
            generator.generate_dataset({"temperature": 0.8, "seed": 42}, path,
                                       training_path=real_path, resume=True)
            self.assertEqual(len(pd.read_parquet(path)), 1)

    def test_resume_rejects_invalid_saved_data_without_generation(self):
        invalid = [
            pd.DataFrame({"text": ["missing label column"]}),
            pd.DataFrame({"text": [" "], "intent": [0]}),
            pd.DataFrame({"text": ["unknown label"], "intent": [9]}),
            pd.DataFrame({"text": ["null label"], "intent": [None]}),
            pd.DataFrame({"text": ["Same", " SAME "], "intent": [0, 0]}),
            pd.DataFrame({"text": ["one", "two", "three"], "intent": [0, 0, 0]}),
        ]
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat") as chat:
            real_path = Path(temp) / "real.parquet"
            pd.DataFrame({"text": ["unused"] * 2, "intent": [0, 0]}).to_parquet(real_path)
            path = Path(temp) / "synthetic.parquet"
            for data in invalid:
                data.to_parquet(path, index=False)
                before = path.read_bytes()
                with self.subTest(data=data), self.assertRaises(ValueError):
                    generator.generate_dataset({"temperature": 0.8, "seed": 42}, path,
                                               training_path=real_path, resume=True)
                self.assertEqual(path.read_bytes(), before)
            chat.assert_not_called()

    def test_main_rejects_missing_or_invalid_definition_before_calling_model(self):
        params = {"example_num": 1, "intent_label": "time", "temperature": 0.8, "seed": 42}
        with patch.object(generator.ollama, "chat") as chat:
            with self.assertRaisesRegex(KeyError, "intent_definition"):
                generator.main(params)
            for value in (None, "", "   ", 123, pd.Series(["A definition"])):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-empty string"):
                    generator.main({**params, "intent_definition": value})
            chat.assert_not_called()

    def test_dataset_rejects_bad_definitions_before_writing(self):
        invalid = [
            pd.DataFrame({"label": ["time"], "definition": ["Wrong label"]}),
            pd.DataFrame({"label": ["restaurant_reviews"] * 2, "definition": ["First", "Second"]}),
            pd.DataFrame({"label": ["restaurant_reviews"], "definition": [None]}),
            pd.DataFrame({"label": ["restaurant_reviews"], "definition": [" "]}),
            pd.DataFrame({"label": ["restaurant_reviews"], "other": ["Missing column"]}),
        ]
        with TemporaryDirectory() as temp, patch.object(generator.ollama, "chat") as chat:
            real_path = Path(temp) / "real.parquet"
            pd.DataFrame({"text": ["unused"], "intent": [0]}).to_parquet(real_path)
            path = Path(temp) / "synthetic.parquet"
            for data in invalid:
                data.to_parquet(self.definitions_path, index=False)
                with self.subTest(data=data), self.assertRaises(ValueError):
                    generator.generate_dataset({"temperature": 0.8, "seed": 42}, path, training_path=real_path)
                self.assertFalse(path.exists())
            chat.assert_not_called()

    def test_existing_batch_is_not_overwritten(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "batch.parquet"
            content = json.dumps({"examples": ["A request"]})
            generator.save_batch(content, 9, 1, path)
            with self.assertRaises(FileExistsError):
                generator.save_batch(content, 9, 1, path)


if __name__ == "__main__":
    unittest.main()
