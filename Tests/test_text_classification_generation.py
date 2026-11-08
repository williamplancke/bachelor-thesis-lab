from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from Scripts.DataGeneration import text_classification as generator
from importlib import import_module

NEWS = import_module("Scripts.DataGeneration.AG-News.settings").SETTINGS
SENTIMENT = import_module("Scripts.DataGeneration.SST-2.settings").SETTINGS


class ClassificationGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = deepcopy(SENTIMENT)
        self.settings["name"] = "Fixture"
        self.directory = self.root / "Data" / "Fixture"
        self.directory.mkdir(parents=True)
        self.train_path = self.directory / "train-00000-of-00001.parquet"
        pd.DataFrame({"sentence": [f"original class {label} row {i}" for label in range(2) for i in range(12)],
                      "label": [label for label in range(2) for _ in range(12)],
                      "idx": range(24)}).to_parquet(self.train_path, index=False)
        patcher = patch.object(generator, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.call_count = 0
        output = patch("sys.stdout", new=StringIO())
        output.start()
        self.addCleanup(output.stop)

    def fake_chat(self, **kwargs):
        self.call_count += 1
        count = kwargs["format"]["properties"]["examples"]["minItems"]
        texts = [f"generated request {self.call_count} item {i}" for i in range(count)]
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": texts})))

    def test_imported_mappings_and_dataset_specific_prompts(self):
        self.assertEqual(NEWS["labels"], ["World", "Sports", "Business", "Sci/Tech"])
        self.assertEqual(SENTIMENT["labels"], ["negative", "positive"])
        self.assertIn("headline", NEWS["generation_instructions"])
        self.assertIn("fragments", SENTIMENT["generation_instructions"])

    def test_read_normalizes_column_names_not_labels_or_text(self):
        frame = generator.read_frame(self.train_path, self.settings)
        self.assertEqual(frame.columns.tolist(), ["text", "label"])
        self.assertEqual(frame.iloc[0].to_dict(), {"text": "original class 0 row 0", "label": 0})

    def test_metadata_mapping_mismatch_is_rejected(self):
        table = pq.read_table(self.train_path)
        metadata = {b"huggingface": json.dumps({"info": {"features": {
            "label": {"names": ["positive", "negative"]}}}}).encode()}
        pq.write_table(table.replace_schema_metadata(metadata), self.train_path)
        with self.assertRaisesRegex(ValueError, "mapping"):
            generator.read_frame(self.train_path, self.settings)

    def test_unknown_labels_and_blank_text_are_rejected(self):
        for text, label in [("text", -1), ("text", 2), (" ", 0), (None, 0)]:
            pd.DataFrame({"sentence": [text], "label": [label]}).to_parquet(self.train_path, index=False)
            with self.subTest(text=text, label=label), self.assertRaises(ValueError):
                generator.read_frame(self.train_path, self.settings)

    def test_grouped_holdout_is_reproducible_and_duplicate_safe(self):
        frame = pd.read_parquet(self.train_path)
        frame = pd.concat([frame, pd.DataFrame({"sentence": [" ORIGINAL CLASS 0 ROW 0 "], "label": [1], "idx": [24]})], ignore_index=True)
        frame.to_parquet(self.train_path, index=False)
        self.settings["validation_fraction"] = 0.25
        train, validation = generator.training_partition(self.settings)
        again_train, again_validation = generator.training_partition(self.settings)
        pd.testing.assert_frame_equal(train, again_train)
        pd.testing.assert_frame_equal(validation, again_validation)
        self.assertFalse(set(train.text.map(generator.normalize)) & set(validation.text.map(generator.normalize)))
        self.assertEqual(len(train) + len(validation), 25)
        self.assertFalse(set(train.index) & set(validation.index))
        self.assertEqual(set(train.label), {0, 1})
        self.assertEqual(set(validation.label), {0, 1})

    def test_plan_reads_training_only_and_never_calls_model_or_writes(self):
        before = self.train_path.read_bytes()
        with patch.object(generator.ollama, "chat") as chat, patch.object(generator, "read_frame", wraps=generator.read_frame) as read:
            output = self.directory / "plan"
            identity = generator.generate_dataset(self.settings, output=output)
        self.assertEqual(identity["targets"], {"0": 12, "1": 12})
        self.assertFalse(output.exists())
        self.assertEqual([call.args[0] for call in read.call_args_list], [self.train_path])
        self.assertEqual(before, self.train_path.read_bytes())
        chat.assert_not_called()

    def test_audit_writes_targets_definitions_and_split_membership(self):
        self.settings["validation_fraction"] = 0.25
        before = self.train_path.read_bytes()
        output = generator.audit(self.settings)
        targets = pd.read_csv(output / "generation_targets.csv")
        train, _ = generator.training_partition(self.settings)
        self.assertEqual(targets.required_synthetic_count.sum(), len(train))
        definitions = pd.read_parquet(output / "label_definitions.parquet")
        self.assertEqual(definitions.label_name.tolist(), self.settings["labels"])
        membership = pd.read_parquet(output / "split_membership.parquet")
        self.assertEqual(len(membership), 24)
        self.assertTrue(membership.source_row.is_unique)
        self.assertEqual(before, self.train_path.read_bytes())
        with self.assertRaises(FileExistsError):
            generator.audit(self.settings)

    def test_raw_condition_omits_definitions(self):
        definitions = {0: "Distinctive definition sentinel"}
        raw = generator.build_prompt(self.settings, 0, 2, "raw", definitions)
        enriched = generator.build_prompt(self.settings, 0, 2, "with-definition", definitions)
        self.assertNotIn("Distinctive definition sentinel", raw)
        self.assertIn("Distinctive definition sentinel", enriched)
        with self.assertRaisesRegex(ValueError, "Raw condition"):
            generator.generate_dataset(self.settings, condition="raw", definitions_path="unused")

    def test_generation_writes_matched_counts_and_original_label_ids(self):
        with patch.object(generator.ollama, "chat", side_effect=self.fake_chat):
            output = generator.generate_dataset(self.settings, generate=True, examples_per_class=3, batch_size=2)
        frame = pd.read_parquet(output / "data.parquet")
        self.assertEqual(frame.columns.tolist(), ["text", "label"])
        self.assertEqual(frame.label.value_counts().to_dict(), {0: 3, 1: 3})
        self.assertEqual(str(frame.label.dtype), "int64")
        self.assertFalse(frame.text.map(generator.normalize).duplicated().any())
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["status"], "completed")
        self.assertEqual(config["counts"], {"0": 3, "1": 3})
        self.assertEqual(len((output / "generation_log.jsonl").read_text().splitlines()), 4)
        with patch.object(generator.ollama, "chat") as chat, self.assertRaises(FileExistsError):
            generator.generate_dataset(self.settings, generate=True, examples_per_class=3, batch_size=2)
        chat.assert_not_called()

    def test_resume_preserves_saved_rows_and_rejects_changed_config(self):
        def fail_second_call(**kwargs):
            if self.call_count == 1:
                raise ConnectionError("fixture interrupted")
            return self.fake_chat(**kwargs)
        with patch.object(generator.ollama, "chat", side_effect=fail_second_call), self.assertRaises(ConnectionError):
            generator.generate_dataset(self.settings, generate=True, examples_per_class=2, batch_size=2)
        output = self.directory / "synthetic_train_with_definition"
        saved = pd.read_parquet(output / "data.parquet")
        with patch.object(generator.ollama, "chat", side_effect=self.fake_chat):
            generator.generate_dataset(self.settings, generate=True, examples_per_class=2, batch_size=2, resume=True)
        final = pd.read_parquet(output / "data.parquet")
        pd.testing.assert_frame_equal(final.iloc[:len(saved)], saved)
        with patch.object(generator.ollama, "chat") as chat:
            generator.generate_dataset(self.settings, generate=True, examples_per_class=2, batch_size=2, resume=True)
            chat.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Resume settings"):
                generator.generate_dataset(self.settings, generate=True, examples_per_class=2, batch_size=2, temperature=0.2, resume=True)

    def test_real_overlap_and_cross_class_duplicates_are_filtered(self):
        batches = [
            ["ORIGINAL CLASS 0 ROW 0"], ["unique first example"],
            [" UNIQUE FIRST EXAMPLE "], ["unique second example"],
        ]
        responses = [SimpleNamespace(message=SimpleNamespace(content=json.dumps({"examples": b}))) for b in batches]
        with patch.object(generator.ollama, "chat", side_effect=responses) as chat:
            output = generator.generate_dataset(self.settings, generate=True, examples_per_class=1, batch_size=1, max_retries=1)
        self.assertEqual(pd.read_parquet(output / "data.parquet").text.tolist(), ["unique first example", "unique second example"])
        self.assertEqual(chat.call_count, 4)

    def test_failed_attempts_are_bounded_and_partial_output_is_marked(self):
        response = SimpleNamespace(message=SimpleNamespace(content='{"examples": ["original class 0 row 0"]}'))
        with patch.object(generator.ollama, "chat", return_value=response) as chat, self.assertRaisesRegex(ValueError, "after 2 attempts"):
            generator.generate_dataset(self.settings, generate=True, examples_per_class=1, batch_size=1, max_retries=1)
        self.assertEqual(chat.call_count, 2)
        output = self.directory / "synthetic_train_with_definition"
        self.assertFalse((output / "data.parquet").exists())
        self.assertEqual(json.loads((output / "config.json").read_text())["status"], "failed")

    def test_definition_file_mapping_is_validated(self):
        path = self.directory / "definitions.parquet"
        pd.DataFrame({"label": [0, 1], "label_name": ["positive", "negative"], "definition": ["x", "y"]}).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "IDs/names"):
            generator.definitions_for(self.settings, path)

    def test_definition_inference_uses_training_partition_only(self):
        self.settings["validation_fraction"] = 0.25
        train, holdout = generator.training_partition(self.settings)
        response = SimpleNamespace(message=SimpleNamespace(content='{"definition": "A concise class definition."}'))
        with patch.object(generator.ollama, "chat", return_value=response) as chat:
            path = generator.infer_definitions(self.settings, generate=True, samples_per_class=100)
        prompts = [call.kwargs["messages"][0]["content"] for call in chat.call_args_list]
        self.assertEqual(chat.call_count, 2)
        for text in holdout.text:
            self.assertFalse(any(json.dumps(text) in prompt for prompt in prompts))
        self.assertEqual(pd.read_parquet(path).label.tolist(), [0, 1])
        self.assertTrue(path.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
