import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Scripts.DataGeneration import finetuning_pipeline as pipeline


class FinetuningPipelineTests(unittest.TestCase):
    def test_default_plan_never_calls_model_or_creates_output(self):
        with TemporaryDirectory() as temp, patch.object(pipeline.ollama, "chat") as chat:
            output = Path(temp) / "untouched"
            plan = pipeline.run(pipeline.DEFAULT_SPEC, output=output)
            self.assertEqual(plan["total_candidates"], 44)
            self.assertEqual(plan["classes"], 11)
            self.assertFalse(output.exists())
            chat.assert_not_called()

    def test_full_requires_explicit_output(self):
        with patch.object(pipeline.ollama, "chat") as chat:
            with self.assertRaisesRegex(ValueError, "explicit NEW"):
                pipeline.run(pipeline.DEFAULT_SPEC, mode="full")
            chat.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        with TemporaryDirectory() as temp, patch.object(pipeline.ollama, "chat") as chat:
            with self.assertRaises(FileExistsError):
                pipeline.run(pipeline.DEFAULT_SPEC, mode="trial", output=Path(temp))
            chat.assert_not_called()

    def test_rejects_bad_counts_or_labels(self):
        with patch.object(pipeline.ollama, "chat") as chat:
            for kwargs in ({"examples": 0}, {"batch_size": 0}, {"labels": ["unknown"]}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    pipeline.run(pipeline.DEFAULT_SPEC, **kwargs)
            chat.assert_not_called()

    def test_trial_cap_requires_explicit_full_mode(self):
        with patch.object(pipeline.ollama, "chat") as chat:
            with self.assertRaisesRegex(ValueError, "capped"):
                pipeline.run(pipeline.DEFAULT_SPEC, mode="trial", examples=13)
            chat.assert_not_called()

    def test_style_and_word_range_are_in_prompt_and_schema(self):
        spec, _ = pipeline.load_spec(pipeline.DEFAULT_SPEC)
        prompt = pipeline.build_prompt(spec, spec["intents"][0], 4, offset=8)
        self.assertIn("polite indirect request", prompt)
        schema = pipeline.response_schema(4, spec["batch_styles"][2])
        self.assertIn("pattern", schema["properties"]["examples"]["items"]["properties"]["text"])

    def test_response_validation(self):
        invalid = [[], {}, {"examples": []}, {"examples": ["text"]},
                   {"examples": [{"text": "x", "action": " ", "referent": "y"}]},
                   {"examples": [{"text": "x" * 301, "action": "a", "referent": "b"}]},
                   {"examples": [{"text": "x", "action": "a", "referent": "b", "intent": 999}]}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                pipeline.validate_response(json.dumps(value), 1)

    def test_flags_punctuation_overlap_and_all_duplicates(self):
        rows = [{"text": "Check my score!"}, {"text": "CHECK my score."}, {"text": "New request"}]
        pipeline.flag_candidates(rows, {"train": {pipeline.canonical("check my score")}})
        self.assertIn("duplicate_in_run", rows[0]["mechanical_flags"])
        self.assertIn("duplicate_in_run", rows[1]["mechanical_flags"])
        self.assertIn("overlap:train", rows[0]["mechanical_flags"])
        self.assertEqual(rows[2]["mechanical_flags"], [])

    def test_prompt_uses_only_neighbors_and_advances_coverage(self):
        spec, _ = pipeline.load_spec(pipeline.DEFAULT_SPEC)
        target = spec["intents"][0]
        first = pipeline.build_prompt(spec, target, 4)
        second = pipeline.build_prompt(spec, target, 4, ["prior candidate"], offset=4)
        self.assertIn(target["coverage"][0], first)
        self.assertNotIn(target["coverage"][4], first)
        self.assertIn(target["coverage"][4], second)
        self.assertIn("prior candidate", second)
        self.assertNotIn("The USER asks the ASSISTANT", first)
        self.assertNotIn("user utterance", first)

    def test_batches_keep_numeric_ids_seeds_and_review_fields_separate(self):
        calls = []
        def fake_chat(**kwargs):
            calls.append(kwargs)
            n = kwargs["format"]["properties"]["examples"]["minItems"]
            content = json.dumps({"examples": [{"text": f"candidate {len(calls)} {i}",
                "action": "lookup", "referent": "payment"} for i in range(n)]})
            return SimpleNamespace(message=SimpleNamespace(content=content), model_dump_json=lambda **_: content)
        with TemporaryDirectory() as temp, patch.object(pipeline.ollama, "chat", side_effect=fake_chat), \
                patch.object(pipeline, "overlap_sources", return_value={}):
            output = Path(temp) / "trial"
            pipeline.run(pipeline.DEFAULT_SPEC, "trial", 90, 5, output, ["min_payment"])
            rows = json.loads((output / "candidates.json").read_text())
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(len(rows), 5)
            self.assertEqual({r["intent"] for r in rows}, {143})
            self.assertEqual([c["options"]["seed"] for c in calls], [90, 91])
            self.assertIn("candidate 1 0", calls[1]["messages"][0]["content"])
            self.assertTrue(all("lookup" not in r["text"] for r in rows))
            self.assertFalse(config["training_approved"])
            self.assertEqual(config["status"], "awaiting_semantic_review")
            self.assertFalse(list(output.glob("*.parquet")))

    def test_full_mode_exports_filtered_parquet_and_keeps_raw_candidates(self):
        content = json.dumps({"examples": [
            {"text": "An overlapping request", "action": "lookup", "referent": "payment"},
            {"text": "A fresh minimum payment request", "action": "lookup", "referent": "payment"},
        ]})
        response = SimpleNamespace(message=SimpleNamespace(content=content), model_dump_json=lambda **_: content)
        with TemporaryDirectory() as temp, patch.object(pipeline.ollama, "chat", return_value=response), \
                patch.object(pipeline, "overlap_sources", return_value={"train": {"an overlapping request"}}):
            output = Path(temp) / "full"
            pipeline.run(pipeline.DEFAULT_SPEC, "full", examples=2, output=output, labels=["min_payment"])
            frame = pipeline.pd.read_parquet(output / "data.parquet")
            self.assertEqual(frame.columns.tolist(), ["text", "intent"])
            self.assertEqual(frame["text"].tolist(), ["A fresh minimum payment request"])
            self.assertEqual(frame["intent"].tolist(), [143])
            self.assertEqual(str(frame["intent"].dtype), "int64")
            self.assertEqual(len(json.loads((output / "candidates.json").read_text())), 2)
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["status"], "completed_unreviewed")
            self.assertEqual(config["exported_examples"], 1)
            self.assertEqual(config["excluded_examples"], 1)
            self.assertEqual(config["exported_per_class"], {"143": 1})
            self.assertFalse(config["training_approved"])
            self.assertFalse(config["semantic_review_performed"])

    def test_export_excludes_all_duplicate_members_and_empty_canonical_text(self):
        rows = [{"text": "Same request!", "intent": 1},
                {"text": "same request", "intent": 2},
                {"text": "???", "intent": 1},
                {"text": "Unique request", "intent": 3}]
        pipeline.flag_candidates(rows, {})
        with TemporaryDirectory() as temp:
            summary = pipeline.export_parquet(rows, temp)
            frame = pipeline.pd.read_parquet(Path(temp) / "data.parquet")
            self.assertEqual(frame.to_dict("records"), [{"text": "Unique request", "intent": 3}])
            self.assertEqual(summary["excluded_examples"], 3)
            before = (Path(temp) / "data.parquet").read_bytes()
            with self.assertRaises(FileExistsError):
                pipeline.export_parquet(rows, temp)
            self.assertEqual((Path(temp) / "data.parquet").read_bytes(), before)

    def test_export_rejects_empty_selection_and_invalid_ids(self):
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "No candidates"):
                pipeline.export_parquet([], temp)
            for intent in (True, -1, "143", 1.5):
                with self.subTest(intent=intent), self.assertRaisesRegex(ValueError, "integers"):
                    pipeline.export_parquet([{"text": "A request", "intent": intent, "mechanical_flags": []}], temp)
            self.assertFalse((Path(temp) / "data.parquet").exists())

    def test_failure_records_status_and_raw_response(self):
        response = SimpleNamespace(message=SimpleNamespace(content='{"examples": []}'),
                                   model_dump_json=lambda **_: '{"examples": []}')
        with TemporaryDirectory() as temp, patch.object(pipeline.ollama, "chat", return_value=response), \
                patch.object(pipeline, "overlap_sources", return_value={}):
            output = Path(temp) / "trial"
            with self.assertRaises(ValueError):
                pipeline.run(pipeline.DEFAULT_SPEC, "trial", output=output, labels=["min_payment"])
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["status"], "failed")
            self.assertEqual(len(list(output.glob("*_response.json"))), 1)
            self.assertFalse(list(output.glob("*.parquet")))


if __name__ == "__main__":
    unittest.main()
