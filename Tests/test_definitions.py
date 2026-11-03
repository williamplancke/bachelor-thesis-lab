import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Scripts.DataGeneration import definitions


def response(value):
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.params = {"temperature": 0.2, "seed": 42}

    def test_extracts_definition_with_system_prompt_and_schema(self):
        with patch.object(definitions.ollama, "chat", return_value=response({"definition": " Asking for help. "})) as chat:
            self.assertEqual(definitions.get_response("Examples", self.params), "Asking for help.")
            self.assertEqual(chat.call_args.kwargs["messages"][0]["role"], "system")
            self.assertEqual(chat.call_args.kwargs["format"], definitions.DEFINITION_SCHEMA)

    def test_retries_headings_with_feedback_and_fresh_seed(self):
        replies = [response({"definition": "Label: bookings"}),
                   response({"definition": "DEFINITION : Asking about bookings."}),
                   response({"definition": "Asking whether a restaurant accepts reservations."})]
        with patch.object(definitions.ollama, "chat", side_effect=replies) as chat:
            result = definitions.get_response("Examples", self.params)
            self.assertEqual(result, "Asking whether a restaurant accepts reservations.")
            self.assertEqual([call.kwargs["options"]["seed"] for call in chat.call_args_list], [42, 43, 44])
            self.assertIn("previous response was invalid", chat.call_args.kwargs["messages"][1]["content"])

    def test_rejects_invalid_responses_after_bounded_attempts(self):
        invalid = [[], {}, {"definition": " "}, {"definition": 3},
                   {"definition": "Valid", "label": "extra"}, {"definition": "Label: time"}]
        for value in invalid:
            with self.subTest(value=value), patch.object(definitions.ollama, "chat", return_value=response(value)) as chat:
                with self.assertRaisesRegex(ValueError, "after 2 attempts"):
                    definitions.get_response("Examples", self.params, max_retries=1)
                self.assertEqual(chat.call_count, 2)

    def test_retries_malformed_json(self):
        malformed = SimpleNamespace(message=SimpleNamespace(content="not JSON"))
        with patch.object(definitions.ollama, "chat", side_effect=[malformed, response({"definition": "Asking for the time."})]):
            self.assertEqual(definitions.get_response("Examples", self.params), "Asking for the time.")


if __name__ == "__main__":
    unittest.main()
