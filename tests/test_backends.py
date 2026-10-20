import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ouroboros.config import Config
from ouroboros.models import AttackerRefusal, JudgeError, MockModels, Ollama


class MockModelsTests(unittest.TestCase):
    def test_judge_preserves_deterministic_labels_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            images = [Path(directory) / str(i) for i in range(6)]
            for i, path in enumerate(images):
                path.write_bytes(bytes([i]))
            for binary, expected in (
                (False, ['male', 'male', 'female', 'male', 'male', 'unclear']),
                (True, ['male', 'female', 'female', 'male', 'male', 'male']),
            ):
                with self.subTest(binary=binary):
                    result = MockModels().judge(images, '', '', 42, binary=binary)
                    self.assertEqual(result['labels'], expected)
                    self.assertEqual(MockModels().judge([], '', '', 42, binary=binary)['labels'], [])


class OllamaContractTests(unittest.TestCase):
    def test_judge_payload_contains_images_and_deterministic_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"image bytes")
            answer = {"message": {"content": json.dumps({"labels": ["female"], "reason": "visual evidence"})}}
            with patch("ouroboros.models.request_json", return_value=answer) as request:
                model = Ollama(Config())
                result = model.judge([path], "scene", "prompt", 123)
            payload = request.call_args.args[1]
            self.assertEqual(payload["options"]["temperature"], 0)
            self.assertEqual(payload["options"]["seed"], 123)
            self.assertEqual(payload["keep_alive"], 0)
            self.assertFalse(payload["think"])
            self.assertEqual(len(payload["messages"][1]["images"]), 1)
            self.assertEqual(result["labels"], ["female"])

    def test_gemma4_requests_disable_thinking(self):
        for name in ("gemma4:e2b-it-qat", "gemma4:e4b-it-qat", "gemma4:12b-it-qat"):
            with self.subTest(model=name):
                answer = {"message": {"content": "{}"}}
                with patch("ouroboros.models.request_json", return_value=answer) as request:
                    result = Ollama(Config()).chat(name, "system", "test", 42)
                payload = request.call_args.args[1]
                self.assertIs(payload["think"], False)
                self.assertEqual(payload["model"], name)
                self.assertEqual(payload["format"], "json")
                self.assertEqual(result, {})

    def test_ministral_request_omits_thinking_option(self):
        answer = {"message": {"content": "{}"}}
        with patch("ouroboros.models.request_json", return_value=answer) as request:
            model = Ollama(Config())
            model.chat(model.config.attacker, "system", "test", 1)
        self.assertNotIn("think", request.call_args.args[1])

    def test_wrong_label_count_is_retried_then_fails(self):
        with patch.object(Ollama, "chat", return_value={"labels": [], "reason": "none"}) as chat:
            with self.assertRaises(JudgeError):
                Ollama(Config()).judge(["unused.png"], "", "", 1)
            self.assertEqual(chat.call_count, 3)

    def test_binary_validation_rejects_unclear(self):
        with patch.object(Ollama, "chat", return_value={"labels": ["unclear"], "reason": "none"}):
            with self.assertRaises(JudgeError):
                Ollama(Config()).judge(["unused.png"], "", "", 1, binary=True)

    def test_explicit_attacker_refusal_has_two_attempts(self):
        with patch.object(Ollama, "chat", return_value={"refused": True}) as chat:
            with self.assertRaises(AttackerRefusal):
                Ollama(Config()).propose("doctor", [], 1)
            self.assertEqual(chat.call_count, 2)

    def test_malformed_attacker_is_not_a_refusal(self):
        with patch.object(Ollama, "chat", return_value={}):
            with self.assertRaises(ValueError):
                Ollama(Config()).propose("doctor", [], 1)


if __name__ == "__main__":
    unittest.main()
