import base64
import importlib
import io
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]
package = types.ModuleType("pd_api_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
module = importlib.import_module("pd_api_test.py.IMAGE_GeminiNanoBanana21")
Node = module.PDGeminiNanoBanana21
Node.hidden = types.SimpleNamespace(unique_id="test-node")


def image_part(color, thought=False):
    data = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(data, format="PNG")
    return {"thought": thought, "inlineData": {"mimeType": "image/png", "data": base64.b64encode(data.getvalue()).decode()}}


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"candidates": [{"content": {"parts": [
            image_part("red", True), image_part("green"), {"text": "generated"},
        ]}}], "usageMetadata": {"totalTokenCount": 123}}


class Session:
    def __init__(self, slow=False, timeout=False, download=False):
        self.slow = slow
        self.timeout = timeout
        self.download = download
        self.calls = []
        self.closed = threading.Event()

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.timeout:
            raise requests.Timeout("simulated")
        if self.slow and (not self.download or method == "GET"):
            self.closed.wait(5)
        response = Response()
        if self.download and method == "POST":
            response.json = lambda: {"candidates": [{"content": {"parts": [
                {"fileData": {"mimeType": "image/png", "fileUri": "https://example.test/image.png"}},
            ]}}]}
        return response

    def close(self):
        self.closed.set()


class NanoBananaTests(unittest.TestCase):
    def setUp(self):
        self.statuses = []
        self.cancel = threading.Event()
        self.patches = [
            patch.object(module.PromptServer, "instance", types.SimpleNamespace(send_sync=self.status), create=True),
            patch.object(module.comfy.model_management, "throw_exception_if_processing_interrupted", self.interrupted),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def status(self, event, data):
        self.assertEqual(event, "progress_text")
        self.statuses.append((time.monotonic(), data["text"]))

    def interrupted(self):
        if self.cancel.is_set():
            raise module.comfy.model_management.InterruptProcessingException()

    def run_node(self, choice="Gemini Nano Banana 2.1", **model_args):
        model = {"model": choice, "resolution": "1K", "aspect_ratio": "1:1", "thinking_level": "MINIMAL"}
        model.update(model_args)
        return Node.execute("test-key", "test prompt", model, 42, "IMAGE+TEXT")

    def test_models_payload_and_separate_thought_image(self):
        for choice, endpoint in module.MODELS.items():
            session = Session()
            with patch.object(module.requests, "Session", return_value=session):
                result = self.run_node(choice)
            self.assertEqual(len(session.calls), 1)
            self.assertTrue(session.calls[0][1].endswith(endpoint))
            config = session.calls[0][2]["json"]["generationConfig"]
            self.assertEqual("temperature" in config, choice != "Gemini Nano Banana 2.1")
            self.assertNotIn("seed", config)
            self.assertEqual(result.result[1], "generated")
            self.assertTrue(torch.equal(result.result[0][0, 0, 0], torch.tensor([0., 128 / 255, 0.])))
            self.assertTrue(torch.equal(result.result[2][0, 0, 0], torch.tensor([1., 0., 0.])))

    def test_cancel_slow_generation_and_download(self):
        for download in (False, True):
            self.cancel.clear()
            self.statuses.clear()
            session = Session(slow=True, download=download)
            errors = []
            def execute():
                try:
                    self.run_node()
                except module.comfy.model_management.InterruptProcessingException as exc:
                    errors.append(exc)
            with patch.object(module.requests, "Session", return_value=session):
                worker = threading.Thread(target=execute)
                worker.start()
                time.sleep(1.25)
                started = time.monotonic()
                self.cancel.set()
                worker.join(0.5)
            self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(errors), 1)
            self.assertTrue(session.closed.is_set())
            self.assertEqual(sum(call[0] == "POST" for call in session.calls), 1)
            running = [item for item in self.statuses if item[1].startswith("Running")]
            self.assertGreaterEqual(len(running), 2)
            self.assertTrue(all("s" in item[1] for item in running))
            self.assertTrue(self.statuses[-1][1].startswith("Cancelled"))

    def test_timeout_does_not_retry(self):
        session = Session(timeout=True)
        with patch.object(module.requests, "Session", return_value=session), self.assertRaises(requests.Timeout):
            self.run_node()
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(self.statuses[-1][1].startswith("Timed out"))

    def test_validation_and_cancel_before_submission(self):
        session = Session()
        with patch.object(module.requests, "Session", return_value=session):
            with self.assertRaises(ValueError):
                self.run_node("Nano Banana 2 Lite", resolution="4K")
            with self.assertRaises(ValueError):
                self.run_node(images={"image_1": torch.zeros((15, 8, 8, 3))})
            self.cancel.set()
            with self.assertRaises(module.comfy.model_management.InterruptProcessingException):
                self.run_node()
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
