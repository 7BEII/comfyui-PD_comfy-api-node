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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = [sys.argv[0], "--cpu"]
package = types.ModuleType("pd_openai_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
module = importlib.import_module("pd_openai_test.py.OPENAI_latest_apikey")
from comfy_api_nodes.apis.openai import InputFileContent, ModelResponseProperties, Reasoning

Chat = module.PDOpenAIChatLatestAPIKey
ImageNode = module.PDOpenAIGPTImage25APIKey
for node in (Chat, ImageNode):
    node.hidden = types.SimpleNamespace(unique_id="test-node")


def png(size=(8, 8)):
    data = io.BytesIO()
    Image.new("RGBA", size, (255, 0, 0, 80)).save(data, format="PNG")
    return data.getvalue()


class Response:
    def __init__(self, data):
        self.data = data
        self.content = png()

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, slow=None, timeout=False, queued=False, url_image=False):
        self.calls = []
        self.closed = threading.Event()
        self.slow = slow
        self.timeout = timeout
        self.queued = queued
        self.url_image = url_image

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.timeout:
            raise requests.Timeout("simulated")
        if method == self.slow:
            self.closed.wait(5)
        if "/responses" in url:
            if method == "POST" and self.queued:
                return Response({"id": "resp_test", "status": "queued"})
            return Response({"id": "resp_test", "status": "completed", "usage": {"total_tokens": 12},
                             "output": [{"type": "message", "content": [
                                 {"type": "output_text", "text": "one"}, {"type": "output_text", "text": "two"},
                             ]}]})
        if method == "GET":
            return Response({})
        if self.url_image:
            return Response({"data": [{"url": "https://example.test/image.png"}]})
        return Response({"data": [{"b64_json": base64.b64encode(png()).decode()},
                                  {"b64_json": base64.b64encode(png((10, 12))).decode()}]})

    def close(self):
        self.closed.set()


class OpenAITests(unittest.TestCase):
    def setUp(self):
        self.cancel = threading.Event()
        self.events = []
        self.patches = [
            patch.object(module.PromptServer, "instance", types.SimpleNamespace(send_sync=lambda event, data: self.events.append(data)), create=True),
            patch.object(module.comfy.model_management, "throw_exception_if_processing_interrupted", self.check),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def check(self):
        if self.cancel.is_set():
            raise module.comfy.model_management.InterruptProcessingException()

    def chat(self, **kwargs):
        return Chat.execute("test-key", "test prompt", kwargs.pop("model", "gpt-6-astra"), **kwargs)

    def image(self, **kwargs):
        model = {"model": "gpt-image-2.5-flare", "size": "1024x1024", "quality": "low", "background": "transparent"}
        model.update(kwargs)
        return ImageNode.execute("test-key", "test prompt", model, 1, 0)

    def test_chat_models_files_options_and_poll(self):
        self.assertEqual(len(list(module.SupportedOpenAIModel)), 18)
        for model in module.SupportedOpenAIModel:
            session = Session(queued=True)
            with patch.object(module.requests, "Session", return_value=session), patch.object(module._Request, "sleep"):
                result = self.chat(model=model.value, images=torch.zeros((1, 8, 8, 3)),
                    files=[InputFileContent(type="input_file", filename="test.pdf", file_data="data:application/pdf;base64,AA==")],
                    advanced_options=ModelResponseProperties(instructions="system", max_output_tokens=32))
            self.assertEqual(result.result, ("one\ntwo",))
            self.assertEqual([item[0] for item in session.calls], ["POST", "GET"])
            body = session.calls[0][2]["json"]
            self.assertEqual(body["model"], model.value)
            self.assertEqual(body["instructions"], "system")
            self.assertEqual([item["type"] for item in body["input"][0]["content"]], ["input_text", "input_image", "input_file"])
            self.assertNotIn("previous_response_id", body)

    def test_image_models_generation_edit_mask_and_alpha(self):
        for model in module.GPT_IMAGE_MODELS:
            session = Session()
            with patch.object(module.requests, "Session", return_value=session):
                result = self.image(model=model, background="auto")
            self.assertEqual(result.result[0].shape, (2, 8, 8, 4))
            self.assertAlmostEqual(result.result[0][0, 0, 0, 3].item(), 80 / 255)
            self.assertTrue(session.calls[0][1].endswith("/images/generations"))
            self.assertNotIn("seed", session.calls[0][2]["json"])
        session = Session()
        with patch.object(module.requests, "Session", return_value=session):
            self.image(size="Custom", custom_width=1024, custom_height=1024,
                       images={"image_1": torch.zeros((1, 8, 8, 3))}, mask=torch.ones((1, 8, 8)))
        self.assertTrue(session.calls[0][1].endswith("/images/edits"))
        self.assertEqual([item[0] for item in session.calls[0][2]["files"]], ["image", "mask"])
        with Image.open(io.BytesIO(session.calls[0][2]["files"][1][1][1])) as mask:
            self.assertEqual(mask.getpixel((0, 0))[3], 0)

    def test_cancellation_generation_poll_and_download(self):
        cases = [(self.chat, Session(slow="POST")), (self.chat, Session(slow="GET", queued=True)),
                 (self.image, Session(slow="POST")), (self.image, Session(slow="GET", url_image=True))]
        for action, session in cases:
            self.cancel.clear()
            self.events.clear()
            errors = []
            def run():
                try:
                    action()
                except module.comfy.model_management.InterruptProcessingException as exc:
                    errors.append(exc)
            with patch.object(module.requests, "Session", return_value=session):
                worker = threading.Thread(target=run)
                worker.start()
                time.sleep(1.25)
                started = time.monotonic()
                self.cancel.set()
                worker.join(0.5)
            self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(errors), 1)
            self.assertEqual(sum(item[0] == "POST" for item in session.calls), 1)
            self.assertTrue(session.closed.is_set())
            self.assertTrue(self.events[-1]["text"].startswith("Cancelled"))
            self.assertGreaterEqual(sum(item["text"].startswith("Running") for item in self.events), 2)

    def test_timeout_and_invalid_inputs_never_retry(self):
        session = Session(timeout=True)
        with patch.object(module.requests, "Session", return_value=session), self.assertRaises(requests.Timeout):
            self.chat()
        self.assertEqual(len(session.calls), 1)
        session = Session()
        with patch.object(module.requests, "Session", return_value=session):
            for action in (
                lambda: self.chat(model="gpt-4.1", advanced_options=ModelResponseProperties(reasoning=Reasoning(effort="high"))),
                lambda: self.image(images={"image_1": torch.zeros((17, 8, 8, 3))}),
                lambda: self.image(mask=torch.zeros((1, 8, 8))),
                lambda: self.image(size="Custom", custom_width=1025, custom_height=1024),
            ):
                with self.assertRaises(ValueError):
                    action()
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
