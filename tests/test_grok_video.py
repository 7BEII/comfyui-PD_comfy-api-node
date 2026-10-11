import importlib
import io
import json
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.argv = [sys.argv[0], "--cpu"]
package = types.ModuleType("pd_grok_video_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
module = importlib.import_module("pd_grok_video_test.py.VIDEO_Grok_apikey")
common = importlib.import_module("pd_grok_video_test.py.OPENAI_latest_apikey")
Node = module.PDGrokVideoAPIKey
Node.hidden = types.SimpleNamespace(unique_id="test-node")


def make_video():
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


class Response:
    def __init__(self, result=None):
        self.result = result or {}
        self.headers = {"X-Comfy-Credits-Used": "12.3"}
        self.content = make_video()

    def json(self):
        return self.result

    def raise_for_status(self):
        pass


class Session:
    def __init__(self, slow=None, timeout=False, failed=False):
        self.calls = []
        self.closed = threading.Event()
        self.slow, self.timeout, self.failed = slow, timeout, failed

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        stage = "submit" if method == "POST" else ("download" if "example.test" in url else "poll")
        if self.timeout:
            raise requests.Timeout("simulated")
        if stage == self.slow:
            self.closed.wait(5)
        if stage == "submit":
            return Response({"request_id": "test-task"})
        if stage == "poll":
            if self.failed:
                return Response({"status": "failed", "error": "simulated failure"})
            return Response({"status": "done", "video": {"url": "https://example.test/video.mp4"},
                             "usage": {"cost_in_usd_ticks": 1000}})
        return Response()

    def close(self):
        self.closed.set()


class GrokVideoTests(unittest.TestCase):
    def setUp(self):
        self.cancel = threading.Event()
        self.events = []
        self.patches = [
            patch.object(common.PromptServer, "instance", types.SimpleNamespace(send_sync=lambda event, data: self.events.append(data)), create=True),
            patch.object(common.comfy.model_management, "throw_exception_if_processing_interrupted", self.check),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def check(self):
        if self.cancel.is_set():
            raise common.comfy.model_management.InterruptProcessingException()

    def run_node(self, **kwargs):
        values = dict(comfy_api_key="test-key", model="grok-imagine-video-1.5-lite", prompt="test prompt",
                      resolution="720p", aspect_ratio="auto", duration=6, seed=42)
        values.update(kwargs)
        return Node.execute(**values)

    def test_models_text_and_image_video_output(self):
        for model in ("grok-imagine-video", "grok-imagine-video-1.5", "grok-imagine-video-1.5-lite"):
            for image in (None, torch.zeros((1, 8, 8, 3))):
                session = Session()
                with patch.object(common.requests, "Session", return_value=session):
                    result = self.run_node(model=model, image=image)
                self.assertEqual(result.result[0].get_dimensions(), (16, 16))
                info = json.loads(result.result[1])
                self.assertEqual(info["request_id"], "test-task")
                self.assertEqual(info["credits_used"], "12.3")
                self.assertEqual([item[0] for item in session.calls], ["POST", "GET", "GET"])
                payload = session.calls[0][2]["json"]
                self.assertEqual(payload["model"], model)
                self.assertEqual(payload["seed"], 42)
                self.assertEqual("image" in payload, image is not None)
                self.assertNotIn("aspect_ratio", payload)
                self.assertEqual(session.calls[-1][2]["headers"], {})

    def test_cancel_submit_poll_download(self):
        for stage in ("submit", "poll", "download"):
            self.cancel.clear()
            self.events.clear()
            errors = []
            session = Session(slow=stage)
            def run():
                try:
                    self.run_node()
                except common.comfy.model_management.InterruptProcessingException as exc:
                    errors.append(exc)
            with patch.object(common.requests, "Session", return_value=session):
                worker = threading.Thread(target=run)
                worker.start()
                time.sleep(1.25)
                started = time.monotonic()
                self.cancel.set()
                worker.join(0.5)
            self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(errors), 1)
            self.assertTrue(session.closed.is_set())
            self.assertEqual(sum(item[0] == "POST" for item in session.calls), 1)
            self.assertTrue(self.events[-1]["text"].startswith("Cancelled"))
            self.assertGreaterEqual(sum(item["text"].startswith("Running") for item in self.events), 2)

    def test_timeout_failure_and_validation(self):
        for session, error in ((Session(timeout=True), requests.Timeout), (Session(failed=True), ValueError)):
            with patch.object(common.requests, "Session", return_value=session), self.assertRaises(error):
                self.run_node()
            self.assertEqual(sum(item[0] == "POST" for item in session.calls), 1)
        session = Session()
        with patch.object(common.requests, "Session", return_value=session):
            for kwargs in ({"model": "grok-imagine-video", "resolution": "1080p"},
                           {"image": torch.zeros((2, 8, 8, 3))}, {"prompt": ""}, {"duration": 16}):
                with self.assertRaises(ValueError):
                    self.run_node(**kwargs)
            self.cancel.set()
            with self.assertRaises(common.comfy.model_management.InterruptProcessingException):
                self.run_node()
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
