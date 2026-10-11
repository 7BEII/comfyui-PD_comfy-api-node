import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np
import requests
import torch
from PIL import Image

import comfy.model_management
from comfy.utils import common_upscale
from comfy_api.latest import IO
from comfy_api_nodes.nodes_openai import (
    OpenAIChatNode, OpenAIGPTImageNodeV2, GPT_IMAGE_MODELS,
    SUPPORTED_REASONING_EFFORTS, SupportedOpenAIModel, downscale_image_tensor,
)
from server import PromptServer

from ._image_gpt_common import tensor_to_bytes, validate_gpt_image_2_size


BASE_URL = "https://api.comfy.org"


class _Request:
    def __init__(self, node_id, api_key):
        if not api_key.strip():
            raise ValueError("请填写 Comfy.org API Key。")
        self.node_id = node_id
        self.started = time.monotonic()
        self.session = requests.Session()
        self.headers = {"X-API-KEY": api_key.strip(), "Accept": "application/json"}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.response_id = ""
        self.usage = {}
        self.last_status = 0

    def status(self, text):
        PromptServer.instance.send_sync("progress_text", {
            "node": self.node_id,
            "text": f"{text} | {time.monotonic() - self.started:.1f}s {self.response_id}",
        })

    def check(self):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if time.monotonic() - self.started > 600:
            raise requests.Timeout("Operation exceeded 600 seconds")
        if time.monotonic() - self.last_status >= 1:
            self.status("Running")
            self.last_status = time.monotonic()

    def request(self, method, url, **kwargs):
        self.check()
        headers = self.headers if url.startswith(BASE_URL + "/") else {}
        self.future = self.executor.submit(self.session.request, method, url,
                                           headers=headers, timeout=(15, 180), **kwargs)
        while True:
            self.check()
            try:
                response = self.future.result(timeout=0.1)
                response.raise_for_status()
                return response
            except FutureTimeout:
                if self.future.done():
                    raise

    def sleep(self, seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.check()
            time.sleep(min(0.1, max(0, until - time.monotonic())))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.session.close()
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        if exc_type is not None and issubclass(exc_type, comfy.model_management.InterruptProcessingException):
            self.status("Cancelled")
        elif exc_type is not None and issubclass(exc_type, (requests.Timeout, FutureTimeout)):
            self.status("Timed out")
        elif exc_type is not None:
            self.status("Failed")
        else:
            self.status(f"Completed | Usage: {self.usage}")


def _schema(official, node_id, name, category):
    schema = official.define_schema()
    schema.node_id = node_id
    schema.display_name = name
    schema.category = category
    schema.inputs.insert(0, IO.String.Input("comfy_api_key", default="", tooltip="Comfy.org API Key"))
    schema.hidden = [IO.Hidden.unique_id]
    schema.is_api_node = False
    return schema


class PDOpenAIChatLatestAPIKey(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = _schema(OpenAIChatNode, "PDOpenAIChatLatestAPIKey", "PD: OpenAI ChatGPT Latest (ComfyUI Key)", "PD_Tools/Text_Generation")
        # The official persist_context widget is deprecated and does nothing.
        schema.inputs = [item for item in schema.inputs if item.id != "persist_context"]
        return schema

    @classmethod
    def execute(cls, comfy_api_key, prompt, model, images=None, files=None, advanced_options=None):
        if model not in [item.value for item in SupportedOpenAIModel]:
            raise ValueError(f"Unknown model: {model}")
        if not prompt.strip():
            raise ValueError("请填写提示词。")
        options = advanced_options.model_dump(exclude_none=True) if advanced_options else {}
        effort = (options.get("reasoning") or {}).get("effort")
        if effort is not None and effort not in SUPPORTED_REASONING_EFFORTS[model]:
            raise ValueError(f"{model} 不支持 reasoning_effort={effort}，请选择 default 或该模型支持的等级。")
        content = OpenAIChatNode.create_input_message_contents(prompt, images, files)
        payload = {**options, "model": model, "input": [{"role": "user", "content": [
            item.model_dump(exclude_none=True) for item in content
        ]}], "store": True, "stream": False}
        with _Request(cls.hidden.unique_id, comfy_api_key) as api:
            result = api.request("POST", BASE_URL + "/proxy/openai/v1/responses", json=payload).json()
            api.response_id = result.get("id", "")
            while result.get("status") in ("queued", "in_progress"):
                if not api.response_id:
                    raise ValueError("OpenAI response missing ID")
                api.sleep(1)
                result = api.request("GET", BASE_URL + "/proxy/openai/v1/responses/" + api.response_id).json()
            api.check()
            if result.get("error") or result.get("status") not in ("completed", "incomplete"):
                raise ValueError(f"OpenAI response {result.get('status')}: {result.get('error')}")
            texts = [item["text"] for output in result.get("output", []) if output.get("type") == "message"
                     for item in output.get("content", []) if item.get("type") == "output_text" and item.get("text")]
            if not texts:
                raise ValueError("OpenAI 未返回文字。")
            api.usage = result.get("usage", {})
            return IO.NodeOutput("\n".join(texts))


class PDOpenAIGPTImage25APIKey(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(OpenAIGPTImageNodeV2, "PDOpenAIGPTImage25APIKey", "PD: OpenAI GPT Image 2.5 (ComfyUI Key)", "PD_Tools/Image_Generation")

    @classmethod
    def execute(cls, comfy_api_key, prompt, model, n, seed):
        model_id = model["model"]
        if model_id not in GPT_IMAGE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        if not prompt.strip():
            raise ValueError("请填写提示词。")
        if not 1 <= n <= 8:
            raise ValueError("图片数量须为 1–8。")
        size = model["size"]
        if size == "Custom":
            size = validate_gpt_image_2_size(size, model["custom_width"], model["custom_height"])
        payload = {"model": model_id, "prompt": prompt, "n": n, "size": size,
                   "quality": model["quality"], "background": model["background"], "moderation": "low"}
        if model_id == "gpt-image-2" and payload["background"] == "transparent":
            raise ValueError("gpt-image-2 不支持透明背景。")
        images = [item for image in (model.get("images") or {}).values() if image is not None for item in image]
        if len(images) > 16:
            raise ValueError("最多支持 16 张参考图。")
        mask = model.get("mask")
        if mask is not None and (len(images) != 1 or mask.shape != (1, *images[0].shape[:2])):
            raise ValueError("遮罩要求恰好一张参考图，且尺寸相同。")
        with _Request(cls.hidden.unique_id, comfy_api_key) as api:
            uploads = []
            for index, image in enumerate(images):
                api.check()
                uploads.append(("image" if len(images) == 1 else "image[]",
                                (f"image_{index}.png", tensor_to_bytes(image).getvalue(), "image/png")))
            if mask is not None:
                rgba = torch.zeros((*images[0].shape[:2], 4))
                rgba[:, :, 3] = 1 - mask[0].cpu()
                scaled = downscale_image_tensor(rgba.unsqueeze(0), total_pixels=2048 * 2048).squeeze(0)
                data = io.BytesIO()
                Image.fromarray((scaled.numpy() * 255).astype(np.uint8)).save(data, format="PNG")
                uploads.append(("mask", ("mask.png", data.getvalue(), "image/png")))
            if uploads:
                result = api.request("POST", BASE_URL + "/proxy/openai/images/edits", data=payload, files=uploads).json()
            else:
                result = api.request("POST", BASE_URL + "/proxy/openai/images/generations", json=payload).json()
            tensors = []
            for item in result.get("data", []):
                api.check()
                if item.get("b64_json"):
                    data = base64.b64decode(item["b64_json"])
                elif item.get("url"):
                    url = item["url"]
                    if url.startswith("/proxy/"):
                        url = BASE_URL + url
                    data = api.request("GET", url).content
                else:
                    raise ValueError("图片响应缺少 URL 或 base64 数据。")
                api.check()
                with Image.open(io.BytesIO(data)) as image:
                    tensors.append(torch.from_numpy(np.array(image.convert("RGBA")).astype(np.float32) / 255.0))
            if not tensors:
                raise ValueError("OpenAI 未返回图片。")
            height, width = tensors[0].shape[:2]
            for index, image in enumerate(tensors):
                if image.shape[:2] != (height, width):
                    tensors[index] = common_upscale(image.movedim(-1, 0).unsqueeze(0), width, height, "bilinear", "center")[0].movedim(0, -1)
            api.check()
            api.usage = result.get("usage", {})
            return IO.NodeOutput(torch.stack(tensors))


NODE_CLASS_MAPPINGS = {
    "PDOpenAIChatLatestAPIKey": PDOpenAIChatLatestAPIKey,
    "PDOpenAIGPTImage25APIKey": PDOpenAIGPTImage25APIKey,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIChatLatestAPIKey": "PD: OpenAI ChatGPT Latest (ComfyUI Key)",
    "PDOpenAIGPTImage25APIKey": "PD: OpenAI GPT Image 2.5 (ComfyUI Key)",
}
