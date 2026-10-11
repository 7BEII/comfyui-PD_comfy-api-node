import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np
import requests
import torch
from PIL import Image

import comfy.model_management
from comfy_api.latest import io as IO
from comfy_api_nodes.nodes_gemini import GeminiNanoBanana2V2
from server import PromptServer

from .IMAGE_GeminiNanoBanana2 import tensor_to_base64


MODELS = {
    "Gemini Nano Banana 2.1": "gemini-nano-banana-2.1",
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image",
    "Nano Banana 2 Lite": "gemini-3.1-flash-lite-image",
}


class PDGeminiNanoBanana21(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = GeminiNanoBanana2V2.define_schema()
        schema.node_id = "PDGeminiNanoBanana21"
        schema.display_name = "PD: Nano Banana 2.1 (ComfyUI Key)"
        schema.category = "PD_Tools/Image_Generation"
        schema.inputs.insert(0, IO.String.Input("comfy_api_key", default="", tooltip="Comfy.org API Key"))
        for item in schema.inputs:
            if item.id == "seed":
                item.tooltip = "Matches the official node. The current Comfy.org endpoint does not receive seed; a fixed seed does not guarantee repeatable output."
            elif item.id in ("temperature", "top_p"):
                item.tooltip += " Ignored by Nano Banana 2.1, matching the official endpoint."
        schema.hidden = [IO.Hidden.unique_id]
        schema.is_api_node = False
        return schema

    @classmethod
    def execute(cls, comfy_api_key, prompt, model, seed, response_modalities,
                system_prompt="", temperature=1.0, top_p=0.95):
        if not comfy_api_key.strip():
            raise ValueError("请填写 ComfyUI API Key。")
        if not prompt.strip():
            raise ValueError("请填写提示词。")
        choice = model["model"]
        model_id = MODELS[choice]
        if choice == "Nano Banana 2 Lite" and model["resolution"] != "1K":
            raise ValueError("Nano Banana 2 Lite 仅支持 1K。")
        if choice != "Gemini Nano Banana 2.1" and model["thinking_level"] == "MEDIUM":
            raise ValueError("MEDIUM 仅适用于 Nano Banana 2.1。")
        images = list((model.get("images") or {}).values())
        if sum(image.shape[0] for image in images if image is not None) > 14:
            raise ValueError("最多支持 14 张参考图。")
        parts = [{"text": prompt}]
        for image in images:
            if image is not None:
                for item in image:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    parts.append({"inlineData": {"mimeType": "image/png", "data": tensor_to_base64(item)}})
        for part in model.get("files") or []:
            parts.append(part.model_dump(exclude_none=True) if hasattr(part, "model_dump") else part)
        image_config = {"imageSize": model["resolution"]}
        if model["aspect_ratio"] != "auto":
            image_config["aspectRatio"] = model["aspect_ratio"]
        config = {
            "responseModalities": ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"],
            "imageConfig": image_config,
            "thinkingConfig": {"thinkingLevel": model["thinking_level"]},
        }
        # The official 2.1 endpoint omits sampling parameters and seed.
        if choice != "Gemini Nano Banana 2.1":
            config.update(temperature=temperature, topP=top_p)
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        started = time.monotonic()
        session = requests.Session()
        executor = ThreadPoolExecutor(max_workers=1)
        future = None

        def status(text):
            PromptServer.instance.send_sync("progress_text", {
                "node": cls.hidden.unique_id,
                "text": f"{text} | {time.monotonic() - started:.1f}s",
            })

        def request(method, url, **kwargs):
            nonlocal future
            comfy.model_management.throw_exception_if_processing_interrupted()
            request_started = time.monotonic()
            future = executor.submit(session.request, method, url, timeout=(15, 180), **kwargs)
            last_status = 0.0
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                if time.monotonic() - request_started > 195:
                    raise requests.Timeout("Request exceeded 195 seconds")
                if time.monotonic() - last_status >= 1:
                    status("Running")
                    last_status = time.monotonic()
                try:
                    return future.result(timeout=0.1)
                except FutureTimeout:
                    if future.done():
                        raise

        def decode(data):
            comfy.model_management.throw_exception_if_processing_interrupted()
            with Image.open(io.BytesIO(data)) as image:
                return torch.from_numpy(np.array(image.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

        try:
            response = request("POST", f"https://api.comfy.org/proxy/vertexai/gemini/{model_id}",
                               headers={"X-API-KEY": comfy_api_key.strip()}, json=payload)
            response.raise_for_status()
            result = response.json()
            generated, thoughts, texts = [], [], []
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    if part.get("text"):
                        texts.append(part["text"])
                    inline = part.get("inlineData") or part.get("inline_data")
                    file_data = part.get("fileData") or part.get("file_data")
                    if inline and inline.get("mimeType", inline.get("mime_type", "")).startswith("image/"):
                        image = decode(base64.b64decode(inline["data"]))
                    elif file_data and file_data.get("mimeType", file_data.get("mime_type", "")).startswith("image/"):
                        downloaded = request("GET", file_data.get("fileUri") or file_data["file_uri"])
                        downloaded.raise_for_status()
                        image = decode(downloaded.content)
                    else:
                        continue
                    (thoughts if part.get("thought") else generated).append(image)
            if not generated:
                raise ValueError("Gemini 未返回成品图：" + ("\n".join(texts) or str(result.get("promptFeedback", {}))))
            output = torch.cat(generated)
            thought = thoughts[0] if thoughts else torch.zeros((1, 1024, 1024, 3))
            comfy.model_management.throw_exception_if_processing_interrupted()
            usage = result.get("usageMetadata", {})
            status(f"Completed | Tokens: {usage.get('totalTokenCount', 'N/A')}")
            return IO.NodeOutput(output, "\n".join(texts), thought)
        except comfy.model_management.InterruptProcessingException:
            session.close()
            if future is not None:
                future.cancel()
            status("Cancelled")
            raise
        except (requests.Timeout, FutureTimeout):
            status("Timed out")
            raise
        except Exception:
            status("Failed")
            raise
        finally:
            session.close()
            executor.shutdown(wait=False, cancel_futures=True)


NODE_CLASS_MAPPINGS = {"PDGeminiNanoBanana21": PDGeminiNanoBanana21}
NODE_DISPLAY_NAME_MAPPINGS = {"PDGeminiNanoBanana21": "PD: Nano Banana 2.1 (ComfyUI Key)"}
