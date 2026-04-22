"""
PD: Nano Banana 2 (ComfyUI Key)
Google Gemini 3.1 Flash Image node with limited retries and Chinese pricing output.
"""

import base64
import io
import time

import numpy as np
import requests
import torch
from PIL import Image
from server import PromptServer
import comfy.utils

from ._image_gpt_common import format_credits_value, format_rmb_value


GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input regardless of format, intent, or abstraction as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


def tensor_to_base64(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]

    array = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def bytes_to_tensor(data):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)
    except Exception as e:
        print(f"[PD_NB2] Image decode error: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDGeminiNanoBanana2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Paste your ComfyUI API Key here"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A futuristic cityscape with flying cars and neon lights",
                }),
                "model": (["Nano Banana 2 (Gemini 3.1 Flash Image)"], {
                    "default": "Nano Banana 2 (Gemini 3.1 Flash Image)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0x7FFFFFFF,
                }),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "auto",
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "1K",
                }),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"], {
                    "default": "IMAGE",
                }),
                "thinking_level": (["MINIMAL", "HIGH"], {
                    "default": "MINIMAL"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "files": ("GEMINI_INPUT_FILES",),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": GEMINI_IMAGE_SYS_PROMPT,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "info", "price")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(
        self,
        comfy_api_key,
        prompt,
        model,
        seed,
        aspect_ratio,
        resolution,
        response_modalities,
        thinking_level,
        unique_id,
        images=None,
        files=None,
        system_prompt="",
    ):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update(10)

        if not comfy_api_key or not comfy_api_key.strip():
            return (torch.zeros((1, 512, 512, 3)), "错误: 缺少 ComfyUI API Key", "￥0.00")

        api_key = comfy_api_key.strip()
        api_model = "gemini-3.1-flash-image-preview"
        self._send_status(unique_id, "准备请求...")

        parts = [{"text": prompt}]

        if images is not None:
            for i in range(min(images.shape[0], 14)):
                b64_str = tensor_to_base64(images[i])
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_str
                    }
                })

        if files:
            for file_part in files:
                if hasattr(file_part, "model_dump"):
                    parts.append(file_part.model_dump(exclude_none=True))
                else:
                    parts.append(file_part)

        image_config = {
            "imageSize": resolution,
            "imageOutputOptions": {"mimeType": "image/png"}
        }
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        generation_config = {
            "responseModalities": (["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
            "imageConfig": image_config,
            "thinkingConfig": {"thinkingLevel": thinking_level}
        }

        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": generation_config
        }

        if system_prompt and system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt.strip()}]
            }

        headers = {
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.1",
        }

        base_url = f"https://api.comfy.org/proxy/vertexai/gemini/{api_model}"

        self._send_status(unique_id, "正在发送请求...")
        pbar.update(30)

        start_time = time.time()
        try:
            response = self._post_with_retries(base_url, headers, payload, unique_id)
            pbar.update(80)

            if response.status_code != 200:
                error_body = response.text
                error_msg = f"API Error {response.status_code}: {error_body}"
                self._send_status(unique_id, "请求失败")
                return (torch.zeros((1, 512, 512, 3)), error_msg, "N/A")

            result = response.json()
            image_tensors = []
            text_parts = []

            self._send_status(unique_id, "正在处理响应...")

            candidates = result.get("candidates", [])
            for cand in candidates:
                content = cand.get("content", {})
                parts_out = content.get("parts", [])
                for part in parts_out:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data:
                        data_b64 = inline_data.get("data")
                        if data_b64:
                            raw_data = base64.b64decode(data_b64)
                            image_tensors.append(bytes_to_tensor(raw_data))

                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])

            if not image_tensors:
                return (torch.zeros((1, 512, 512, 3)), "错误: 响应中没有图片", "N/A")

            final_image = torch.cat(image_tensors, dim=0)
            final_text = "\n".join(text_parts) if text_parts else "无文字描述"
            cost_info = self._calculate_cost(result, api_model, time.time() - start_time)

            pbar.update(100)
            self._send_status(unique_id, "成功")

            return (final_image, final_text, cost_info)

        except Exception as e:
            self._send_status(unique_id, "发生异常")
            return (torch.zeros((1, 512, 512, 3)), f"Error: {str(e)}", "N/A")

    def _send_status(self, node_id, text):
        try:
            PromptServer.instance.send_sync("progress_text", {"text": text, "node": node_id})
        except Exception:
            pass

    def _post_with_retries(self, url, headers, payload, unique_id, retries=3, timeout=90):
        last_error = None
        for attempt in range(retries):
            try:
                if attempt > 0:
                    self._send_status(unique_id, f"网络波动，正在重试 {attempt + 1}/{retries}...")
                return requests.post(
                    url,
                    headers={**headers, "Connection": "close"},
                    json=payload,
                    timeout=timeout,
                )
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                last_error = exc
                if attempt == retries - 1:
                    raise
                time.sleep(min(4, attempt + 1))
        raise last_error

    def _calculate_cost(self, result, model, duration):
        usage = result.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        details = usage.get("candidatesTokensDetails", [])
        thoughts_tokens = usage.get("thoughtsTokenCount", 0)

        prices = {"input": 0.5, "output_text": 3.0, "output_image": 60.0}

        text_tokens = thoughts_tokens
        img_tokens = 0

        for d in details:
            modality = d.get("modality")
            count = d.get("tokenCount", 0)
            if modality == "IMAGE":
                img_tokens += count
            else:
                text_tokens += count

        if not details and candidates_tokens > 0:
            text_tokens += candidates_tokens

        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        text_cost = (text_tokens / 1_000_000) * prices["output_text"]
        img_cost = (img_tokens / 1_000_000) * prices["output_image"]
        total_cost = input_cost + text_cost + img_cost

        info = [
            f"模型: {model}",
            f"状态: 成功",
            f"耗时: {duration:.2f} 秒",
            f"Tokens: 输入:{prompt_tokens:,} | 输出文本:{text_tokens:,} | 输出图像:{img_tokens:,}",
            f"实际价格(USD): ${total_cost:.6f}",
            f"实际价格(RMB): {format_rmb_value(total_cost)}",
            f"实际价格(Credits): {format_credits_value(total_cost)}",
        ]
        return "\n".join(info)


NODE_CLASS_MAPPINGS = {"PDGeminiNanoBanana2": PDGeminiNanoBanana2}
NODE_DISPLAY_NAME_MAPPINGS = {"PDGeminiNanoBanana2": "PD: Nano Banana 2 (ComfyUI Key)"}
