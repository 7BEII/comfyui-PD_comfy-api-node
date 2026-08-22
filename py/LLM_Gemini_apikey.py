"""Google Gemini multimodal text node using a ComfyUI API key."""

import queue
import threading
import time

import requests

from comfy_api.latest import Types
from comfy_api_nodes.util import audio_to_base64_string, tensor_to_base64_string, video_to_base64_string
from comfy_api_nodes.util._helpers import is_processing_interrupted
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted

from ._image_gpt_common import USD_TO_CNY_RATE, format_credits_value


GEMINI_MODELS = {
    "Gemini 3.1 Pro": "gemini-3.1-pro-preview",
    "Gemini 3.1 Flash-Lite": "gemini-3.1-flash-lite-preview",
}

# USD per million tokens: (input, text/thinking output).
GEMINI_TOKEN_PRICES = {
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.1-flash-lite-preview": (0.25, 1.5),
}


def _part_to_dict(part):
    if hasattr(part, "model_dump"):
        return part.model_dump(by_alias=True, exclude_none=True)
    return part


def _response_text(result):
    text = []
    for candidate in result.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text"):
                text.append(str(part["text"]))
    return "\n".join(text)


def _estimated_price(model_id, usage):
    rates = GEMINI_TOKEN_PRICES.get(model_id)
    if rates is None or not isinstance(usage, dict):
        return None

    input_tokens = int(usage.get("promptTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    thinking_tokens = int(usage.get("thoughtsTokenCount") or 0)
    input_rate, output_rate = rates
    return (input_tokens * input_rate + (output_tokens + thinking_tokens) * output_rate) / 1_000_000.0


def _post_with_interrupt(url, headers, payload):
    result_queue = queue.Queue(maxsize=1)

    def send_request():
        try:
            result_queue.put(requests.post(url, headers=headers, json=payload, timeout=300))
        except Exception as error:
            result_queue.put(error)

    threading.Thread(target=send_request, daemon=True).start()
    while True:
        if is_processing_interrupted():
            raise ProcessingInterrupted("Task cancelled")
        try:
            result = result_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if isinstance(result, Exception):
            raise result
        return result


class PDGeminiAPIKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your ComfyUI API Key",
                    "tooltip": "The key is stored in the workflow when the workflow is saved.",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "model": (list(GEMINI_MODELS), {
                    "default": "Gemini 3.1 Pro",
                }),
                "thinking_level": (["LOW", "HIGH"], {
                    "default": "HIGH",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "max_output_tokens": ("INT", {
                    "default": 32768,
                    "min": 16,
                    "max": 65536,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "control_after_generate": True,
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "audio_1": ("AUDIO",),
                "video_1": ("VIDEO",),
                "files": ("GEMINI_INPUT_FILES",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "price")
    FUNCTION = "generate_text"
    CATEGORY = "PD_Tools/Text_Generation"

    def generate_text(
        self,
        comfy_api_key,
        prompt,
        model,
        thinking_level,
        temperature,
        top_p,
        max_output_tokens,
        seed,
        system_prompt,
        image_1=None,
        audio_1=None,
        video_1=None,
        files=None,
    ):
        comfy_api_key = comfy_api_key.strip()
        if not comfy_api_key:
            return ("Error: ComfyUI API Key is required", "")
        if not prompt.strip():
            return ("Error: prompt is required", "")

        model_id = GEMINI_MODELS[model]
        parts = [{"text": prompt}]

        if image_1 is not None:
            if is_processing_interrupted():
                raise ProcessingInterrupted("Task cancelled")
            images = image_1 if image_1.ndim == 4 else image_1.unsqueeze(0)
            for image in images:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": tensor_to_base64_string(image),
                    }
                })

        if audio_1 is not None:
            if is_processing_interrupted():
                raise ProcessingInterrupted("Task cancelled")
            waveform = audio_1["waveform"]
            for index in range(waveform.shape[0]):
                audio = {
                    "waveform": waveform[index].unsqueeze(0),
                    "sample_rate": audio_1["sample_rate"],
                }
                parts.append({
                    "inlineData": {
                        "mimeType": "audio/mpeg",
                        "data": audio_to_base64_string(audio, container_format="mp3", codec_name="libmp3lame"),
                    }
                })

        if video_1 is not None:
            if is_processing_interrupted():
                raise ProcessingInterrupted("Task cancelled")
            parts.append({
                "inlineData": {
                    "mimeType": "video/mp4",
                    "data": video_to_base64_string(
                        video_1,
                        container_format=Types.VideoContainer.MP4,
                        codec=Types.VideoCodec.H264,
                    ),
                }
            })

        if files:
            parts.extend(_part_to_dict(part) for part in files)

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "topP": float(top_p),
                "maxOutputTokens": int(max_output_tokens),
                "seed": int(seed) if seed > 0 else None,
                "thinkingConfig": {"thinkingLevel": thinking_level},
            },
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}

        url = f"https://api.comfy.org/proxy/vertexai/gemini/{model_id}"
        headers = {
            "X-API-KEY": comfy_api_key,
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.1",
        }

        started = time.time()
        try:
            response = _post_with_interrupt(url, headers, payload)
            if response.status_code != 200:
                return (f"API Error {response.status_code}: {response.text[:1000]}", "")

            result = response.json()
            text = _response_text(result)
            if not text:
                return (f"Error: no text output in response: {str(result)[:1000]}", "")

            usage = result.get("usageMetadata") or {}
            usd = _estimated_price(model_id, usage)
            price_lines = [
                f"Model: {result.get('modelVersion') or model_id}",
                f"Time: {time.time() - started:.2f}s",
                f"Input tokens: {int(usage.get('promptTokenCount') or 0)}",
                f"Output tokens: {int(usage.get('candidatesTokenCount') or 0)}",
                f"Thinking tokens: {int(usage.get('thoughtsTokenCount') or 0)}",
            ]
            if usd is not None:
                price_lines.extend([
                    f"Estimated price (USD): ${usd:.6f}",
                    f"Estimated price (RMB): 约 ￥{usd * USD_TO_CNY_RATE:.6f}",
                    f"Estimated price (Credits): {format_credits_value(usd)}",
                ])
            else:
                price_lines.append("Estimated price: unavailable")
            return (text, "\n".join(price_lines))
        except requests.exceptions.Timeout:
            return ("Error: request timeout (300s)", "")
        except requests.exceptions.ConnectionError as error:
            return (f"Error: connection failed - {error}", "")
        except ProcessingInterrupted:
            raise
        except Exception as error:
            return (f"Error: {error}", "")


NODE_CLASS_MAPPINGS = {
    "PDGeminiAPIKey": PDGeminiAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGeminiAPIKey": "PD: Google Gemini (ComfyUI API Key)",
}
