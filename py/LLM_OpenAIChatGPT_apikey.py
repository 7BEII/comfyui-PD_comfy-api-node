"""
PD: OpenAI ChatGPT text node using a ComfyUI API key.
"""

import base64
import time

import requests

from ._image_gpt_common import format_credits_value, format_rmb_value, tensor_to_bytes


BASE_URL = "https://api.comfy.org"
RESPONSES_ENDPOINT = "/proxy/openai/v1/responses"

OPENAI_CHAT_MODELS = [
    "gpt-5.5-pro",
    "gpt-5.5",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    "o3",
    "o1-pro",
    "o1",
]

OPENAI_CHAT_PRICES_PER_MILLION = {
    "gpt-5.5-pro": (30.0, 180.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.4),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "o4-mini": (1.1, 4.4),
    "o3": (10.0, 40.0),
    "o1-pro": (150.0, 600.0),
    "o1": (15.0, 60.0),
}


def _headers(api_key):
    return {
        "X-API-KEY": api_key.strip(),
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "ComfyUI-PD-Node/2.1",
    }


def _image_content_from_tensor(image):
    contents = []
    batch_size = image.shape[0] if image.ndim == 4 else 1
    for i in range(batch_size):
        tensor = image[i : i + 1] if image.ndim == 4 else image
        image_bytes = tensor_to_bytes(tensor).getvalue()
        contents.append({
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}",
        })
    return contents


def _extract_text(response):
    for output in response.get("output") or []:
        if output.get("type") != "message":
            continue
        for content in output.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    return ""


def _estimate_price(model, usage):
    if not isinstance(usage, dict):
        return None
    rates = OPENAI_CHAT_PRICES_PER_MILLION.get(model)
    if rates is None:
        return None

    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0


class PDOpenAIChatGPTAPIKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your ComfyUI API Key",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Text input to the model.",
                }),
                "model": (OPENAI_CHAT_MODELS, {
                    "default": "gpt-5",
                    "tooltip": "OpenAI model used through the Comfy API proxy.",
                }),
                "system_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional instructions for the model.",
                }),
                "max_output_tokens": ("INT", {
                    "default": 4096,
                    "min": 16,
                    "max": 16384,
                    "step": 1,
                    "tooltip": "Upper bound for generated tokens.",
                }),
                "truncation": (["auto", "disabled"], {
                    "default": "auto",
                    "tooltip": "Context truncation strategy.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional image batch to use as context.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "info", "price")
    FUNCTION = "generate_text"
    CATEGORY = "PD_Tools/Text_Generation"

    def generate_text(
        self,
        comfy_api_key,
        prompt,
        model,
        system_prompt,
        max_output_tokens,
        truncation,
        images=None,
    ):
        if not comfy_api_key or not comfy_api_key.strip():
            return ("", "Error: ComfyUI API Key is required", "")
        if not prompt or not prompt.strip():
            return ("", "Error: prompt is required", "")

        content = [{"type": "input_text", "text": prompt}]
        if images is not None:
            content.extend(_image_content_from_tensor(images))

        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "store": True,
            "stream": False,
            "max_output_tokens": int(max_output_tokens),
            "truncation": truncation,
        }
        if system_prompt and system_prompt.strip():
            payload["instructions"] = system_prompt.strip()

        start_time = time.time()

        try:
            response = requests.post(
                f"{BASE_URL}{RESPONSES_ENDPOINT}",
                headers=_headers(comfy_api_key),
                json=payload,
                timeout=180,
            )
            if response.status_code != 200:
                return ("", f"API Error {response.status_code}: {response.text[:500]}", "")

            result = response.json()
            response_id = result.get("id")
            if response_id and result.get("status") not in ("completed", "incomplete"):
                result = self._poll_response(comfy_api_key, response_id)

            if result.get("error"):
                error = result["error"]
                return ("", f"Error: {error.get('code')}: {error.get('message')}", "")

            text = _extract_text(result)
            if not text:
                return ("", f"Error: no text output in response: {str(result)[:500]}", "")

            elapsed = time.time() - start_time
            usage = result.get("usage") or {}
            usd = _estimate_price(model, usage)

            info_lines = [
                f"Model: {model}",
                "Status: Success",
                f"Response ID: {response_id or ''}",
                f"Time: {elapsed:.2f}s",
            ]
            if usage:
                info_lines.append(f"Usage: {usage}")

            if usd is None:
                price = "Estimated price: token-based"
            else:
                price = "\n".join([
                    f"Estimated price (USD): ${usd:.6f}",
                    f"Estimated price (RMB): {format_rmb_value(usd)}",
                    f"Estimated price (Credits): {format_credits_value(usd)}",
                ])

            return (text, "\n".join(info_lines), price)

        except requests.exceptions.Timeout:
            return ("", "Error: Request timeout", "")
        except requests.exceptions.ConnectionError as e:
            return ("", f"Error: Connection failed - {str(e)}", "")
        except Exception as e:
            import traceback

            traceback.print_exc()
            return ("", f"Error: {str(e)}", "")

    def _poll_response(self, api_key, response_id):
        status_url = f"{BASE_URL}{RESPONSES_ENDPOINT}/{response_id}"
        for attempt in range(180):
            time.sleep(2 if attempt else 1)
            response = requests.get(status_url, headers=_headers(api_key), timeout=120)
            if response.status_code != 200:
                raise RuntimeError(f"Poll Error {response.status_code}: {response.text[:500]}")
            result = response.json()
            status = result.get("status")
            if status in ("completed", "incomplete", "failed", "cancelled"):
                return result
        raise RuntimeError("OpenAI response timed out while polling")


NODE_CLASS_MAPPINGS = {
    "PDOpenAIChatGPTAPIKey": PDOpenAIChatGPTAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIChatGPTAPIKey": "PD: OpenAI ChatGPT (comfyui_apikey)",
}
