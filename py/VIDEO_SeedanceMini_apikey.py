"""
PD: Seedance 2.0 Text-to-Video node using a ComfyUI API key.
"""

import io
import time

import requests
import torch

try:
    from comfy_api.latest import InputImpl
except Exception:
    InputImpl = None

from ._image_gpt_common import empty_image, format_credits_value, format_rmb_value


BASE_URL = "https://api.comfy.org"
TASK_ENDPOINT = "/proxy/byteplus/api/v3/contents/generations/tasks"
SEEDANCE2_STATUS_ENDPOINT = "/proxy/byteplus-seedance2/api/v3/contents/generations/tasks"
SEEDANCE_MODELS = {
    "Seedance 2.0": "dreamina-seedance-2-0-260128",
    "Seedance 2.0 Fast": "dreamina-seedance-2-0-fast-260128",
    "Seedance 2.0 Mini": "dreamina-seedance-2-0-mini",
}
SEEDANCE_MODEL_RESOLUTIONS = {
    "Seedance 2.0": ["480p", "720p", "1080p", "4k"],
    "Seedance 2.0 Fast": ["480p", "720p"],
    "Seedance 2.0 Mini": ["480p", "720p"],
}
SEEDANCE2_PRICE_PER_1K_TOKENS = {
    ("dreamina-seedance-2-0-260128", "480p"): 0.007,
    ("dreamina-seedance-2-0-260128", "720p"): 0.007,
    ("dreamina-seedance-2-0-260128", "1080p"): 0.0077,
    ("dreamina-seedance-2-0-260128", "4k"): 0.004,
    ("dreamina-seedance-2-0-fast-260128", "480p"): 0.0056,
    ("dreamina-seedance-2-0-fast-260128", "720p"): 0.0056,
    ("dreamina-seedance-2-0-mini", "480p"): 0.0035,
    ("dreamina-seedance-2-0-mini", "720p"): 0.0035,
}


def _download_video_to_output(video_url, headers):
    request_url = video_url
    request_headers = {}
    if isinstance(video_url, str) and video_url.startswith("/"):
        request_url = f"{BASE_URL}{video_url}"
        request_headers = headers

    response = requests.get(request_url, headers=request_headers, timeout=300)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    buffer.seek(0)

    if InputImpl is not None:
        return InputImpl.VideoFromFile(buffer)

    return buffer


def _estimate_seedance_usd(model_id, resolution, duration, usage=None):
    token_rates = {
        "480p": 10044,
        "720p": 21600,
        "1080p": 48800,
        "4k": 195200,
    }
    price_per_1k = SEEDANCE2_PRICE_PER_1K_TOKENS.get((model_id, resolution))
    if price_per_1k is None:
        return 0.0

    if isinstance(usage, dict):
        total_tokens = usage.get("total_tokens") or usage.get("completion_tokens") or 0
        if total_tokens:
            return total_tokens * price_per_1k / 1000.0

    return duration * token_rates.get(resolution, token_rates["720p"]) * price_per_1k / 1000.0


class PDSeedanceMiniTextToVideoAPIKey:
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
                    "tooltip": "Text prompt for Seedance 2.0 video generation.",
                }),
                "model": (list(SEEDANCE_MODELS.keys()), {
                    "default": "Seedance 2.0 Fast",
                    "tooltip": "Seedance 2.0 for quality, Fast for speed, Mini for lowest cost.",
                }),
                "resolution": (["480p", "720p", "1080p", "4k"], {
                    "default": "720p",
                    "tooltip": "Seedance 2.0 supports 480p/720p/1080p/4k. Fast and Mini support 480p/720p.",
                }),
                "ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {
                    "default": "16:9",
                    "tooltip": "Output video aspect ratio.",
                }),
                "duration": ("INT", {
                    "default": 7,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Output duration in seconds.",
                }),
                "generate_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable audio generation for the output video.",
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to add a watermark.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "Seed controls rerun behavior; Seedance results are still non-deterministic.",
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "info", "price")
    FUNCTION = "generate_video"
    CATEGORY = "PD_Tools/Video_Generation"

    def generate_video(
        self,
        comfy_api_key,
        prompt,
        model,
        resolution,
        ratio,
        duration,
        generate_audio,
        watermark,
        seed,
    ):
        if not comfy_api_key or not comfy_api_key.strip():
            return (None, "Error: ComfyUI API Key is required", "")

        if not prompt or not prompt.strip():
            return (None, "Error: prompt is required", "")

        allowed_resolutions = SEEDANCE_MODEL_RESOLUTIONS.get(model, [])
        if resolution not in allowed_resolutions:
            return (None, f"Error: {model} supports {', '.join(allowed_resolutions)} only", "")

        model_id = SEEDANCE_MODELS[model]
        api_key = comfy_api_key.strip()
        headers = {
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "close",
            "User-Agent": "ComfyUI-PD-Node/2.1",
        }

        payload = {
            "model": model_id,
            "content": [
                {
                    "type": "text",
                    "text": prompt.strip(),
                }
            ],
            "generate_audio": bool(generate_audio),
            "resolution": resolution,
            "ratio": ratio,
            "duration": int(duration),
            "seed": int(seed),
            "watermark": bool(watermark),
        }

        start_time = time.time()

        try:
            create_response = requests.post(
                f"{BASE_URL}{TASK_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if create_response.status_code != 200:
                error_msg = f"API Error {create_response.status_code}: {create_response.text[:500]}"
                print(f"[PD_SEEDANCE_MINI] {error_msg}")
                return (None, error_msg, "")

            create_result = create_response.json()
            task_id = create_result.get("id")
            if not task_id:
                return (None, f"Error: task id missing in response: {str(create_result)[:500]}", "")

            status_url = f"{BASE_URL}{SEEDANCE2_STATUS_ENDPOINT}/{task_id}"
            final_result = None
            max_attempts = 480

            for attempt in range(max_attempts):
                time.sleep(9 if attempt else 2)
                status_response = requests.get(status_url, headers=headers, timeout=120)

                if status_response.status_code != 200:
                    error_msg = f"Poll Error {status_response.status_code}: {status_response.text[:500]}"
                    print(f"[PD_SEEDANCE_MINI] {error_msg}")
                    return (None, error_msg, "")

                status_result = status_response.json()
                status = status_result.get("status")
                if status == "succeeded":
                    final_result = status_result
                    break
                if status in ("failed", "cancelled"):
                    error = status_result.get("error") or {}
                    message = error.get("message") or error.get("code") or str(status_result)
                    return (None, f"Error: Seedance task {status}: {message}", "")

            if final_result is None:
                return (None, "Error: Seedance task timed out while polling", "")

            content = final_result.get("content") or {}
            video_url = content.get("video_url")
            if not video_url:
                return (None, f"Error: video_url missing in response: {str(final_result)[:500]}", "")

            video = _download_video_to_output(video_url, headers)
            elapsed = time.time() - start_time
            usage = final_result.get("usage") or {}
            usd = _estimate_seedance_usd(model_id, resolution, duration, usage)

            info_lines = [
                f"Model: {model}",
                f"Model ID: {model_id}",
                "Status: Success",
                f"Task ID: {task_id}",
                f"Time: {elapsed:.2f}s",
                f"Resolution: {resolution}",
                f"Ratio: {ratio}",
                f"Duration: {duration}s",
                f"Audio: {'on' if generate_audio else 'off'}",
                f"Watermark: {'on' if watermark else 'off'}",
            ]

            if usage:
                info_lines.append(f"Usage: {usage}")

            price_lines = [
                f"Estimated price (USD): ${usd:.6f}",
                f"Estimated price (RMB): {format_rmb_value(usd)}",
                f"Estimated price (Credits): {format_credits_value(usd)}",
            ]

            return (video, "\n".join(info_lines), "\n".join(price_lines))

        except requests.exceptions.Timeout:
            return (None, "Error: Request timeout", "")
        except requests.exceptions.ConnectionError as e:
            return (None, f"Error: Connection failed - {str(e)}", "")
        except Exception as e:
            import traceback

            traceback.print_exc()
            return (None, f"Error: {str(e)}", "")


NODE_CLASS_MAPPINGS = {
    "PDSeedanceMiniTextToVideoAPIKey": PDSeedanceMiniTextToVideoAPIKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDSeedanceMiniTextToVideoAPIKey": "PD: Seedance 2.0 T2V (comfyui_apikey)"
}
