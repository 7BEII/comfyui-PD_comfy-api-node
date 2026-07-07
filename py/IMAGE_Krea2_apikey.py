"""
PD: Krea 2 image-generation nodes using a ComfyUI API key.
"""

import re
import time

import requests

from ._image_gpt_common import (
    empty_image,
    format_credits_value,
    format_rmb_value,
    load_image_to_tensor,
    tensor_to_bytes,
)


BASE_URL = "https://api.comfy.org"
KREA_MODEL_ENDPOINTS = {
    "Krea 2 Medium": "/proxy/krea/generate/image/krea/krea-2/medium",
    "Krea 2 Medium Turbo": "/proxy/krea/generate/image/krea/krea-2/medium-turbo",
    "Krea 2 Large": "/proxy/krea/generate/image/krea/krea-2/large",
}
KREA_ASPECT_RATIOS = ["1:1", "4:3", "3:2", "16:9", "2.35:1", "4:5", "2:3", "9:16"]
KREA_RESOLUTIONS = ["1K"]
KREA_CREATIVITY_LEVELS = ["raw", "low", "medium", "high"]
KREA_QUEUED_STATUSES = {"backlogged", "queued", "scheduled", "running", "processing", "in_progress"}
KREA_DONE_STATUSES = {"succeeded", "success", "completed", "complete", "done"}
KREA_FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
KREA_PRICE_USD = {
    "Krea 2 Medium Turbo": {"text": 0.015, "style": 0.0175, "moodboard": 0.02},
    "Krea 2 Medium": {"text": 0.03, "style": 0.035, "moodboard": 0.04},
    "Krea 2 Large": {"text": 0.06, "style": 0.065, "moodboard": 0.07},
}
KREA_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _headers(api_key, content_type="application/json"):
    headers = {
        "X-API-KEY": api_key.strip(),
        "Accept": "application/json",
        "Connection": "close",
        "User-Agent": "ComfyUI-PD-Node/2.1",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _estimate_krea_price(model, has_style_reference, has_moodboard):
    rates = KREA_PRICE_USD.get(model, KREA_PRICE_USD["Krea 2 Medium"])
    if has_moodboard:
        return rates["moodboard"]
    if has_style_reference:
        return rates["style"]
    return rates["text"]


def _upload_krea_style_asset(api_key, image):
    image_bytes = tensor_to_bytes(image, max_pixels=2048 * 2048).getvalue()
    response = requests.post(
        f"{BASE_URL}/proxy/krea/assets",
        headers=_headers(api_key, content_type=None),
        files=[("file", ("style_reference.png", image_bytes, "image/png"))],
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Krea asset upload error {response.status_code}: {response.text[:500]}")

    result = response.json()
    image_url = result.get("image_url")
    if not image_url:
        raise RuntimeError(f"Krea asset response missing image_url: {str(result)[:500]}")
    return image_url


def _poll_krea_job(api_key, job_id):
    status_url = f"{BASE_URL}/proxy/krea/jobs/{job_id}"
    for attempt in range(240):
        time.sleep(3 if attempt else 1)
        response = requests.get(status_url, headers=_headers(api_key), timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Krea poll error {response.status_code}: {response.text[:500]}")

        result = response.json()
        status = str(result.get("status") or "").lower()
        if status in KREA_DONE_STATUSES:
            return result
        if status in KREA_FAILED_STATUSES:
            raise RuntimeError(f"Krea job {status}: {str(result)[:500]}")
        if status not in KREA_QUEUED_STATUSES:
            print(f"[PD_KREA2] Waiting on unknown Krea status: {status}")

    raise RuntimeError("Krea job timed out while polling")


class PDKrea2ImageAPIKey:
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
                    "tooltip": "Text prompt for Krea 2 image generation.",
                }),
                "model": (list(KREA_MODEL_ENDPOINTS.keys()), {
                    "default": "Krea 2 Medium Turbo",
                    "tooltip": "Medium Turbo is fastest; Large is best for photorealism.",
                }),
                "aspect_ratio": (KREA_ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "Output aspect ratio.",
                }),
                "resolution": (KREA_RESOLUTIONS, {
                    "default": "1K",
                    "tooltip": "Resolution scale.",
                }),
                "creativity": (KREA_CREATIVITY_LEVELS, {
                    "default": "medium",
                    "tooltip": "Prompt interpretation strength.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "tooltip": "Random seed for reproducibility.",
                }),
                "moodboard_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional Krea moodboard UUID from the Krea website.",
                }),
                "moodboard_strength": ("FLOAT", {
                    "default": 0.35,
                    "min": -0.5,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "Moodboard influence; ignored when moodboard_id is empty.",
                }),
            },
            "optional": {
                "style_reference": ("KREA_STYLE_REF", {
                    "tooltip": "Optional chained style references from PD: Krea 2 Style Reference.",
                }),
            },
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
        aspect_ratio,
        resolution,
        creativity,
        seed,
        moodboard_id,
        moodboard_strength,
        style_reference=None,
    ):
        if not comfy_api_key or not comfy_api_key.strip():
            return (empty_image(), "Error: ComfyUI API Key is required", "")
        if not prompt or not prompt.strip():
            return (empty_image(), "Error: prompt is required", "")

        endpoint = KREA_MODEL_ENDPOINTS.get(model)
        if endpoint is None:
            return (empty_image(), f"Error: unknown Krea model: {model}", "")

        moodboard_id = (moodboard_id or "").strip()
        if moodboard_id and not KREA_UUID_RE.match(moodboard_id):
            return (empty_image(), f"Error: moodboard_id must be a UUID: {moodboard_id}", "")

        image_style_references = None
        if style_reference:
            if len(style_reference) > 10:
                return (empty_image(), f"Error: Krea accepts at most 10 style references, got {len(style_reference)}", "")
            image_style_references = [
                {"url": ref["url"], "strength": float(ref["strength"])}
                for ref in style_reference
            ]

        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "seed": int(seed),
            "creativity": creativity,
        }
        if moodboard_id:
            payload["moodboards"] = [{"id": moodboard_id, "strength": float(moodboard_strength)}]
        if image_style_references:
            payload["image_style_references"] = image_style_references

        start_time = time.time()

        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                headers=_headers(comfy_api_key),
                json=payload,
                timeout=120,
            )
            if response.status_code != 200:
                return (empty_image(), f"API Error {response.status_code}: {response.text[:500]}", "")

            initial = response.json()
            job_id = initial.get("job_id")
            if not job_id:
                return (empty_image(), f"Error: job_id missing in response: {str(initial)[:500]}", "")

            job = _poll_krea_job(comfy_api_key, job_id)
            result = job.get("result") or {}
            urls = result.get("urls") or []
            if not urls:
                return (empty_image(), f"Error: Krea job {job_id} completed without image urls", "")

            image = load_image_to_tensor(urls[0])
            elapsed = time.time() - start_time
            usd = _estimate_krea_price(model, bool(image_style_references), bool(moodboard_id))

            info_lines = [
                f"Model: {model}",
                "Status: Success",
                f"Job ID: {job_id}",
                f"Time: {elapsed:.2f}s",
                f"Aspect Ratio: {aspect_ratio}",
                f"Resolution: {resolution}",
                f"Creativity: {creativity}",
                f"Style References: {len(image_style_references or [])}",
                f"Moodboard: {'on' if moodboard_id else 'off'}",
            ]
            price_lines = [
                f"Estimated price (USD): ${usd:.6f}",
                f"Estimated price (RMB): {format_rmb_value(usd)}",
                f"Estimated price (Credits): {format_credits_value(usd)}",
            ]
            return (image, "\n".join(info_lines), "\n".join(price_lines))

        except requests.exceptions.Timeout:
            return (empty_image(), "Error: Request timeout", "")
        except requests.exceptions.ConnectionError as e:
            return (empty_image(), f"Error: Connection failed - {str(e)}", "")
        except Exception as e:
            import traceback

            traceback.print_exc()
            return (empty_image(), f"Error: {str(e)}", "")


class PDKrea2StyleReferenceAPIKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your ComfyUI API Key",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Reference image whose style influences Krea 2 generation.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Reference strength; negative values invert the style influence.",
                }),
            },
            "optional": {
                "style_reference": ("KREA_STYLE_REF", {
                    "tooltip": "Optional incoming chain; this node appends one more reference.",
                }),
            },
        }

    RETURN_TYPES = ("KREA_STYLE_REF",)
    RETURN_NAMES = ("style_reference",)
    FUNCTION = "make_style_reference"
    CATEGORY = "PD_Tools/Image_Generation"

    def make_style_reference(self, comfy_api_key, image, strength, style_reference=None):
        if not comfy_api_key or not comfy_api_key.strip():
            raise ValueError("ComfyUI API Key is required")

        chain = list(style_reference) if style_reference else []
        if len(chain) >= 10:
            raise ValueError("Krea accepts at most 10 style references in one generation")

        url = _upload_krea_style_asset(comfy_api_key, image)
        chain.append({"url": url, "strength": float(strength)})
        return (chain,)


NODE_CLASS_MAPPINGS = {
    "PDKrea2ImageAPIKey": PDKrea2ImageAPIKey,
    "PDKrea2StyleReferenceAPIKey": PDKrea2StyleReferenceAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDKrea2ImageAPIKey": "PD: Krea 2 Image (comfyui_apikey)",
    "PDKrea2StyleReferenceAPIKey": "PD: Krea 2 Style Reference (comfyui_apikey)",
}
