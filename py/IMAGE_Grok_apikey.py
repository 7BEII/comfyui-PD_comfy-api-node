import base64
import time

import comfy.model_management as mm
import requests
import torch

from .IMAGE_GrokEdit_apikey import (
    ASPECT_RATIOS,
    BASE_URL,
    MODELS,
    QUALITY_MODELS,
    request_with_interrupt,
    send_status,
)
from ._image_gpt_common import (
    empty_image,
    format_credits_value,
    format_rmb_value,
    load_image_to_tensor,
    usd_to_comfy_credits,
)


GENERATION_ENDPOINT = "/proxy/xai/v1/images/generations"


def estimate_price(model, resolution, quality, number_of_images):
    is_1k = resolution.lower() == "1k"
    if model == "grok-imagine-image-2.0":
        rate = (0.04 if is_1k else 0.06) if quality == "low" else (0.06 if is_1k else 0.08)
    elif model == "grok-imagine-image-quality":
        rate = 0.05 if is_1k else 0.07
    elif model == "grok-imagine-image-pro":
        rate = 0.07
    else:
        rate = 0.02
    return rate * number_of_images


def format_price(result, response, model, resolution, quality, number_of_images):
    usage = result.get("usage") or {}
    ticks = usage.get("cost_in_usd_ticks")
    credits_header = response.headers.get("X-Comfy-Credits-Used")

    credits = None
    try:
        credits = float(credits_header) if credits_header is not None else None
    except ValueError:
        pass

    usd = ticks / 10_000_000_000 if isinstance(ticks, (int, float)) else None
    if usd is None and credits is not None:
        usd = credits / usd_to_comfy_credits(1.0)
    if credits is None and usd is not None:
        credits = usd_to_comfy_credits(usd)

    if usd is None:
        usd = estimate_price(model, resolution, quality, number_of_images)
        return "\n".join([
            f"Estimated price (USD): ${usd:.6f}",
            f"Estimated price (RMB): {format_rmb_value(usd)}",
            f"Estimated price (Credits): {format_credits_value(usd)}",
        ])

    lines = [
        f"Actual price (USD): ${usd:.6f}",
        f"Actual price (RMB): {format_rmb_value(usd)}",
    ]
    if credits is not None:
        lines.append(f"Credits used: {credits:.2f} credits")
    return "\n".join(lines)


class PDGrokImageAPIKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your ComfyUI API Key",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (MODELS, {"default": "grok-imagine-image-2.0"}),
                "aspect_ratio": (ASPECT_RATIOS[1:], {"default": "1:1"}),
                "resolution": (["1K", "2K"], {"default": "1K"}),
                "quality": (["medium", "low"], {"default": "medium"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "show_price")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(
        self,
        api_key,
        prompt,
        model,
        aspect_ratio,
        resolution,
        quality,
        number_of_images,
        seed,
        unique_id,
    ):
        if not api_key or not api_key.strip():
            return (empty_image(), "Error: ComfyUI API Key is required")
        if not prompt or not prompt.strip():
            return (empty_image(), "Error: prompt is required")

        payload = {
            "model": model,
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "n": number_of_images,
            "seed": seed,
            "response_format": "url",
            "resolution": resolution.lower(),
            "quality": quality if model in QUALITY_MODELS else None,
        }
        headers = {
            "X-API-KEY": api_key.strip(),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.1",
        }

        start_time = time.monotonic()
        session = requests.Session()
        try:
            response = request_with_interrupt(
                session,
                session.post,
                f"{BASE_URL}{GENERATION_ENDPOINT}",
                unique_id,
                "Waiting for Grok server",
                start_time,
                headers=headers,
                json=payload,
                timeout=600,
            )
            if response.status_code != 200:
                send_status(unique_id, f"Failed\nTime elapsed: {int(time.monotonic() - start_time)}s")
                return (empty_image(), f"API Error {response.status_code}: {response.text[:500]}")

            result = response.json()
            output_images = []
            for item in result.get("data") or []:
                if item.get("url"):
                    download = request_with_interrupt(
                        session,
                        session.get,
                        item["url"],
                        unique_id,
                        "Downloading result",
                        start_time,
                        timeout=120,
                    )
                    download.raise_for_status()
                    output_images.append(load_image_to_tensor(download.content))
                elif item.get("b64_json"):
                    output_images.append(load_image_to_tensor(base64.b64decode(item["b64_json"])))

            if not output_images:
                send_status(unique_id, f"Failed\nTime elapsed: {int(time.monotonic() - start_time)}s")
                return (empty_image(), f"Error: response contained no images: {str(result)[:500]}")

            image = torch.cat(output_images, dim=0)
            price = format_price(result, response, model, resolution, quality, number_of_images)
            elapsed = time.monotonic() - start_time
            send_status(unique_id, f"Completed\nTime elapsed: {int(elapsed)}s")
            print(f"[PD_GROK_IMAGE] {model} completed {len(output_images)} image(s) in {elapsed:.2f}s")
            return (image, price)
        except mm.InterruptProcessingException:
            raise
        except requests.exceptions.Timeout:
            send_status(unique_id, f"Timed out\nTime elapsed: {int(time.monotonic() - start_time)}s")
            return (empty_image(), "Error: request timeout (600s)")
        except requests.exceptions.ConnectionError as error:
            send_status(unique_id, f"Connection failed\nTime elapsed: {int(time.monotonic() - start_time)}s")
            return (empty_image(), f"Error: connection failed - {error}")
        except Exception as error:
            send_status(unique_id, f"Failed\nTime elapsed: {int(time.monotonic() - start_time)}s")
            return (empty_image(), f"Error: {error}")
        finally:
            session.close()


NODE_CLASS_MAPPINGS = {
    "PDGrokImageAPIKey": PDGrokImageAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGrokImageAPIKey": "PD-grok-text to image",
}
