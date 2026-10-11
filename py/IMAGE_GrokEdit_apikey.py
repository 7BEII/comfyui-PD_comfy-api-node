import base64
import concurrent.futures
import time

import comfy.model_management as mm
import requests
import torch
from server import PromptServer

from ._image_gpt_common import (
    empty_image,
    format_credits_value,
    format_rmb_range,
    format_rmb_value,
    load_image_to_tensor,
    tensor_to_bytes,
    usd_to_comfy_credits,
)


BASE_URL = "https://api.comfy.org"
EDIT_ENDPOINT = "/proxy/xai/v1/images/edits"
MODELS = [
    "grok-imagine-image-2.0",
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image",
]
ASPECT_RATIOS = [
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
]
QUALITY_MODELS = {"grok-imagine-image-2.0"}
ASPECT_RATIO_NEEDS_MULTIPLE = {"grok-imagine-image-quality", "grok-imagine-image"}
MAX_INPUT_IMAGES = {
    "grok-imagine-image-2.0": 3,
    "grok-imagine-image-quality": 3,
    "grok-imagine-image-pro": 1,
    "grok-imagine-image": 3,
}


def tensor_to_data_url(image):
    encoded = base64.b64encode(tensor_to_bytes(image).getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def flatten_images(*inputs):
    images = []
    for tensor in inputs:
        if tensor is None:
            continue
        if tensor.ndim == 4:
            images.extend(tensor[index] for index in range(tensor.shape[0]))
        else:
            images.append(tensor)
    return images


def send_status(node_id, text):
    if node_id is None:
        return
    try:
        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        pass


def request_with_interrupt(session, method, url, node_id, status, start_time, **kwargs):
    mm.throw_exception_if_processing_interrupted()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(method, url, **kwargs)
    last_elapsed = -1
    try:
        while True:
            try:
                return future.result(timeout=0.2)
            except concurrent.futures.TimeoutError:
                mm.throw_exception_if_processing_interrupted()
                elapsed = int(time.monotonic() - start_time)
                if elapsed != last_elapsed:
                    send_status(node_id, f"{status}\nTime elapsed: {elapsed}s")
                    last_elapsed = elapsed
    except mm.InterruptProcessingException:
        future.cancel()
        session.close()
        send_status(node_id, f"Cancelled\nTime elapsed: {int(time.monotonic() - start_time)}s")
        raise
    except BaseException:
        future.cancel()
        session.close()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def estimate_price(model, resolution, quality, number_of_images):
    is_1k = resolution.lower() == "1k"
    if model == "grok-imagine-image-2.0":
        rate = (0.04 if is_1k else 0.06) if quality == "low" else (0.06 if is_1k else 0.08)
        base = 0.01
    elif model == "grok-imagine-image-quality":
        rate = 0.05 if is_1k else 0.07
        base = 0.01
    elif model == "grok-imagine-image-pro":
        rate = 0.07
        base = 0.002
    else:
        rate = 0.02
        base = 0.002

    output_price = rate * number_of_images
    if model == "grok-imagine-image-pro":
        price = base + output_price
        return price, price
    return base + output_price, 3 * base + output_price


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

    if usd is not None:
        lines = [
            f"Actual price (USD): ${usd:.6f}",
            f"Actual price (RMB): {format_rmb_value(usd)}",
        ]
        if credits is not None:
            lines.append(f"Credits used: {credits:.2f} credits")
        return "\n".join(lines)

    min_usd, max_usd = estimate_price(model, resolution, quality, number_of_images)
    if min_usd == max_usd:
        return "\n".join([
            f"Estimated price (USD): ${min_usd:.6f}",
            f"Estimated price (RMB): {format_rmb_value(min_usd)}",
            f"Estimated price (Credits): {format_credits_value(min_usd)}",
        ])
    return "\n".join([
        f"Estimated price (USD): ${min_usd:.6f}-${max_usd:.6f}",
        f"Estimated price (RMB): {format_rmb_range(min_usd, max_usd)}",
        f"Estimated price (Credits): 约 {usd_to_comfy_credits(min_usd):.2f}-{usd_to_comfy_credits(max_usd):.2f} credits",
    ])


class PDGrokImageEditAPIKey:
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
                "image_1": ("IMAGE",),
                "resolution": (["1K", "2K"], {"default": "1K"}),
                "quality": (["medium", "low"], {"default": "medium"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "show_price")
    FUNCTION = "edit_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def edit_image(
        self,
        api_key,
        prompt,
        model,
        image_1,
        resolution,
        quality,
        number_of_images,
        seed,
        aspect_ratio,
        unique_id,
        image_2=None,
        image_3=None,
    ):
        if not api_key or not api_key.strip():
            return (empty_image(), "Error: ComfyUI API Key is required")
        if not prompt or not prompt.strip():
            return (empty_image(), "Error: prompt is required")

        images = flatten_images(image_1, image_2, image_3)
        max_images = MAX_INPUT_IMAGES[model]
        if len(images) > max_images:
            return (empty_image(), f"Error: {model} supports at most {max_images} input image(s)")
        if aspect_ratio != "auto" and model in ASPECT_RATIO_NEEDS_MULTIPLE and len(images) == 1:
            return (empty_image(), "Error: custom aspect ratio requires multiple input images for this model")

        payload = {
            "model": model,
            "images": [{"url": tensor_to_data_url(image)} for image in images],
            "prompt": prompt.strip(),
            "resolution": resolution.lower(),
            "n": number_of_images,
            "seed": seed,
            "aspect_ratio": None if aspect_ratio == "auto" or model == "grok-imagine-image-pro" else aspect_ratio,
            "quality": quality if model in QUALITY_MODELS else None,
        }
        headers = {
            "X-API-KEY": api_key.strip(),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "close",
            "User-Agent": "ComfyUI-PD-Node/2.1",
        }

        start_time = time.monotonic()
        session = requests.Session()
        try:
            response = request_with_interrupt(
                session,
                session.post,
                f"{BASE_URL}{EDIT_ENDPOINT}",
                unique_id,
                "Grok image editing",
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
            print(f"[PD_GROK_EDIT] Completed {len(output_images)} image(s) in {elapsed:.2f}s")
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
    "PDGrokImageEditAPIKey": PDGrokImageEditAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGrokImageEditAPIKey": "PD-grok-image-edit",
}
