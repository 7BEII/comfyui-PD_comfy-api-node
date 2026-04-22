import base64
import io

import numpy as np
import requests
import torch
from PIL import Image


GPT_IMAGE_EDIT_MAX_PIXELS = 2048 * 2048
GPT_IMAGE_2_PRESET_SIZES = [
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
    "auto",
]

# USD/CNY reference rate used for RMB estimation in node info output.
# Based on the most recent publicly indexed April 2026 daily spot pages we could verify:
# 2026-04-03: 1 USD ≈ 6.88154 CNY
USD_TO_CNY_RATE = 6.8815


def empty_image():
    return torch.zeros((1, 512, 512, 3))


def tensor_to_bytes(tensor, max_pixels=GPT_IMAGE_EDIT_MAX_PIXELS):
    if tensor.ndim == 4:
        tensor = tensor[0]

    height, width = tensor.shape[:2]
    current_pixels = height * width
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_height = max(1, int(height * scale))
        new_width = max(1, int(width * scale))
        array = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        array = np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        image = Image.fromarray(array)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered


def load_image_to_tensor(image_data):
    try:
        if isinstance(image_data, str) and image_data.startswith("http"):
            response = requests.get(image_data, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            img = Image.open(io.BytesIO(image_data))

        img = img.convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)
    except Exception:
        return empty_image()


def mask_to_png_bytes(mask):
    mask_tensor = mask
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]

    mask_h, mask_w = mask_tensor.shape
    rgba = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
    mask_np = mask_tensor.cpu().numpy()
    alpha = ((1.0 - mask_np) * 255).astype(np.uint8)
    rgba[..., 3] = alpha

    mask_pil = Image.fromarray(rgba)
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format="PNG")
    mask_buffer.seek(0)
    return mask_buffer


def decode_response_images(result):
    data_list = result.get("data", [])
    image_tensors = []
    for item in data_list:
        if "url" in item:
            img_tensor = load_image_to_tensor(item["url"])
            image_tensors.append(img_tensor)
        elif "b64_json" in item:
            img_data = base64.b64decode(item["b64_json"])
            img_tensor = load_image_to_tensor(img_data)
            image_tensors.append(img_tensor)
    return image_tensors


def estimate_price_range_1_or_1_5(model, quality, n):
    ranges = {
        "gpt-image-1": {
            "low": (0.011, 0.016),
            "medium": (0.042, 0.063),
            "high": (0.167, 0.250),
        },
        "gpt-image-1.5": {
            "low": (0.009, 0.013),
            "medium": (0.034, 0.050),
            "high": (0.133, 0.200),
        },
    }
    model_ranges = ranges.get(model, ranges["gpt-image-1"])
    min_cost, max_cost = model_ranges.get(quality, model_ranges["medium"])
    return (min_cost * n, max_cost * n)


def calculate_token_price_1_or_1_5(model, n):
    if model == "gpt-image-1":
        cost = (1000 * 10 + 2000 * 40) / 1_000_000
    else:
        cost = (1000 * 8 + 2000 * 32) / 1_000_000
    return cost * n


def estimate_price_range_2(quality, n):
    ranges = {
        "auto": (0.005, 0.250),
        "low": (0.005, 0.030),
        "medium": (0.041, 0.120),
        "high": (0.165, 0.250),
    }
    min_cost, max_cost = ranges.get(quality, ranges["auto"])
    return (min_cost * n, max_cost * n)


def calculate_token_price_2(n):
    cost = (1000 * 8 + 2000 * 30) / 1_000_000
    return cost * n


def resolve_gpt_image_2_size(size, custom_width, custom_height):
    if custom_width > 0 and custom_height > 0:
        return f"{custom_width}x{custom_height}"
    return size


def usd_to_cny(usd_amount):
    return usd_amount * USD_TO_CNY_RATE


def format_rmb_range(min_usd, max_usd):
    return f"约￥{usd_to_cny(min_usd):.2f}-￥{usd_to_cny(max_usd):.2f}"


def format_rmb_value(usd_amount):
    return f"约￥{usd_to_cny(usd_amount):.2f}"


def validate_gpt_image_2_size(size, custom_width=0, custom_height=0):
    resolved_size = resolve_gpt_image_2_size(size, custom_width, custom_height)
    if resolved_size == "auto":
        return resolved_size

    try:
        width_str, height_str = resolved_size.lower().split("x")
        width = int(width_str)
        height = int(height_str)
    except Exception as exc:
        raise ValueError(f"Invalid size: {resolved_size}") from exc

    if width > 3840 or height > 3840:
        raise ValueError("Width and height must not exceed 3840")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("Width and height must be multiples of 16")

    shorter_edge = min(width, height)
    longer_edge = max(width, height)
    if shorter_edge == 0 or (longer_edge / shorter_edge) > 3:
        raise ValueError("Aspect ratio must not exceed 3:1")

    total_pixels = width * height
    if total_pixels < 655360 or total_pixels > 8294400:
        raise ValueError("Total pixels must be between 655360 and 8294400")

    return resolved_size
