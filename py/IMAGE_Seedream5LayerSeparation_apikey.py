"""Seedream 5.0 Pro layer separation using a ComfyUI API key."""

import concurrent.futures
import io
import logging

import numpy as np
import requests
import torch
from PIL import Image

from ._image_gpt_common import tensor_to_bytes


BASE_URL = "https://api.comfy.org"
IMAGE_ENDPOINT = "/proxy/byteplus/api/v3/images/generations"
MODEL = "seedream-5-0-pro-260628"


def _headers(api_key):
    return {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "close",
        "User-Agent": "ComfyUI-PD-Node/2.1",
    }


def _response_json(response, operation):
    if response.status_code not in (200, 201):
        raise RuntimeError(f"{operation} error {response.status_code}: {response.text[:500]}")
    return response.json()


def _upload_image(api_key, image):
    image_bytes = tensor_to_bytes(image, max_pixels=2048 * 2048).getvalue()
    create_response = requests.post(
        f"{BASE_URL}/customers/storage",
        headers=_headers(api_key),
        json={"file_name": "seedream-layer-input.png", "content_type": "image/png"},
        timeout=120,
    )
    upload = _response_json(create_response, "ComfyUI storage")
    upload_url = upload.get("upload_url")
    download_url = upload.get("download_url")
    if not upload_url or not download_url:
        raise RuntimeError(f"ComfyUI storage response is missing URLs: {str(upload)[:500]}")

    put_response = requests.put(
        upload_url,
        headers={"Content-Type": "image/png"},
        data=image_bytes,
        timeout=300,
    )
    if put_response.status_code not in (200, 201, 204):
        raise RuntimeError(f"Image upload error {put_response.status_code}: {put_response.text[:500]}")
    return download_url


def _download_image(url, api_key, mode):
    request_url = f"{BASE_URL}{url}" if url.startswith("/") else url
    headers = _headers(api_key) if url.startswith("/") else {}
    response = requests.get(request_url, headers=headers, timeout=180)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert(mode)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array.copy())


def _z_index(item):
    value = item.get("z_index")
    if isinstance(value, bool):
        return 1_000_000
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    return 1_000_000


class PDSeedream5LayerSeparationAPIKey:
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
                    "tooltip": "Exactly one image, at least 512x512 pixels.",
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Leave empty to separate all major elements automatically.",
                }),
                "size": (["auto", "1K", "1.5K", "2K"], {"default": "auto"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1,
                    "control_after_generate": True,
                }),
                "prompt_optimization": (["standard", "fast"], {"default": "standard"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "crop_layers": ("BOOLEAN", {
                    "default": False,
                    "label_on": "minimal size",
                    "label_off": "full canvas",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "BOUNDING_BOX", "LAYERS")
    RETURN_NAMES = ("base_image", "base_mask", "layers", "masks", "bboxes", "layer_stack")
    FUNCTION = "separate_layers"
    CATEGORY = "PD_Tools/Image_Generation"

    def separate_layers(
        self,
        comfy_api_key,
        image,
        prompt,
        size,
        seed,
        prompt_optimization,
        watermark,
        crop_layers,
    ):
        api_key = (comfy_api_key or "").strip()
        if not api_key:
            raise ValueError("ComfyUI API Key is required")
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Only a single input image is supported")

        height, width = image.shape[1:3]
        if width < 512 or height < 512:
            raise ValueError(f"Input image must be at least 512x512 pixels, got {width}x{height}")
        ratio = width / height
        if ratio < 1 / 16 or ratio > 16:
            raise ValueError(f"Input image aspect ratio must be between 1:16 and 16:1, got {width}:{height}")

        image_url = _upload_image(api_key, image)
        payload = {
            "model": MODEL,
            "prompt": prompt.strip() or None,
            "image": image_url,
            "size": size,
            "seed": int(seed),
            "response_format": "url",
            "output_format": "png",
            "layer_decomposition": True,
            "watermark": bool(watermark),
            "optimize_prompt_options": {"mode": prompt_optimization},
        }
        response = requests.post(
            f"{BASE_URL}{IMAGE_ENDPOINT}",
            headers=_headers(api_key),
            json=payload,
            timeout=1800,
        )
        result = _response_json(response, "Seedream layer separation")
        if result.get("error"):
            error = result["error"]
            raise RuntimeError(
                f"ByteDance request failed. Code: {error.get('code')}, message: {error.get('message')}"
            )

        data = [item for item in (result.get("data") or []) if isinstance(item, dict)]
        if not data or "url" not in data[0]:
            raise RuntimeError("Unexpected response: no base image returned")
        base_item = data[0]
        if _z_index(base_item) not in (0, 1_000_000):
            raise RuntimeError("Unexpected response: the first item is not the base image")

        layer_items = [item for item in data[1:] if "url" in item]
        if not layer_items:
            raise RuntimeError("The model returned no layers. Try a different prompt or input image")
        layer_items.sort(key=_z_index)

        base_image = _download_image(str(base_item["url"]), api_key, "RGB").unsqueeze(0).contiguous()
        height, width = base_image.shape[1:3]
        specs = []
        for item in layer_items:
            flags = []
            bbox = item.get("bounding_box")
            absolute = bbox.get("absolute") if isinstance(bbox, dict) else None
            if (
                isinstance(absolute, (list, tuple))
                and len(absolute) == 4
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in absolute)
            ):
                left, top, right, bottom = (int(round(value)) for value in absolute)
                rect_w, rect_h = right - left, bottom - top
                if rect_w > width or rect_h > height:
                    rect_w, rect_h = min(rect_w, width), min(rect_h, height)
                    flags.append("bbox_clamped")
                if rect_w <= 0 or rect_h <= 0:
                    flags.append("bbox_degenerate")
            else:
                left, top, rect_w, rect_h = 0, 0, width, height
                flags.append("bbox_missing")
            specs.append({
                "item": item,
                "flags": flags,
                "left": left,
                "top": top,
                "rect_w": rect_w,
                "rect_h": rect_h,
            })

        if crop_layers:
            canvas_w = max((spec["rect_w"] for spec in specs if "bbox_degenerate" not in spec["flags"]), default=1)
            canvas_h = max((spec["rect_h"] for spec in specs if "bbox_degenerate" not in spec["flags"]), default=1)
        else:
            canvas_w, canvas_h = width, height

        base_mask = torch.zeros((1, height, width))
        layers = torch.zeros((len(specs), canvas_h, canvas_w, 3))
        masks = torch.ones((len(specs), canvas_h, canvas_w))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            downloads = list(executor.map(
                lambda spec: _download_image(str(spec["item"]["url"]), api_key, "RGBA"),
                specs,
            ))

        stack_items = [{
            "image": base_image,
            "type": "raster",
            "x": 0,
            "y": 0,
            "z_index": 0,
            "name": "background",
        }]
        boxes = []
        for index, (spec, rgba) in enumerate(zip(specs, downloads)):
            item = spec["item"]
            flags = spec["flags"]
            left, top = spec["left"], spec["top"]
            rect_w, rect_h = spec["rect_w"], spec["rect_h"]
            native_size = f"{rgba.shape[1]}x{rgba.shape[0]}"

            if "bbox_degenerate" not in flags:
                if (rgba.shape[1], rgba.shape[0]) != (rect_w, rect_h):
                    rgba = rgba.clone()
                    rgba[..., :3] *= rgba[..., 3:4]
                    rgba = torch.nn.functional.interpolate(
                        rgba.permute(2, 0, 1).unsqueeze(0),
                        size=(rect_h, rect_w),
                        mode="bilinear",
                        antialias=True,
                    ).squeeze(0).permute(1, 2, 0)
                    alpha = rgba[..., 3:4]
                    rgba = torch.cat([rgba[..., :3] / alpha.clamp(min=1e-6), alpha], dim=-1).clamp(0, 1)
                    flags.append("resized_to_bbox")

                if crop_layers:
                    layers[index, :rect_h, :rect_w] = rgba[..., :3]
                    masks[index, :rect_h, :rect_w] = 1.0 - rgba[..., 3]
                else:
                    x0, y0 = max(left, 0), max(top, 0)
                    x1, y1 = min(left + rect_w, width), min(top + rect_h, height)
                    if x0 < x1 and y0 < y1:
                        patch = rgba[y0 - top:y1 - top, x0 - left:x1 - left]
                        layers[index, y0:y1, x0:x1] = patch[..., :3]
                        masks[index, y0:y1, x0:x1] = 1.0 - patch[..., 3]
                    else:
                        flags.append("bbox_out_of_canvas")

                stack_item = {
                    "image": rgba[..., :3].unsqueeze(0).contiguous(),
                    "type": "raster",
                    "x": left,
                    "y": top,
                    "z_index": _z_index(item) if _z_index(item) != 1_000_000 else index + 1,
                    "mask": (1.0 - rgba[..., 3]).unsqueeze(0),
                }
                if isinstance(item.get("name"), str):
                    stack_item["name"] = item["name"]
                stack_items.append(stack_item)

            abnormal = [flag for flag in flags if flag != "resized_to_bbox"]
            if abnormal:
                logging.warning(
                    "Seedream layer separation: layer %d (%r) flagged %s",
                    index + 1,
                    item.get("name"),
                    ", ".join(abnormal),
                )
            z_index = _z_index(item)
            rect_x, rect_y = (0, 0) if crop_layers else (left, top)
            boxes.append({
                "x": left if crop_layers else 0,
                "y": top if crop_layers else 0,
                "width": canvas_w,
                "height": canvas_h,
                "metadata": {
                    "name": item.get("name"),
                    "desc": item.get("description"),
                    "z_index": z_index if z_index != 1_000_000 else None,
                    "native_size": native_size,
                    "content_rect": [rect_x, rect_y, max(rect_w, 0), max(rect_h, 0)],
                    "flags": flags,
                },
            })

        bboxes = [boxes]
        layer_stack = {"version": 1, "canvas": (width, height), "layers": stack_items}
        return (base_image, base_mask, layers, masks, bboxes, layer_stack)


NODE_CLASS_MAPPINGS = {
    "PDSeedream5LayerSeparationAPIKey": PDSeedream5LayerSeparationAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDSeedream5LayerSeparationAPIKey": "PD: Seedream 5.0 Pro Layer Separation (comfyui_apikey)",
}
