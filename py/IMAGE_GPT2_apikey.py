"""
PD: OpenAI GPT Image 2 Node (ComfyUI API Key)
"""

import time

import requests
import torch

from ._image_gpt_common import (
    GPT_IMAGE_2_PRESET_SIZES,
    calculate_token_price_2,
    decode_response_images,
    empty_image,
    estimate_price_range_2,
    format_rmb_range,
    format_rmb_value,
    mask_to_png_bytes,
    resolve_gpt_image_2_size,
    tensor_to_bytes,
    validate_gpt_image_2_size,
)


class PDOpenAIGPTImage2APIKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Your ComfyUI API Key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text prompt for GPT Image 2"
                }),
                "model": (["gpt-image-2"], {
                    "default": "gpt-image-2"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "tooltip": "not implemented yet in backend"
                }),
                "quality": (["auto", "low", "medium", "high"], {
                    "default": "auto",
                    "tooltip": "GPT Image 2 quality"
                }),
                "background": (["auto", "opaque"], {
                    "default": "auto",
                    "tooltip": "GPT Image 2 does not support transparent background"
                }),
                "size": (GPT_IMAGE_2_PRESET_SIZES, {
                    "default": "1024x1024",
                    "tooltip": "Preset output size"
                }),
                "custom_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3840,
                    "step": 16,
                    "tooltip": "Override width when both width and height are non-zero"
                }),
                "custom_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3840,
                    "step": 16,
                    "tooltip": "Override height when both width and height are non-zero"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "How many images to generate"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional reference image for image editing."
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask for inpainting (white areas will be replaced)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(self, api_key, prompt, model, seed, quality, background, size, custom_width, custom_height, num_images, unique_id, image=None, mask=None):
        if not api_key or not api_key.strip():
            error_msg = "Error: ComfyUI API Key is required"
            print(f"[IMAGE_GPT2_APIKEY] {error_msg}")
            return (empty_image(), error_msg)

        try:
            resolved_size = validate_gpt_image_2_size(size, custom_width, custom_height)
        except Exception as e:
            return (empty_image(), f"Error: {str(e)}")

        api_key = api_key.strip()
        is_edit_mode = image is not None
        base_url = "https://api.comfy.org/proxy/openai/images/edits" if is_edit_mode else "https://api.comfy.org/proxy/openai/images/generations"
        start_time = time.time()

        try:
            if is_edit_mode:
                files = {
                    "image": ("image.png", tensor_to_bytes(image), "image/png")
                }
                if mask is not None:
                    files["mask"] = ("mask.png", mask_to_png_bytes(mask), "image/png")

                data = {
                    "model": model,
                    "prompt": prompt,
                    "n": str(num_images),
                    "size": resolved_size,
                    "quality": quality,
                    "background": background,
                    "moderation": "low"
                }
                headers = {
                    "X-API-KEY": api_key,
                    "Accept": "application/json",
                    "User-Agent": "ComfyUI-PD-Node/2.1"
                }
                response = requests.post(base_url, headers=headers, files=files, data=data, timeout=300)
            else:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": num_images,
                    "size": resolved_size,
                    "quality": quality,
                    "background": background,
                    "moderation": "low"
                }
                headers = {
                    "X-API-KEY": api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "ComfyUI-PD-Node/2.1"
                }
                response = requests.post(base_url, headers=headers, json=payload, timeout=300)

            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[IMAGE_GPT2_APIKEY] {error_msg}")
                return (empty_image(), error_msg)

            result = response.json()
            image_tensors = decode_response_images(result)
            if not image_tensors:
                error_msg = "Error: Failed to process output images"
                print(f"[IMAGE_GPT2_APIKEY] {error_msg}")
                return (empty_image(), error_msg)

            final_image = torch.cat(image_tensors, dim=0)
            duration = time.time() - start_time
            min_cost, max_cost = estimate_price_range_2(quality, num_images)
            token_cost = calculate_token_price_2(num_images)
            size_note = resolve_gpt_image_2_size(size, custom_width, custom_height)

            info_str = (
                f"模型: {model}\n"
                f"显示名称: OpenAI GPT Image 2\n"
                f"模式: {'编辑' if is_edit_mode else '生成'}\n"
                f"状态: 成功\n"
                f"耗时: {duration:.2f} 秒\n"
                f"图片数量: {len(image_tensors)}\n"
                f"尺寸预设: {size}\n"
                f"实际尺寸: {size_note}\n"
                f"质量: {quality}\n"
                f"背景: {background}\n"
                f"预估价格区间(USD): ~${min_cost:.3f}-${max_cost:.3f}\n"
                f"预估价格区间(RMB): {format_rmb_range(min_cost, max_cost)}\n"
                f"Token 公式估算(USD): ${token_cost:.4f}\n"
                f"Token 公式估算(RMB): {format_rmb_value(token_cost)}"
            )
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            return (empty_image(), "Error: Request timeout (300s)")
        except requests.exceptions.ConnectionError as e:
            return (empty_image(), f"Error: Connection failed - {str(e)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (empty_image(), f"Error: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "PDOpenAIGPTImage2APIKey": PDOpenAIGPTImage2APIKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIGPTImage2APIKey": "PD: GPT Image 2 (apikey)"
}
