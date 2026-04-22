"""
PD: OpenAI GPT Image 2 Node (ComfyUI Auth Token)
"""

import time

import requests
import torch

from ._image_gpt_common import (
    build_openai_edit_files,
    calculate_exact_cost_gpt_image_2_from_usage,
    GPT_IMAGE_2_PRESET_SIZES,
    calculate_token_price_2,
    decode_response_images,
    empty_image,
    estimate_price_range_2,
    format_credits_value,
    format_rmb_range,
    format_rmb_value,
    mask_to_png_bytes,
    post_with_retries,
    resolve_gpt_image_2_size,
    validate_gpt_image_2_size,
)


class PDOpenAIGPTImage2AuthToken:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "auth_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Paste your Auth Token here"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describe what you want to generate or edit"
                }),
                "model": (["gpt-image-2"], {
                    "default": "gpt-image-2",
                    "tooltip": "Locked GPT Image 2 model"
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF
                }),
            },
            "optional": {
                "image": ("IMAGE", {"default": None, "tooltip": "Input image for editing"}),
                "mask": ("MASK", {"default": None, "tooltip": "Mask for inpainting"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(self, auth_token, prompt, model, quality, background, size, custom_width, custom_height, num_images, seed, unique_id, image=None, mask=None):
        if not auth_token or not auth_token.strip():
            error_msg = "Error: Auth Token is required"
            print(f"[IMAGE_GPT2_AUTHTOKEN] {error_msg}")
            return (empty_image(), error_msg)

        try:
            resolved_size = validate_gpt_image_2_size(size, custom_width, custom_height)
        except Exception as e:
            return (empty_image(), f"Error: {str(e)}")

        auth_token = auth_token.strip()
        if auth_token.lower().startswith("bearer "):
            auth_token = auth_token[7:].strip()

        base_url = "https://api.comfy.org"
        is_edit_mode = image is not None
        start_time = time.time()

        try:
            if is_edit_mode:
                url = f"{base_url}/proxy/openai/images/edits"
                files = build_openai_edit_files(image, mask)

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
                    "Authorization": f"Bearer {auth_token}",
                    "Connection": "close",
                    "User-Agent": "ComfyUI-PD-Node/2.1"
                }
                response = post_with_retries(url, headers, data=data, files=files, timeout=300)
            else:
                url = f"{base_url}/proxy/openai/images/generations"
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
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                    "User-Agent": "ComfyUI-PD-Node/2.1"
                }
                response = post_with_retries(url, headers, json_payload=payload, timeout=300)

            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[IMAGE_GPT2_AUTHTOKEN] {error_msg}")
                return (empty_image(), error_msg)

            result = response.json()
            image_tensors = decode_response_images(result)
            if not image_tensors:
                error_msg = "Error: Failed to process output images"
                print(f"[IMAGE_GPT2_AUTHTOKEN] {error_msg}")
                return (empty_image(), error_msg)

            final_image = torch.cat(image_tensors, dim=0)
            duration = time.time() - start_time
            min_cost, max_cost = estimate_price_range_2(quality, num_images)
            token_cost = calculate_token_price_2(num_images)
            size_note = resolve_gpt_image_2_size(size, custom_width, custom_height)
            exact_cost = calculate_exact_cost_gpt_image_2_from_usage(result)

            lines = [
                f"模型: {model}",
                f"显示名称: OpenAI GPT Image 2",
                f"模式: {'编辑' if is_edit_mode else '生成'}",
                f"状态: 成功",
                f"耗时: {duration:.2f} 秒",
                f"图片数量: {len(image_tensors)}",
                f"尺寸预设: {size}",
                f"实际尺寸: {size_note}",
                f"质量: {quality}",
                f"背景: {background}",
            ]

            if exact_cost is not None:
                lines.extend([
                    f"实际价格(USD): ${exact_cost['usd']:.4f}",
                    f"实际价格(RMB): {format_rmb_value(exact_cost['usd'])}",
                    f"实际价格(Credits): {format_credits_value(exact_cost['usd'])}",
                    f"图像输入 Tokens: {exact_cost['image_tokens']}",
                    f"文本输入 Tokens: {exact_cost['text_tokens']}",
                    f"输出 Tokens: {exact_cost['output_tokens']}",
                ])
            else:
                lines.extend([
                    f"预估价格区间(USD): ~${min_cost:.3f}-${max_cost:.3f}",
                    f"预估价格区间(RMB): {format_rmb_range(min_cost, max_cost)}",
                    f"Token 公式估算(USD): ${token_cost:.4f}",
                    f"Token 公式估算(RMB): {format_rmb_value(token_cost)}",
                ])

            info_str = "\n".join(lines)
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
    "PDOpenAIGPTImage2AuthToken": PDOpenAIGPTImage2AuthToken
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIGPTImage2AuthToken": "PD: GPT Image 2 (comfyui_AuthToken)"
}
