"""
PD: OpenAI GPT Image Node (ComfyUI Auth Token)
基于 gemini_image_api 的 Auth Token 认证方式
支持文生图和图片编辑
"""

import torch
import requests
import time
import io
import base64
import numpy as np
from PIL import Image


def tensor_to_bytes(tensor, max_pixels=2048*2048):
    """将 Tensor 转换为 PNG BytesIO，并进行下采样"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # 下采样大图片
    height, width = tensor.shape[:2]
    current_pixels = height * width
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # 使用 PIL 进行高质量缩放
        array = np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        array = np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered


def load_image_to_tensor(image_data):
    """将图片数据（URL 或 bytes）转换为 Tensor"""
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
    except Exception as e:
        print(f"[IMAGE_GPT_AUTHTOKEN] Error loading image: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDOpenAIGPTImageAuthToken:
    """
    PD: OpenAI GPT Image (Auth Token)
    使用 ComfyUI Auth Token 调用 GPT Image API
    """

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
                "model": (["gpt-image-1", "gpt-image-1.5"], {
                    "default": "gpt-image-1",
                    "tooltip": "GPT Image model version"
                }),
                "quality": (["low", "medium", "high"], {
                    "default": "low",
                    "tooltip": "Image quality, affects cost and generation time"
                }),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto",
                    "tooltip": "Background transparency (GPT Image only)"
                }),
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024"], {
                    "default": "auto",
                    "tooltip": "Image size"
                }),
                "n": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 8, 
                    "step": 1,
                    "tooltip": "Number of images to generate"
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

    def generate_image(self, auth_token, prompt, model, quality, background, size, n, seed, unique_id, image=None, mask=None):
        """生成或编辑图片"""
        
        # 验证 Token
        if not auth_token or not auth_token.strip():
            error_msg = "Error: Auth Token is required"
            print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
        
        # 清理 Token
        auth_token = auth_token.strip()
        if auth_token.lower().startswith("bearer "):
            auth_token = auth_token[7:].strip()

        # API 基础地址
        base_url = "https://api.comfy.org"
        
        # 判断模式：编辑 vs 生成
        is_edit_mode = image is not None
        
        start_time = time.time()
        
        try:
            if is_edit_mode:
                # 图片编辑模式 - 使用 multipart/form-data
                url = f"{base_url}/proxy/openai/images/edits"
                
                # 准备文件
                files = {}
                img_bytes = tensor_to_bytes(image)
                files['image'] = ('image.png', img_bytes, 'image/png')
                
                # 处理 mask
                if mask is not None:
                    m_tensor = mask
                    if m_tensor.ndim == 3:
                        m_tensor = m_tensor[0]
                    
                    mask_h, mask_w = m_tensor.shape
                    rgba = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
                    
                    mask_np = m_tensor.cpu().numpy()
                    alpha = ((1.0 - mask_np) * 255).astype(np.uint8)
                    rgba[..., 3] = alpha
                    
                    mask_pil = Image.fromarray(rgba)
                    mask_buffer = io.BytesIO()
                    mask_pil.save(mask_buffer, format="PNG")
                    mask_buffer.seek(0)
                    files['mask'] = ('mask.png', mask_buffer, 'image/png')

                # 表单数据 - multipart/form-data 模式下所有值都应该是字符串
                data = {
                    "model": model,
                    "prompt": prompt,
                    "n": str(n),
                    "size": size,
                    "quality": quality,
                    "background": background,
                    "moderation": "low"
                }
                
                # 请求头
                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "User-Agent": "ComfyUI-PD-Node/2.0"
                }
                
                print(f"[IMAGE_GPT_AUTHTOKEN] Sending edit request to {url}")
                print(f"[IMAGE_GPT_AUTHTOKEN] Model: {model}, Size: {size}, Quality: {quality}, Background: {background}")
                print(f"[IMAGE_GPT_AUTHTOKEN] Image size: {image.shape}, Mask: {'Yes' if mask is not None else 'No'}")
                
                response = requests.post(url, headers=headers, files=files, data=data, timeout=300)
                
            else:
                # 文生图模式 - 使用 JSON
                url = f"{base_url}/proxy/openai/images/generations"
                
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "quality": quality,
                    "background": background,
                    "moderation": "low",
                    "response_format": "url"
                }
                
                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "ComfyUI-PD-Node/2.0"
                }
                
                print(f"[IMAGE_GPT_AUTHTOKEN] Sending generation request to {url}")
                print(f"[IMAGE_GPT_AUTHTOKEN] Model: {model}, Size: {size}, Quality: {quality}, Background: {background}")
                print(f"[IMAGE_GPT_AUTHTOKEN] Prompt length: {len(prompt)} chars")
                
                response = requests.post(url, headers=headers, json=payload, timeout=300)

            # 检查响应
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)

            result = response.json()
            data_list = result.get("data", [])
            
            if not data_list:
                error_msg = "Error: No image data in response"
                print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)

            # 提取图片
            image_tensors = []
            for item in data_list:
                if "url" in item:
                    img_tensor = load_image_to_tensor(item["url"])
                    image_tensors.append(img_tensor)
                elif "b64_json" in item:
                    img_data = base64.b64decode(item["b64_json"])
                    img_tensor = load_image_to_tensor(img_data)
                    image_tensors.append(img_tensor)

            if not image_tensors:
                error_msg = "Error: Failed to process output images"
                print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)

            final_image = torch.cat(image_tensors, dim=0)
            
            # 计算耗时和成本
            end_time = time.time()
            duration = end_time - start_time
            
            # 估算成本
            if model == "gpt-image-1":
                # GPT Image 1: $10/1M input tokens, $40/1M output tokens
                # 估算：约 1000 tokens input, 2000 tokens output per image
                cost = (1000 * 10 + 2000 * 40) / 1_000_000
            elif model == "gpt-image-1.5":
                # GPT Image 1.5: $8/1M input tokens, $32/1M output tokens
                cost = (1000 * 8 + 2000 * 32) / 1_000_000
            else:
                cost = 0.10  # 默认估算
            total_cost = cost * n

            info_str = (
                f"Model: {model}\n"
                f"Mode: {'Edit' if is_edit_mode else 'Generate'}\n"
                f"Status: Success\n"
                f"Time: {duration:.2f}s\n"
                f"Images: {len(image_tensors)}\n"
                f"Size: {size}\n"
                f"Quality: {quality}\n"
                f"Background: {background}\n"
                f"Est. Cost: ${total_cost:.4f}"
            )

            print(f"[IMAGE_GPT_AUTHTOKEN] Success! Generated {len(image_tensors)} images in {duration:.2f}s")
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout (300s)"
            print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error: Connection failed - {str(e)}"
            print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[IMAGE_GPT_AUTHTOKEN] {error_msg}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, 512, 512, 3)), error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PDOpenAIGPTImageAuthToken": PDOpenAIGPTImageAuthToken
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIGPTImageAuthToken": "PD: GPT Image (comfyui_AuthToken)"
}
