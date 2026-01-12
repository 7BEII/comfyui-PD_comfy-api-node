"""
PD: OpenAI GPT Image Node (ComfyUI API Key)
使用 ComfyUI API Key 调用 ComfyUI 代理 API
参数和官方节点完全一样
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
        print(f"[IMAGE_GPT_APIKEY] Error loading image: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDOpenAIGPTImageAPIKey:
    """
    PD: OpenAI GPT Image (API Key)
    使用 ComfyUI API Key 调用 ComfyUI 代理 API
    """

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
                    "tooltip": "Text prompt for GPT Image"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 2**31 - 1,
                    "step": 1,
                    "tooltip": "not implemented yet in backend"
                }),
                "quality": (["low", "medium", "high"], {
                    "default": "low",
                    "tooltip": "Image quality, affects cost and generation time"
                }),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto",
                    "tooltip": "Return image with or without background"
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
                "model": (["gpt-image-1", "gpt-image-1.5"], {
                    "default": "gpt-image-1"
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

    def generate_image(self, api_key, prompt, seed, quality, background, size, n, unique_id, image=None, mask=None, model="gpt-image-1"):
        """生成或编辑图片"""
        
        # 验证 API Key
        if not api_key or not api_key.strip():
            error_msg = "Error: ComfyUI API Key is required"
            print(f"[IMAGE_GPT_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
        
        # 清理 API Key
        api_key = api_key.strip()

        # 判断模式：编辑 vs 生成
        is_edit_mode = image is not None
        
        # ComfyUI 代理 API URL
        if is_edit_mode:
            base_url = "https://api.comfy.org/proxy/openai/images/edits"
        else:
            base_url = "https://api.comfy.org/proxy/openai/images/generations"
        
        start_time = time.time()
        
        try:
            if is_edit_mode:
                # 图片编辑模式 - 使用 multipart/form-data
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
                    "X-API-KEY": api_key,
                    "Accept": "application/json",
                    "User-Agent": "ComfyUI-PD-Node/2.0"
                }
                
                print(f"[IMAGE_GPT_APIKEY] Sending edit request")
                print(f"[IMAGE_GPT_APIKEY] Model: {model}, Size: {size}, Quality: {quality}")
                print(f"[IMAGE_GPT_APIKEY] Image size: {image.shape}, Mask: {'Yes' if mask is not None else 'No'}")
                
                response = requests.post(base_url, headers=headers, files=files, data=data, timeout=300)
                
            else:
                # 文生图模式 - 使用 JSON
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": n,
                    "size": size,
                    "quality": quality,
                    "background": background,
                    "response_format": "url"
                }
                
                headers = {
                    "X-API-KEY": api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "ComfyUI-PD-Node/2.0"
                }
                
                print(f"[IMAGE_GPT_APIKEY] Sending generation request")
                print(f"[IMAGE_GPT_APIKEY] Model: {model}, Size: {size}, Quality: {quality}")
                print(f"[IMAGE_GPT_APIKEY] Prompt length: {len(prompt)} chars")
                
                response = requests.post(base_url, headers=headers, json=payload, timeout=300)

            # 检查响应
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[IMAGE_GPT_APIKEY] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)

            result = response.json()
            data_list = result.get("data", [])
            
            if not data_list:
                error_msg = "Error: No image data in response"
                print(f"[IMAGE_GPT_APIKEY] {error_msg}")
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
                print(f"[IMAGE_GPT_APIKEY] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)

            final_image = torch.cat(image_tensors, dim=0)
            
            # 计算耗时和成本
            end_time = time.time()
            duration = end_time - start_time
            
            # 估算成本
            cost_map = {
                "gpt-image-1": {"low": 0.02, "medium": 0.04, "high": 0.08},
                "gpt-image-1.5": {"low": 0.03, "medium": 0.06, "high": 0.12}
            }
            cost = cost_map.get(model, {}).get(quality, 0.04)
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
                f"Est. Cost: ${total_cost:.3f}"
            )

            print(f"[IMAGE_GPT_APIKEY] Success! Generated {len(image_tensors)} images in {duration:.2f}s")
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout (300s)"
            print(f"[IMAGE_GPT_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error: Connection failed - {str(e)}"
            print(f"[IMAGE_GPT_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[IMAGE_GPT_APIKEY] {error_msg}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, 512, 512, 3)), error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PDOpenAIGPTImageAPIKey": PDOpenAIGPTImageAPIKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIGPTImageAPIKey": "PD: GPT Image (apikey)"
}
