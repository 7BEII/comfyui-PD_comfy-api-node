"""
PD: Gemini Image Generation Node (ComfyUI Auth Token)
基于 gemini_image_api 的 Auth Token 认证方式
支持文生图和图生图
"""

import torch
import requests
import time
import io
import base64
import numpy as np
from PIL import Image

# 默认系统提示词（来自 ComfyUI 官方 API 节点）
GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


def tensor_to_base64(tensor):
    """将 Tensor 转换为 base64 字符串"""
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    array = np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def bytes_to_tensor(data):
    """将图片字节数据转换为 Tensor"""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0)
    except Exception as e:
        print(f"[GEMINI_AUTHTOKEN] Image decode error: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDGeminiImageGenAuthToken:
    """
    PD: Gemini Image Generation (Auth Token)
    使用 ComfyUI Auth Token 调用 Gemini Image API
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
                    "default": "A futuristic city with flying cars",
                    "tooltip": "Describe what you want to generate"
                }),
                "model": ([
                    "gemini-2.5-flash-image",
                    "gemini-2.5-flash-image-preview"
                ], {
                    "default": "gemini-2.5-flash-image"
                }),
                "aspect_ratio": ([
                    "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"
                ], {
                    "default": "auto"
                }),
                "resolution": (["auto", "1K", "2K", "4K"], {
                    "default": "1K"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xFFFFFFFFFFFFFFFF
                }),
            },
            "optional": {
                "image_ref": ("IMAGE", {
                    "tooltip": "Reference image(s) for image-to-image generation"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": GEMINI_IMAGE_SYS_PROMPT,
                    "tooltip": "Optional system instructions"
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

    def generate_image(self, auth_token, prompt, model, aspect_ratio, resolution, seed, unique_id, image_ref=None, system_prompt=""):
        """生成图片"""
        
        print(f"[GEMINI_AUTHTOKEN] ========== 开始执行 ==========")
        print(f"[GEMINI_AUTHTOKEN] Prompt: {prompt[:50]}...")
        print(f"[GEMINI_AUTHTOKEN] Model: {model}")
        
        # 验证 Token
        if not auth_token or not auth_token.strip():
            error_msg = "Error: Auth Token is required"
            print(f"[GEMINI_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
        
        # 清理 Token
        auth_token = auth_token.strip()
        if auth_token.lower().startswith("bearer "):
            auth_token = auth_token[7:].strip()

        # 构建 API URL (不需要添加 :generateContent 后缀，API 会自动处理)
        base_url = f"https://api.comfy.org/proxy/vertexai/gemini/{model}"
        
        # 构建请求内容
        parts = []
        
        # 添加系统提示词（如果有）
        if system_prompt and system_prompt.strip():
            parts.append({"text": system_prompt.strip()})
        
        # 添加用户提示词
        parts.append({"text": prompt})
        
        # 添加参考图片（如果有）
        if image_ref is not None:
            for i in range(image_ref.shape[0]):
                b64_str = tensor_to_base64(image_ref[i])
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_str
                    }
                })
                print(f"[GEMINI_AUTHTOKEN] Added reference image {i+1}")

        # 生成配置
        generation_config = {
            "responseModalities": ["IMAGE"]
        }
        
        # 图片配置
        image_config = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if resolution != "auto":
            image_config["resolution"] = resolution
        
        if image_config:
            generation_config["imageConfig"] = image_config
        
        # 完整请求体
        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": generation_config
        }

        # 请求头
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.0"
        }
        
        start_time = time.time()
        
        try:
            print(f"[GEMINI_AUTHTOKEN] Sending request to {model}")
            print(f"[GEMINI_AUTHTOKEN] Aspect Ratio: {aspect_ratio}, Resolution: {resolution}")
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=120)
            
            # 检查响应
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[GEMINI_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            result = response.json()
            
            # 提取图片
            image_tensors = []
            candidates = result.get("candidates", [])
            
            if not candidates:
                error_msg = "Error: No candidates in response"
                print(f"[GEMINI_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            for cand in candidates:
                content = cand.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    # 支持两种字段名格式
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    
                    if inline_data:
                        mime = inline_data.get("mime_type") or inline_data.get("mimeType")
                        data_b64 = inline_data.get("data")
                        
                        if mime in ["image/png", "image/jpeg"] and data_b64:
                            raw_data = base64.b64decode(data_b64)
                            image_tensors.append(bytes_to_tensor(raw_data))
                            print(f"[GEMINI_AUTHTOKEN] Extracted image ({mime})")
            
            if not image_tensors:
                error_msg = f"Error: No images found in response"
                print(f"[GEMINI_AUTHTOKEN] {error_msg}")
                print(f"[GEMINI_AUTHTOKEN] Response: {str(result)[:500]}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            final_image = torch.cat(image_tensors, dim=0)
            
            # 计算耗时和成本
            end_time = time.time()
            duration = end_time - start_time
            
            # 估算成本（根据模型）
            cost_per_img = 0.04
            if "pro" in model.lower():
                cost_per_img = 0.08
            total_cost = cost_per_img * len(image_tensors)
            
            info_str = (
                f"Model: {model}\n"
                f"Status: Success\n"
                f"Time: {duration:.2f}s\n"
                f"Images: {len(image_tensors)}\n"
                f"Aspect Ratio: {aspect_ratio}\n"
                f"Resolution: {resolution}\n"
                f"Est. Cost: ${total_cost:.4f}"
            )
            
            print(f"[GEMINI_AUTHTOKEN] Success! Generated {len(image_tensors)} images in {duration:.2f}s")
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout (120s)"
            print(f"[GEMINI_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error: Connection failed - {str(e)}"
            print(f"[GEMINI_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GEMINI_AUTHTOKEN] {error_msg}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, 512, 512, 3)), error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PDGeminiImageGenAuthToken": PDGeminiImageGenAuthToken
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGeminiImageGenAuthToken": "PD: Gemini Image (comfyui_AuthToken)"
}
