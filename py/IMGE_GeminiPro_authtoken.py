"""
PD: Gemini Pro Image Generation Node (ComfyUI Auth Token)
Nano Banana Pro - 基于 Gemini 3 Pro Image
支持高分辨率图像生成（1K/2K/4K）
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
        print(f"[GEMINI_PRO_AUTHTOKEN] Image decode error: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDGeminiProImageGenAuthToken:
    """
    PD: Gemini Pro Image Generation (Auth Token)
    Nano Banana Pro - 使用 ComfyUI Auth Token 调用 Gemini 3 Pro Image API
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
                    "tooltip": "Text prompt describing the image to generate or the edits to apply"
                }),
                "model": (["gemini-3-pro-image-preview"], {
                    "default": "gemini-3-pro-image-preview"
                }),
                "aspect_ratio": ([
                    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {
                    "default": "auto",
                    "tooltip": "If set to 'auto', matches your input image's aspect ratio"
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "1K",
                    "tooltip": "Target output resolution. For 2K/4K the native Gemini upscaler is used"
                }),
                "response_modalities": (["IMAGE+TEXT", "IMAGE"], {
                    "default": "IMAGE+TEXT",
                    "tooltip": "Choose 'IMAGE' for image-only output, or 'IMAGE+TEXT' for both"
                }),
                "seed": ("INT", {
                    "default": 42, 
                    "min": 0, 
                    "max": 0xFFFFFFFFFFFFFFFF
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional reference image(s) (up to 14)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": GEMINI_IMAGE_SYS_PROMPT,
                    "tooltip": "Foundational instructions that dictate an AI's behavior"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(self, auth_token, prompt, model, aspect_ratio, resolution, response_modalities, seed, unique_id, images=None, system_prompt=""):
        """生成图片"""
        
        print(f"[GEMINI_PRO_AUTHTOKEN] ========== 开始执行 ==========")
        print(f"[GEMINI_PRO_AUTHTOKEN] Prompt: {prompt[:50]}...")
        print(f"[GEMINI_PRO_AUTHTOKEN] Model: {model}")
        
        # 验证 Token
        if not auth_token or not auth_token.strip():
            error_msg = "Error: Auth Token is required"
            print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
        
        # 清理 Token
        auth_token = auth_token.strip()
        if auth_token.lower().startswith("bearer "):
            auth_token = auth_token[7:].strip()

        # 构建 API URL
        base_url = f"https://api.comfy.org/proxy/vertexai/gemini/{model}"
        
        # 构建请求内容
        parts = []
        
        # 添加用户提示词
        parts.append({"text": prompt})
        
        # 添加参考图片（如果有，最多14张）
        if images is not None:
            num_images = min(images.shape[0], 14)
            for i in range(num_images):
                b64_str = tensor_to_base64(images[i])
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_str
                    }
                })
                print(f"[GEMINI_PRO_AUTHTOKEN] Added reference image {i+1}/{num_images}")
            
            if images.shape[0] > 14:
                print(f"[GEMINI_PRO_AUTHTOKEN] Warning: Only first 14 images will be used (provided {images.shape[0]})")

        # 生成配置
        generation_config = {
            "responseModalities": ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]
        }
        
        # 图片配置
        image_config = {
            "imageSize": resolution
        }
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        
        generation_config["imageConfig"] = image_config
        
        # 系统指令
        system_instruction = None
        if system_prompt and system_prompt.strip():
            system_instruction = {
                "parts": [{"text": system_prompt.strip()}]
            }
        
        # 完整请求体
        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": generation_config
        }
        
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # 请求头
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.0"
        }
        
        start_time = time.time()
        
        try:
            print(f"[GEMINI_PRO_AUTHTOKEN] Sending request to {model}")
            print(f"[GEMINI_PRO_AUTHTOKEN] Aspect Ratio: {aspect_ratio}, Resolution: {resolution}")
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=180)
            
            # 检查响应
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            result = response.json()
            
            # 提取图片和文本
            image_tensors = []
            text_parts = []
            candidates = result.get("candidates", [])
            
            if not candidates:
                error_msg = "Error: No candidates in response"
                print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            for cand in candidates:
                content = cand.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    # 提取图片
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    
                    if inline_data:
                        mime = inline_data.get("mime_type") or inline_data.get("mimeType")
                        data_b64 = inline_data.get("data")
                        
                        if mime in ["image/png", "image/jpeg"] and data_b64:
                            raw_data = base64.b64decode(data_b64)
                            image_tensors.append(bytes_to_tensor(raw_data))
                            print(f"[GEMINI_PRO_AUTHTOKEN] Extracted image ({mime})")
                    
                    # 提取文本
                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])
            
            if not image_tensors:
                error_msg = f"Error: No images found in response"
                print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
                print(f"[GEMINI_PRO_AUTHTOKEN] Response: {str(result)[:500]}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            final_image = torch.cat(image_tensors, dim=0)
            final_text = "\n".join(text_parts) if text_parts else "No text output"
            
            # 计算耗时和成本
            end_time = time.time()
            duration = end_time - start_time
            
            # 估算成本（Gemini 3 Pro Image）
            # $2/1M input tokens, $12/1M output text tokens, $120/1M output image tokens
            cost_per_img = 0.12  # 估算
            total_cost = cost_per_img * len(image_tensors)
            
            info_str = (
                f"Model: {model}\n"
                f"Status: Success\n"
                f"Time: {duration:.2f}s\n"
                f"Images: {len(image_tensors)}\n"
                f"Aspect Ratio: {aspect_ratio}\n"
                f"Resolution: {resolution}\n"
                f"Est. Cost: ${total_cost:.4f}\n"
                f"---\n"
                f"{final_text}"
            )
            
            print(f"[GEMINI_PRO_AUTHTOKEN] Success! Generated {len(image_tensors)} images in {duration:.2f}s")
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout (180s)"
            print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error: Connection failed - {str(e)}"
            print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GEMINI_PRO_AUTHTOKEN] {error_msg}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, 512, 512, 3)), error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PDGeminiProImageGenAuthToken": PDGeminiProImageGenAuthToken
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGeminiProImageGenAuthToken": "PD: Gemini Pro Image (comfyui_AuthToken)"
}
