"""
PD: Gemini Image Generation Node (ComfyUI API Key)
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
        print(f"[GEMINI_APIKEY] Image decode error: {e}")
        return torch.zeros((1, 512, 512, 3))


class PDGeminiImageGenAPIKey:
    """
    PD: Gemini Image Generation (API Key)
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
                    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {
                    "default": "auto"
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

    def generate_image(self, api_key, prompt, model, aspect_ratio, response_modalities, seed, unique_id, image_ref=None, system_prompt=""):
        """生成图片"""
        
        # 验证 API Key
        if not api_key or not api_key.strip():
            error_msg = "Error: ComfyUI API Key is required"
            print(f"[GEMINI_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
        
        # 清理 API Key
        api_key = api_key.strip()

        # ComfyUI 代理 API URL
        base_url = f"https://api.comfy.org/proxy/vertexai/gemini/{model}"
        
        # 构建请求内容
        parts = []
        
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
                print(f"[GEMINI_APIKEY] Added reference image {i+1}")

        # 生成配置
        generation_config = {
            "responseModalities": ["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]
        }
        
        # 图片配置
        if aspect_ratio != "auto":
            generation_config["imageConfig"] = {
                "aspectRatio": aspect_ratio
            }
        
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
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.0"
        }
        
        start_time = time.time()
        
        try:
            print(f"[GEMINI_APIKEY] Sending request to {model}")
            print(f"[GEMINI_APIKEY] Aspect Ratio: {aspect_ratio}")
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=120)
            
            # 检查响应
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text[:200]}"
                print(f"[GEMINI_APIKEY] {error_msg}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            result = response.json()
            
            # 提取图片和文本
            image_tensors = []
            text_parts = []
            candidates = result.get("candidates", [])
            
            if not candidates:
                error_msg = "Error: No candidates in response"
                print(f"[GEMINI_APIKEY] {error_msg}")
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
                            print(f"[GEMINI_APIKEY] Extracted image ({mime})")
                    
                    # 提取文本
                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])
            
            if not image_tensors:
                error_msg = f"Error: No images found in response"
                print(f"[GEMINI_APIKEY] {error_msg}")
                print(f"[GEMINI_APIKEY] Response: {str(result)[:500]}")
                return (torch.zeros((1, 512, 512, 3)), error_msg)
            
            final_image = torch.cat(image_tensors, dim=0)
            final_text = "\n".join(text_parts) if text_parts else "No text output"
            
            # 计算耗时和成本
            end_time = time.time()
            duration = end_time - start_time
            
            # 估算成本
            cost_per_img = 0.04
            total_cost = cost_per_img * len(image_tensors)
            
            info_str = (
                f"Model: {model}\n"
                f"Status: Success\n"
                f"Time: {duration:.2f}s\n"
                f"Images: {len(image_tensors)}\n"
                f"Aspect Ratio: {aspect_ratio}\n"
                f"Est. Cost: ${total_cost:.4f}\n"
                f"---\n"
                f"{final_text}"
            )
            
            print(f"[GEMINI_APIKEY] Success! Generated {len(image_tensors)} images in {duration:.2f}s")
            return (final_image, info_str)

        except requests.exceptions.Timeout:
            error_msg = "Error: Request timeout (120s)"
            print(f"[GEMINI_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Error: Connection failed - {str(e)}"
            print(f"[GEMINI_APIKEY] {error_msg}")
            return (torch.zeros((1, 512, 512, 3)), error_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GEMINI_APIKEY] {error_msg}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((1, 512, 512, 3)), error_msg)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "PDGeminiImageGenAPIKey": PDGeminiImageGenAPIKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGeminiImageGenAPIKey": "PD: Gemini Image (apikey)"
}
