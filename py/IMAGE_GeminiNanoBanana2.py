"""
PD: Nano Banana 2 (ComfyUI Key)
基于 Google Gemini 3.1 Flash Image 模型的图像生成/编辑节点
支持进度反馈、Token 成本计算、及其独立输出接口
"""

import torch
import base64
import json
import time
import uuid
import requests
import numpy as np
import io
from PIL import Image
from typing import Optional, List
from server import PromptServer
import comfy.utils

# 默认系统提示词
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
        print(f"[PD_NB2] Image decode error: {e}")
        return torch.zeros((1, 512, 512, 3))

class PDGeminiNanoBanana2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {
                    "default": "", 
                    "multiline": False, 
                    "placeholder": "Paste your ComfyUI API Key here"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A futuristic cityscape with flying cars and neon lights",
                }),
                "model": (["Nano Banana 2 (Gemini 3.1 Flash Image)"], {
                    "default": "Nano Banana 2 (Gemini 3.1 Flash Image)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0x7FFFFFFF, 
                }),
                "aspect_ratio": (["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {
                    "default": "auto",
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "1K",
                }),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"], {
                    "default": "IMAGE",
                }),
                "thinking_level": (["MINIMAL", "HIGH"], {
                    "default": "MINIMAL"
                }),
            },
            "optional": {
                "images": ("IMAGE", ),
                "files": ("GEMINI_INPUT_FILES", ),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": GEMINI_IMAGE_SYS_PROMPT,
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "info", "price")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools/Image_Generation"

    def generate_image(self, comfy_api_key, prompt, model, seed, aspect_ratio, resolution, response_modalities, thinking_level, unique_id, images=None, files=None, system_prompt=""):
        pbar = comfy.utils.ProgressBar(100)
        pbar.update(10)
        
        if not comfy_api_key or not comfy_api_key.strip():
            return (torch.zeros((1, 512, 512, 3)), "Error: Missing ComfyUI API Key", "$0.00")
        
        api_key = comfy_api_key.strip()
        api_model = "gemini-3.1-flash-image-preview"

        # 发送进度文字到前端
        self._send_status(unique_id, "Preparing tokens...")

        # 构建请求内容
        parts = [{"text": prompt}]
        
        if images is not None:
            for i in range(min(images.shape[0], 14)):
                b64_str = tensor_to_base64(images[i])
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_str
                    }
                })
        
        # Files 输入处理
        if files:
            for file_part in files:
                if hasattr(file_part, "model_dump"):
                    parts.append(file_part.model_dump(exclude_none=True))
                else:
                    parts.append(file_part)

        # 核心参数修正：匹配官方 Pydantic 模型的 camelCase 结构
        # 且不发送 seed (官方 GeminiNanoBanana2.execute 并没有发送 seed 到 API)
        image_config = {
            "imageSize": resolution,
            "imageOutputOptions": {"mimeType": "image/png"}
        }
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio

        generation_config = {
            "responseModalities": (["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
            "imageConfig": image_config,
            "thinkingConfig": {"thinkingLevel": thinking_level}
        }

        payload = {
            "contents": [{
                "role": "user",
                "parts": parts
            }],
            "generationConfig": generation_config
        }
        
        if system_prompt and system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt.strip()}]
            }

        headers = {
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-PD-Node/2.0"
        }
        
        base_url = f"https://api.comfy.org/proxy/vertexai/gemini/{api_model}"
        
        self._send_status(unique_id, "Sending request to Google...")
        pbar.update(30)
        
        start_time = time.time()
        try:
            # 打印发送的 Payload 调试 (可选)
            # print(f"[PD_NB2] Payload: {json.dumps(payload)}")
            
            response = requests.post(base_url, headers=headers, json=payload, timeout=120)
            pbar.update(80)
            
            if response.status_code != 200:
                error_body = response.text
                error_msg = f"API Error {response.status_code}: {error_body}"
                self._send_status(unique_id, "Error occurred")
                return (torch.zeros((1, 512, 512, 3)), error_msg, "N/A")
            
            result = response.json()
            image_tensors = []
            text_parts = []
            
            self._send_status(unique_id, "Processing response...")
            
            candidates = result.get("candidates", [])
            for cand in candidates:
                content = cand.get("content", {})
                parts_out = content.get("parts", [])
                for part in parts_out:
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if inline_data:
                        data_b64 = inline_data.get("data")
                        if data_b64:
                            raw_data = base64.b64decode(data_b64)
                            image_tensors.append(bytes_to_tensor(raw_data))
                    
                    if "text" in part and part["text"]:
                        text_parts.append(part["text"])

            if not image_tensors:
                return (torch.zeros((1, 512, 512, 3)), "Error: No images found in response", "N/A")
            
            final_image = torch.cat(image_tensors, dim=0)
            final_text = "\n".join(text_parts) if text_parts else "No description"
            
            # 计算成本
            cost_info = self._calculate_cost(result, api_model, time.time() - start_time)
            
            pbar.update(100)
            self._send_status(unique_id, "Success")
            
            return (final_image, final_text, cost_info)

        except Exception as e:
            self._send_status(unique_id, "Exception caught")
            return (torch.zeros((1, 512, 512, 3)), f"Error: {str(e)}", "N/A")

    def _send_status(self, node_id, text):
        """发送状态文字到节点下方显示"""
        try:
            PromptServer.instance.send_sync("progress_text", {"text": text, "node": node_id})
        except:
            pass

    def _calculate_cost(self, result, model, duration):
        """基于 API 响应计算成本"""
        usage = result.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        details = usage.get("candidatesTokensDetails", [])
        thoughts_tokens = usage.get("thoughtsTokenCount", 0)
        
        # 价格配置 (3.1 Flash Image Preview)
        prices = {"input": 0.5, "output_text": 3.0, "output_image": 60.0}
        
        text_tokens = thoughts_tokens
        img_tokens = 0
        
        for d in details:
            modality = d.get("modality")
            count = d.get("tokenCount", 0)
            if modality == "IMAGE":
                img_tokens += count
            else:
                text_tokens += count
        
        if not details and candidates_tokens > 0:
            text_tokens += candidates_tokens
                
        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        text_cost = (text_tokens / 1_000_000) * prices["output_text"]
        img_cost = (img_tokens / 1_000_000) * prices["output_image"]
        total_cost = input_cost + text_cost + img_cost

        info = [
            f"Model: {model}",
            f"Status: Success",
            f"Time: {duration:.2f}s",
            f"Tokens: In:{prompt_tokens:,} | Out Text:{text_tokens:,} | Out Img:{img_tokens:,}",
            f"Est. Cost: ${total_cost:.6f}"
        ]
        return "\n".join(info)

# 注册
NODE_CLASS_MAPPINGS = {"PDGeminiNanoBanana2": PDGeminiNanoBanana2}
NODE_DISPLAY_NAME_MAPPINGS = {"PDGeminiNanoBanana2": "PD: Nano Banana 2 (ComfyUI Key)"}
