import torch
import base64
from io import BytesIO
import json
import time
import uuid
from typing import Optional

# 引入 ComfyUI 官方 API 依赖
from comfy_api_nodes.apis import (
    GeminiContent,
    GeminiPart,
    GeminiMimeType,
    GeminiInlineData,
    GeminiGenerateContentResponse,
)
from comfy_api_nodes.apis.gemini_api import (
    GeminiImageGenerationConfig, 
    GeminiImageGenerateContentRequest,
    GeminiImageConfig
)
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    bytesio_to_image_tensor,
    validate_string,
)
from server import PromptServer

class PDGeminiImageGenComfyKey:
    """
    PD: Gemini Image Generation (ComfyUI Key)
    已移除多余 Text 输出，Auto 模式下参考原图比例
    """
    
    # 定价表 (仅供估算)
    PRICING = {
        "flash": { "input": 0.075, "output": 0.30, "image_gen": 0.04 },
        "pro":   { "input": 1.25,  "output": 5.00, "image_gen": 0.04 },
        "default": { "input": 0.10, "output": 0.40, "image_gen": 0.04 }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Paste ComfyUI API Key here"}),
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city with flying cars"}),
                "model": (["gemini-2.5-flash-image", "gemini-2.5-flash-image-preview", "gemini-1.5-pro", "gemini-1.5-flash"], {"default": "gemini-2.5-flash-image"}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "image_ref": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    # --- 修改：只保留 image 和 cost_info ---
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "cost_info")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools"

    async def generate_image(self, comfy_api_key, prompt, model, aspect_ratio, seed, unique_id, image_ref=None):
        # 1. 基础校验
        if not comfy_api_key:
            return (torch.zeros((1, 512, 512, 3)), "Error: Missing Key")
        validate_string(prompt, strip_whitespace=True, min_length=1)

        # 2. 构建请求
        parts = [self._create_text_part(prompt)]
        if image_ref is not None:
            parts.extend(self._create_image_parts(image_ref))

        # 3. 配置 Ratio (Auto = None，让模型自己决定)
        img_config = None
        if aspect_ratio != "auto":
            img_config = GeminiImageConfig(aspectRatio=aspect_ratio)

        gen_config = GeminiImageGenerationConfig(
            responseModalities=["TEXT", "IMAGE"],
            imageConfig=img_config
        )

        auth_kwargs = {"comfy_api_key": comfy_api_key}

        request_payload = GeminiImageGenerateContentRequest(
            contents=[GeminiContent(role="user", parts=parts)],
            generationConfig=gen_config
        )

        try:
            # 4. 发起请求
            endpoint = self._get_image_endpoint(model)
            response = await SynchronousOperation(
                endpoint=endpoint,
                request=request_payload,
                auth_kwargs=auth_kwargs,
            ).execute()

            # 5. 解析结果
            output_image = self._extract_image_from_response(response)
            
            # 计算生成的图片数量
            image_count = output_image.shape[0] if output_image is not None else 0
            if output_image.shape[1] == 1024 and torch.all(output_image == 0): 
                 image_count = 0 # 空图不计费

            # 6. 计算成本
            info_str = self._calculate_cost(response, model, image_count)

            # --- 返回：只返回图片和成本信息 ---
            return (output_image, info_str)

        except Exception as e:
            print(f"Gemini Image Gen Error: {e}")
            return (torch.zeros((1, 512, 512, 3)), f"Error: {str(e)}")

    # -----------------------------------------------------------
    # 成本计算逻辑
    # -----------------------------------------------------------
    def _calculate_cost(self, response, model_name, image_count):
        """计算 Token 消耗和预估美金成本"""
        
        price_tier = self.PRICING["default"]
        if "flash" in model_name.lower():
            price_tier = self.PRICING["flash"]
        elif "pro" in model_name.lower():
            price_tier = self.PRICING["pro"]
            
        info = [f"Model: {model_name}"]
        total_cost = 0.0

        # Token 费用
        if hasattr(response, 'usageMetadata') and response.usageMetadata:
            usage = response.usageMetadata
            p_tokens = getattr(usage, 'promptTokenCount', getattr(usage, 'prompt_token_count', 0))
            c_tokens = getattr(usage, 'candidatesTokenCount', getattr(usage, 'candidates_token_count', 0))
            
            input_cost = (p_tokens / 1_000_000) * price_tier["input"]
            output_cost = (c_tokens / 1_000_000) * price_tier["output"]
            
            total_cost += input_cost + output_cost
            
            info.append(f"Input Tokens: {p_tokens} (~${input_cost:.6f})")
            info.append(f"Output Tokens: {c_tokens} (~${output_cost:.6f})")
        else:
            info.append("Token Usage: Unavailable")

        # 生图费用
        if image_count > 0:
            img_cost = image_count * price_tier["image_gen"]
            total_cost += img_cost
            info.append(f"Images Gen: {image_count} (~${img_cost:.4f})")
        
        info.append("-" * 20)
        info.append(f"Total Est. Cost: ${total_cost:.6f}")
        
        return "\n".join(info)

    # -----------------------------------------------------------
    # 辅助函数
    # -----------------------------------------------------------
    def _create_text_part(self, text):
        return GeminiPart(text=text)

    def _create_image_parts(self, image_tensor):
        parts = []
        for i in range(image_tensor.shape[0]):
            img_b64 = tensor_to_base64_string(image_tensor[i].unsqueeze(0))
            parts.append(GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.image_png,
                    data=img_b64
                )
            ))
        return parts

    def _get_image_endpoint(self, model_name):
        GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"
        return ApiEndpoint(
            path=f"{GEMINI_BASE_ENDPOINT}/{model_name}",
            method=HttpMethod.POST,
            request_model=GeminiImageGenerateContentRequest,
            response_model=GeminiGenerateContentResponse,
        )

    def _extract_image_from_response(self, response):
        image_tensors = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inlineData and part.inlineData.mimeType == "image/png":
                            image_data = base64.b64decode(part.inlineData.data)
                            tensor_img = bytesio_to_image_tensor(BytesIO(image_data))
                            image_tensors.append(tensor_img)
        
        if not image_tensors:
            return torch.zeros((1, 1024, 1024, 3))
        return torch.cat(image_tensors, dim=0)

NODE_CLASS_MAPPINGS = {
    "PDGeminiImageGenComfyKey": PDGeminiImageGenComfyKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDGeminiImageGenComfyKey": "PD: Gemini Image Gen (With Cost Info)"
}