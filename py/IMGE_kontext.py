import torch
import asyncio
import aiohttp
import time
import json
from io import BytesIO
import numpy as np
from PIL import Image
from typing import Optional, Any, Dict

# 引入 Pydantic (ComfyUI 自带)
from pydantic import BaseModel, Field

# 基础工具依赖
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    tensor_to_base64_string,
    process_image_response,
    validate_string,
)
from server import PromptServer

# ==========================================
# 1. 本地定义请求与响应模型
# ==========================================

class LocalFluxKontextRequest(BaseModel):
    prompt: str
    prompt_upsampling: bool = False
    seed: int = 0
    aspect_ratio: str = "16:9"
    guidance: float = 3.0
    steps: int = 50
    input_image: Optional[str] = None

class LocalBFLResponse(BaseModel):
    id: str
    status: Optional[str] = None
    polling_url: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# ==========================================
# 2. 节点主逻辑
# ==========================================

class PDFluxKontextProOfficial:
    """
    PD: Flux.1 Kontext [pro] (ComfyUI Key)
    修改版：aspect_ratio 改为 STRING 输入，方便连接外部节点
    """
    
    ESTIMATED_COST = 0.05 

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Paste ComfyUI API Key here"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Specify what and how to edit."}),
                
                # --- 修改点：改为 STRING 类型，支持外部连接 ---
                "aspect_ratio": ("STRING", {"default": "16:9", "multiline": False, "tooltip": "Valid values: 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 4:5"}),
                # -----------------------------------------
                
                "guidance": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 99.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 150}),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "input_image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "cost_info")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools"

    async def generate_image(self, comfy_api_key, prompt, aspect_ratio, guidance, steps, seed, prompt_upsampling, unique_id, input_image=None):
        # 1. 基础校验
        if not comfy_api_key:
            return (torch.zeros((1, 512, 512, 3)), "Error: Missing Key")
        
        if input_image is None:
            validate_string(prompt, strip_whitespace=False)

        # 2. 准备图片
        image_b64 = None
        if input_image is not None:
            image_b64 = tensor_to_base64_string(input_image[0].unsqueeze(0)[:, :, :, :3])

        # 3. 准备请求体
        request_payload = LocalFluxKontextRequest(
            prompt=prompt,
            prompt_upsampling=prompt_upsampling,
            guidance=round(guidance, 1),
            steps=steps,
            seed=seed,
            # 这里直接接受输入的字符串，API 会自己校验是否合法 (如 16:9, 1:1 等)
            aspect_ratio=aspect_ratio,
            input_image=image_b64
        )

        # 4. 鉴权配置
        auth_kwargs = {"comfy_api_key": comfy_api_key}
        
        # 5. Endpoint
        endpoint = ApiEndpoint(
            path="/proxy/bfl/flux-kontext-pro/generate",
            method=HttpMethod.POST,
            request_model=LocalFluxKontextRequest, 
            response_model=LocalBFLResponse, 
        )

        try:
            # 6. 发起任务
            initial_response = await SynchronousOperation(
                endpoint=endpoint,
                request=request_payload,
                auth_kwargs=auth_kwargs,
            ).execute()

            # 7. 处理 Polling URL (补全逻辑)
            polling_url = initial_response.polling_url
            if not polling_url and initial_response.id:
                polling_url = f"/proxy/bfl/get_result?id={initial_response.id}"
            
            if not polling_url:
                raise Exception("Failed to retrieve Polling URL from response.")

            # 8. 轮询下载
            output_image = await self._poll_until_generated(
                polling_url, 
                node_id=unique_id
            )

            cost_info = f"Model: Flux.1 Kontext [pro]\nSteps: {steps}\nStatus: Success\nEst. Cost: ${self.ESTIMATED_COST}"

            return (output_image, cost_info)

        except Exception as e:
            print(f"BFL Kontext Error: {e}")
            return (torch.zeros((1, 512, 512, 3)), f"Error: {str(e)}")

    # -----------------------------------------------------------
    # 轮询逻辑
    # -----------------------------------------------------------
    async def _poll_until_generated(self, polling_url, timeout=360, node_id=None):
        start_time = time.time()
        retry_pending_seconds = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                if time.time() - start_time > timeout:
                    raise Exception("BFL API Timeout")

                if node_id:
                    elapsed = time.time() - start_time
                    PromptServer.instance.send_progress_text(f"BFL Generating... ({elapsed:.1f}s)", node_id)

                async with session.get(polling_url) as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result.get("status")
                        
                        if status == "Ready":
                            img_url = result["result"]["sample"]
                            if node_id:
                                PromptServer.instance.send_progress_text(f"Downloading: {img_url}", node_id)
                            
                            async with session.get(img_url) as img_resp:
                                content = await img_resp.content.read()
                                return process_image_response(content)
                                
                        elif status in ["Request Moderated", "Content Moderated"]:
                            raise Exception(f"Content Moderated: {status}")
                        elif status == "Error":
                            raise Exception(f"BFL API Error: {result}")
                        elif status == "Pending":
                            await asyncio.sleep(retry_pending_seconds)
                            continue
                        await asyncio.sleep(retry_pending_seconds)
                    
                    elif response.status == 202:
                        await asyncio.sleep(retry_pending_seconds)
                    else:
                        try:
                            err_text = await response.text()
                        except:
                            err_text = str(response.status)
                        raise Exception(f"Polling Error {response.status}: {err_text}")

NODE_CLASS_MAPPINGS = {
    "PDFluxKontextProOfficial": PDFluxKontextProOfficial
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDFluxKontextProOfficial": "PD: Flux.1 Kontext Pro (ComfyUI Key)"
}