import torch
import io
import time  # 新增：引入时间模块
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# 导入 ComfyUI 基础类型
from comfy.comfy_types.node_typing import IO

# 导入 API 节点相关工具
from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
)
from comfy_api_nodes.apinode_utils import (
    validate_and_cast_response,
    validate_string,
    downscale_image_tensor,
)
from comfy_api_nodes.apis import (
    OpenAIImageGenerationRequest,
    OpenAIImageEditRequest,
    OpenAIImageGenerationResponse,
)

class PDOpenAIGPTImageKey:
    """
    PD: OpenAI GPT Image 1 (ComfyUI Key)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comfy_api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Paste ComfyUI API Key here"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "quality": (["low", "medium", "high"], {"default": "low"}),
                "background": (["opaque", "transparent"], {"default": "opaque"}),
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status_info")
    FUNCTION = "generate_image"
    CATEGORY = "PD_Tools"

    async def generate_image(
        self, 
        comfy_api_key, 
        prompt, 
        seed, 
        quality, 
        background, 
        size, 
        n, 
        unique_id, 
        image=None, 
        mask=None
    ):
        # 1. 基础校验
        if not comfy_api_key:
            return (torch.zeros((1, 512, 512, 3)), "Error: Missing API Key")

        validate_string(prompt, strip_whitespace=False)
        
        model = "gpt-image-1"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        files = []

        # 2. 处理图像 (Image/Mask)
        if image is not None:
            path = "/proxy/openai/images/edits"
            request_class = OpenAIImageEditRequest
            content_type = "multipart/form-data"
            batch_size = image.shape[0]
            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()
                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                
                key_name = "image[]" if batch_size > 1 else "image"
                files.append((key_name, (f"image_{i}.png", img_byte_arr, "image/png")))

        if mask is not None:
            if image is None:
                return (torch.zeros((1, 512, 512, 3)), "Error: Need Image for Mask")
            
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()
            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0)).squeeze()
            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

        # 3. 准备请求
        auth_kwargs = {"comfy_api_key": comfy_api_key}
        request_payload = request_class(
            model=model,
            prompt=prompt,
            quality=quality,
            background=background,
            n=n,
            seed=seed,
            size=size,
        )

        endpoint = ApiEndpoint(
            path=path,
            method=HttpMethod.POST,
            request_model=request_class,
            response_model=OpenAIImageGenerationResponse,
        )

        # 4. 执行并计时
        start_time = time.time() # 开始计时
        try:
            operation = SynchronousOperation(
                endpoint=endpoint,
                request=request_payload,
                files=files if files else None,
                content_type=content_type,
                auth_kwargs=auth_kwargs,
            )

            response = await operation.execute()
            end_time = time.time() # 结束计时
            
            elapsed_time = end_time - start_time

            # 5. 计算费用 (预估)
            # OpenAI 绘图不使用 Token，按张计费。
            # 标准(Low/Standard): ~$0.04, 高清(High/HD): ~$0.08 (参考 DALL-E 3 定价)
            unit_price = 0.04
            if quality == "high":
                unit_price = 0.08
            elif quality == "medium": 
                unit_price = 0.06 # 预估中间值
            
            total_cost = unit_price * n

            img_tensor = await validate_and_cast_response(response, node_id=unique_id)
            
            # 6. 格式化输出信息
            # 注意：绘图模型通常没有 Token 概念，这里显示为 N/A 或直接显示金额
            status_msg = (
                f"Model: {model}\n"
                f"Status: Success\n"
                f"生成时间：{elapsed_time:.2f} s\n"
                f"生成费用：${total_cost:.3f} 美金 (Est.)\n"
                f"Size: {size}"
            )
            
            return (img_tensor, status_msg)

        except Exception as e:
            print(f"PD GPT Image Error: {e}")
            return (torch.zeros((1, 512, 512, 3)), f"Error: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "PDOpenAIGPTImageKey": PDOpenAIGPTImageKey
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDOpenAIGPTImageKey": "PD: OpenAI GPT Image 1 (ComfyUI Key)"
}