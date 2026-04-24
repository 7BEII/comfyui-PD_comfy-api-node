import torch
from comfy_api_nodes.apis import GeminiPart, GeminiInlineData
from comfy_api_nodes.util.conversions import tensor_to_base64_string

class PD_ComfyPlusImage:
    """
    将最多9个尺寸不一的图片按顺序整合打包为 GEMINI_INPUT_FILES，
    可以直接连接给支持 files 的大模型节点 (如 Nano Banana Pro 等)，
    从而完美突破 ComfyUI Batch 必须相同尺寸的限制。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # 第1张图片强制必填
            "required": {
                "image1": ("IMAGE", {"tooltip": "必须输入的第1张图片"}),
            },
            # 第 2-9 张及 previous_files 作为可选填
            "optional": {
                "image2": ("IMAGE", {"tooltip": "可选输入的第2张图片"}),
                "image3": ("IMAGE", {"tooltip": "可选输入的第3张图片"}),
                "image4": ("IMAGE", {"tooltip": "可选输入的第4张图片"}),
                "image5": ("IMAGE", {"tooltip": "可选输入的第5张图片"}),
                "image6": ("IMAGE", {"tooltip": "可选输入的第6张图片"}),
                "image7": ("IMAGE", {"tooltip": "可选输入的第7张图片"}),
                "image8": ("IMAGE", {"tooltip": "可选输入的第8张图片"}),
                "image9": ("IMAGE", {"tooltip": "可选输入的第9张图片"}),
                "previous_files": ("GEMINI_INPUT_FILES", {"tooltip": "如超过9张图需继续打包，可连接上一个该节点输出的files"}),
            }
        }
    
    RETURN_TYPES = ("GEMINI_INPUT_FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "pack_images"
    CATEGORY = "PD_Tools"

    def pack_images(self, image1, image2=None, image3=None, image4=None, 
                    image5=None, image6=None, image7=None, image8=None, 
                    image9=None, previous_files=None):
        
        files_list = []
        
        # 如果需要无缝串联别的节点输出的 files ，优先加进去
        if previous_files is not None:
            files_list.extend(previous_files)
            
        images = [image1, image2, image3, image4, image5, image6, image7, image8, image9]
        
        for img in images:
            if img is not None:
                # 兼容万一单条线里输入了 Batch(N) 图片的特殊情况，分别解开处理
                for i in range(img.shape[0]):
                    try:
                        # 转换成 Base64
                        img_b64 = tensor_to_base64_string(img[i].unsqueeze(0))
                        # 构造为 Gemini 认证的 Part
                        new_part = GeminiPart(
                            inlineData=GeminiInlineData(
                                mimeType="image/png",
                                data=img_b64
                            )
                        )
                        # 转成字典防止 Pydantic 跨模块传递对象时校验失败
                        files_list.append(new_part.model_dump(exclude_none=True))
                    except Exception as e:
                        print(f"[Error] PD_comfyplus_image pack failed for an image: {e}")
        
        return (files_list,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "PD_comfyplus_image": PD_ComfyPlusImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_comfyplus_image": "PD_comfyplus_image (Multi-Size Pack)"
}
