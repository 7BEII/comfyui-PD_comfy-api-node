import io
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image


def _image_bytes(image):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("阿里云换脸节点每个输入只支持一张图片")

    array = image[0].detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(array, "RGB").save(output, format="JPEG", quality=95)
    output.seek(0)
    return output


def _image_tensor(data):
    with Image.open(io.BytesIO(data)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _create_client():
    try:
        from alibabacloud_credentials.client import Client as CredentialClient
        from alibabacloud_facebody20191230.client import Client
        from alibabacloud_tea_openapi import models as open_api_models
    except ImportError as error:
        raise RuntimeError(
            "缺少阿里云 SDK，请运行: pip install alibabacloud_facebody20191230 alibabacloud_credentials"
        ) from error

    config = open_api_models.Config(credential=CredentialClient())
    config.endpoint = "facebody.cn-shanghai.aliyuncs.com"
    config.region_id = "cn-shanghai"
    return Client(config)


class PDAliyunFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_image": ("IMAGE", {"tooltip": "保留姿势、表情和背景的底图"}),
                "face_image": ("IMAGE", {"tooltip": "提供新人物面部特征的人脸图"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("换脸结果",)
    FUNCTION = "swap_face"
    CATEGORY = "PD/API/阿里云"
    DESCRIPTION = "将人脸图的面部特征融合到底图。请确保已获得两张图片中人物的授权。"

    def swap_face(self, template_image, face_image):
        from alibabacloud_facebody20191230 import models
        from alibabacloud_tea_util import models as util_models

        client = _create_client()
        runtime = util_models.RuntimeOptions()
        template_id = None

        try:
            template_request = models.AddFaceImageTemplateAdvanceRequest(
                image_urlobject=_image_bytes(template_image)
            )
            template_response = client.add_face_image_template_advance(template_request, runtime)
            template_id = template_response.body.data.template_id
            if not template_id:
                raise RuntimeError("阿里云未返回人脸模板 ID")

            merge_request = models.MergeImageFaceAdvanceRequest(
                template_id=template_id,
                image_urlobject=_image_bytes(face_image),
            )
            merge_response = client.merge_image_face_advance(merge_request, runtime)
            image_url = merge_response.body.data.image_url
            if not image_url:
                raise RuntimeError("阿里云未返回换脸结果地址")

            with urlopen(image_url, timeout=60) as response:
                return (_image_tensor(response.read()),)
        except Exception as error:
            code = getattr(error, "code", None)
            message = getattr(error, "message", None) or str(error)
            detail = f" [{code}]" if code else ""
            raise RuntimeError(f"阿里云换脸失败{detail}: {message}") from error
        finally:
            if template_id:
                try:
                    delete_request = models.DeleteFaceImageTemplateRequest(template_id=template_id)
                    client.delete_face_image_template_with_options(delete_request, runtime)
                except Exception as error:
                    print(f"[PD_NODE] 阿里云临时人脸模板清理失败: {error}")


NODE_CLASS_MAPPINGS = {
    "PD_AliyunFaceSwap": PDAliyunFaceSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PD_AliyunFaceSwap": "PD 阿里云换脸（双图）",
}
