import io
import json

from comfy_api.latest import IO, VideoFromFile
from comfy_api_nodes.nodes_grok import GrokVideoNode, tensor_to_base64_string

from .OPENAI_latest_apikey import BASE_URL, _Request


class PDGrokVideoAPIKey(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = GrokVideoNode.define_schema()
        schema.node_id = "PDGrokVideoAPIKey"
        schema.display_name = "PD- grok video comfyapi"
        schema.category = "PD_Tools/Video_Generation"
        schema.inputs.insert(0, IO.String.Input("comfy_api_key", default="", tooltip="Comfy.org API Key"))
        schema.outputs.append(IO.String.Output(display_name="info"))
        schema.hidden = [IO.Hidden.unique_id]
        schema.is_api_node = False
        schema.price_badge.expr = schema.price_badge.expr.replace(
            '"usd": $is15 ? $total * 1.43 : $total}',
            '"usd": $is15 ? $total * 1.43 : $total, "format": {"approximate": true}}',
        )
        return schema

    @classmethod
    def execute(cls, comfy_api_key, model, prompt, resolution, aspect_ratio, duration, seed, image=None):
        if model not in ("grok-imagine-video", "grok-imagine-video-1.5", "grok-imagine-video-1.5-lite"):
            raise ValueError(f"Unknown Grok video model: {model}")
        if resolution == "1080p" and model == "grok-imagine-video":
            raise ValueError("grok-imagine-video 不支持 1080p。")
        if not 1 <= duration <= 15:
            raise ValueError("视频时长须为 1–15 秒。")
        if image is not None and image.shape[0] != 1:
            raise ValueError("Grok Video 只支持一张起始图。")
        if not prompt.strip() and (image is None or model == "grok-imagine-video"):
            raise ValueError("请填写提示词。")
        payload = {"model": model, "prompt": prompt, "resolution": resolution,
                   "duration": duration, "seed": seed}
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        with _Request(cls.hidden.unique_id, comfy_api_key) as api:
            api.check()
            if image is not None:
                payload["image"] = {"url": "data:image/png;base64," + tensor_to_base64_string(image)}
            created = api.request("POST", BASE_URL + "/proxy/xai/v1/videos/generations", json=payload)
            api.response_id = created.json().get("request_id", "")
            if not api.response_id:
                raise ValueError("Grok 响应缺少 request_id。")
            while True:
                response = api.request("GET", BASE_URL + "/proxy/xai/v1/videos/" + api.response_id)
                result = response.json()
                status = result.get("status")
                if status in ("failed", "error", "cancelled", "canceled", "expired") or result.get("error"):
                    raise ValueError(f"Grok video {status}: {result.get('error', '')}")
                video = result.get("video") or {}
                if video.get("url") and status in (None, "done", "complete", "completed", "success", "succeeded"):
                    break
                api.sleep(2)
            api.usage = result.get("usage") or {}
            info = {"request_id": api.response_id, "model": model, "usage": api.usage}
            credits = response.headers.get("X-Comfy-Credits-Used") or created.headers.get("X-Comfy-Credits-Used")
            if credits is not None:
                info["credits_used"] = credits
            url = video["url"]
            if url.startswith("/proxy/"):
                url = BASE_URL + url
            downloaded = api.request("GET", url)
            api.check()
            output = VideoFromFile(io.BytesIO(downloaded.content))
            # Reject invalid downloads before reporting success.
            output.get_dimensions()
            api.check()
            return IO.NodeOutput(output, json.dumps(info, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {"PDGrokVideoAPIKey": PDGrokVideoAPIKey}
NODE_DISPLAY_NAME_MAPPINGS = {"PDGrokVideoAPIKey": "PD- grok video comfyapi"}
