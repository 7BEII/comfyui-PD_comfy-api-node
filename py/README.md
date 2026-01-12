# PD ComfyUI API Nodes

ComfyUI 自定义节点集合，用于调用各种 AI 图像生成 API（GPT Image、Gemini Image、Flux Kontext）。

## 📦 节点列表

### ComfyUI AuthToken 版本（通过 ComfyUI 代理）
ComfyUI AuthToken 版本（通过 ComfyUI 代理）
   - PD: GPT Image (comfyui_AuthToken)
   - PD: Gemini Image (comfyui_AuthToken)
   - PD: Flux Kontext (comfyui_AuthToken)

### 直接 API Key 版本直接调用官方comfyui的key

API Key 版本：
- 只保留 ComfyUI API Key 输入 
- 参数和调用逻辑跟 ComfyUI 官方自带的 API 节点一样 - 只是在调用时可以手动填入 key

| 节点名称 | 文件 | 说明 |
|---------|------|------|
| **PD: GPT Image (apikey)** | `IMAGE_GPT_apikey.py` | OpenAI GPT Image 生成/编辑 |
| **PD: Gemini Image (apikey)** | `IMGE_GeminiNode_apikey.py` | Google Gemini 图像生成 |
| **PD: Gemini Pro Image (apikey)** | `IMGE_GeminiPro_apikey.py` | Google Gemini 3 Pro 高分辨率图像生成 |
| **PD: Flux Kontext (apikey)** | `IMGE_kontext_apikey.py` | Flux Kontext 图像编辑 |

---

## 🔑 两种版本的区别

### ComfyUI AuthToken 版本 (comfyui_AuthToken)

**认证方式**：使用 ComfyUI Auth Token，pandy自制油猴插件获取。

**认证头格式**：`Authorization: Bearer {auth_token}`

**使用示例**：
```
auth_token: eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzOGVhNmEw...
```

### API Key 版本 (apikey)

**认证方式**：使用 ComfyUI 官方 API Key

**认证头格式**：`X-API-KEY: {api_key}` （与 ComfyUI 官方 API 节点完全一致）

**使用示例**：
```
api_key: comfy_xxxxxxxxxxxxxxxxxxxxx
```

**特点**：
- 参数和调用逻辑跟 ComfyUI 官方自带的 API 节点一样
- 只是在调用时可以手动填入 key
- 直接使用官方账号的 key 申请下来填入使用


## 📖 节点详细说明

### 1. GPT Image 节点

**功能**：
- 文生图（Text-to-Image）
- 图片编辑（Image Editing）
- 图片修复（Inpainting with Mask）

**支持模型**：

**AuthToken 版本**：
- `gpt-image-1`：OpenAI GPT Image 1（推荐）
- `gpt-image-1.5`：OpenAI GPT Image 1.5（更高质量）


### 3. Gemini Pro Image 节点（Nano Banana Pro）

**功能**：
- 高分辨率图像生成（1K/2K/4K）
- 文生图（Text-to-Image）
- 图生图（Image-to-Image）
- 支持多张参考图片（最多14张）
- 同时输出图像和文本说明

**支持模型**：
- `gemini-3-pro-image-preview`：Gemini 3 Pro Image（最高质量）

**参数说明**：
- `prompt`：文本提示词，描述要生成的图像或编辑指令
- `model`：gemini-3-pro-image-preview
- `aspect_ratio`：宽高比
  - auto：自动匹配输入图片比例，无图片时默认 16:9
  - 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- `resolution`：目标输出分辨率
  - 1K：标准分辨率
  - 2K：使用原生 Gemini 放大器
  - 4K：使用原生 Gemini 放大器（最高质量）
- `response_modalities`：输出模式
  - IMAGE+TEXT：同时返回图像和文本说明
  - IMAGE：仅返回图像
- `images`（可选）：参考图片（最多14张，使用 Batch Images 节点）
- `system_prompt`（可选）：系统提示词，默认使用官方优化的图像生成提示词

**输出**：
- `image`：生成的图像
- `text`：文本说明（包含模型的思考过程和描述）

**成本估算**：
- Gemini 3 Pro Image: 约 $0.12/张
  - 基于 token 使用量：$2/1M input + $12/1M output text + $120/1M output image

**特殊说明（apikey 版本）**：
- 需要提供 `project_id`（Google Cloud 项目 ID）
- 需要提供 `location`（区域，默认 us-central1）
- 超时时间：180秒（因为高分辨率生成需要更长时间）

**使用建议**：
- 适合需要高质量、高分辨率图像的场景
- 2K/4K 分辨率会使用原生放大器，质量更好但成本更高
- IMAGE+TEXT 模式可以看到模型的思考过程，有助于理解生成结果
- 可以提供多张参考图片来引导生成风格

---

### 4. Flux Kontext 节点

**功能**：
- 图片编辑（Image Editing）
- 基于上下文的智能修改

**支持模型**：
- `flux-kontext-pro`：快速版本
- `flux-kontext-max`：高质量版本

**参数说明**：
- `prompt`：编辑指令
- `aspect_ratio`：宽高比
- `guidance`：提示词遵循度（0.1-99.0）
- `steps`：生成步数（1-150）
- `prompt_upsampling`：AI 增强提示词
- `input_image`（可选）：输入图片

**成本估算**：
- Kontext Pro: $0.05/张
- Kontext Max: $0.10/张

**特点**：
- 异步生成，需要轮询结果
- 最长等待时间：10 分钟
- 轮询间隔：2 秒

---

## 🚀 使用建议

### 选择 ComfyUI AuthToken 版本，如果你：
- ✅ 已经有 ComfyUI Auth Token
- ✅ 想要统一管理多个 AI 模型
- ✅ 喜欢通过 ComfyUI 生态系统
- ✅ 不想配置多个平台的 API Key

### 选择直接 API Key 版本，如果你：
- ✅ 已经有各个平台的 API Key
- ✅ 想要直接控制 API 调用
- ✅ 不想依赖 ComfyUI 代理服务
- ✅ 需要更灵活的配置（如 Gemini 的 project_id）

---

## 📝 使用示例

### 示例 1：使用 ComfyUI AuthToken 生成图片（GPT Image）

```
节点：PD: GPT Image (comfyui_AuthToken)

参数：
- auth_token: eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzOGVhNmEw...
- prompt: A futuristic city with flying cars at sunset
- model: gpt-image-1
- quality: medium
- background: auto
- size: 1024x1024
- n: 1
```

### 示例 2：使用直接 API Key 生成图片（DALL-E）

```
节点：PD: GPT Image (apikey)

参数：
- api_key: sk-proj-abc123...
- prompt: A futuristic city with flying cars at sunset
- model: dall-e-3
- quality: hd
- size: 1024x1792
```

### 示例 3：使用 Gemini 生成图片（apikey 版本）

```
节点：PD: Gemini Image (apikey)

参数：
- api_key: AIzaSyD...
- project_id: my-project-123
- location: us-central1
- prompt: A beautiful landscape with mountains
- model: gemini-2.5-flash-image
- aspect_ratio: 16:9
- resolution: 2K
```

### 示例 4：使用 Flux Kontext 编辑图片

```
节点：PD: Flux Kontext (comfyui_AuthToken)

参数：
- auth_token: eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzOGVhNmEw...
- prompt: Change the sky to sunset colors
- model: flux-kontext-pro
- aspect_ratio: 16:9
- guidance: 3.0
- steps: 50
- input_image: [连接输入图片]
```

---

## ⚠️ 注意事项

1. **API Key 安全**：
   - 不要在公开的工作流中暴露 API Key
   - 建议使用环境变量或配置文件管理

2. **成本控制**：
   - 所有节点都会显示预估成本
   - 实际成本以各平台账单为准
   - 建议设置 API 使用限额

3. **超时设置**：
   - GPT/Gemini: 120 秒超时
   - Flux Kontext: 10 分钟超时（包含轮询）

4. **错误处理**：
   - 所有节点都有详细的错误日志
   - 检查 ComfyUI 控制台查看详细信息

5. **网络要求**：
   - 需要稳定的网络连接
   - 某些地区可能需要代理访问

---

## 🔧 故障排除

### 问题 1：认证失败（401 错误）

**ComfyUI AuthToken 版本**：
- 检查 Auth Token 是否正确
- Token 可能已过期，重新获取
- 认证头格式：`Authorization: Bearer {auth_token}`

**直接 API Key 版本**：
- 检查 API Key 格式是否正确
- 确认 API Key 有效且有余额
- 认证头格式：`X-API-KEY: {api_key}` （与 ComfyUI 官方 API 节点一致）
- Gemini 需要确认 Project ID 和 Location 正确

### 问题 2：请求超时

- 检查网络连接
- 尝试增加超时时间
- Flux Kontext 可能需要更长时间，属于正常现象

### 问题 3：图片无法生成 / 返回黑图

**症状**：节点返回 512x512 的黑色图片

**可能原因**：
1. **API 端点错误**（已修复）：
   - 旧版本的 Gemini AuthToken 节点使用了错误的端点格式
   - 错误：`https://api.comfy.org/proxy/vertexai/gemini/{model}:generateContent`
   - 正确：`https://api.comfy.org/proxy/vertexai/gemini/{model}`
   - **解决方案**：更新到最新版本（v2.1+）

2. **模型名称不支持**：
   - 检查控制台是否有 "Unsupported model" 错误
   - 确认使用的模型在支持列表中
   - Gemini AuthToken 版本支持：`gemini-2.5-flash-image`, `gemini-2.5-flash-image-preview`

3. **提示词违规**：
   - 检查提示词是否符合内容政策
   - 查看控制台日志获取详细错误信息

4. **API 配额不足**：
   - 确认账户有足够的配额/余额
   - 检查 API 使用限制

**调试步骤**：
1. 打开 ComfyUI 控制台
2. 查找 `[GEMINI_AUTHTOKEN]` 或 `[GEMINI_APIKEY]` 开头的日志
3. 检查是否有 API Error 信息
4. 根据错误信息采取相应措施

### 问题 4：认证失败（Gemini 节点）

- 确认已启用 Vertex AI API
- 确认 Project ID 正确
- 确认 Location 支持 Gemini 模型
- 检查 API Key 权限

---

## 📚 相关链接

- [ComfyUI 官网](https://www.comfy.org)
- [OpenAI Platform](https://platform.openai.com)
- [Google Cloud Console](https://console.cloud.google.com)
- [Black Forest Labs](https://blackforestlabs.ai)
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**最后更新**：2025-01-12
**版本**：2.5

**更新日志**：
- 2025-01-12 v2.5: 修复 API Key 版本认证头问题（401 错误），使用 `X-API-KEY` 与官方一致
- 2025-01-12 v2.4: 新增 Gemini Pro Image (Nano Banana Pro) 节点，支持高分辨率生成
- 2025-01-12 v2.3: 更新 GPT Image AuthToken 节点参数，匹配官方 ComfyUI API 节点
- 2025-01-12 v2.2: 添加官方 ComfyUI API 节点的默认 system prompt
- 2025-01-12 v2.1: 修复 Gemini AuthToken 版本黑图问题，更新模型列表
- 2025-01-11 v2.0: 初始版本，支持 AuthToken 和 API Key 两种认证方式
