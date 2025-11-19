# ComfyUI PD Comfy API Node

一个功能强大的 ComfyUI 自定义节点套件，集成了多个 AI 图像生成 API，支持通过 ComfyUI API Key 进行统一认证和调用。

## ✨ 功能特性

- 🎨 **多模型支持**：集成 Gemini 和 Flux.1 Kontext Pro 等先进图像生成模型
- 🔑 **统一认证**：使用 ComfyUI API Key 进行统一认证，简化配置流程
- 💰 **成本追踪**：实时计算和显示 API 调用成本，帮助控制预算
- 🖼️ **图像处理**：支持图像输入和输出，兼容 ComfyUI 标准张量格式（B H W C）
- ⚙️ **灵活配置**：支持多种宽高比、引导强度、步数等参数调节
- 🔄 **异步处理**：采用异步请求和轮询机制，提升处理效率

## 📦 安装方法

### 方式一：通过 Git 克隆（推荐）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/7BEII/comfyui-PD_comfy-api-node.git
cd comfyui-PD_comfy-api-node
```

### 方式二：手动安装

1. 下载或克隆本仓库到 `ComfyUI/custom_nodes` 目录
2. 确保目录结构如下：
   ```
   ComfyUI/custom_nodes/comfyui-PD_comfy-api-node/
   ├── __init__.py
   ├── py/
   │   ├── IMGE_GeminiNode.py
   │   └── IMGE_kontext.py
   └── requirements.txt
   ```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 使用方法

1. **获取 ComfyUI API Key**
   - 访问 ComfyUI API 服务获取您的 API Key
   - 在节点中输入您的 API Key

2. **在 ComfyUI 中使用节点**
   - 重启 ComfyUI 后，在节点菜单中找到 `PD_Tools` 分类
   - 选择需要的节点并配置参数
   - 连接输入输出，执行工作流

## 📋 节点说明

### 1. PD: Gemini Image Gen (With Cost Info)

**功能**：使用 Google Gemini 模型生成图像

**输入参数**：
- `comfy_api_key` (必需): ComfyUI API Key
- `prompt` (必需): 图像生成提示词
- `model` (可选): 模型选择
  - `gemini-2.5-flash-image` (默认)
  - `gemini-2.5-flash-image-preview`
  - `gemini-1.5-pro`
  - `gemini-1.5-flash`
- `aspect_ratio` (可选): 宽高比
  - `auto` (自动，参考原图比例)
  - `1:1`, `16:9`, `9:16`, `4:3`, `3:4`
- `seed` (可选): 随机种子 (默认: 0)
- `image_ref` (可选): 参考图像 (IMAGE 类型)

**输出**：
- `image`: 生成的图像 (IMAGE 类型，格式: B H W C)
- `cost_info`: 成本信息字符串，包含：
  - 使用的模型
  - 输入/输出 Token 数量
  - 图像生成数量
  - 预估总成本（美元）

**特性**：
- 支持参考图像输入
- Auto 模式下自动参考原图比例
- 实时成本计算和显示

### 2. PD: Flux.1 Kontext Pro (ComfyUI Key)

**功能**：使用 Black Forest Labs 的 Flux.1 Kontext Pro 模型生成或编辑图像

**输入参数**：
- `comfy_api_key` (必需): ComfyUI API Key
- `prompt` (必需): 图像生成/编辑提示词
- `aspect_ratio` (必需): 宽高比字符串
  - 支持: `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9`, `4:5`
- `guidance` (可选): 引导强度 (默认: 3.0, 范围: 0.1-99.0)
- `steps` (可选): 生成步数 (默认: 50, 范围: 1-150)
- `seed` (可选): 随机种子 (默认: 1234)
- `prompt_upsampling` (可选): 提示词增强 (默认: False)
- `input_image` (可选): 输入图像，用于图像编辑 (IMAGE 类型)

**输出**：
- `image`: 生成的图像 (IMAGE 类型，格式: B H W C)
- `cost_info`: 成本信息字符串

**特性**：
- 支持文本生成图像
- 支持图像编辑（提供 input_image）
- 异步轮询机制，自动等待生成完成
- 实时进度显示

## 🔧 技术细节

### 张量格式

所有图像输入输出均遵循 ComfyUI 标准格式：
- **图像**: `B H W C` (批次, 高度, 宽度, 通道)
- **遮罩**: `B H W` (批次, 高度, 宽度)

### 依赖项

- `torch`: PyTorch 深度学习框架
- `google-generativeai`: Google Gemini API 客户端
- `aiohttp`: 异步 HTTP 客户端
- `pydantic`: 数据验证库
- `PIL`: Python 图像处理库
- `numpy`: 数值计算库

### 架构设计

- **模块化设计**：每个节点独立文件，便于维护和扩展
- **动态加载**：自动扫描 `py/` 目录下的节点模块
- **错误处理**：完善的异常捕获和错误提示
- **异步支持**：使用异步 I/O 提升性能

## 💡 使用示例

### 示例 1: 使用 Gemini 生成图像

```
1. 添加 "PD: Gemini Image Gen (With Cost Info)" 节点
2. 输入您的 ComfyUI API Key
3. 输入提示词，例如: "A futuristic city with flying cars"
4. 选择模型和宽高比
5. 连接输出到图像显示节点
6. 执行工作流
```

### 示例 2: 使用 Flux.1 Kontext Pro 编辑图像

```
1. 添加 "PD: Flux.1 Kontext Pro (ComfyUI Key)" 节点
2. 输入您的 ComfyUI API Key
3. 连接输入图像到 input_image 端口
4. 输入编辑提示词，例如: "Add a sunset sky in the background"
5. 设置宽高比和其他参数
6. 连接输出并执行
```

## ⚠️ 注意事项

1. **API Key 安全**：请妥善保管您的 ComfyUI API Key，不要分享给他人
2. **成本控制**：使用前请了解各模型的定价，注意控制 API 调用成本
3. **网络连接**：需要稳定的网络连接以访问 API 服务
4. **生成时间**：某些模型可能需要较长的生成时间，请耐心等待
5. **内容审核**：生成的图像需要符合 API 服务的内容政策

## 🐛 故障排除

### 节点未显示
- 确保已正确安装到 `custom_nodes` 目录
- 检查 `__init__.py` 文件是否存在
- 查看 ComfyUI 控制台的错误信息

### API 调用失败
- 验证 API Key 是否正确
- 检查网络连接
- 查看控制台错误信息

### 图像格式错误
- 确保输入图像格式为 `B H W C`
- 检查图像通道数是否为 3 (RGB)

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持 Gemini 图像生成
- 支持 Flux.1 Kontext Pro 图像生成
- 实现成本计算功能

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📮 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**注意**：本项目为 ComfyUI 自定义节点，需要配合 ComfyUI 使用。确保您已正确安装 ComfyUI 环境。
