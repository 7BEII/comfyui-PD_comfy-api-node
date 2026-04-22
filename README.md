# ComfyUI PD Comfy API Node

ComfyUI 自定义节点套件，集成 Gemini 和 GPT 图像生成 API，支持两种认证方式。

## ✨ 功能特性

- 🎨 **多模型支持**：Gemini 2.5 Flash Image、GPT Image 1/1.5
- 🔑 **双认证方式**：支持 API Key 和 Auth Token 两种调用方式
- 💰 **成本追踪**：实时显示 API 调用成本
- 🖼️ **图像处理**：支持文生图、图生图、图像编辑
- ⚙️ **灵活配置**：多种宽高比、质量、分辨率选项

## 📦 安装方法

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/7BEII/comfyui-PD_comfy-api-node.git
pip install -r comfyui-PD_comfy-api-node/requirements.txt
```

重启 ComfyUI 后，在节点菜单的 `PD_Tools/Image_Generation` 分类中找到节点。

## 🔑 两种认证方式

### 方式一：API Key（推荐）

使用 ComfyUI 官方 API Key，简单直接。

**获取方法**：
1. 访问 [ComfyUI API](https://www.comfy.org/api-keys)
2. 登录并创建 API Key
3. 复制 Key 并粘贴到节点中

## 📝 更新日志

### v2.0.0 -2026.0422
- 更新chatgpt image2 节点
- PD: Nano Banana 2 (ComfyUI Key) 节点更新，增加重试3次机制，放置出黑图，稳定性增加。
### v2.0.0 -2026.0112
- 新增 Auth Token 认证方式
- 新增 GPT Image 节点
- 优化成本计算
- 改进错误处理

### v1.0.0
- 初始版本
- 支持 Gemini 图像生成
- 支持 API Key 认证

### 节点介绍
#### PD: GPT Image 2 (comfyui_apikey)
- 支持输入多图参考
- 支持单图输入，GPT image2 节点comfyui 格式。
![GPT Image 2 示例 1](image/Pasted%20image%2020260422192905.png)
![GPT Image 2 示例 2](image/Pasted%20image%2020260422192828.png)
#### PD: Gemini Image (ComfyUI Key)
![alt text](image.png)

#### PD: Gemini Pro Image (ComfyUI Key)
![alt text](image-1.png)
#### PD: Nano Banana 2 (ComfyUI Key)
![alt text](image-2.png)


### 方式二：Auth Token

使用浏览器 Auth Token，适合高级用户。

**获取方法**：
1. 安装油猴脚本（Tampermonkey）：具体B站有教程：小熊猫pandy
2. 安装 Token 抓取脚本  脚本: \comfyui-PD_comfy api node\temperay\PD_comfyui_token_grabber.user.js
3. 访问 ComfyUI 网站并登录
4. 脚本会自动抓取 Token   
5. 复制 Token 并粘贴到节点中

![Auth Token 节点示例](image/comfyui-node-auth.png)


**优点**：
- ✅ 无需申请 API Key
- ✅ 使用现有账号权限

## ⚠️ 注意事项

1. **API Key 安全**：请妥善保管 API Key，不要分享给他人
2. **成本控制**：注意查看成本信息，控制 API 调用次数
3. **网络连接**：需要稳定的网络连接
4. **生成时间**：高质量图像可能需要较长时间

## 🐛 故障排除

### 节点未显示
- 检查是否正确安装到 `custom_nodes` 目录
- 重启 ComfyUI
- 查看控制台错误信息

### API 调用失败
- 验证 API Key 或 Auth Token 是否正确
- 检查网络连接
- 查看控制台详细错误信息

### 图像格式错误
- 确保输入图像格式为 ComfyUI 标准格式（B H W C）
- 检查图像通道数是否为 3 (RGB)


## 🤝 贡献
欢迎提交 Issue 和 Pull Request！
---

