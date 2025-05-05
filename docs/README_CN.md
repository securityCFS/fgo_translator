# FGO 对话翻译器

[English](../README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md)

一个 Python 工具，用于将《Fate/Grand Order》（FGO）对话从日语翻译为其他语言，支持 GPT 或 Google 翻译。此工具主要由 `Cursor` 开发，因此部分功能（如搜索）尚未优化，但在简单翻译场景下仍然非常实用。

对话脚本数据来源于 [Atlas Academy](https://apps.atlasacademy.io/db)。要查询关卡名称，请访问 [此页面](https://apps.atlasacademy.io/db/JP/wars)。

提供了一个简单的命令行示例脚本 `demo.py`，详细用法见 `demo.ipynb`。核心实现请参考 `dialogue_loader.py` 和 `db_loader.py`。

## 功能

- 将 FGO 对话从日语翻译为：
  - 英语  
  - 简体中文  
  - 繁体中文  
- 支持 GPT 与 Google 翻译  
- 交互式命令行界面  
- 自动检测关卡与脚本  
- 保存用户偏好（API 密钥、语言设置）  
- 可定制导出目录  
- 多语言界面  

## 环境要求

- Python 3.8 或更高  
- pip（Python 包管理器）  

## 安装

1. 克隆仓库：  

```bash
git clone https://github.com/yourusername/fgo_translator.git
cd fgo_translator
```

2. 创建虚拟环境（推荐）：  

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：  

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行示例脚本：  

```bash
python demo.py
```

2. 按照交互提示操作：  

- 选择目标语言  
- 选择翻译方式（GPT 或 Google 翻译）  
- 如果使用 GPT：  
  - 输入 API 基础 URL（默认：https://api.openai.com/v1）  
  - 输入 API 密钥  
  - 选择 API 类型（OpenAI 或 自定义）  
  - 输入基础模型名称（默认：gpt-4）  
  - 选择认证方式  
- 输入战役名称（可在 https://apps.atlasacademy.io/db/JP/wars 查找）  
- 选择翻译所有关卡或搜索特定关卡  
- 指定导出目录（可选）  

3. 翻译结果将保存在指定目录下。  

## 配置

工具会将你的偏好保存在 SQLite 数据库（`user_preferences.db`）中，包括：  

- 选择的语言  
- API 配置  
- 翻译方式  

你可以在后续会话中复用或更新这些设置。  

## 翻译方式

### GPT 翻译

- 需要 OpenAI API 密钥（支持所有 OpenAI/requests 兼容 API，推荐使用 [阿里云](https://bailian.console.aliyun.com/) 获取免费 Token）  
- 支持自定义 API 端点  
- 可配置模型与认证方式  

### Google 翻译

- 免费使用  
- 无需 API 密钥  
- 仅限于 Google 翻译能力  

## 目录结构

```
fgo_translator/
├── demo.py              # 主演示脚本
├── dialogue_loader.py   # 核心翻译功能
├── db_loader.py         # 数据库交互
├── requirements.txt     # Python 依赖
├── README.md            # 英文说明
└── docs/                # 文档目录
    ├── README_CN.md     # 简体中文文档
    └── README_TW.md     # 繁体中文文档
```

## 许可证

本项目基于 MIT 协议，详情见 LICENSE 文件。  

## 致谢

- 感谢 [Atlas Academy](https://apps.atlasacademy.io/) 提供 FGO 数据库  
- 感谢 OpenAI 提供 GPT API  
- 感谢 Google 翻译 API  
- 感谢 [Cursor](https://www.cursor.com/) 提供 AI 代码助手