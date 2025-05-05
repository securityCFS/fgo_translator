# FGO 對話翻譯器

[English](../README.md) | [簡體中文](README_CN.md) | [繁體中文](README_TW.md)

一個 Python 工具，用於將《Fate/Grand Order》（FGO）對話從日語翻譯為其他語言，支持 GPT 或 Google 翻譯。此工具主要由 `Cursor` 開發，因此部分功能（如搜尋）尚未優化，但在簡單翻譯場景下仍非常實用。

對話腳本資料來源於 [Atlas Academy](https://apps.atlasacademy.io/db)。要查詢戰役名稱，請訪問 [此頁面](https://apps.atlasacademy.io/db/JP/wars)。

提供了一個簡單的命令列示範腳本 `demo.py`，詳細用法見 `demo.ipynb`。核心實作請參考 `dialogue_loader.py` 和 `db_loader.py`。

## 功能

- 將 FGO 對話從日語翻譯為：
  - 英語  
  - 簡體中文  
  - 繁體中文  
- 支持 GPT 與 Google 翻譯  
- 互動式命令列介面  
- 自動偵測關卡與腳本  
- 保存用戶偏好（API 金鑰、語言設定）  
- 可自訂匯出目錄  
- 多語言介面  

## 環境需求

- Python 3.8 或更高  
- pip（Python 套件管理器）  

## 安裝

1. 克隆倉庫：  

```bash
git clone https://github.com/yourusername/fgo_translator.git
cd fgo_translator
```

2. 建立虛擬環境（推薦）：  

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安裝相依套件：  

```bash
pip install -r requirements.txt
```

## 使用方法

1. 執行示範腳本：  

```bash
python demo.py
```

2. 按照互動提示進行：  

- 選擇目標語言  
- 選擇翻譯方式（GPT 或 Google 翻譯）  
- 若使用 GPT：  
  - 輸入 API 基礎 URL（預設：https://api.openai.com/v1）  
  - 輸入 API 金鑰  
  - 選擇 API 類型（OpenAI 或 自訂）  
  - 輸入基礎模型名稱（預設：gpt-4）  
  - 選擇認證方式  
- 輸入戰役名稱（可於 https://apps.atlasacademy.io/db/JP/wars 查詢）  
- 選擇翻譯所有關卡或搜尋特定關卡  
- 指定匯出目錄（可選）  

3. 翻譯結果將保存在指定目錄。  

## 配置

工具會將你的偏好保存在 SQLite 資料庫（`user_preferences.db`）中，包括：  

- 選擇的語言  
- API 設定  
- 翻譯方式  

你可在後續使用或更新這些設定。  

## 翻譯方式

### GPT 翻譯

- 需要 OpenAI API 金鑰（支持所有 OpenAI/requests 兼容 API，推薦使用 [阿里雲](https://bailian.console.aliyun.com/) 取得免費 Token）  
- 支持自訂 API 端點  
- 可配置模型與認證方式  

### Google 翻譯

- 免費使用  
- 無需 API 金鑰  
- 限於 Google 翻譯能力  

## 目錄結構

```
fgo_translator/
├── demo.py              # 主示範腳本
├── dialogue_loader.py   # 核心翻譯功能
├── db_loader.py         # 資料庫互動
├── requirements.txt     # Python 相依
├── README.md            # 英文說明
└── docs/                # 文件資料夾
    ├── README_CN.md     # 簡體中文文件
    └── README_TW.md     # 繁體中文文件
```

## 授權

本專案採用 MIT 授權，詳情請見 LICENSE 檔。  

## 致謝

- 感謝 [Atlas Academy](https://apps.atlasacademy.io/) 提供 FGO 資料庫  
- 感謝 OpenAI 提供 GPT API  
- 感謝 Google 翻譯 API  
- 感謝 [Cursor](https://www.cursor.com/) 提供 AI 程式碼助手  