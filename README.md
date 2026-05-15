# FGO Dialogue Translator

[English](README.md) | [简体中文](docs/README_CN.md) | [繁體中文](docs/README_TW.md)

**Live demo:** [securitycfs.github.io/fgo_translator](https://securitycfs.github.io/fgo_translator/)

A static web tool for translating Fate/Grand Order (FGO) story dialogues from Japanese. It runs without an app server for normal browser use; Atlas data, translation requests, and optional cache lookups are made directly from the browser. API keys remain in local browser storage. Script data is fetched directly from [Atlas Academy](https://apps.atlasacademy.io/db).

Also includes a Python / Flask backend (`app.py`) for local use with APIs that block cross-origin browser requests (e.g. DashScope / 通义千问).

---

## Quick Start (Web)

1. Open the [live demo](https://securitycfs.github.io/fgo_translator/)
2. Search for a war, event, or quest name (for example: `奏章Ⅳ`, `ネロ祭`, `Camelot`, or `Snowfield`)
3. Pick a quest and phase
4. Choose a translation engine (see below), click **Start Translation**
5. Click **Gaming Mode** to read the story with a visual-novel-style UI

---

## Recent UI and workflow updates

- First-run guided tutorial with skip/replay support.
- English/Chinese interface toggle.
- Settings guidance for configuring your own API endpoint and token.
- Default recent-task discovery now loads 5 tasks for a faster first search.
- Task search filters out entries that do not contain dialogue scripts.
- Atlas official/synchronized translations can be loaded when available for the selected script.
- Translated scripts can be reused from the GitHub-hosted cache.
- Gaming Mode includes mobile gestures, fixed-height dialogue history, responsive text fitting, and a small in-game guide.

---

## Translation Engines

### Free Engine — Google Translate (no key required)
Select **Free Engine (Google Translate)** in the Engine dropdown. No configuration needed. Quality is lower than LLM-based translation but instant and free.

### Gemini (recommended for browser use)
Google's API supports cross-origin requests from browsers.

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/)
2. Open **Settings** (⚙ icon) → API Type: `Gemini` → paste key → Save
3. Default model: `gemini-2.5-flash` (fast and free-tier eligible)

| Field | Value |
|---|---|
| API Type | Gemini |
| API Key | your key from AI Studio |
| API Base URL | `https://generativelanguage.googleapis.com/v1beta` |
| Model | `gemini-2.5-flash` |

### DeepSeek
DeepSeek's API is OpenAI-compatible but **blocks browser requests (no CORS)**. Use it with a local server or Cloudflare Worker proxy.

- Docs: [api-docs.deepseek.com](https://api-docs.deepseek.com/zh-cn/)
- Console: [platform.deepseek.com](https://platform.deepseek.com/)

| Field | Value |
|---|---|
| API Type | OpenAI Compatible |
| API Base URL | `https://api.deepseek.com` |
| Model | `deepseek-chat` |

### 通义千问 / DashScope (Aliyun)
Also OpenAI-compatible, also CORS-blocked in browsers. Use local Flask server (`python app.py`).

- Console: [bailian.console.aliyun.com](https://bailian.console.aliyun.com/)
- Docs: [help.aliyun.com/zh/model-studio](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api)

| Field | Value |
|---|---|
| API Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Model | `qwen-plus` |

### Moonshot (Kimi)
- Console: [platform.moonshot.cn](https://platform.moonshot.cn/)
- Docs: [platform.moonshot.cn/docs](https://platform.moonshot.cn/docs/api/chat)

| Field | Value |
|---|---|
| API Base URL | `https://api.moonshot.cn/v1` |
| Model | `moonshot-v1-8k` |

### OpenAI
- Console: [platform.openai.com](https://platform.openai.com/)

| Field | Value |
|---|---|
| API Base URL | `https://api.openai.com/v1` |
| Model | `gpt-4o-mini` |

> **Note:** All non-Gemini APIs block browser requests due to CORS. For those, either run the local Flask server (`python app.py`) or set up a [Cloudflare Worker](https://developers.cloudflare.com/workers/) proxy.

---

## CORS workarounds for OpenAI-compatible APIs

If you see a *Network/CORS error* in the browser:

1. **Use Gemini** — Google's API has CORS enabled, works in browsers directly.
2. **Use Free Engine** — Google Translate public endpoint, no key needed.
3. **Run `python app.py` locally** — Flask proxies all API calls server-side.
4. **Cloudflare Worker proxy** — Deploy a tiny worker that forwards requests and adds CORS headers.

---

## Local Flask server

```bash
git clone https://github.com/securityCFS/fgo_translator.git
cd fgo_translator
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

Requires Python 3.8+. Works with any API provider.

---

## Gaming Mode

After translation, click **Gaming Mode** to open an immersive visual-novel reader:
- Character sprites and portraits from Atlas Academy CDN
- Choice branches with translated options
- History panel (H key)
- Auto-advance (A key), Skip (S key)
- FGO-style formatting: ruby text, size tags, inline images

---

## Directory Structure

```
fgo_translator/
├── index.html           # Static web app entry point
├── gaming.html          # Gaming mode visual-novel UI
├── js/
│   ├── api.js           # Atlas Academy API helpers + script parser
│   └── translate.js     # Translation engine (Gemini / OpenAI / Free)
├── app.py               # Flask backend (local use)
├── dialogue_loader.py   # Core script parsing logic
├── db_loader.py         # Atlas Academy data fetching
├── notebooks/           # Colab / Jupyter demos
└── docs/                # Translated READMEs and project notice
```

---

## Data Source

All script data is fetched live from [Atlas Academy](https://apps.atlasacademy.io/). No game files are bundled. Character sprites and portraits are loaded directly from `static.atlasacademy.io`.

## Privacy and project scope

- API keys and auth tokens are stored only in browser local storage.
- Translated scripts may be synchronized to the project's GitHub-hosted cache so other users can reuse them.
- Official/synchronized translations from CN/TW/other servers are shown when Atlas exposes them for a script.
- This project does not unpack game files, modify the game client, or provide game modification features.
- Please support the official Fate/Grand Order project: [fate-go.jp](https://www.fate-go.jp/).
- A short Chinese project notice is available at [`docs/project-notice.md`](docs/project-notice.md).

---

## License

MIT License

## Acknowledgments

- [Atlas Academy](https://apps.atlasacademy.io/) for the FGO database and CDN
- Google — Gemini API & Google Translate
- [Cursor](https://www.cursor.com/) / GitHub Copilot — AI coding assistants

