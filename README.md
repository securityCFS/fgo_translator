# FGO Dialogue Translator

[English](README.md) | [简体中文](docs/README_CN.md) | [繁體中文](docs/README_TW.md)

A web-first translator and visual-novel reader for Fate/Grand Order (FGO) story scripts. It can translate Japanese dialogue with LLM/VLM-compatible APIs, Google Translate, or official translations that are already synchronized in Atlas Academy for CN/TW/other servers.

The static web app runs directly in the browser and is deployed through GitHub Pages:

**Live demo:** [securitycfs.github.io/fgo_translator](https://securitycfs.github.io/fgo_translator/)

A Flask backend is still provided in `app.py` for local use, especially with API providers that block direct browser requests through CORS. The older command-line demo remains available in `demo.py`, and implementation details live in `dialogue_loader.py`, `db_loader.py`, and `translation_cache.py`.

## Features

- Static browser app with no required server for normal use.
- Local Flask mode for OpenAI-compatible APIs that need a server-side proxy.
- Atlas Academy task/script discovery with a default quick search for 5 recent tasks.
- Filters out tasks without dialogue scripts before showing playable/translation candidates.
- Japanese-to-English / Simplified Chinese / Traditional Chinese translation.
- Google Translate fallback when no API key is configured.
- Atlas official/synchronized translations when available for the selected script.
- GitHub-hosted translated script cache for sharing completed translations.
- Bilingual UI toggle for English and Chinese.
- Guided first-run tutorial with skip/replay support.
- FGO-style Gaming Mode with mobile gestures, fixed dialogue history area, auto/skip/back controls, and responsive text fitting.
- Browser-local settings for API keys, model options, language, and UI preferences.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fgo_translator.git
cd fgo_translator
```

2. Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Web Usage

1. Open the [live demo](https://securitycfs.github.io/fgo_translator/) or serve the `gh-pages` static worktree locally.
2. Use the default recent-task search, or search by event/war/quest name.
3. Pick a task and phase with an available dialogue script.
4. Choose a translation source:
   - Atlas official translation if present.
   - Google Translate for a free no-key fallback.
   - Gemini/OpenAI-compatible API after configuring credentials in Settings.
5. Start translation, then open Gaming Mode for the visual-novel reader.

The in-app tutorial walks through API setup, recent-task search, selecting a phase, loading Atlas translations, starting translation, and using Gaming Mode.

## Usage

1. Run the demo script:
```bash
python demo.py
```

2. Follow the interactive prompts:
   - Select target language
   - Choose translation method (GPT or Google Translate)
   - If using GPT:
     - Enter API base URL (default: https://api.openai.com/v1)
     - Enter API key
     - Choose API type (OpenAI or Custom)
     - Enter base model name (default: gpt-4)
     - Select authentication type
   - Enter war name (find at https://apps.atlasacademy.io/db/JP/wars)
   - Choose whether to translate all quests or search for specific ones
   - Specify export directory (optional)

3. The translated dialogues will be saved in the specified directory.

## Configuration

The web app saves all API credentials and preferences only in browser local storage. They are not uploaded to this repository or to the hosted static page.

The local Python/Flask tools may use local files such as `user_preferences.db` for CLI/demo preferences.

## Translation Methods

### GPT Translation
- Requires OpenAI API key (Supporting all OpenAI/requests compatible APIs, recommend to use [aliyun](https://bailian.console.aliyun.com/) for free tokens)
- Supports custom API endpoints
- Configurable model and authentication

### Google Translate
- Free to use
- No API key required
- Limited to Google Translate's capabilities

### Atlas / Hosted Translations
- Uses Atlas Academy open data for script and asset metadata.
- Shows official/synchronized translations from CN/TW/other servers when Atlas exposes them.
- Can read and reuse translated scripts from the GitHub-hosted cache when available.

## Project Scope

- This project does not unpack game files, modify the game client, or provide any game modification features.
- Art assets are loaded from publicly accessible Atlas Academy/CDN pages.
- Translated scripts may be synchronized to the project's GitHub-hosted cache so other users can reuse them.
- Please support the official Fate/Grand Order project: [fate-go.jp](https://www.fate-go.jp/).

## Directory Structure

```
fgo_translator/
├── app.py                         # Flask backend and local proxy mode
├── templates/
│   ├── index.html                 # Flask web app entry point
│   └── gaming.html                # Gaming Mode reader
├── dialogue_loader.py             # Atlas dialogue/script loading
├── translation_cache.py           # GitHub-hosted translation cache helpers
├── db_loader.py                   # Atlas Academy data fetching
├── demo.py                        # Command-line demo script
├── test/                          # Regression tests
└── docs/
    ├── project-notice.md          # Project scope and data-source notice
    ├── README_CN.md               # Simplified Chinese documentation
    └── README_TW.md               # Traditional Chinese documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Atlas Academy](https://apps.atlasacademy.io/) for providing the FGO database
- OpenAI for GPT API
- Google Translate API
- [Cursor](https://www.cursor.com/) for providing the AI code assistant

