# FGO Dialogue Translator

[English](README.md) | [简体中文](docs/README_CN.md) | [繁體中文](docs/README_TW.md)

A Python tool for translating Fate/Grand Order (FGO) dialogues from Japanese to other languages using GPT or Google Translate. This tool is mainly created by `Cursor`, therefore some parts like search function is not optimized. However, this tool is still useful for simple translation.

The scripts data are extracted from [Atlas Academy](https://apps.atlasacademy.io/db). To search for war names, you can go to [this page](https://apps.atlasacademy.io/db/JP/wars).

A simple command-line demo is provided in `demo.py`. Detailed usage is shown in `demo.ipynb`. For detailed implementation, please refer to `dialogue_loader.py` and `db_loader.py`.

## Features

- Translate FGO dialogues from Japanese to:
  - English
  - Simplified Chinese
  - Traditional Chinese
- Support for both GPT and Google Translate
- Interactive command-line interface
- Automatic quest and script detection
- Saves user preferences (API keys, language settings)
- Customizable export directory
- Multi-language interface

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

The tool saves your preferences in a SQLite database (`user_preferences.db`), including:
- Selected language
- API configurations
- Translation method

You can use these saved settings in future sessions or update them as needed.

## Translation Methods

### GPT Translation
- Requires OpenAI API key (Supporting all OpenAI/requests compatible APIs, recommend to use [aliyun](https://bailian.console.aliyun.com/) for free tokens)
- Supports custom API endpoints
- Configurable model and authentication

### Google Translate
- Free to use
- No API key required
- Limited to Google Translate's capabilities

## Directory Structure

```
fgo_translator/
├── demo.py              # Main demo script
├── dialogue_loader.py   # Core translation functionality
├── db_loader.py         # Database interaction
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── docs/               # Documentation
    ├── README_CN.md    # Simplified Chinese documentation
    └── README_TW.md    # Traditional Chinese documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Atlas Academy](https://apps.atlasacademy.io/) for providing the FGO database
- OpenAI for GPT API
- Google Translate API
- [Cursor](https://www.cursor.com/) for providing the AI code assistant
