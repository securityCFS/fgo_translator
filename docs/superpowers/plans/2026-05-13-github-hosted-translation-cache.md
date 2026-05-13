# GitHub-Hosted Translation Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GitHub Pages-backed translation cache that server-generated translations can publish and both Flask and static deployments can read.

**Architecture:** The `master` branch gets a focused Python cache module that handles canonical hashing, cache path construction, JSON validation, GitHub Pages reads, and GitHub Contents API writes. The Flask `/translate` route uses that module per script, translating only misses and writing only trusted server-generated LLM results. The `static-migration` branch gets a matching JavaScript read-only cache helper used before browser-side model calls.

**Tech Stack:** Python 3, Flask, requests, pytest-compatible unittest tests, browser JavaScript, GitHub Pages static JSON, GitHub REST Contents API.

---

## File Structure

- Create `translation_cache.py`: Python cache key/hash/schema/path/client utilities for `master`.
- Create `test/test_translation_cache.py`: unit tests for Python hash, paths, validation, and read/write behavior.
- Modify `app.py`: pass script metadata through `/translate`, use cache hits, translate misses, and publish trusted results.
- Modify `templates/index.html`: include `script_ids`, `script_dialogue_counts`, and `source_region` in `/translate` requests.
- Modify `static-migration` files after porting in that branch:
  - Create `js/translation-cache.js`: browser read-only cache key/hash/path/validation utilities.
  - Modify `index.html`: load `js/translation-cache.js` and check cache before `Translator.translateDialogues()`.

The first implementation slice leaves Atlas/Rayshift and Free Engine paths unchanged. Only server LLM translations are written to GitHub.

---

### Task 1: Python Cache Utilities

**Files:**
- Create: `translation_cache.py`
- Test: `test/test_translation_cache.py`

- [ ] **Step 1: Write failing tests for canonical hashing, path construction, and validation**

Create `test/test_translation_cache.py` with:

```python
import json
import unittest
from unittest.mock import Mock

from translation_cache import (
    TranslationCacheConfig,
    TranslationCacheEntry,
    TranslationCacheKey,
    TranslationCacheClient,
    canonical_source_hash,
    normalize_provider,
    normalize_target_language,
    sanitize_path_segment,
)


class TranslationCacheUtilityTests(unittest.TestCase):
    def test_canonical_source_hash_is_stable_for_equivalent_dialogues(self):
        dialogues = [
            {"speaker": "A", "content": "line\r\none", "ignored": "x"},
            {"speaker": "B", "content": " two "},
        ]
        equivalent = [
            {"content": "line\none", "speaker": "A"},
            {"speaker": "B", "content": " two "},
        ]

        self.assertEqual(canonical_source_hash(dialogues), canonical_source_hash(equivalent))

    def test_normalize_target_language_maps_ui_values(self):
        self.assertEqual(normalize_target_language("Chinese (Simplified)"), "zh-CN")
        self.assertEqual(normalize_target_language("Chinese (Traditional)"), "zh-TW")
        self.assertEqual(normalize_target_language("English"), "en")

    def test_provider_and_path_segments_are_safe(self):
        self.assertEqual(normalize_provider("OpenAI"), "openai")
        self.assertEqual(sanitize_path_segment("gpt-4.1/mini:test model"), "gpt-4.1_mini_test_model")

    def test_key_builds_deterministic_relative_path(self):
        key = TranslationCacheKey(
            script_id="0400041440",
            source_region="JP",
            source_hash="abcdef123456",
            target_language="zh-CN",
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_version="fgo-v1",
        )

        self.assertEqual(
            key.relative_path(),
            "v1/JP/0400041440/abcdef123456/zh-CN/gemini/gemini-2.5-flash/fgo-v1.json",
        )

    def test_entry_validation_rejects_mismatch_and_error_placeholders(self):
        key = TranslationCacheKey(
            script_id="1",
            source_region="JP",
            source_hash="hash",
            target_language="zh-CN",
            provider="gemini",
            model="gemini",
            prompt_version="fgo-v1",
        )
        good = {
            "schema_version": 1,
            "script_id": "1",
            "source_region": "JP",
            "source_hash": "hash",
            "target_language": "zh-CN",
            "provider": "gemini",
            "model": "gemini",
            "prompt_version": "fgo-v1",
            "dialogue_count": 1,
            "trusted_generation": True,
                "translations": [{"speaker": "A", "translated_content": "hello"}],
        }
        self.assertTrue(TranslationCacheEntry.from_json(good, key, 1))

        bad_count = dict(good, dialogue_count=2)
        self.assertIsNone(TranslationCacheEntry.from_json(bad_count, key, 1))

        bad_error = dict(good, translations=[{"speaker": "A", "translated_content": "[Translation Error: boom]"}])
        self.assertIsNone(TranslationCacheEntry.from_json(bad_error, key, 1))


class TranslationCacheClientTests(unittest.TestCase):
    def test_read_hit_returns_valid_entry(self):
        key = TranslationCacheKey("1", "JP", "hash", "zh-CN", "gemini", "gemini", "fgo-v1")
        payload = {
            "schema_version": 1,
            "script_id": "1",
            "source_region": "JP",
            "source_hash": "hash",
            "target_language": "zh-CN",
            "provider": "gemini",
            "model": "gemini",
            "prompt_version": "fgo-v1",
            "dialogue_count": 1,
            "trusted_generation": True,
            "translations": [{"speaker": "A", "translated_content": "hello"}],
        }
        session = Mock()
        session.get.return_value = Mock(status_code=200, json=Mock(return_value=payload))
        client = TranslationCacheClient(
            TranslationCacheConfig(base_url="https://example.github.io/cache"),
            session=session,
        )

        entry = client.read(key, expected_dialogue_count=1)

        self.assertIsNotNone(entry)
        self.assertEqual(entry.translations[0]["translated_content"], "hello")

    def test_write_uses_github_contents_api_when_configured(self):
        key = TranslationCacheKey("1", "JP", "hash", "zh-CN", "gemini", "gemini", "fgo-v1")
        entry = TranslationCacheEntry(
            key=key,
            dialogue_count=1,
            translations=[{"speaker": "A", "translated_content": "hello"}],
        )
        session = Mock()
        session.get.return_value = Mock(status_code=404)
        session.put.return_value = Mock(status_code=201, json=Mock(return_value={}))
        client = TranslationCacheClient(
            TranslationCacheConfig(
                base_url="https://example.github.io/cache",
                repo="owner/repo",
                branch="main",
                token="secret",
                write_enabled=True,
            ),
            session=session,
        )

        self.assertTrue(client.write(entry))
        put_url = session.put.call_args.args[0]
        self.assertIn("/repos/owner/repo/contents/v1/JP/1/hash/zh-CN/gemini/gemini/fgo-v1.json", put_url)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist**

Run:

```bash
python -m unittest test.test_translation_cache -v
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'translation_cache'`.

- [ ] **Step 3: Implement `translation_cache.py`**

Create `translation_cache.py` with:

```python
import base64
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_PROMPT_VERSION = "fgo-v1"


def _normalise_line_endings(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def canonical_source_payload(dialogues: List[Dict]) -> List[Dict[str, str]]:
    return [
        {
            "speaker": _normalise_line_endings(item.get("speaker", "")),
            "content": _normalise_line_endings(item.get("content", "")),
        }
        for item in dialogues
    ]


def canonical_source_hash(dialogues: List[Dict]) -> str:
    payload = canonical_source_payload(dialogues)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sanitize_path_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    safe = safe.strip("._")
    return safe or "unknown"


def normalize_target_language(value: str) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    mapping = {
        "chinese": "zh-CN",
        "chinese simplified": "zh-CN",
        "chinese (simplified)": "zh-CN",
        "simplified chinese": "zh-CN",
        "zh-cn": "zh-CN",
        "中文": "zh-CN",
        "简体中文": "zh-CN",
        "chinese traditional": "zh-TW",
        "chinese (traditional)": "zh-TW",
        "traditional chinese": "zh-TW",
        "zh-tw": "zh-TW",
        "繁體中文": "zh-TW",
        "繁体中文": "zh-TW",
        "english": "en",
        "en": "en",
        "japanese": "ja",
        "ja": "ja",
        "korean": "ko",
        "ko": "ko",
    }
    return mapping.get(lowered, sanitize_path_segment(raw))


def normalize_provider(api_type: str, model: str = "") -> str:
    raw = str(api_type or "openai").strip().lower()
    model_l = str(model or "").lower()
    if raw == "gemini" or "gemini" in model_l:
        return "gemini"
    if "deepseek" in model_l:
        return "deepseek"
    if "qwen" in model_l or "qwq" in model_l:
        return "qwen"
    if "claude" in model_l:
        return "claude"
    if raw in {"openai", "custom"}:
        return raw
    return sanitize_path_segment(raw.lower())


@dataclass(frozen=True)
class TranslationCacheKey:
    script_id: str
    source_region: str
    source_hash: str
    target_language: str
    provider: str
    model: str
    prompt_version: str = DEFAULT_PROMPT_VERSION

    def relative_path(self) -> str:
        parts = [
            "v1",
            sanitize_path_segment(self.source_region.upper()),
            sanitize_path_segment(self.script_id),
            sanitize_path_segment(self.source_hash),
            sanitize_path_segment(self.target_language),
            sanitize_path_segment(self.provider),
            sanitize_path_segment(self.model),
            f"{sanitize_path_segment(self.prompt_version)}.json",
        ]
        return "/".join(parts)


@dataclass
class TranslationCacheEntry:
    key: TranslationCacheKey
    dialogue_count: int
    translations: List[Dict[str, str]]
    generated_at: Optional[str] = None

    def to_json(self) -> Dict:
        generated_at = self.generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "schema_version": SCHEMA_VERSION,
            "script_id": self.key.script_id,
            "source_region": self.key.source_region,
            "source_hash": self.key.source_hash,
            "target_language": self.key.target_language,
            "provider": self.key.provider,
            "model": self.key.model,
            "prompt_version": self.key.prompt_version,
            "dialogue_count": self.dialogue_count,
            "trusted_generation": True,
            "generator": {
                "app": "fgo_translator",
                "branch_mode": "server",
                "generated_at": generated_at,
            },
            "translations": [
                {
                    "speaker": str(item.get("speaker", "")),
                    "translated_content": str(item.get("translated_content", "")),
                }
                for item in self.translations
            ],
        }

    @classmethod
    def from_json(cls, data: Dict, key: TranslationCacheKey, expected_dialogue_count: int) -> Optional["TranslationCacheEntry"]:
        try:
            if data.get("schema_version") != SCHEMA_VERSION:
                return None
            if data.get("trusted_generation") is not True:
                return None
            expected = {
                "script_id": key.script_id,
                "source_region": key.source_region,
                "source_hash": key.source_hash,
                "target_language": key.target_language,
                "provider": key.provider,
                "model": key.model,
                "prompt_version": key.prompt_version,
            }
            for field, value in expected.items():
                if str(data.get(field, "")) != str(value):
                    return None
            translations = data.get("translations")
            if not isinstance(translations, list):
                return None
            dialogue_count = int(data.get("dialogue_count", -1))
            if dialogue_count != expected_dialogue_count or dialogue_count != len(translations):
                return None
            cleaned = []
            for item in translations:
                if not isinstance(item, dict):
                    return None
                text = str(item.get("translated_content", ""))
                if "[Translation Error:" in text:
                    return None
                cleaned.append({
                    "speaker": str(item.get("speaker", "")),
                    "translated_content": text,
                })
            return cls(key=key, dialogue_count=dialogue_count, translations=cleaned)
        except Exception:
            return None


@dataclass
class TranslationCacheConfig:
    base_url: str = ""
    repo: str = ""
    branch: str = "main"
    token: str = ""
    prompt_version: str = DEFAULT_PROMPT_VERSION
    enabled: bool = False
    write_enabled: bool = False

    @classmethod
    def from_env(cls) -> "TranslationCacheConfig":
        base_url = os.getenv("TRANSLATION_CACHE_BASE_URL", "").rstrip("/")
        repo = os.getenv("TRANSLATION_CACHE_REPO", "")
        token = os.getenv("TRANSLATION_CACHE_TOKEN", "")
        enabled = os.getenv("TRANSLATION_CACHE_ENABLED", "").lower()
        write_enabled = os.getenv("TRANSLATION_CACHE_WRITE_ENABLED", "").lower()
        return cls(
            base_url=base_url,
            repo=repo,
            branch=os.getenv("TRANSLATION_CACHE_BRANCH", "main"),
            token=token,
            prompt_version=os.getenv("TRANSLATION_CACHE_PROMPT_VERSION", DEFAULT_PROMPT_VERSION),
            enabled=(enabled not in {"0", "false", "no"} and bool(base_url)),
            write_enabled=(write_enabled not in {"0", "false", "no"} and bool(repo and token)),
        )


class TranslationCacheClient:
    def __init__(self, config: TranslationCacheConfig, session=None):
        self.config = config
        self.session = session or requests.Session()

    def read(self, key: TranslationCacheKey, expected_dialogue_count: int) -> Optional[TranslationCacheEntry]:
        if not self.config.base_url:
            return None
        url = f"{self.config.base_url.rstrip('/')}/{key.relative_path()}"
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return TranslationCacheEntry.from_json(response.json(), key, expected_dialogue_count)
        except Exception as exc:
            logger.warning("Translation cache read failed for %s: %s", key.relative_path(), exc)
            return None

    def write(self, entry: TranslationCacheEntry) -> bool:
        if not (self.config.write_enabled and self.config.repo and self.config.token):
            return False
        path = entry.key.relative_path()
        get_url = f"https://api.github.com/repos/{self.config.repo}/contents/{quote(path)}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.config.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            existing = self.session.get(get_url, headers=headers, params={"ref": self.config.branch}, timeout=15)
            if existing.status_code == 200:
                logger.info("Translation cache entry already exists: %s", path)
                return True
            if existing.status_code not in {404}:
                logger.warning("Translation cache existence check failed for %s: HTTP %s", path, existing.status_code)
                return False

            body = json.dumps(entry.to_json(), ensure_ascii=False, indent=2)
            payload = {
                "message": f"Add translation cache {entry.key.script_id} {entry.key.target_language} {entry.key.model}",
                "content": base64.b64encode(body.encode("utf-8")).decode("ascii"),
                "branch": self.config.branch,
            }
            response = self.session.put(get_url, headers=headers, json=payload, timeout=20)
            if response.status_code in {200, 201}:
                return True
            logger.warning("Translation cache write failed for %s: HTTP %s %s", path, response.status_code, response.text[:200])
            return False
        except Exception as exc:
            logger.warning("Translation cache write failed for %s: %s", path, exc)
            return False
```

- [ ] **Step 4: Run the utility tests and verify they pass**

Run:

```bash
python -m unittest test.test_translation_cache -v
```

Expected: PASS.

---

### Task 2: Flask Cache Integration

**Files:**
- Modify: `app.py`
- Modify: `templates/index.html`
- Test: `test/test_translation_cache.py`

- [ ] **Step 1: Write failing tests for cached script orchestration**

Append to `test/test_translation_cache.py`:

```python
from app import _split_dialogues_by_script_counts, _merge_script_translations


class TranslationRouteHelperTests(unittest.TestCase):
    def test_split_dialogues_by_script_counts_keeps_script_order(self):
        dialogues = [
            {"speaker": "A", "content": "1"},
            {"speaker": "B", "content": "2"},
            {"speaker": "C", "content": "3"},
        ]

        result = _split_dialogues_by_script_counts(dialogues, ["s1", "s2"], [2, 1])

        self.assertEqual(result["s1"], dialogues[:2])
        self.assertEqual(result["s2"], dialogues[2:])

    def test_merge_script_translations_keeps_original_order(self):
        result = _merge_script_translations(
            ["s1", "s2"],
            {
                "s1": [{"speaker": "A", "translated_content": "one"}],
                "s2": [{"speaker": "B", "translated_content": "two"}],
            },
        )

        self.assertEqual([item["translated_content"] for item in result], ["one", "two"])
```

- [ ] **Step 2: Run the tests and verify they fail because helpers are missing**

Run:

```bash
python -m unittest test.test_translation_cache.TranslationRouteHelperTests -v
```

Expected: FAIL or ERROR with import error for `_split_dialogues_by_script_counts`.

- [ ] **Step 3: Add route helper imports and helper functions to `app.py`**

Add imports near the top:

```python
from translation_cache import (
    TranslationCacheClient,
    TranslationCacheConfig,
    TranslationCacheEntry,
    TranslationCacheKey,
    canonical_source_hash,
    normalize_provider,
    normalize_target_language,
)
```

After `user_preferences = load_user_preferences()`, add:

```python
translation_cache_config = TranslationCacheConfig.from_env()
translation_cache_client = TranslationCacheClient(translation_cache_config)
```

Before the `/translate` route, add:

```python
def _split_dialogues_by_script_counts(dialogues, script_ids, script_dialogue_counts):
    if not script_ids or not script_dialogue_counts:
        return {}
    if len(script_ids) != len(script_dialogue_counts):
        return {}
    result = {}
    offset = 0
    for script_id, raw_count in zip(script_ids, script_dialogue_counts):
        count = int(raw_count)
        result[str(script_id)] = dialogues[offset:offset + count]
        offset += count
    if offset != len(dialogues):
        return {}
    return result


def _merge_script_translations(script_ids, translations_by_script):
    merged = []
    for script_id in script_ids:
        merged.extend(translations_by_script.get(str(script_id), []))
    return merged


def _has_translation_errors(translations):
    return any("[Translation Error:" in str(item.get("translated_content", "")) for item in translations)
```

- [ ] **Step 4: Run helper tests and verify they pass**

Run:

```bash
python -m unittest test.test_translation_cache.TranslationRouteHelperTests -v
```

Expected: PASS.

- [ ] **Step 5: Update `/translate` to read/write cache for server LLM translations**

In `app.py`, replace the current LLM branch inside `translate()` with this structure:

```python
script_ids = [str(item) for item in data.get("script_ids", []) if str(item)]
script_dialogue_counts = data.get("script_dialogue_counts", [])
source_region = loader.normalize_region(data.get("source_region", "JP"))
script_dialogues = _split_dialogues_by_script_counts(dialogues, script_ids, script_dialogue_counts)
```

Inside `translation_method == "gpt"`, after resolving `api_type`, `api_base`, `api_key`, and `base_model`:

```python
provider = normalize_provider(api_type, base_model)
normalized_language = normalize_target_language(target_language)
prompt_version = translation_cache_config.prompt_version
translations_by_script = {}
miss_dialogues = []
miss_script_ids = []

if translation_cache_config.enabled and script_dialogues:
    for script_id in script_ids:
        server_source = loader.extract_dialogues(script_id, region=source_region)
        if len(server_source) != len(script_dialogues.get(script_id, [])):
            server_source = script_dialogues.get(script_id, [])
        source_hash = canonical_source_hash(server_source)
        key = TranslationCacheKey(
            script_id=script_id,
            source_region=source_region,
            source_hash=source_hash,
            target_language=normalized_language,
            provider=provider,
            model=base_model,
            prompt_version=prompt_version,
        )
        entry = translation_cache_client.read(key, expected_dialogue_count=len(server_source))
        if entry:
            translations_by_script[script_id] = [
                {"speaker": t.get("speaker", ""), "content": src.get("content", ""), "translated_content": t.get("translated_content", "")}
                for src, t in zip(server_source, entry.translations)
            ]
        else:
            miss_script_ids.append(script_id)
            miss_dialogues.extend(script_dialogues.get(script_id, []))

    if miss_dialogues:
        miss_translated = loader.gpt_dialogue_translate(...)
        offset = 0
        for script_id in miss_script_ids:
            source = script_dialogues.get(script_id, [])
            translated_slice = miss_translated[offset:offset + len(source)]
            offset += len(source)
            translations_by_script[script_id] = translated_slice
            if len(translated_slice) == len(source) and not _has_translation_errors(translated_slice):
                key = TranslationCacheKey(...)
                translation_cache_client.write(TranslationCacheEntry(key, len(source), translated_slice))
    translated = _merge_script_translations(script_ids, translations_by_script)
else:
    translated = loader.gpt_dialogue_translate(...)
```

Keep the existing no-cache path for requests that do not include script metadata.

- [ ] **Step 6: Update `templates/index.html` request metadata**

In the `/translate` request body, add:

```javascript
script_ids: scriptIds,
script_dialogue_counts: scriptDialogueCounts,
source_region: currentQuestRegion,
```

- [ ] **Step 7: Run route helper and cache utility tests**

Run:

```bash
python -m unittest test.test_translation_cache -v
```

Expected: PASS.

---

### Task 3: Static Branch Read-Only Cache

**Files:**
- Switch/worktree: `static-migration`
- Create: `js/translation-cache.js`
- Modify: `index.html`

- [ ] **Step 1: Work in a `static-migration` worktree or switch safely**

Use an isolated worktree outside the repository if the current branch has unrelated uncommitted changes. Do not overwrite `master` changes.

Run:

```bash
git worktree add "$env:USERPROFILE\.config\superpowers\worktrees\fgo_translator\static-cache" static-migration
```

Expected: a separate checkout for `static-migration`.

- [ ] **Step 2: Add read-only browser cache helper**

Create `js/translation-cache.js` with:

```javascript
const TranslationCache = (() => {
    const DEFAULT_PROMPT_VERSION = 'fgo-v1';

    function getBaseUrl() {
        try {
            const prefs = JSON.parse(localStorage.getItem('fgo_translator_prefs') || '{}');
            return (prefs.translationCacheBaseUrl || prefs.translation_cache_base_url || '').replace(/\/+$/, '');
        } catch {
            return '';
        }
    }

    function sanitizePathSegment(value) {
        const safe = String(value || '').trim().replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '');
        return safe || 'unknown';
    }

    function normalizeTargetLanguage(value) {
        const raw = String(value || '').trim();
        const lowered = raw.toLowerCase();
        const mapping = {
            'chinese': 'zh-CN',
            'chinese simplified': 'zh-CN',
            'chinese (simplified)': 'zh-CN',
            'zh-cn': 'zh-CN',
            '中文': 'zh-CN',
            '简体中文': 'zh-CN',
            'chinese traditional': 'zh-TW',
            'chinese (traditional)': 'zh-TW',
            'zh-tw': 'zh-TW',
            '繁體中文': 'zh-TW',
            '繁体中文': 'zh-TW',
            'english': 'en',
            'en': 'en',
        };
        return mapping[lowered] || sanitizePathSegment(raw);
    }

    function normalizeProvider(apiType, model) {
        const raw = String(apiType || 'openai').toLowerCase();
        const modelLower = String(model || '').toLowerCase();
        if (raw === 'gemini' || modelLower.includes('gemini')) return 'gemini';
        if (modelLower.includes('deepseek')) return 'deepseek';
        if (modelLower.includes('qwen') || modelLower.includes('qwq')) return 'qwen';
        if (modelLower.includes('claude')) return 'claude';
        if (raw === 'openai' || raw === 'custom') return raw;
        return sanitizePathSegment(raw);
    }

    async function sha256Hex(text) {
        const bytes = new TextEncoder().encode(text);
        const hash = await crypto.subtle.digest('SHA-256', bytes);
        return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
    }

    async function canonicalSourceHash(dialogues) {
        const payload = (dialogues || []).map(d => ({
            content: String(d.content || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
            speaker: String(d.speaker || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n'),
        }));
        const json = JSON.stringify(payload.map(item => ({content: item.content, speaker: item.speaker})));
        return sha256Hex(json);
    }

    function relativePath(key) {
        return [
            'v1',
            sanitizePathSegment(String(key.sourceRegion || 'JP').toUpperCase()),
            sanitizePathSegment(key.scriptId),
            sanitizePathSegment(key.sourceHash),
            sanitizePathSegment(key.targetLanguage),
            sanitizePathSegment(key.provider),
            sanitizePathSegment(key.model),
            `${sanitizePathSegment(key.promptVersion || DEFAULT_PROMPT_VERSION)}.json`,
        ].join('/');
    }

    function validate(data, key, expectedDialogueCount) {
        if (!data || data.schema_version !== 1 || data.trusted_generation !== true) return null;
        if (String(data.script_id) !== String(key.scriptId)) return null;
        if (String(data.source_region) !== String(key.sourceRegion)) return null;
        if (String(data.source_hash) !== String(key.sourceHash)) return null;
        if (String(data.target_language) !== String(key.targetLanguage)) return null;
        if (String(data.provider) !== String(key.provider)) return null;
        if (String(data.model) !== String(key.model)) return null;
        if (String(data.prompt_version) !== String(key.promptVersion || DEFAULT_PROMPT_VERSION)) return null;
        if (!Array.isArray(data.translations)) return null;
        if (Number(data.dialogue_count) !== expectedDialogueCount || data.translations.length !== expectedDialogueCount) return null;
        if (data.translations.some(t => String(t.translated_content || '').includes('[Translation Error:'))) return null;
        return data.translations;
    }

    async function readScript({scriptId, sourceRegion, dialogues, targetLanguage, apiType, model, promptVersion = DEFAULT_PROMPT_VERSION}) {
        const baseUrl = getBaseUrl();
        if (!baseUrl) return null;
        const sourceHash = await canonicalSourceHash(dialogues);
        const key = {
            scriptId: String(scriptId),
            sourceRegion: String(sourceRegion || 'JP').toUpperCase(),
            sourceHash,
            targetLanguage: normalizeTargetLanguage(targetLanguage),
            provider: normalizeProvider(apiType, model),
            model: model || 'unknown',
            promptVersion,
        };
        const url = `${baseUrl}/${relativePath(key)}`;
        try {
            const response = await fetch(url, {cache: 'force-cache'});
            if (!response.ok) return null;
            return validate(await response.json(), key, dialogues.length);
        } catch {
            return null;
        }
    }

    return {readScript, canonicalSourceHash, normalizeProvider, normalizeTargetLanguage};
})();
```

- [ ] **Step 3: Load the helper in `index.html`**

Add before `js/translate.js`:

```html
<script src="js/translation-cache.js"></script>
```

- [ ] **Step 4: Check static cache before browser translation**

Inside `_apiTranslate({ dialogues, target_language, translation_method })`, keep the existing browser translation fallback, but the per-script cache lookup must be called from `translateSelectedQuest()` before `_apiTranslate()` because only that scope has `scriptIds` and `scriptDialogueCounts`.

Add a helper in `index.html`:

```javascript
async function tryReadCachedScripts(scriptIds, scriptDialogueCounts, allDialogues, targetLanguage) {
    if (!window.TranslationCache && typeof TranslationCache === 'undefined') return null;
    const prefs = Translator.loadPrefs();
    const apiType = prefs.apiType || prefs.api_type || 'gemini';
    const model = prefs.baseModel || prefs.base_model || '';
    const merged = [];
    let offset = 0;
    for (let i = 0; i < scriptIds.length; i++) {
        const count = scriptDialogueCounts[i];
        const source = allDialogues.slice(offset, offset + count);
        offset += count;
        const cached = await TranslationCache.readScript({
            scriptId: scriptIds[i],
            sourceRegion: currentQuestRegion || 'JP',
            dialogues: source,
            targetLanguage,
            apiType,
            model,
        });
        if (!cached) return null;
        merged.push(...cached.map((t, idx) => ({
            speaker: t.speaker || source[idx]?.speaker || '',
            content: source[idx]?.content || '',
            translated_content: t.translated_content || '',
        })));
    }
    return {translated_dialogues: merged, cache_hit: true};
}
```

Then in `translateSelectedQuest()`, before `_apiTranslate(...)`, add:

```javascript
translateData = await tryReadCachedScripts(scriptIds, scriptDialogueCounts, allDialogues, targetLangValue);
if (!translateData) {
    translateData = await _apiTranslate({
        dialogues: allDialogues,
        translation_method: document.getElementById('translationMethod').value,
        target_language: targetLangValue,
    });
}
```

- [ ] **Step 5: Manually verify static syntax**

Run in the static worktree:

```bash
node --check js/translation-cache.js
```

Expected: no syntax errors.

---

### Task 4: Final Verification And Commits

**Files:**
- All changed files from Tasks 1-3.

- [ ] **Step 1: Run Python tests on `master`**

Run:

```bash
python -m unittest test.test_translation_cache -v
```

Expected: PASS.

- [ ] **Step 2: Run a Flask import smoke test**

Run:

```bash
python -c "import app; print('app import ok')"
```

Expected: prints `app import ok`.

- [ ] **Step 3: Check static JavaScript syntax**

Run:

```bash
node --check js/translation-cache.js
```

Expected: no syntax errors in the `static-migration` worktree.

- [ ] **Step 4: Review diffs**

Run:

```bash
git diff --stat
git diff --cached --stat
```

Expected: only cache-related files changed, plus any pre-existing user change remains unstaged.

- [ ] **Step 5: Commit master changes**

Run:

```bash
git add translation_cache.py test/test_translation_cache.py app.py templates/index.html
git commit -m "Add GitHub-hosted translation cache"
```

Expected: commit succeeds.

- [ ] **Step 6: Commit static-migration changes**

Run in the static worktree:

```bash
git add js/translation-cache.js index.html
git commit -m "Add read-only GitHub translation cache"
```

Expected: commit succeeds on `static-migration`.
