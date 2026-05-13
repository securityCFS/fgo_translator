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


def _normalize_line_endings(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def canonical_source_payload(dialogues: List[Dict]) -> List[Dict[str, str]]:
    return [
        {
            "speaker": _normalize_line_endings(item.get("speaker", "")),
            "content": _normalize_line_endings(item.get("content", "")),
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
    return sanitize_path_segment(raw)


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
            sanitize_path_segment(str(self.source_region).upper()),
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
    def from_json(
        cls,
        data: Dict,
        key: TranslationCacheKey,
        expected_dialogue_count: int,
    ) -> Optional["TranslationCacheEntry"]:
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
            if existing.status_code != 404:
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
            logger.warning(
                "Translation cache write failed for %s: HTTP %s %s",
                path,
                response.status_code,
                response.text[:200],
            )
            return False
        except Exception as exc:
            logger.warning("Translation cache write failed for %s: %s", path, exc)
            return False
