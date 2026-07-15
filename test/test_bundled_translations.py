import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from scripts.build_bundled_translations import (
    _validate_agent_script,
    build_agent_bundles,
)
from translation_cache import TranslationCacheEntry, TranslationCacheKey, canonical_source_hash


class BundledTranslationRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = [{"speaker": "Mash", "content": "Hello [line 3]"}]
        key = TranslationCacheKey(
            script_id="9415800010",
            source_region="JP",
            source_hash=canonical_source_hash(self.source),
            target_language="zh-CN",
            provider="codex-agent",
            model="agent-translation",
            prompt_version="fgo-agent-v1",
        )
        payload = TranslationCacheEntry(
            key=key,
            dialogue_count=1,
            translations=[{"speaker": "玛修", "translated_content": "你好 [line 3]"}],
        ).to_json()
        path = self.root / "zh-CN" / "9415800010.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_check_requires_every_script_in_phase(self):
        with patch.object(app_module, "BUNDLED_TRANSLATION_ROOT", self.root):
            response = app_module.app.test_client().post(
                "/check_bundled_translations",
                json={
                    "script_ids": ["9415800010", "9415800011"],
                    "target_language": "zh-CN",
                },
            )

        data = response.get_json()
        self.assertFalse(data["available"])
        self.assertEqual(data["missing"], ["9415800011"])

    def test_get_validates_source_and_returns_agent_translation(self):
        with (
            patch.object(app_module, "BUNDLED_TRANSLATION_ROOT", self.root),
            patch.object(app_module.loader, "extract_dialogues", return_value=self.source),
        ):
            response = app_module.app.test_client().post(
                "/get_bundled_dialogues",
                json={"script_ids": ["9415800010"], "target_language": "zh-CN"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["bundled"])
        self.assertEqual(data["providers"], ["codex-agent"])
        self.assertEqual(data["translated_dialogues"][0]["translated_content"], "你好 [line 3]")

    def test_get_rejects_source_hash_drift(self):
        with (
            patch.object(app_module, "BUNDLED_TRANSLATION_ROOT", self.root),
            patch.object(
                app_module.loader,
                "extract_dialogues",
                return_value=[{"speaker": "Mash", "content": "Changed"}],
            ),
        ):
            response = app_module.app.test_client().post(
                "/get_bundled_dialogues",
                json={"script_ids": ["9415800010"], "target_language": "zh-CN"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("source hash mismatch", response.get_json()["error"].lower())


class BundledTranslationBuildTests(unittest.TestCase):
    def test_agent_build_normalizes_runtime_script_tokens(self):
        source = {
            "batch_id": "batch-test",
            "scripts": [
                {
                    "script_id": "1",
                    "source_hash": "hash",
                    "dialogues": [
                        {"speaker": "A", "content": "Text [line 3] [%1]"}
                    ],
                }
            ],
        }
        translated = {
            "batch_id": "batch-test",
            "scripts": [
                {
                    "script_id": "1",
                    "source_hash": "hash",
                    "translations": [
                        {
                            "speaker": "A",
                            "translated_content": "Words [line 3] [%1]",
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            translated_dir = root / "translated"
            output_dir = root / "output"
            source_dir.mkdir()
            translated_dir.mkdir()
            (source_dir / "batch-test.json").write_text(
                json.dumps(source), encoding="utf-8"
            )
            (translated_dir / "batch-test.json").write_text(
                json.dumps(translated), encoding="utf-8"
            )

            build_agent_bundles(source_dir, translated_dir, output_dir)

            payload = json.loads((output_dir / "1.json").read_text(encoding="utf-8"))
            runtime_source = [
                {
                    "speaker": "A",
                    "content": "Text \u2014\u2014 \u85e4\u4e38\u7acb\u9999",
                }
            ]
            self.assertEqual(payload["source_hash"], canonical_source_hash(runtime_source))
            self.assertEqual(
                payload["translations"][0]["translated_content"],
                "Words \u2014\u2014 \u85e4\u4e38\u7acb\u9999",
            )

    def test_agent_validation_preserves_formatting_tags(self):
        source = {
            "script_id": "1",
            "source_hash": "hash",
            "dialogues": [{"speaker": "A", "content": "Text [line 3] [%1]"}],
        }
        translated = {
            "script_id": "1",
            "source_hash": "hash",
            "translations": [{"speaker": "甲", "translated_content": "文字 [line 3] [%1]"}],
        }

        _validate_agent_script(source, translated)

        translated["translations"][0]["translated_content"] = "文字 [%1]"
        with self.assertRaisesRegex(ValueError, "Formatting tag mismatch"):
            _validate_agent_script(source, translated)


if __name__ == "__main__":
    unittest.main()
