import unittest
from unittest.mock import Mock

from translation_cache import (
    TranslationCacheConfig,
    TranslationCacheEntry,
    TranslationCacheKey,
    TranslationCacheClient,
    TranslationCacheOption,
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

    def test_list_options_discovers_github_cache_versions(self):
        payload = {
            "schema_version": 1,
            "script_id": "1",
            "source_region": "JP",
            "source_hash": "hash",
            "target_language": "zh-CN",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "prompt_version": "fgo-v1",
            "dialogue_count": 1,
            "trusted_generation": True,
            "translations": [{"speaker": "A", "translated_content": "hello"}],
        }

        def fake_get(url, **kwargs):
            if url.endswith("/contents/v1/JP/1/hash/zh-CN"):
                return Mock(status_code=200, json=Mock(return_value=[
                    {"type": "dir", "name": "deepseek"},
                ]))
            if url.endswith("/contents/v1/JP/1/hash/zh-CN/deepseek"):
                return Mock(status_code=200, json=Mock(return_value=[
                    {"type": "dir", "name": "deepseek-v4-flash"},
                ]))
            if url.endswith("/contents/v1/JP/1/hash/zh-CN/deepseek/deepseek-v4-flash"):
                return Mock(status_code=200, json=Mock(return_value=[
                    {"type": "file", "name": "fgo-v1.json", "download_url": "https://raw.example/fgo-v1.json"},
                ]))
            if url == "https://raw.example/fgo-v1.json":
                return Mock(status_code=200, json=Mock(return_value=payload))
            return Mock(status_code=404, json=Mock(return_value={}))

        session = Mock()
        session.get.side_effect = fake_get
        client = TranslationCacheClient(
            TranslationCacheConfig(
                base_url="https://example.github.io/cache",
                repo="owner/repo",
                branch="main",
            ),
            session=session,
        )

        options = client.list_options(
            script_id="1",
            source_region="JP",
            source_hash="hash",
            target_language="Chinese (Simplified)",
            expected_dialogue_count=1,
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].provider, "deepseek")
        self.assertEqual(options[0].model, "deepseek-v4-flash")
        self.assertEqual(options[0].prompt_version, "fgo-v1")
        self.assertEqual(options[0].label, "deepseek / deepseek-v4-flash / fgo-v1")


from app import _common_cache_options_for_scripts, _merge_script_translations, _split_dialogues_by_script_counts


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

    def test_common_cache_options_require_every_script(self):
        common = TranslationCacheOption(
            provider="deepseek",
            model="deepseek-v4-flash",
            prompt_version="fgo-v1",
            label="deepseek / deepseek-v4-flash / fgo-v1",
            dialogue_count=1,
        )
        only_first = TranslationCacheOption(
            provider="gemini",
            model="gemini-2.5-flash",
            prompt_version="fgo-v1",
            label="gemini / gemini-2.5-flash / fgo-v1",
            dialogue_count=1,
        )

        class FakeCacheClient:
            def list_options(self, script_id, source_region, source_hash, target_language, expected_dialogue_count):
                if script_id == "s1":
                    return [common, only_first]
                return [common]

        options = _common_cache_options_for_scripts(
            {
                "s1": [{"speaker": "A", "content": "one"}],
                "s2": [{"speaker": "B", "content": "two"}],
            },
            source_region="JP",
            target_language="Chinese (Simplified)",
            cache_client=FakeCacheClient(),
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["provider"], "deepseek")
        self.assertEqual(options[0]["script_count"], 2)
        self.assertEqual(options[0]["dialogue_count"], 2)


if __name__ == "__main__":
    unittest.main()
