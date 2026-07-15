import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from dialogue_loader import DialogueLoader
from scripts.prewarm_story_cache import prewarm_story_cache


class StoryScriptCacheTests(unittest.TestCase):
    def test_downloads_raw_text_then_reuses_cache_without_network(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DialogueLoader(cache_dir=Path(temp_dir))
            loader.db_loader._make_request_with_retry = Mock(
                return_value={"script": "https://static.atlasacademy.io/script.txt"}
            )
            loader._get_text_content = Mock(return_value="raw Atlas script\n[token is game text]")

            first = loader.load_script_text("0300080110", region="jp")

            cache_file = (
                Path(temp_dir) / "story_scripts" / "JP" / "0300080110.txt"
            )
            self.assertEqual(first, "raw Atlas script\n[token is game text]")
            self.assertEqual(cache_file.read_text(encoding="utf-8"), first)

            loader.db_loader._make_request_with_retry.reset_mock()
            loader._get_text_content.reset_mock()

            second = loader.load_script_text("0300080110", region="JP")

            self.assertEqual(second, first)
            loader.db_loader._make_request_with_retry.assert_not_called()
            loader._get_text_content.assert_not_called()

    def test_refresh_network_failure_falls_back_to_cached_raw_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DialogueLoader(cache_dir=Path(temp_dir))
            cache_file = (
                Path(temp_dir) / "story_scripts" / "TW" / "0300080120.txt"
            )
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("cached Atlas text", encoding="utf-8")
            loader.db_loader._make_request_with_retry = Mock(
                side_effect=RuntimeError("network unavailable")
            )

            text = loader.load_script_text("0300080120", region="TW", refresh=True)

            self.assertEqual(text, "cached Atlas text")
            self.assertEqual(cache_file.read_text(encoding="utf-8"), "cached Atlas text")

    def test_rejects_script_ids_that_could_escape_cache_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DialogueLoader(cache_dir=Path(temp_dir))

            with self.assertRaises(ValueError):
                loader.load_script_text("../secret", region="JP")


class PrewarmStoryCacheTests(unittest.TestCase):
    def test_prewarms_region_script_matrix_and_reports_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DialogueLoader(cache_dir=Path(temp_dir))

            def fake_quest_request(url):
                if "/nice/JP/quest/3000801/1" in url:
                    return {
                        "scripts": [
                            {"scriptId": "0300080110"},
                            {"scriptId": "0300080120"},
                        ]
                    }
                if "/nice/CN/quest/3000801/1" in url:
                    return {"scripts": [{"scriptId": "0300080110"}]}
                raise AssertionError(f"Unexpected Atlas request: {url}")

            loader.db_loader._make_request_with_retry = Mock(side_effect=fake_quest_request)
            loader.load_script_text = Mock(return_value="raw text")
            loader.gpt_dialogue_translate = Mock(
                side_effect=AssertionError("translation API must not be called")
            )
            output = []

            results = prewarm_story_cache(
                loader,
                quests=["3000801"],
                regions=["JP", "CN"],
                phases=[1],
                output=output.append,
            )

            self.assertEqual(len(results), 3)
            self.assertEqual(sum(result.success for result in results), 3)
            self.assertEqual(sum(not result.success for result in results), 0)
            self.assertIn("JP: success=2 missing=0", output)
            self.assertIn("CN: success=1 missing=0", output)
            self.assertIn("TOTAL: success=3 missing=0", output)
            loader.gpt_dialogue_translate.assert_not_called()
            self.assertEqual(
                loader.load_script_text.call_args_list,
                [
                    call("0300080110", region="JP", refresh=False),
                    call("0300080120", region="JP", refresh=False),
                    call("0300080110", region="CN", refresh=False),
                ],
            )


if __name__ == "__main__":
    unittest.main()
