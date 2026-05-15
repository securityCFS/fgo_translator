from pathlib import Path
import unittest


TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "index.html"


class DownloadButtonTemplateTests(unittest.TestCase):
    def test_download_button_passes_button_for_visible_feedback(self):
        html = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("onclick=\"downloadScript('${scriptId}', this)\"", html)
        self.assertIn("function setDownloadButtonState(button, label)", html)
        self.assertIn("setDownloadButtonState(button, 'Exported')", html)
        self.assertIn("setDownloadButtonState(button, 'No rows')", html)

    def test_shared_cache_selector_can_use_cached_translation(self):
        html = TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('id="translationCacheSelect"', html)
        self.assertIn("loadTranslationCacheOptions()", html)
        self.assertIn("fetch('/translate_cached'", html)


if __name__ == "__main__":
    unittest.main()
