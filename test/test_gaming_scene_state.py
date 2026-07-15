import unittest
from unittest.mock import patch

import app as app_module
from app import _parse_fgo_script
from dialogue_loader import DialogueLoader


class GamingSceneStateTests(unittest.TestCase):
    def test_parser_preserves_slot_position_scale_depth_and_multiple_talkers(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaSet B 1002001 1 Partner]
[charaScale A 1.25]
[charaDepth A 7]
[charaFadein A 0.1 -256,80]
[charaFadein B 0.1 2]
[charaTalk A,B]
＠A：Hero
First line
[k]
[charaMove A 1 0.4]
[charaTalk A]
＠A：Hero
Second line
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue_frames = [frame for frame in frames if frame.get("type") == "dialogue"]

        self.assertEqual(len(dialogue_frames), 2)
        first_by_slot = {sprite["slot"]: sprite for sprite in dialogue_frames[0]["sprites"]}
        self.assertEqual(first_by_slot["A"]["x"], -256)
        self.assertEqual(first_by_slot["A"]["y"], 80)
        self.assertEqual(first_by_slot["A"]["scale"], 1.25)
        self.assertEqual(first_by_slot["A"]["depth"], 7)
        self.assertTrue(first_by_slot["A"]["talking"])
        self.assertTrue(first_by_slot["B"]["talking"])
        self.assertEqual(first_by_slot["B"]["x"], 256)

        second_by_slot = {sprite["slot"]: sprite for sprite in dialogue_frames[1]["sprites"]}
        self.assertEqual(second_by_slot["A"]["x"], 0)
        self.assertTrue(second_by_slot["A"]["talking"])
        self.assertFalse(second_by_slot["B"]["talking"])

    def test_parser_hides_zero_alpha_chara_put(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaPut A -120,40]
[charaFadeTime A 0 0]
＠
Narration
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        self.assertEqual(dialogue["sprites"], [])

    def test_parser_preserves_ensemble_brightness_and_visual_updates(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaSet B 1002001 1 Partner]
[charaFadein A 0.1 0]
[charaFadein B 0.1 2]
[charaTalk off]
[charaMoveScale A 1.2 0.4]
[charaFaceFade B 3 0.2]
[charaCrossFade A 1003001 4 0.4]
＠Narrator
Ensemble scene
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        by_slot = {sprite["slot"]: sprite for sprite in dialogue["sprites"]}

        self.assertTrue(by_slot["A"]["talking"])
        self.assertTrue(by_slot["B"]["talking"])
        self.assertEqual(by_slot["A"]["scale"], 1.2)
        self.assertEqual(by_slot["A"]["entityId"], "1003001")
        self.assertEqual(by_slot["A"]["face"], 4)
        self.assertEqual(by_slot["B"]["face"], 3)

    def test_parser_composes_sub_render_position_scale_and_visibility(self):
        raw = """
[charaSet E 1049000 1 Muramasa]
[charaLayer E sub #A]
[charaFadeinFSR E 0 0,250]
[subRenderScale #A 0.8]
[subRenderDepth #A 6]
[subRenderFadeinFSR #A 0.3 400,-280]
＠E：Muramasa
First
[k]
[subRenderMoveEaseFSR #A 350,-280 0.4 easeOutSine]
＠E：Muramasa
Second
[k]
[subRenderFadeout #A 0.2]
＠Narrator
Hidden
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogues = [frame for frame in frames if frame.get("type") == "dialogue"]

        first_sprite = dialogues[0]["sprites"][0]
        self.assertEqual(first_sprite["x"], 400)
        self.assertEqual(first_sprite["y"], -80)
        self.assertEqual(first_sprite["scale"], 0.8)
        self.assertEqual(first_sprite["depth"], 6)
        self.assertEqual(dialogues[1]["sprites"][0]["x"], 350)
        self.assertEqual(dialogues[1]["sprites"][0]["y"], -80)
        self.assertEqual(dialogues[2]["sprites"], [])


class RegionalDialogueSyntaxTests(unittest.TestCase):
    def test_korean_ascii_choice_markers_are_counted_in_order(self):
        loader = DialogueLoader()
        raw = """
＠마슈
첫 번째 대사
[k]
?1:첫 번째 선택
?2:두 번째 선택
?!
＠마슈
마지막 대사
[k]
""".strip() + "\n"

        dialogues = loader._parse_script_dialogues(raw)

        self.assertEqual(
            [item["content"] for item in dialogues],
            [
                "첫 번째 대사",
                "Choice 1: 첫 번째 선택",
                "Choice 2: 두 번째 선택",
                "Choice 2 Ending",
                "마지막 대사",
            ],
        )


class GamingVisualRouteTests(unittest.TestCase):
    def test_visual_route_uses_cached_script_loader(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaFadein A 0.1 1]
＠Hero
Cached scene
[k]
""".strip()

        with (
            patch.object(app_module.loader, "load_script_text", return_value=raw) as load_script,
            patch.object(app_module, "_fetch_svt_scripts_parallel", return_value={}),
        ):
            response = app_module.app.test_client().post(
                "/parse_script_visual",
                json={"script_id": "0500010010", "region": "JP"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["frames"][0]["sprites"][0]["entityId"], "1001001")
        load_script.assert_called_once_with("0500010010", region="JP")


if __name__ == "__main__":
    unittest.main()
