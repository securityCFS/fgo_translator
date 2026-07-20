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

    def test_parser_preserves_partial_opacity_and_silhouette_filter(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaPut A 1]
[charaFadeTime A 0.4 0.6]
[charaFilter A silhouette 00000080]
＠Hero
Shadow
[k]
[charaFilter A normal]
[charaFadeTime A 0.2 1]
＠Hero
Normal
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogues = [frame for frame in frames if frame.get("type") == "dialogue"]

        shadow = dialogues[0]["sprites"][0]
        self.assertEqual(shadow["opacity"], 0.6)
        self.assertEqual(shadow["filter"], "silhouette")
        self.assertEqual(shadow["filterColor"], "#000000")
        self.assertAlmostEqual(shadow["filterAlpha"], 128 / 255)

        normal = dialogues[1]["sprites"][0]
        self.assertEqual(normal["opacity"], 1.0)
        self.assertEqual(normal["filter"], "normal")
        self.assertEqual(normal["filterAlpha"], 1.0)

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

        frames, entity_ids = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        by_slot = {sprite["slot"]: sprite for sprite in dialogue["sprites"]}

        self.assertTrue(by_slot["A"]["talking"])
        self.assertTrue(by_slot["B"]["talking"])
        self.assertEqual(by_slot["A"]["scale"], 1.2)
        self.assertEqual(by_slot["A"]["entityId"], "1003001")
        self.assertIn("/1003001/1003001_merged.png", by_slot["A"]["url"])
        self.assertIn("1003001", entity_ids)
        self.assertEqual(by_slot["A"]["face"], 4)
        self.assertEqual(by_slot["B"]["face"], 3)

    def test_parser_preserves_sub_render_group_and_local_sprite_coordinates(self):
        raw = """
[charaSet E 1049000 1 Muramasa]
[charaLayer E sub #A]
[charaFadeinFSR E 0 0,250]
[subCameraFilter #A maskEdge cut359_mask16 4 255,255,255,255 0]
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
        self.assertEqual(first_sprite["x"], 0)
        self.assertEqual(first_sprite["y"], 250)
        self.assertEqual(first_sprite["scale"], 1)
        self.assertEqual(first_sprite["subRender"], "#A")
        self.assertEqual(
            dialogues[0]["subRenders"]["#A"],
            {
                "visible": True,
                "x": 400,
                "y": -280,
                "scale": 0.8,
                "depth": 6,
                "mask": "cut359_mask16",
            },
        )
        self.assertEqual(dialogues[1]["sprites"][0]["x"], 0)
        self.assertEqual(dialogues[1]["sprites"][0]["y"], 250)
        self.assertEqual(dialogues[1]["subRenders"]["#A"]["x"], 350)
        self.assertEqual(dialogues[2]["sprites"], [])

    def test_parser_emits_visual_wait_frame_before_dialogue(self):
        raw = """
[scene 245000]
[charaSet A 1001001 1 Hero]
[charaFadein A 0.2 1]
[wt 0.6]
＠Hero
Hello
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")

        self.assertEqual(frames[0]["type"], "stage")
        self.assertEqual(frames[0]["duration"], 0.6)
        self.assertEqual(frames[0]["sprites"][0]["slot"], "A")
        self.assertEqual(frames[1]["type"], "dialogue")

    def test_parser_prefers_matching_speaker_over_stale_explicit_talker(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaSet B 1002001 1 Partner]
[charaFadein A 0.1 0]
[charaFadein B 0.1 2]
[charaTalk A]
＠Partner
Take over the line
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        by_slot = {sprite["slot"]: sprite for sprite in dialogue["sprites"]}

        self.assertFalse(by_slot["A"]["talking"])
        self.assertTrue(by_slot["B"]["talking"])

    def test_parser_accepts_scene_transition_arguments(self):
        raw = """
[scene 292601 0.7]
[fadeout black 0.5]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        transition = next(frame for frame in frames if frame.get("type") == "transition")

        self.assertEqual(
            transition["bg"],
            "https://static.atlasacademy.io/JP/Back/back292601.png",
        )

    def test_parser_uses_widescreen_assets_and_merged_figures(self):
        raw = """
[enableFullScreen]
[scene 105500]
[charaSet A 1098360800 1 グレイ]
[charaFadein A 0.1 1]
＠グレイ
Hello
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")

        self.assertTrue(dialogue["bg"].endswith("back105500_1344_626.png"))
        self.assertTrue(dialogue["sprites"][0]["url"].endswith("1098360800_merged.png"))

    def test_parser_keeps_scene_and_image_sets_as_stage_layers(self):
        raw = """
[enableFullScreen]
[sceneSet A 292600 1]
[imageSet B back292604 1]
[scene 245000]
[charaFadein A 0.1 1]
[charaFadein B 0.1 2]
＠Narrator
Layered scene
[k]
""".strip()

        frames, entity_ids = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        by_slot = {sprite["slot"]: sprite for sprite in dialogue["sprites"]}

        self.assertTrue(dialogue["bg"].endswith("back245000_1344_626.png"))
        self.assertEqual(by_slot["A"]["assetType"], "scene")
        self.assertTrue(by_slot["A"]["url"].endswith("back292600_1344_626.png"))
        self.assertEqual(by_slot["B"]["assetType"], "image")
        self.assertTrue(by_slot["B"]["url"].endswith("/Image/back292604/back292604.png"))
        self.assertTrue(by_slot["A"]["talking"])
        self.assertTrue(by_slot["B"]["talking"])
        self.assertEqual(entity_ids, [])

    def test_parser_infers_speaker_after_talk_mode_is_reenabled(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaSet B 1002001 1 Partner]
[charaFadein A 0.1 0]
[charaFadein B 0.1 2]
[charaTalk off]
[charaTalk on]
＠Partner
Focused line
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogue = next(frame for frame in frames if frame.get("type") == "dialogue")
        by_slot = {sprite["slot"]: sprite for sprite in dialogue["sprites"]}

        self.assertFalse(by_slot["A"]["talking"])
        self.assertTrue(by_slot["B"]["talking"])

    def test_parser_switches_preloaded_variants_without_double_rendering(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaSet B 1001001 1 Hero_演出用]
[charaFadein B 0.1 1]
[charaFace B 3]
＠Hero
Cinematic
[k]
[charaFadein A 0.1 0]
[charaMove A -200,0 0.2]
[charaTalk A]
＠Hero
Normal
[k]
[charaFadeout A 0.1]
[charaTalk off]
＠Hero
Cinematic again
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogues = [frame for frame in frames if frame.get("type") == "dialogue"]

        self.assertEqual([sprite["slot"] for sprite in dialogues[0]["sprites"]], ["B"])
        self.assertEqual([sprite["slot"] for sprite in dialogues[1]["sprites"]], ["A"])
        self.assertEqual([sprite["slot"] for sprite in dialogues[2]["sprites"]], ["B"])

    def test_chara_put_does_not_reveal_a_preloaded_sprite(self):
        raw = """
[charaSet A 1001001 1 Hero]
[charaPut A 1]
＠Narrator
Still hidden
[k]
[charaFadeTime A 0.2 0.6]
＠Hero
Now visible
[k]
""".strip()

        frames, _ = _parse_fgo_script(raw, "JP")
        dialogues = [frame for frame in frames if frame.get("type") == "dialogue"]

        self.assertEqual(dialogues[0]["sprites"], [])
        self.assertEqual(dialogues[1]["sprites"][0]["opacity"], 0.6)


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
