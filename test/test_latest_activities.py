import unittest

from dialogue_loader import DialogueLoader


class LatestActivitiesTests(unittest.TestCase):
    def test_latest_tasks_filter_scripts_that_parse_to_no_dialogue(self):
        loader = DialogueLoader()
        inspected_scripts = []

        def fake_request(url, max_retries=None):
            if url.endswith("/basic/JP/quest/phase/latestEnemyData"):
                return [
                    {"id": 1, "warId": 9, "phase": 1, "openedAt": 200},
                    {"id": 2, "warId": 9, "phase": 1, "openedAt": 100},
                ]
            if url.endswith("/nice/JP/war/9"):
                return {
                    "id": 9,
                    "name": "test war",
                    "maps": [],
                    "spots": [{
                        "id": 90,
                        "quests": [
                            {
                                "id": 1,
                                "name": "battle only",
                                "phaseScripts": [{"phase": 1, "scripts": [{"scriptId": "battle"}]}],
                            },
                            {
                                "id": 2,
                                "name": "story",
                                "phaseScripts": [{"phase": 1, "scripts": [{"scriptId": "story"}]}],
                            },
                        ],
                    }],
                }
            self.fail(f"unexpected URL {url}")

        def fake_extract(script_id, region="JP"):
            inspected_scripts.append(script_id)
            if script_id == "story":
                return [{"speaker": "A", "content": "dialogue"}]
            return []

        loader.db_loader._make_request_with_retry = fake_request
        loader.extract_dialogues = fake_extract

        rows = loader.list_latest_tasks(region="JP", limit=1)

        self.assertEqual(["2"], [row["id"] for row in rows])
        self.assertEqual(["battle", "story"], inspected_scripts)
        self.assertEqual(1, rows[0]["hiddenNoScriptCount"])

    def test_latest_wars_use_lightweight_export_and_enrich_top_rows(self):
        loader = DialogueLoader()
        calls = []

        def fake_request(url, max_retries=None):
            calls.append(url)
            if url.endswith("/export/JP/nice_war.json"):
                self.fail("latest war listing should not require the huge nice_war export")
            if url.endswith("/export/JP/basic_war.json"):
                return [
                    {"id": 1, "name": "old war", "longName": "old war long", "eventId": 10, "eventName": "old event"},
                    {"id": 2, "name": "new war", "longName": "new war long", "eventId": 20, "eventName": "new event"},
                ]
            if url.endswith("/export/JP/basic_event.json"):
                return [
                    {"id": 10, "startedAt": 1000, "endedAt": 1500},
                    {"id": 20, "startedAt": 2000, "endedAt": 2500},
                ]
            if url.endswith("/nice/JP/war/2"):
                return {"id": 2, "name": "new war detail", "longName": "new war long", "eventId": 20, "banner": "banner2"}
            if url.endswith("/nice/JP/war/1"):
                return {"id": 1, "name": "old war detail", "longName": "old war long", "eventId": 10, "banner": "banner1"}
            self.fail(f"unexpected URL {url}")

        loader.db_loader._make_request_with_retry = fake_request

        rows = loader.list_latest_activities(region="JP", activity_type="war", limit=2)

        self.assertEqual(["2", "1"], [row["id"] for row in rows])
        self.assertEqual("banner2", rows[0]["banner"])
        self.assertFalse(any(url.endswith("/export/JP/nice_war.json") for url in calls))

    def test_latest_wars_can_use_event_banner_when_war_detail_fails(self):
        loader = DialogueLoader()

        def fake_request(url, max_retries=None):
            if url.endswith("/export/JP/basic_war.json"):
                return [{"id": 2, "name": "war", "longName": "war long", "eventId": 20, "eventName": "event"}]
            if url.endswith("/export/JP/basic_event.json"):
                return [{"id": 20, "startedAt": 2000, "endedAt": 2500}]
            if url.endswith("/nice/JP/war/2"):
                raise RuntimeError("war detail currently broken")
            if url.endswith("/nice/JP/event/20"):
                return {"id": 20, "name": "event detail", "banner": "event-banner", "noticeBanner": "notice-banner"}
            self.fail(f"unexpected URL {url}")

        loader.db_loader._make_request_with_retry = fake_request

        rows = loader.list_latest_activities(region="JP", activity_type="war", limit=1)

        self.assertEqual("event-banner", rows[0]["banner"])


if __name__ == "__main__":
    unittest.main()
