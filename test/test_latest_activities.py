import unittest

from dialogue_loader import DialogueLoader


class LatestActivitiesTests(unittest.TestCase):
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
