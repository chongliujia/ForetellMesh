import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from foretellmesh.market_dataset import (build_market_dataset, collect_markets,
                                        observed_record, poll_market_labels, replay_capture)
from foretellmesh.schema import ValidationError, parse_record


T = "2026-09-18T08:00:00Z"
LATER = "2026-10-17T08:00:00Z"
ROOT = Path(__file__).resolve().parents[1]


def market(ticker="KXCPIYOY-26SEP-T3"):
    return {"ticker": ticker, "event_ticker": "KXCPIYOY-26SEP", "market_type": "binary",
            "title": "September CPI above 3%?", "yes_sub_title": "Above 3%", "no_sub_title": "3% or below",
            "rules_primary": "Resolves Yes if the official September figure exceeds 3%.", "rules_secondary": "",
            "status": "active", "result": "", "open_time": "2026-09-01T00:00:00Z",
            "created_time": "2026-08-31T00:00:00Z", "updated_time": T,
            "close_time": "2026-10-15T12:29:00Z", "yes_bid_dollars": "0.4500",
            "yes_ask_dollars": "0.5500", "notional_value_dollars": "1.0000", "settlement_ts": None}


EVENT = {"event_ticker": "KXCPIYOY-26SEP", "series_ticker": "KXCPIYOY",
         "title": "September CPI", "category": "Economics"}


class MarketDatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        config = json.loads((ROOT / "configs/kalshi_collection_v1.json").read_text())
        config["series_tickers"] = ["KXCPIYOY"]
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        self.market = market()
        self.capture = self.root / "capture"

    def response(self, url):
        parsed = urlparse(url)
        if parsed.path.endswith("/series/KXCPIYOY"):
            obj = {"series": {"ticker": "KXCPIYOY"}}
        elif parsed.path.endswith("/events/KXCPIYOY-26SEP"):
            obj = {"event": EVENT}
        elif parsed.path.endswith("/markets"):
            self.assertEqual(parse_qs(parsed.query)["status"], ["open"])
            obj = {"markets": [self.market], "cursor": ""}
        else:
            obj = {"market": self.market}
        return json.dumps(obj).encode(), None

    def collect(self):
        with patch("foretellmesh.market_dataset.public_get", side_effect=self.response), patch("foretellmesh.market_dataset.now", return_value=T):
            return collect_markets(self.config, self.capture)

    def poll(self, name="poll"):
        path = self.root / name
        with patch("foretellmesh.market_dataset.public_get", side_effect=self.response), patch("foretellmesh.market_dataset.now", return_value=LATER):
            poll_market_labels(self.capture, path)
        return path

    def settle(self):
        self.market.update(status="finalized", result="yes", settlement_ts="2026-10-15T13:00:00Z",
                           settlement_value_dollars="1.0000")

    def test_capture_freezes_real_inputs_and_replays_without_network(self):
        report = self.collect()
        self.assertEqual(report["observations"], 1)
        _, observations = replay_capture(self.capture)
        record = observations[0]["record"]
        self.assertEqual(record["market"]["probability"], .5)
        self.assertIsNone(record["label"])
        self.assertIn(self.market["rules_primary"], record["question"])
        payload = parse_record(record).forecast_input.to_payload()
        self.assertNotIn("result", payload)
        self.assertEqual(payload["evidence"], [])

    def test_later_label_does_not_mutate_original_input(self):
        self.collect()
        original = replay_capture(self.capture)[1][0]["record"]
        self.settle()
        poll = self.poll()
        report = build_market_dataset(self.capture, [poll], self.root / "dataset")
        self.assertEqual(report["counts"], {"resolved": 1})
        row = json.loads((self.root / "dataset/records.jsonl").read_text())
        self.assertEqual(parse_record(row).forecast_input, parse_record(original).forecast_input)
        self.assertEqual(row["label"]["available_at"], LATER)
        self.assertIsNone(replay_capture(self.capture)[1][0]["record"]["label"])
        second = build_market_dataset(self.capture, [poll], self.root / "replay")
        self.assertEqual(report, second)

    def test_pending_without_fabricated_outcome(self):
        self.collect()
        report = build_market_dataset(self.capture, [], self.root / "dataset")
        self.assertEqual(report["counts"], {"pending_resolution": 1})
        self.assertFalse(report["ready_for_sft"])

    def test_rule_change_quarantines_label(self):
        self.collect()
        self.settle()
        self.market["rules_primary"] += " Amended resolution criteria."
        poll = self.poll()
        report = build_market_dataset(self.capture, [poll], self.root / "dataset")
        self.assertEqual(report["counts"], {"contract_changed_since_observation": 1})
        self.assertIsNone(json.loads((self.root / "dataset/records.jsonl").read_text())["label"])

    def test_close_time_is_never_substituted_for_settlement_time(self):
        self.collect()
        self.settle()
        self.market["settlement_ts"] = None
        report = build_market_dataset(self.capture, [self.poll()], self.root / "dataset")
        self.assertEqual(report["counts"], {"settlement_timestamp_missing": 1})

    def test_historical_resolution_and_nonbinary_payout_are_rejected(self):
        self.collect()
        self.settle()
        self.market["settlement_ts"] = "2026-09-17T00:00:00Z"
        first = self.poll("old_label")
        self.assertEqual(build_market_dataset(self.capture, [first], self.root / "old")["counts"], {"invalid_settlement_timeline": 1})
        self.market["settlement_ts"] = "2026-10-15T13:00:00Z"
        self.market["settlement_value_dollars"] = "0.5000"
        self.assertEqual(build_market_dataset(self.capture, [self.poll()], self.root / "fractional")["counts"], {"settlement_value_conflict": 1})

    def test_conflicting_settlements_are_quarantined(self):
        self.collect()
        self.settle()
        first = self.poll("first")
        self.market.update(result="no", settlement_value_dollars="0.0000")
        second = self.poll("second")
        report = build_market_dataset(self.capture, [first, second], self.root / "dataset")
        self.assertEqual(report["counts"], {"conflicting_settlements": 1})

    def test_reopened_settlement_is_quarantined(self):
        self.collect()
        self.settle()
        first = self.poll("settled")
        self.market.update(status="active", result="", settlement_ts=None, settlement_value_dollars=None)
        second = self.root / "reopened"
        with patch("foretellmesh.market_dataset.public_get", side_effect=self.response), patch(
                "foretellmesh.market_dataset.now", return_value="2026-10-18T00:00:00Z"):
            poll_market_labels(self.capture, second)
        report = build_market_dataset(self.capture, [second, first], self.root / "dataset")
        self.assertEqual(report["counts"], {"settlement_reopened": 1})

    def test_future_metadata_settled_inputs_and_crossed_quotes_are_rejected(self):
        for changes in ({"updated_time": LATER}, {"status": "finalized", "result": "yes"},
                        {"yes_bid_dollars": "0.6", "yes_ask_dollars": "0.4"},
                        {"yes_bid_dollars": "NaN"}, {"notional_value_dollars": "100"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                observed_record({**self.market, **changes}, EVENT, T, T, "fixture")

    def test_empty_quote_sides_are_not_a_market_baseline(self):
        self.market.update(yes_bid_dollars="0.0000", yes_ask_dollars="1.0000")
        self.assertIsNone(observed_record(self.market, EVENT, T, T, "fixture")["market"])

    def test_raw_tamper_detected(self):
        self.collect()
        (self.capture / "raw/0001.json").write_text("{}")
        with self.assertRaisesRegex(ValidationError, "archive hash"):
            replay_capture(self.capture)

    def test_overwrite_refused(self):
        self.collect()
        with self.assertRaisesRegex(ValidationError, "already exists"):
            self.collect()

    def test_pagination_limit_does_not_publish_partial_events(self):
        config = json.loads(self.config.read_text())
        config["max_pages_per_series"] = 1
        self.config.write_text(json.dumps(config))
        normal = self.response
        def paginated(url):
            raw, date = normal(url)
            obj = json.loads(raw)
            if "markets" in obj:
                obj["cursor"] = "more"
            return json.dumps(obj).encode(), date
        with patch("foretellmesh.market_dataset.public_get", side_effect=paginated), patch("foretellmesh.market_dataset.now", return_value=T):
            report = collect_markets(self.config, self.capture)
        self.assertEqual(report["observations"], 0)
        self.assertEqual(report["exclusions"][0]["reason"], "page_limit_reached")


if __name__ == "__main__":
    unittest.main()
