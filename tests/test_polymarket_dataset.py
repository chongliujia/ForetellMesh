import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import strict_json
from foretellmesh.market_quality import audit_market_quality
from foretellmesh.polymarket_dataset import (GAMMA, CLOB, DATA, book_quote, build_polymarket_dataset,
    collect_polymarket, mapping, poll_polymarket_labels, replay_capture, resolution_row)
from foretellmesh.schema import ValidationError, parse_record, timestamp
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
T = "2026-09-18T08:00:00Z"
LATER = "2026-10-30T08:00:00Z"
CONDITION = "0x" + "a" * 64
QUESTION = "0x" + "b" * 64


class PolymarketDatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.capture = self.root / "capture"
        self.config = json.loads((ROOT / "configs/polymarket_collection_v1.json").read_text())
        self.config["event_ids"] = ["606422"]
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.market = {"id": "12", "conditionId": CONDITION, "questionID": "0x" + "c" * 64,
                       "question": "Will the October Fed rate rise?", "description": "Resolve from the October FOMC statement.",
                       "outcomes": '["Yes", "No"]', "clobTokenIds": '["123", "456"]',
                       "active": True, "closed": False, "acceptingOrders": True, "enableOrderBook": True,
                       "negRisk": True, "negRiskOther": False, "createdAt": "2026-06-01T00:00:00Z", "updatedAt": T,
                       "startDate": "2026-06-01T00:00:00Z", "endDate": "2026-10-29T00:00:00Z"}
        self.event = {"id": "606422", "closed": False, "markets": [copy.deepcopy(self.market)]}
        self.clob = {"c": CONDITION, "ao": True, "t": [{"o": "Yes", "t": "123"}, {"o": "No", "t": "456"}]}
        self.book = {"market": CONDITION, "asset_id": "123", "timestamp": str(int(datetime.fromisoformat(T).timestamp() * 1000)),
                     "bids": [{"price": "0.1", "size": "50"}, {"price": "0.45", "size": "20"}],
                     "asks": [{"price": "0.9", "size": "50"}, {"price": "0.50", "size": "20"}]}
        self.state = {"condition_id": CONDITION, "question_id": QUESTION, "status": "posed", "extended_review": False,
                      "was_disputed": False, "new_version_q": True, "price": "69"}

    def response(self, url):
        responses = {GAMMA + "/events/606422": self.event, GAMMA + "/markets/12": self.market,
                     CLOB + "/clob-markets/" + CONDITION: self.clob, CLOB + "/book?token_id=123": self.book,
                     DATA + "/v2/resolutions?condition=" + CONDITION: {"data": [self.state]}}
        return json.dumps(responses[url]).encode(), None

    def collect(self):
        with patch("foretellmesh.polymarket_dataset.public_get", side_effect=self.response), patch(
                "foretellmesh.polymarket_dataset.now", return_value=T):
            return collect_polymarket(self.config_path, self.capture)

    def poll(self, name="poll", time=LATER):
        path = self.root / name
        with patch("foretellmesh.polymarket_dataset.public_get", side_effect=self.response), patch(
                "foretellmesh.polymarket_dataset.now", return_value=time):
            poll_polymarket_labels(self.capture, path)
        return path

    def settle(self):
        self.market.update(closed=True, acceptingOrders=False)
        self.state.update(status="resolved", payouts=[1_000_000, 0], resolved_at="2026-10-29T03:00:00Z",
                          resolved_block=12000, resolution_source="reported")

    def build(self, polls, name="dataset"):
        return build_polymarket_dataset(self.capture, polls, self.root / name)

    def quote(self, **overrides):
        return book_quote({**self.book, **overrides}, {"completed_at": T}, timestamp(T, "t"), CONDITION, "123", self.config)

    def test_capture_uses_best_prices_not_response_order_or_gamma_price(self):
        self.market["outcomePrices"] = '["0.99", "0.01"]'
        result = self.collect()
        self.assertEqual(result["observations"], 1)
        _, rows, rejected = replay_capture(self.capture)
        self.assertFalse(rejected)
        self.assertEqual(rows[0]["record"]["market"]["probability"], .475)
        self.assertIsNone(rows[0]["record"]["label"])
        self.assertEqual(rows[0]["quote_quality"]["best_bid"], .45)

    def test_reversed_outcomes_preserve_yes_token_and_label_index(self):
        self.market.update(outcomes='["No", "Yes"]', clobTokenIds='["456", "123"]')
        self.clob["t"].reverse()
        self.assertEqual(mapping(self.market), ({"No": "456", "Yes": "123"}, 1))
        self.collect()
        self.settle()
        self.state["payouts"] = [0, 1_000_000]
        self.assertEqual(self.build([self.poll()])["counts"], {"resolved": 1})
        row = strict_json((self.root / "dataset/records.jsonl").read_text())
        self.assertEqual(row["label"]["outcome"], 1)

    def test_settlement_and_replay_preserve_model_input(self):
        self.collect()
        original = replay_capture(self.capture)[1][0]["record"]
        self.settle()
        poll = self.poll()
        first = self.build([poll])
        second = self.build([poll], "replay")
        self.assertEqual(first, second)
        row = strict_json((self.root / "dataset/records.jsonl").read_text())
        self.assertEqual(parse_record(row).forecast_input, parse_record(original).forecast_input)
        self.assertEqual(row["label"]["available_at"], LATER)
        for filename in ("records.jsonl", "metadata.jsonl", "manifest.json"):
            self.assertEqual((self.root / "dataset" / filename).read_bytes(), (self.root / "replay" / filename).read_bytes())

    def test_closed_and_extreme_price_are_not_settlement(self):
        self.collect()
        self.market.update(closed=True, outcomePrices='["1", "0"]')
        self.assertEqual(self.build([self.poll()])["counts"], {"pending_resolution": 1})

    def test_unproved_uma_price_is_not_binary_payout(self):
        self.collect()
        self.state.update(status="resolved", price="1000000000000000000", last_update_timestamp="1793300400")
        self.assertEqual(self.build([self.poll()])["counts"], {"binary_payout_proof_missing": 1})

    def test_disputes_and_context_changes_block_labels(self):
        self.collect()
        self.settle()
        good = copy.deepcopy(self.state)
        cases = [("extended_review", True, "disputed_or_unverified_resolution"),
                 ("was_disputed", True, "disputed_or_unverified_resolution"),
                 ("question_id", "0x" + "d" * 64, "resolution_context_changed"),
                 ("new_version_q", False, "resolution_context_changed")]
        for n, (key, value, expected) in enumerate(cases):
            self.state = {**good, key: value}
            self.assertEqual(self.build([self.poll(f"poll{n}")], f"dataset{n}")["counts"], {expected: 1})

    def test_fractional_boolean_missing_timestamp_or_future_settlement(self):
        self.collect()
        self.settle()
        good = copy.deepcopy(self.state)
        cases = [({"payouts": [500_000, 500_000]}, "binary_payout_proof_missing"),
                 ({"payouts": [True, 0]}, "binary_payout_proof_missing"),
                 ({"resolved_at": None}, "settlement_time_or_provenance_missing"),
                 ({"resolved_block": None}, "settlement_time_or_provenance_missing"),
                 ({"resolved_at": "2026-09-01T00:00:00Z"}, "invalid_settlement_timeline"),
                 ({"resolved_at": "2026-12-01T00:00:00Z"}, "invalid_settlement_timeline")]
        for n, (changes, expected) in enumerate(cases):
            self.state = {**good, **changes}
            self.assertEqual(self.build([self.poll(f"poll{n}")], f"dataset{n}")["counts"], {expected: 1})

    def test_rule_change_quarantines_label(self):
        self.collect()
        self.settle()
        self.market["description"] += " Amended rules."
        self.assertEqual(self.build([self.poll()])["counts"], {"contract_changed_since_observation": 1})

    def test_conflicting_and_reopened_settlement(self):
        self.collect()
        self.settle()
        first = self.poll("first")
        self.state["payouts"] = [0, 1_000_000]
        second = self.poll("second", "2026-10-31T00:00:00Z")
        self.assertEqual(self.build([second, first])["counts"], {"conflicting_settlements": 1})
        self.state["status"] = "disputed"
        third = self.poll("third", "2026-11-01T00:00:00Z")
        self.assertEqual(self.build([first, third], "reopened")["counts"], {"settlement_reopened_or_unverifiable": 1})

    def test_stale_missing_thin_and_wide_quotes_have_no_baseline(self):
        changes = [({"timestamp": str(int(self.book["timestamp"]) - 121000)}, "stale_book"),
                   ({"asks": []}, "missing_book_side"),
                   ({"bids": [{"price": "0.45", "size": "1"}]}, "thin_best_level"),
                   ({"asks": [{"price": "0.8", "size": "100"}]}, "wide_spread")]
        for value, reason in changes:
            quote, quality = self.quote(**value)
            self.assertIsNone(quote)
            self.assertIn(reason, quality["issues"])

    def test_bad_book_identity_levels_clock_and_crossing_rejected(self):
        changes = [{"market": "0x" + "e" * 64}, {"asset_id": "456"}, {"timestamp": str(int(self.book["timestamp"]) + 1)},
                   {"bids": [{"price": "NaN", "size": "1"}]}, {"asks": [{"price": "0.50", "size": "0"}]},
                   {"bids": [{"price": "0.51", "size": "10"}]}, {"timestamp": "1789718400"}]
        for changeset in changes:
            with self.subTest(changeset=changeset), self.assertRaises(ValidationError):
                self.quote(**changeset)

    def test_mapping_conflict_is_audited_exclusion(self):
        self.clob["t"][0]["t"] = "456"
        result = self.collect()
        self.assertEqual(result["observations"], 0)
        self.assertIn("CLOB token", result["exclusions"][0]["reason"])

    def test_token_order_mismatch_cannot_invent_payout_index(self):
        self.clob["t"].reverse()
        self.assertEqual(self.collect()["observations"], 0)

    def test_slow_snapshot_is_excluded(self):
        times = [T] * 8 + ["2026-09-18T08:04:00Z"] * 2
        with patch("foretellmesh.polymarket_dataset.public_get", side_effect=self.response), patch(
                "foretellmesh.polymarket_dataset.now", side_effect=times):
            result = collect_polymarket(self.config_path, self.capture)
        self.assertEqual(result["observations"], 0)
        self.assertIn("too slow", result["exclusions"][0]["reason"])

    def test_duplicate_event_member_and_budget_fail_without_output(self):
        self.event["markets"].append(copy.deepcopy(self.market))
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            self.collect()
        self.assertFalse(self.capture.exists())
        self.config["max_markets"] = 1
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValidationError, "exceeds budget"):
            self.collect()
        self.assertFalse(self.capture.exists())

    def test_future_metadata_and_proposed_resolution_excluded(self):
        self.market["updatedAt"] = LATER
        result = self.collect()
        self.assertIn("future metadata", result["exclusions"][0]["reason"])
        self.capture = self.root / "capture2"
        self.market["updatedAt"] = T
        self.state["status"] = "proposed"
        self.assertEqual(self.collect()["observations"], 0)

    def test_resolution_identity_is_mandatory_and_unique(self):
        for value in ({"data": [{**self.state, "condition_id": "wrong"}]}, {"data": [self.state, self.state]}):
            with self.assertRaises(ValidationError):
                resolution_row(value, CONDITION)

    def test_raw_archive_tampering_rejected(self):
        self.collect()
        raw = self.capture / "raw/0000.json"
        raw.write_text(raw.read_text() + " ")
        with self.assertRaisesRegex(ValidationError, "hash mismatch"):
            replay_capture(self.capture)

    def test_label_poll_must_be_later_and_bound_to_capture(self):
        self.collect()
        poll = self.poll(time=T)
        with self.assertRaisesRegex(ValidationError, "must follow"):
            self.build([poll])
        manifest_path = poll / "manifest.json"
        manifest = strict_json(manifest_path.read_text())
        manifest["capture_requests_sha256"] = "wrong"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, "different capture"):
            self.build([poll])

    def test_quality_grouping_keeps_contracts_and_blocks_training(self):
        self.collect()
        self.build([])
        index = {"schema_version": "1", "source": "forecastbench", "entries": [
            {"event_id": "fb:1", "question": "Unrelated election outcome"}]}
        index["entries_sha256"] = canonical_hash(index["entries"])
        index_path = self.root / "index.json"
        index_path.write_text(json.dumps(index))
        report = audit_market_quality([self.root / "dataset"], ROOT / "configs/market_event_groups_v1.json", index_path, self.root / "audit")
        self.assertEqual(report["counts"]["ready_for_sft"], 0)
        self.assertEqual(report["counts"]["verified_outcomes"], 0)
        row = strict_json((self.root / "audit/grouped_staging_records.jsonl").read_text())
        original = replay_capture(self.capture)[1][0]["record"]
        self.assertEqual(row["event_group_id"], "macro:us:fomc:2026-10")
        self.assertEqual(parse_record(row).forecast_input, parse_record(original).forecast_input)
        (self.root / "dataset/records.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValidationError, "hash mismatch"):
            audit_market_quality([self.root / "dataset"], ROOT / "configs/market_event_groups_v1.json", index_path, self.root / "bad")


if __name__ == "__main__":
    unittest.main()
