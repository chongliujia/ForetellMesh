import copy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_bytes
from foretellmesh.market_quality import audit_market_quality
from foretellmesh.schema import ValidationError
from foretellmesh.sft_data import jsonl
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


class MarketQualityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.groups = ROOT / "configs/market_event_groups_v1.json"
        self.index = self.root / "heldout.json"
        entries = [{"event_id": "polymarket:12", "question": "A held out question"}]
        self.index.write_text(json.dumps({"source": "forecastbench", "schema_version": "1", "entries": entries,
                                          "entries_sha256": canonical_hash(entries)}))

    def bundle(self, platform, group, question, extra=None):
        root = self.root / platform
        root.mkdir()
        source = "foretellmesh_" + platform
        record = {"sample_id": platform + ":sample", "dataset_source": source, "dataset_version": "fixture",
                  "event_id": platform + ":12", "event_group_id": group, "question": question,
                  "observation_time": "2026-09-18T08:00:00Z", "market": None, "evidence": [], "label": None}
        if extra:
            record.update(extra)
        meta = {"sample_id": record["sample_id"], "status": "pending_resolution"}
        artifacts = {"records.jsonl": jsonl([record]), "metadata.jsonl": jsonl([meta])}
        manifest = {"kind": "prospective_prediction_market_dataset", "schema_version": "1", "dataset_source": source,
                    "dataset_version": "fixture", "record_count": 1, "capture_requests_sha256": "capture",
                    "counts": {"pending_resolution": 1}, "artifact_hashes": {k: sha256_bytes(v.encode()) for k, v in artifacts.items()}}
        for name, data in artifacts.items():
            (root / name).write_text(data)
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root

    def audit(self, bundles):
        return audit_market_quality(bundles, self.groups, self.index, self.root / "audit")

    def test_related_platform_contracts_share_split_group_without_merging(self):
        a = self.bundle("kalshi", "kalshi:KXFED-26OCT", "Will October Fed rate exceed 4%?")
        b = self.bundle("polymarket", "polymarket:606422", "Will October Fed rate increase?")
        report = self.audit([a, b])
        self.assertEqual(report["counts"]["records"], 2)
        self.assertEqual(report["counts"]["native_groups"], 2)
        self.assertEqual(report["counts"]["split_groups"], 1)
        self.assertEqual(report["counts"]["exact_heldout_overlaps"], 1)
        self.assertEqual(report["counts"]["ready_for_benchmark"], 0)
        rows = [json.loads(x) for x in (self.root / "audit/grouped_staging_records.jsonl").read_text().splitlines()]
        self.assertEqual(len({r["event_id"] for r in rows}), 2)
        self.assertEqual({r["event_group_id"] for r in rows}, {"macro:us:fomc:2026-10"})

    def test_unknown_group_remains_explicitly_unreviewed(self):
        bundle = self.bundle("kalshi", "kalshi:unknown", "Unknown event")
        report = self.audit([bundle])
        self.assertEqual(report["counts"]["blockers"]["cross_platform_group_unmapped"], 1)
        self.assertEqual(report["counts"]["ready_for_sft"], 0)

    def test_duplicate_group_mapping_is_invalid(self):
        bundle = self.bundle("kalshi", "kalshi:KXFED-26OCT", "Fed?")
        groups = json.loads(self.groups.read_text())
        groups["groups"][1]["native_event_groups"].append("kalshi:KXFED-26OCT")
        self.groups = self.root / "groups.json"
        self.groups.write_text(json.dumps(groups))
        with self.assertRaisesRegex(ValidationError, "multiple groups"):
            self.audit([bundle])
        self.assertFalse((self.root / "audit").exists())

    def test_timestamp_invalid_data_cannot_pass_hash_validation_alone(self):
        bundle = self.bundle("kalshi", "kalshi:KXFED-26OCT", "Fed?", {"evidence": [{
            "evidence_id": "future", "text": "Post observation information", "source": "fixture",
            "published_at": "2026-10-01T00:00:00Z", "available_at": "2026-10-01T00:00:00Z"}]})
        with self.assertRaises(ValidationError):
            self.audit([bundle])
        self.assertFalse((self.root / "audit").exists())

    def test_metadata_label_disposition_must_match(self):
        bundle = self.bundle("kalshi", "kalshi:KXFED-26OCT", "Fed?", {"label": {
            "outcome": 1, "resolution_time": "2026-10-29T00:00:00Z", "available_at": "2026-10-30T00:00:00Z"}})
        with self.assertRaisesRegex(ValidationError, "label disposition"):
            self.audit([bundle])

    def test_heldout_index_tampering_and_duplicate_bundles_rejected(self):
        bundle = self.bundle("kalshi", "kalshi:KXFED-26OCT", "Fed?")
        with self.assertRaisesRegex(ValidationError, "duplicate sample_id"):
            self.audit([bundle, bundle])
        index = json.loads(self.index.read_text())
        index["entries"][0]["question"] = "Altered"
        self.index.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValidationError, "held-out index"):
            self.audit([bundle])


if __name__ == "__main__":
    unittest.main()
