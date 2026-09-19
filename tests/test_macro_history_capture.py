import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_bytes
from foretellmesh.macro_history_capture import KALSHI, capture, read_history
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash


class HistoricalArchiveIntegrityTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.config = {'source_hashes': {'inventory': 'fixed'}, 'workers': 4}
        self.config_path = self.root/'config.json'
        self.urls = ['https://www.federalreserve.gov/fixture']
        self.input_patch = patch('foretellmesh.macro_history_capture.inputs', return_value=(self.config, {}, [], []))
        self.plan_patch = patch('foretellmesh.macro_history_capture.request_plan', return_value=self.urls)
        self.input_patch.start(); self.addCleanup(self.input_patch.stop)
        self.plan_patch.start(); self.addCleanup(self.plan_patch.stop)

    def fetched(self, url):
        raw = b'{"market_settled_ts":"2026-08-01T00:00:00Z"}' if url.endswith('/cutoff') else b'archived public response'
        return {'url': url, 'status': 200, 'sha256': sha256_bytes(raw),
                'started_at': '2026-09-19T00:00:00Z', 'completed_at': '2026-09-19T00:00:01Z'}, raw

    def archive(self, name='capture'):
        root = self.root/name
        with patch('foretellmesh.macro_history_capture.fetched', side_effect=self.fetched):
            capture(self.config_path, root)
        return root

    def rewrite(self, root, fn):
        path = root/'manifest.json'; manifest = json.loads(path.read_text()); fn(manifest)
        manifest['requests_sha256'] = canonical_hash(manifest['requests']); path.write_text(json.dumps(manifest))

    def test_reuse_preserves_body_and_original_capture_time_without_network(self):
        original = self.archive(); reused = self.root/'reused'
        with patch('foretellmesh.macro_history_capture.fetched', side_effect=AssertionError('unexpected network')):
            report = capture(self.config_path, reused, reuse=original)
        self.assertEqual(report['reused_responses'], 2)
        self.assertEqual(read_history(self.config_path, original)[-1], read_history(self.config_path, reused)[-1])

    def test_raw_body_tampering_prevents_read_and_reuse(self):
        root = self.archive(); (root/'raw/0001.bin').write_bytes(b'tampered')
        with self.assertRaises(ValidationError):read_history(self.config_path, root)
        with self.assertRaises(ValidationError):capture(self.config_path, self.root/'reuse', reuse=root)
        self.assertFalse((self.root/'reuse').exists())

    def test_rehashed_missing_request_still_fails_coverage(self):
        root = self.archive(); self.rewrite(root, lambda m: m['requests'].pop())
        with self.assertRaisesRegex(ValidationError, 'coverage'):read_history(self.config_path, root)

    def test_rehashed_reference_escape_and_backwards_clock_rejected(self):
        for name, change in [('escape', {'file': '../config.json'}),
                             ('clock', {'completed_at': '2025-01-01T00:00:00Z'})]:
            root = self.archive(name); self.rewrite(root, lambda m: m['requests'][1].update(change))
            with self.subTest(name=name), self.assertRaises(ValidationError):read_history(self.config_path, root)

    def test_failed_requests_stay_in_denominator(self):
        def fail(url):
            ref, raw = self.fetched(url)
            if not url.endswith('/cutoff'):ref['status'] = 503
            return ref, raw
        root = self.root/'failed'
        with patch('foretellmesh.macro_history_capture.fetched', side_effect=fail):report = capture(self.config_path, root)
        self.assertEqual(report['requests'], 2); self.assertEqual(report['failed_requests'], 1)
        self.assertEqual(len(read_history(self.config_path, root)[-1]), 2)


if __name__ == '__main__':unittest.main()
