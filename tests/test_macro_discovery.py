from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from foretellmesh.data import sha256_file
from foretellmesh.macro_discovery import build, capture, load_plan, read_capture
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


class MacroDiscoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.plan = json.loads((ROOT/'configs/macro_evaluation_discovery_v1.json').read_text())
        chosen = {'macro:us:cpi_yoy:2026-02', 'macro:us:fomc:2026-03', 'macro:us:fomc:2026-09'}
        self.plan['groups'] = [g for g in self.plan['groups'] if g['event_group_id'] in chosen]
        self.plan['source_refs'] = []
        self.config = self.root/'config.json'; self.archive = self.root/'capture'; self.index = self.root/'index.json'
        entries = [{'event_id': 'polymarket:'+self.condition('67284'), 'question': 'Different wording'}]
        self.index.write_text(json.dumps({'schema_version': '1', 'source': 'forecastbench',
            'entries': entries, 'entries_sha256': canonical_hash(entries)}))

    @staticmethod
    def condition(event):return '0x'+format(int(event), '064x')

    def fetch(self, url):
        p = urlparse(url); query = parse_qs(p.query)
        if p.hostname == 'gamma-api.polymarket.com':
            event = p.path.split('/')[-1]
            body = {'id': event, 'title': 'Fed decision', 'markets': [{'id': str(int(event)*10),
                    'conditionId': self.condition(event),
                    'question': 'Fed decision '+('March' if event == '67284' else 'September')+' 2026 '+event, 'closed': True,
                    'closedTime': '2026-05-20T00:00:00Z' if event == '432066' else None,
                    'endDate': '2026-09-16T00:00:00Z'}]}
        elif p.hostname == 'data-api.polymarket.com':
            event = query['event_id'][0]
            body = {'data': [{'condition_id': self.condition(event), 'status': 'resolved',
                             'last_update_timestamp': '1779287837' if event == '432066' else '1789640000'}]}
        elif '/historical/markets' in p.path:
            ticker = query['event_ticker'][0]
            body = {'markets': [{'event_ticker': ticker, 'ticker': ticker+'-T1', 'title': ticker}], 'cursor': ''}
        else:
            ticker = p.path.split('/')[-1]; body = {'event': {'event_ticker': ticker}, 'markets': []}
        return 200, json.dumps(body).encode(), None

    def collect(self, fetch=None):
        self.config.write_text(json.dumps(self.plan))
        with patch('foretellmesh.macro_discovery.fetch', side_effect=fetch or self.fetch):
            return capture(self.config, self.archive)

    def test_inventory_preserves_missing_slots_groups_and_has_no_labels(self):
        self.collect(); report = build(self.archive, self.index, self.root/'built')
        self.assertEqual(report['planned_groups'], 3)
        self.assertEqual(report['groups_with_polymarket'], 2)
        self.assertEqual(report['groups_with_kalshi'], 3)
        self.assertEqual(report['pending_groups_after_identity_exclusions'], 2)
        self.assertEqual(report['pending_contracts_after_identity_exclusions'], 3)
        self.assertEqual(report['exact_overlap_excluded_groups'], ['macro:us:fomc:2026-03'])
        self.assertEqual(report['admitted_rows'], 0); self.assertIsNone(report['score_metrics'])
        self.assertEqual(report['native_event_flag_counts']['resolved_status_predates_planned_observations'], 1)
        self.assertEqual([r['id'] for r in report['quarantined_native_events']], ['432066'])
        inventory = json.loads((self.root/'built/inventory.json').read_text())
        march = next(g for g in inventory['groups'] if g['period'] == '2026-03')
        self.assertEqual({n['platform'] for n in march['native_events']}, {'polymarket', 'kalshi'})
        self.assertEqual(march['disposition'], 'exclude_from_new_cohort')
        self.assertNotIn('outcome', (self.root/'built/inventory.json').read_text())
        self.assertEqual(report, build(self.archive, self.index, self.root/'rebuilt'))
        self.assertEqual((self.root/'built/inventory.json').read_bytes(), (self.root/'rebuilt/inventory.json').read_bytes())

    def test_failed_endpoint_and_incomplete_pagination_remain_visible(self):
        def fetch(url):
            if '/events/67284' in url:return 503, b'unavailable', None
            status, raw, date = self.fetch(url)
            if '/historical/markets' in url:
                obj = json.loads(raw); obj['cursor'] = 'next'; raw = json.dumps(obj).encode()
            return status, raw, date
        result = self.collect(fetch); self.assertEqual(result['failures'], 1)
        report = build(self.archive, self.index, self.root/'built')
        self.assertEqual(report['native_event_flag_counts']['historical_contract_inventory_truncated'], 3)
        self.assertTrue(any(m['reason'] == 'request_failed' for m in report['missing']))
        self.assertEqual(report['planned_groups'], 3)

    def test_current_resolution_result_never_emits_a_label_or_changes_scope(self):
        self.collect(); before = build(self.archive, self.index, self.root/'before')
        self.archive = self.root/'changed'
        def fetch(url):
            status, raw, date = self.fetch(url)
            if '/v2/resolutions' in url:
                obj = json.loads(raw)
                for r in obj['data']:r['price'] = '0'; r['payouts'] = [0, 1]
                raw = json.dumps(obj).encode()
            return status, raw, date
        self.collect(fetch); after = build(self.archive, self.index, self.root/'after')
        self.assertEqual(before['planned_groups'], after['planned_groups'])
        self.assertEqual(before['exact_overlap_excluded_groups'], after['exact_overlap_excluded_groups'])
        a = json.loads((self.root/'before/inventory.json').read_text())
        b = json.loads((self.root/'after/inventory.json').read_text())
        for obj in (a, b):
            for g in obj['groups']:
                for n in g['native_events']:
                    n.pop('source_ref', None); n.pop('historical_source_ref', None)
        self.assertEqual(a, b)

    def test_raw_tampering_missing_request_and_overwrite_rejected(self):
        self.collect()
        with self.assertRaisesRegex(ValidationError, 'exists'):capture(self.config, self.archive)
        path = self.archive/'raw/000.bin'; original = path.read_bytes(); path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValidationError, 'body hash'):read_capture(self.archive)
        path.write_bytes(original)
        mpath = self.archive/'manifest.json'; manifest = json.loads(mpath.read_text()); manifest['requests'].pop()
        manifest['requests_sha256'] = canonical_hash(manifest['requests']); mpath.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, 'missing or reordered'):read_capture(self.archive)

    def test_wrong_kalshi_country_duplicate_group_future_or_tuning_policy_rejected(self):
        original = deepcopy(self.plan)
        variants = []
        wrong = deepcopy(original); wrong['groups'][0]['kalshi_event_tickers'] = ['KXUE-26FEB']; variants.append(wrong)
        duplicate = deepcopy(original); duplicate['groups'].append(duplicate['groups'][0]); variants.append(duplicate)
        future = deepcopy(original); future['groups'][0]['calendar_reference_time'] = '2027-01-01T00:00:00Z'; variants.append(future)
        variants.append({**original, 'prompt_tuning': True})
        for plan in variants:
            self.config.write_text(json.dumps(plan))
            with self.assertRaises(ValidationError):load_plan(self.config)

    def test_calendar_provenance_cannot_silently_change(self):
        source = self.root/'calendar'; source.write_text('calendar')
        self.plan['source_refs'] = [{'path': str(source), 'sha256': sha256_file(source), 'url': 'fixture'}]
        source.write_text('revised calendar')
        with self.assertRaisesRegex(ValidationError, 'source changed'):self.collect()

    def test_kalshi_nested_and_historical_markets_are_merged_by_ticker(self):
        def fetch(url):
            status, raw, date = self.fetch(url); obj = json.loads(raw)
            if '/events/KX' in url:
                ticker = obj['event']['event_ticker']
                obj['event']['markets'] = [{'event_ticker': ticker, 'ticker': ticker+'-T1', 'title': ticker},
                                          {'event_ticker': ticker, 'ticker': ticker+'-T2', 'title': ticker}]
            return status, json.dumps(obj).encode(), date
        self.collect(fetch)
        report = build(self.archive, self.index, self.root/'built')
        self.assertEqual(report['contracts_by_platform']['kalshi'], 6)
        self.assertEqual(report['groups_with_kalshi'], 3)


if __name__ == '__main__':unittest.main()
