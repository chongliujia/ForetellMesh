from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file
from foretellmesh.historical_market import kalshi_history_quote, poly_price_url
from foretellmesh.macro_history import build, expected_outcome
from foretellmesh.macro_history_capture import kalshi_url, request_plan
from foretellmesh.schema import ValidationError, timestamp
from test_historical_market import CONDITION, QID, TX, ADAPTER, INIT, init_html
from test_macro_releases import bls_fixture


class HistoricalPriceSchemaTests(unittest.TestCase):
    def test_archived_close_is_fixed_point_dollars_and_live_schema_stays_explicit(self):
        t = timestamp('2026-07-13T12:30:00Z', 'time')
        policy = {'price_window_hours': 6, 'max_price_age_seconds': 10800}
        row = {'end_period_ts': int(t.timestamp()), 'yes_bid': {'close': '0.5300'}, 'yes_ask': {'close': '0.5400'}}
        self.assertIsNone(kalshi_history_quote({'candlesticks': [row]}, t, policy)[0])
        quote, _ = kalshi_history_quote({'candlesticks': [row]}, t, policy, historical=True)
        self.assertEqual(quote['probability'], .535)
        for value in (53, '53', .53, True, '0.53', '1.5300'):
            bad = deepcopy(row); bad['yes_bid']['close'] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                kalshi_history_quote({'candlesticks': [bad]}, t, policy, historical=True)
        row['end_period_ts'] += 1
        self.assertIsNone(kalshi_history_quote({'candlesticks': [row]}, t, policy, historical=True)[0])

    def test_cpi_equal_is_different_from_kalshi_strictly_above(self):
        item = {'platform': 'polymarket', 'group': {'family': 'cpi_yoy', 'period': '2026-06'},
                'market': {'question': 'Will annual inflation be 3.5% in June?',
                           'description': '12-month change before seasonal adjustment for June 2026'}}
        self.assertEqual(expected_outcome(item, {'value': '3.5'}), 1)
        wrong_period = deepcopy(item); wrong_period['group']['period'] = '2026-07'
        with self.assertRaisesRegex(ValidationError, 'period'):expected_outcome(wrong_period, {'value': '3.5'})
        for question in ('Will annual inflation be ≤3.5% in June?', 'Will annual inflation be 3.5% or more in June?'):
            item['market']['question'] = question; self.assertEqual(expected_outcome(item, {'value': '3.5'}), 1)
        item = {'platform': 'kalshi', 'group': {'family': 'cpi_yoy', 'period': '2026-06'}, 'market': {
            'event_ticker': 'KXCPIYOY-26JUN', 'ticker': 'KXCPIYOY-26JUN-T3.5', 'strike_type': 'greater',
            'market_type': 'binary', 'notional_value_dollars': '1', 'floor_strike': 3.5,
            'rules_primary': 'CPI increases by more than 3.5% in the twelve months ending June 2026.'}}
        self.assertEqual(expected_outcome(item, {'value': '3.5'}), 0)
        item['market']['floor_strike'] = 3.6
        with self.assertRaises(ValidationError):expected_outcome(item, {'value': '3.5'})

    def test_fomc_open_ended_bucket_and_nonstandard_increment(self):
        item = {'platform': 'polymarket', 'group': {'family': 'fomc', 'period': '2026-01'},
                'market': {'question': 'Fed increases interest rates by 25+ bps after January 2026 meeting?', 'description': ''}}
        self.assertEqual(expected_outcome(item, {'value': '4.25'}, {'value': '3.75'}), 1)
        with self.assertRaisesRegex(ValidationError, 'nonstandard'):
            expected_outcome(item, {'value': '3.875'}, {'value': '3.75'})


class MacroHistoryBuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.bls = self.root/'bls'; self.bls.mkdir(); refs = []
        for name, obj in [('prior', bls_fixture(period='2026-05', date='June 10, 2026', day='Wednesday', value='4.2')),
                          ('result', bls_fixture())]:
            p = self.bls/(name+'.json'); p.write_text(json.dumps(obj)); refs.append({'file': p.name, 'sha256': sha256_file(p)})
        (self.bls/'manifest.json').write_text(json.dumps({'refs': refs}))
        self.group = {'event_group_id': 'macro:us:cpi_yoy:2026-06', 'family': 'cpi_yoy', 'period': '2026-06',
            'planned_observation_times': ['2026-07-07T12:30:00Z', '2026-07-13T12:30:00Z'],
            'disposition': 'evaluation_only_pending_evidence', 'missing': []}
        self.policy = {'price_window_hours': 6, 'max_price_age_seconds': 10800, 'polymarket_bucket_seconds': 10800, 'kalshi_period_minutes': 60}
        self.config = {'fed_releases': [], 'price_policy': self.policy, 'max_requests': 1600, 'comparison_bucket_seconds': 1800}
        self.cp = self.root/'config.json'; self.cp.write_text(json.dumps(self.config))
        self.capture = self.root/'capture'; self.capture.mkdir(); (self.capture/'manifest.json').write_text('{}')
        self.cutoff = {'market_settled_ts': '2026-08-01T00:00:00Z'}
        ref = {'completed_at': '2026-09-19T00:00:00Z'}
        pm = {'id': '12', 'conditionId': CONDITION, 'negRiskRequestID': QID, 'resolvedBy': ADAPTER,
              'question': 'Will annual inflation be 3.5% in June?', 'description': '12-month change before seasonal adjustment for June 2026',
              'startDate': INIT, 'outcomes': '["Yes", "No"]', 'clobTokenIds': '["123", "456"]', 'closed': True}
        state = {'question_id': QID, 'condition_id': CONDITION, 'transaction_hash': TX, 'status': 'resolved',
                 'price': '1000000000000000000', 'extended_review': False, 'was_disputed': False}
        km = {'ticker': 'KXCPIYOY-26JUN-T3.5', 'event_ticker': 'KXCPIYOY-26JUN', 'strike_type': 'greater',
              'floor_strike': 3.5, 'notional_value_dollars': '1', 'market_type': 'binary', 'open_time': INIT,
              'close_time': '2026-07-14T12:29:00Z', 'settlement_ts': '2026-07-14T13:00:00Z',
              'status': 'finalized', 'result': 'no', 'settlement_value_dollars': '0',
              'rules_primary': 'CPI increases by more than 3.5% in the twelve months ending June 2026.'}
        self.selected = [{'group': self.group, 'platform': 'polymarket', 'market': pm, 'market_ref': ref, 'state': state, 'state_ref': ref},
                         {'group': self.group, 'platform': 'kalshi', 'market': km, 'market_ref': ref}]
        self.responses = {}
        def add(url, obj, binary=False):self.responses[url] = ({'status': 200, 'url': url}, obj if binary else json.dumps(obj).encode())
        add('https://clob.polymarket.com/clob-markets/'+CONDITION, {'c': CONDITION, 't': [{'o': 'Yes', 't': '123'}, {'o': 'No', 't': '456'}]})
        add('https://polygonscan.com/tx/'+TX, init_html('q: title: '+pm['question']+', description: '+pm['description']+', creator: fixture'), True)
        for obs in self.group['planned_observation_times']:
            t = timestamp(obs, 'time'); sec = int(t.timestamp())
            add(poly_price_url('123', t, self.policy), {'data': [{'timestamp': sec-10800, 'resolution_seconds': 10800, 'price': .6},
                {'timestamp': sec-100, 'resolution_seconds': 10800, 'price': .99}, {'timestamp': sec, 'resolution_seconds': 0, 'price': 1}],
                'pagination': {'has_more': False}})
            add(kalshi_url(km, obs, self.policy, self.cutoff), {'ticker': km['ticker'], 'candlesticks': [
                {'end_period_ts': sec, 'yes_bid': {'close': '0.5300'}, 'yes_ask': {'close': '0.5400'}}]})

    def build(self, name):
        supplied = self.config, {'bls_archive': self.bls}, [self.group], self.selected, self.cutoff, self.responses
        with patch('foretellmesh.macro_history.read_history', return_value=supplied):
            return build(self.cp, self.capture, self.root/name)

    def test_outcomes_are_separate_and_full_denominator_survives(self):
        report = self.build('one'); self.assertEqual(report['counts']['observation_rows'], 4)
        self.assertEqual(report['counts']['initial_questions'], 1)
        self.assertEqual(report['counts']['exact_settlement_labels'], 2)
        self.assertEqual(report['counts']['usable_historical_quotes'], 4)
        rows = [json.loads(s) for s in (self.root/'one/observations.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertEqual(row['prior_evidence']['period'], '2026-05'); self.assertEqual(row['prior_evidence']['value'], '4.2')
            self.assertNotIn('label', row); self.assertNotIn('crosscheck', row); self.assertFalse(row['ready_for_scoring'])
        self.assertEqual(report, self.build('two'))
        self.selected[0]['state']['price'] = '0'
        changed = self.build('changed')
        after = [json.loads(s) for s in (self.root/'changed/observations.jsonl').read_text().splitlines()]
        for a, b in zip(rows, after):
            a.pop('blockers'); b.pop('blockers'); self.assertEqual(a, b)
        self.assertLess(changed['counts']['official_outcome_crosschecked_rows'], report['counts']['official_outcome_crosschecked_rows'])

    def test_coarse_and_fine_queries_keep_identical_observation_cutoffs(self):
        urls = request_plan(self.config, [self.group], self.selected, self.cutoff)
        price_urls = [url for url in urls if 'prices-history' in url]
        self.assertEqual(len(price_urls), 4)
        self.assertEqual(sum('bucket_seconds=10800' in url for url in price_urls), 2)
        self.assertEqual(sum('bucket_seconds=1800' in url for url in price_urls), 2)
        self.config.update(polymarket_price_source='clob', max_requests=2200)
        urls = request_plan(self.config, [self.group], self.selected, self.cutoff)
        self.assertEqual(sum('clob.polymarket.com/prices-history?' in url for url in urls), 2)

    def test_clob_selection_is_uniform_without_data_api_fallback(self):
        self.config['polymarket_price_source'] = 'clob'
        from foretellmesh.macro_prices import clob_price_url
        for obs in self.group['planned_observation_times']:
            url = clob_price_url('123', timestamp(obs, 'time'), self.policy)
            self.responses[url] = ({'status': 200, 'url': url}, b'{"history":[]}')
        report = self.build('clob_missing')
        self.assertEqual(report['counts']['quotes_by_platform'], {'kalshi': 2})
        self.assertEqual(report['counts']['blockers']['clob_price_semantics_review_required'], 2)


if __name__ == '__main__':unittest.main()
