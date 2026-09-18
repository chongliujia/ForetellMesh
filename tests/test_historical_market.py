from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from foretellmesh.historical_market import (build_historical_markets, capture_historical_markets,
    fed_upper_bound, historical_poly_label, kalshi_history_quote, load_plan, poly_history_quote)
from foretellmesh.historical_sources import (allowed_url, fed_statement, initialized_question, read_archive)
from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
CONDITION = "0x" + "a" * 64
QID = "0x" + "b" * 64
TX = "0x" + "c" * 64
ADAPTER = "0x" + "d" * 40
NOW = "2026-09-18T09:00:00Z"
INIT = "2026-03-20T12:00:00Z"
QUESTION = "Will there be no change in Fed interest rates after the July 2026 meeting?"
RULES = "Resolve from the FOMC statement for the July meeting."


def official_html(day, body):
    return (f'<p class="article__time">{day}</p><p class="releaseTime">For release at 2:00 p.m. EDT'
            '<div class="col-xs-12 col-sm-8 col-md-8">'
            '<p>The Federal Open Market Committee approved the following statement.</p>'
            f'<p>{body}</p><p>For media inquiries contact the Board.</p></div>'
            f'<div id="lastUpdate">Last Update: {day}</div>').encode()


def init_html(ancillary=None, tx=TX, qid=QID, adapter=ADAPTER, when=INIT):
    payload = (ancillary or 'q: title: ' + QUESTION + ', description: ' + RULES + ', creator: fixture').encode()
    data = (128).to_bytes(32, 'big') + bytes(96) + len(payload).to_bytes(32, 'big') + payload
    sec = int(timestamp(when, 'time').timestamp())
    return (f'<div>Transaction Hash: {tx}</div><div>Status: Success</div><div>Block: 123456</div>'
            f'<span id="showUtcLocalDate" data-timestamp="{sec}"></span><div>Address {adapter}</div>'
            f'<div>QuestionInitialized (index_topic_1 bytes32 questionID) View Source Topics '
            f'1: questionID Dec Decode Hex {qid[2:]} 2: requestTimestamp Dec Decode Hex {sec} '
            f'Data Dec Hex 0x{data.hex()}</div>').encode()


class HistoricalMarketTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = json.loads((ROOT / 'configs/historical_fomc_collection_v1.json').read_text())
        self.plan['events'] = self.plan['events'][:1]
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(self.plan))
        self.capture = self.root / 'capture'
        self.index = self.root / 'index.json'
        entries = [{'event_id': 'fb:else', 'question': 'An unrelated event'}]
        self.index.write_text(json.dumps({'source': 'forecastbench', 'entries': entries, 'entries_sha256': canonical_hash(entries)}))
        self.market = {'id': '12', 'conditionId': CONDITION, 'questionID': '0x' + 'e' * 64,
                       'negRiskRequestID': QID, 'resolvedBy': ADAPTER, 'question': QUESTION, 'description': RULES,
                       'outcomes': '["Yes", "No"]', 'clobTokenIds': '["123", "456"]',
                       'startDate': INIT, 'closed': True}
        self.state = {'condition_id': CONDITION, 'question_id': QID, 'status': 'resolved', 'extended_review': False,
                      'was_disputed': False, 'new_version_q': False, 'price': '1000000000000000000', 'transaction_hash': TX,
                      'payouts': [1_000_000, 0], 'resolved_at': '2026-07-29T20:00:00Z', 'resolved_block': 500,
                      'resolution_source': 'reported'}
        self.kmarket = {'ticker': 'KXFED-26JUL-T4.00', 'event_ticker': 'KXFED-26JUL', 'market_type': 'binary',
                        'notional_value_dollars': '1', 'title': 'Fed rate above 4%?', 'yes_sub_title': 'Above 4%',
                        'rules_primary': 'Above 4% after July meeting.', 'strike_type': 'greater', 'floor_strike': 4,
                        'open_time': INIT, 'close_time': '2026-07-29T17:55:00Z', 'status': 'finalized', 'result': 'no',
                        'settlement_ts': '2026-07-29T18:21:00Z', 'settlement_value_dollars': '0'}
        self.failed_books = False

    def fetch(self, url):
        p = urlparse(url)
        query = parse_qs(p.query)
        if p.hostname == 'www.federalreserve.gov':
            day = 'June 17, 2026' if '20260617' in url else 'July 29, 2026'
            return 200, official_html(day, 'The Committee decided to maintain the target range for the federal funds rate at 3-1/2 to 3-3/4 percent.'), None
        if p.hostname == 'polygonscan.com':
            return 200, init_html(), None
        if p.path.endswith('.pdf'):
            return 200, b'PDF archived for review; never used as a timestamp assertion', None
        if p.path.endswith('/historical/cutoff'):
            obj = {'market_settled_ts': '2026-07-20T00:00:00Z'}
        elif p.path.endswith('/series/KXFED'):
            obj = {'series': {'ticker': 'KXFED'}}
        elif p.hostname == 'gamma-api.polymarket.com':
            obj = {'id': '287395', 'markets': [self.market]}
        elif p.hostname == 'clob.polymarket.com':
            obj = {'c': CONDITION, 't': [{'o': 'Yes', 't': '123'}, {'o': 'No', 't': '456'}]}
        elif p.path.endswith('/v2/resolutions'):
            obj = {'data': [self.state]}
        elif p.path.endswith('/v2/prices-history'):
            if self.failed_books:
                return 503, b'unavailable', None
            end = int(query['end'][0]) - 1
            obj = {'data': [{'timestamp': end-1800, 'resolution_seconds': 1800, 'price': .6},
                            {'timestamp': end-600, 'resolution_seconds': 1800, 'price': .9},
                            {'timestamp': end, 'resolution_seconds': 0, 'price': 1}],
                   'pagination': {'has_more': False, 'next_cursor': None}}
        elif p.path.endswith('/candlesticks'):
            obj = {'candlesticks': [{'end_period_ts': int(query['end_ts'][0]),
                                   'yes_bid': {'close_dollars': '.20'}, 'yes_ask': {'close_dollars': '.25'}}]}
        else:
            obj = {'event': {'event_ticker': 'KXFED-26JUL'}, 'markets': [self.kmarket]}
        return 200, json.dumps(obj).encode(), None

    def collect(self):
        with patch('foretellmesh.historical_sources.fetch', side_effect=self.fetch), patch(
                'foretellmesh.historical_sources.now', return_value=NOW), patch('foretellmesh.historical_market.now', return_value=NOW):
            return capture_historical_markets(self.config, self.capture)

    def build(self, name='built'):
        return build_historical_markets(self.capture, self.index, self.root / name)

    def test_whole_pipeline_separates_inputs_outcomes_and_unknown_rule_versions(self):
        self.collect()
        result = self.build()
        counts = result['counts']
        self.assertEqual(counts['candidates'], 4)
        self.assertEqual(counts['market_contracts'], 2)
        self.assertEqual(counts['event_groups'], 1)
        self.assertEqual(counts['archive_proven_initial_questions'], 2)
        self.assertEqual(counts['exact_settlement_labels'], 4)
        self.assertEqual(counts['outcomes_cross_checked'], 4)
        self.assertEqual(counts['excluded_zero_width_price_points'], 2)
        self.assertEqual(counts['ready_for_sft'], 0)
        inputs = [json.loads(x) for x in (self.root / 'built/review_inputs.jsonl').read_text().splitlines()]
        self.assertEqual(len(inputs), 2)
        for row in inputs:
            self.assertNotIn('label', row['input'])
            self.assertNotIn('outcome', row['input'])
            self.assertEqual(row['input']['market']['probability'], .6)
            self.assertEqual(len(row['input']['evidence']), 1)
        second = self.build('replay')
        self.assertEqual(second, result)
        for path in (self.root / 'built').iterdir():
            self.assertEqual(path.read_bytes(), (self.root / 'replay' / path.name).read_bytes())

    def test_resolution_outcome_changes_never_change_model_visible_context(self):
        self.collect()
        self.build()
        inputs = (self.root / 'built/review_inputs.jsonl').read_bytes()
        self.state.update(price='0', payouts=[0, 1_000_000])
        self.capture = self.root / 'capture_changed'
        self.collect()
        self.build('changed')
        self.assertEqual(inputs, (self.root / 'changed/review_inputs.jsonl').read_bytes())
        rows = [json.loads(x) for x in (self.root / 'changed/outcomes.jsonl').read_text().splitlines()]
        self.assertTrue(any(x['status'] == 'official_result_disagrees_with_platform' for x in rows))

    def test_old_uma_result_cannot_invent_exact_settlement_time(self):
        self.state.pop('resolved_at')
        self.state.pop('resolved_block')
        self.state.pop('payouts')
        self.state['last_update_timestamp'] = '1785355200'
        self.collect()
        report = self.build()
        self.assertEqual(report['counts']['exact_settlement_labels'], 2)
        self.assertEqual(report['counts']['outcomes_cross_checked'], 4)
        rows = [json.loads(x) for x in (self.root / 'built/candidates.jsonl').read_text().splitlines()]
        self.assertTrue(all(x['record']['label'] is None for x in rows if x['record']['dataset_source'].endswith('polymarket')))

    def test_current_question_change_does_not_pass_initialization_proof(self):
        self.market['description'] += ' New amendment.'
        self.collect()
        report = self.build()
        self.assertEqual(report['counts']['archive_proven_initial_questions'], 0)
        self.assertEqual(report['counts']['review_only_inputs'], 0)

    def test_future_or_date_mismatched_official_source_rejected(self):
        url = 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm'
        body = official_html('June 17, 2026', 'The target range for the federal funds rate at 3-1/2 to 3-3/4 percent.')
        with self.assertRaisesRegex(ValidationError, 'does not match'):
            fed_statement(body, url, '2026-06-17T19:00:00Z')
        revised = body.replace(b'Last Update: June 17, 2026', b'Last Update: July 17, 2026')
        with self.assertRaisesRegex(ValidationError, 'later revision'):
            fed_statement(revised, url, '2026-06-17T18:00:00Z')
        self.plan['events'][0]['evidence'][0]['published_at'] = '2026-08-01T00:00:00Z'
        with self.assertRaisesRegex(ValidationError, 'outcome-time evidence'):
            load_plan(self.plan)

    def test_post_observation_evidence_is_filtered_even_if_before_event(self):
        self.plan['events'][0]['evidence'] = [{'url': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260723a.htm',
                                             'published_at': '2026-07-23T18:00:00Z'}]
        self.config.write_text(json.dumps(self.plan))
        original = self.fetch
        def modified(url):
            if '20260723' in url:
                return 200, official_html('July 23, 2026', 'The target range for the federal funds rate at 3-1/2 to 3-3/4 percent.'), None
            return original(url)
        with patch('foretellmesh.historical_sources.fetch', side_effect=modified), patch(
                'foretellmesh.historical_sources.now', return_value=NOW), patch('foretellmesh.historical_market.now', return_value=NOW):
            capture_historical_markets(self.config, self.capture)
        self.build()
        rows = [json.loads(x) for x in (self.root / 'built/candidates.jsonl').read_text().splitlines()]
        for row in rows:
            if row['record']['observation_time'].startswith('2026-07-22'):
                self.assertEqual(row['record']['evidence'], [])

    def test_public_source_failures_are_archived_and_not_silently_filled(self):
        self.failed_books = True
        report = self.collect()
        self.assertEqual(len(report['failed_requests']), 2)
        replay = self.build()
        self.assertEqual(replay['counts']['historical_quotes'], 2)
        self.assertEqual(replay['counts']['review_only_inputs'], 0)

    def test_initialization_receipt_checks_identity_success_adapter_and_time(self):
        good = initialized_question(init_html(), tx_hash=TX, request_id=QID, adapter=ADAPTER)
        self.assertEqual(good['published_at'], INIT)
        for changes in ({'tx_hash': '0x'+'f'*64}, {'request_id': '0x'+'f'*64}, {'adapter': '0x'+'f'*40}):
            args = {'tx_hash': TX, 'request_id': QID, 'adapter': ADAPTER, **changes}
            with self.assertRaises(ValidationError):
                initialized_question(init_html(), **args)
        with self.assertRaises(ValidationError):
            initialized_question(init_html().replace(b'Status: Success', b'Status: Failed'), tx_hash=TX, request_id=QID, adapter=ADAPTER)

    def test_archive_tamper_and_overwrite_rejected(self):
        self.collect()
        with self.assertRaisesRegex(ValidationError, 'already exists'):
            self.collect()
        path = self.capture / 'raw/0000.bin'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValidationError, 'hash mismatch'):
            read_archive(self.capture)

    def test_price_zero_width_and_straddling_intervals_never_leak(self):
        t = timestamp('2026-07-22T18:00:00Z', 't')
        sec = int(t.timestamp())
        page = {'data': [{'timestamp': sec-1800, 'resolution_seconds': 1800, 'price': .4},
                         {'timestamp': sec-100, 'resolution_seconds': 1800, 'price': .99},
                         {'timestamp': sec, 'resolution_seconds': 0, 'price': 1}],
                'pagination': {'has_more': False}}
        quote, quality = poly_history_quote([page], t, self.plan)
        self.assertEqual(quote['probability'], .4)
        self.assertEqual(quality['excluded_zero_width_points'], 1)
        self.assertEqual(quality['excluded_future_or_straddling_points'], 1)
        page['data'] = page['data'][1:]
        self.assertIsNone(poly_history_quote([page], t, self.plan)[0])

    def test_price_truncation_conflicts_boolean_fields_and_staleness(self):
        t = timestamp('2026-07-22T18:00:00Z', 't')
        sec = int(t.timestamp())
        point = {'timestamp': sec-1800, 'resolution_seconds': 1800, 'price': .4}
        page = {'data': [point], 'pagination': {'has_more': True}}
        self.assertIsNone(poly_history_quote([page], t, self.plan)[0])
        page['pagination']['has_more'] = False
        page['data'].append({**point, 'price': .5})
        with self.assertRaisesRegex(ValidationError, 'conflicting'):
            poly_history_quote([page], t, self.plan)
        page['data'] = [{**point, 'resolution_seconds': True}]
        with self.assertRaises(ValidationError):
            poly_history_quote([page], t, self.plan)
        page['data'] = [{**point, 'timestamp': sec-15000}]
        self.assertIsNone(poly_history_quote([page], t, self.plan)[0])

    def test_kalshi_future_candle_and_crossed_quote_rejected(self):
        t = timestamp('2026-07-22T18:00:00Z', 't')
        point = {'end_period_ts': int(t.timestamp()) + 1, 'yes_bid': {'close_dollars': '.4'}, 'yes_ask': {'close_dollars': '.5'}}
        self.assertIsNone(kalshi_history_quote({'candlesticks': [point]}, t, self.plan)[0])
        point['end_period_ts'] -= 1
        point['yes_bid']['close_dollars'] = '.6'
        with self.assertRaises(ValidationError):
            kalshi_history_quote({'candlesticks': [point]}, t, self.plan)

    def test_official_rate_math_and_no_nonbinary_label_shortcut(self):
        self.assertEqual(fed_upper_bound('The target range for the federal funds rate at 3-1/2 to 3-3/4 percent.'), Fraction(15, 4))
        state = {**self.state, 'payouts': [500_000, 500_000], 'price': '500000000000000000'}
        label, ledger = historical_poly_label(self.market, state, NOW, timestamp(INIT, 't'), 1)
        self.assertIsNone(label)
        self.assertEqual(ledger['status'], 'non_binary_or_missing_payout')

    def test_payout_conflicts_cannot_fall_back_to_binary_oracle_price(self):
        for payout in ([500_000, 500_000], [False, 1_000_000], [1_000_000], '1000000'):
            state = {**self.state, 'payouts': payout, 'price': '1000000000000000000'}
            label, ledger = historical_poly_label(self.market, state, NOW, timestamp(INIT, 't'), 1)
            self.assertIsNone(label)
            self.assertIsNone(ledger['outcome'])
        state = {**self.state, 'payouts': [1_000_000, 0], 'price': '0'}
        label, ledger = historical_poly_label(self.market, state, NOW, timestamp(INIT, 't'), 1)
        self.assertIsNone(label)
        self.assertEqual(ledger['status'], 'conflicting_platform_resolution_fields')

    def test_native_title_is_checked_against_heldout_questions(self):
        index = json.loads(self.index.read_text())
        index['entries'] = [{'event_id': 'different-native-id', 'question': QUESTION}]
        index['entries_sha256'] = canonical_hash(index['entries'])
        self.index.write_text(json.dumps(index))
        self.collect()
        report = self.build()
        self.assertEqual(report['counts']['blockers']['exact_heldout_overlap'], 2)

    def test_read_only_url_scope(self):
        self.assertFalse(allowed_url('https://clob.polymarket.com/order'))
        self.assertFalse(allowed_url('https://gamma-api.polymarket.com.evil.test/events/1'))
        self.assertFalse(allowed_url('https://polygonscan.com/tx/not-a-hash'))
        self.assertFalse(allowed_url('https://external-api.kalshi.com/trade-api/v2/portfolio/orders'))


if __name__ == '__main__':
    unittest.main()
