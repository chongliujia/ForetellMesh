from datetime import timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_bytes, sha256_file
from foretellmesh.pma_proof_pilot import capture, inputs
from foretellmesh.pma_proof_replay import (build, audit_contract, expected_outcome, observation,
                                         REVIEW_ADAPTER, REVIEW_OPERATOR)
from foretellmesh.polygon_proof_audit import INITIALIZED, GET_UPDATES, GET_QUESTION, EMPTY_UPDATES
from foretellmesh.polygon_settlement_audit import CTF, CTF_RESOLVED, UMA_RESOLVED, NEG_RISK, REPORTED, DENOMINATOR, NUMERATOR
from foretellmesh.schema import ValidationError, timestamp, iso
from foretellmesh.synthetic_sft import canonical_hash
from test_historical_market import official_html

QID, NATIVE, CONDITION, TX = ['0x'+c*64 for c in 'abcd']
CREATOR = '0x'+'e'*40
QUESTION = 'No change in Fed interest rates after May 2025 meeting?'
RULES = 'Resolve from the upper bound in the May 2025 FOMC statement.'
RELEASE = '2025-05-07T18:00:00Z'
PRIOR = '2025-03-19T18:00:00Z'
RATE = 'The Committee maintained the target range for the federal funds rate at 4-1/4 to 4-1/2 percent.'


def words(*values):
    return '0x'+''.join(f'{n:064x}' for n in values)


def abi_text(value):
    raw = value.encode()
    return f'{len(raw):064x}'+raw.hex()+'00'*((-len(raw)) % 32)


class PmaProofPilotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.market = {'id': '1', 'question': QUESTION, 'description': RULES, 'conditionId': CONDITION,
            'questionID': NATIVE, 'negRiskRequestID': QID, 'resolvedBy': REVIEW_ADAPTER, 'negRisk': True,
            'outcomes': '["Yes","No"]', 'clobTokenIds': '["11","22"]', 'events': [{'id': '2'}]}
        self.state = {'question_id': QID, 'condition_id': CONDITION, 'status': 'resolved', 'was_disputed': False,
            'extended_review': False, 'transaction_hash': TX, 'last_update_timestamp': str(int(timestamp(RELEASE, 'release').timestamp())+7200),
            'price': str(10**18)}
        self.ancillary = ('q: title: '+QUESTION+', description: '+RULES+', res_data: p1: 0, p2: 1. '
            f'Updates made by the question creator via the bulletin board at {REVIEW_ADAPTER} should be considered.,initializer:'+CREATOR[2:])
        self.group = {'event_group_id': 'macro:us:fomc:2025-05', 'period': '2025-05', 'native_event_id': '2',
            'group_rationale': 'same scheduled release',
            'release': {'url': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20250507a.htm', 'published_at': RELEASE},
            'prior_release': {'url': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm', 'published_at': PRIOR}}
        catalog = self.root/'catalog'; catalog.mkdir()
        candidate = {'market_id': '1', 'condition_id': CONDITION, 'question': QUESTION, 'tokens': {'No': '22', 'Yes': '11'}, 'benchmark_matches': []}
        self.dump(catalog/'macro_candidates.jsonl', candidate)
        self.dump(catalog/'report.json', {'artifact_hashes': {'macro_candidates.jsonl': sha256_file(catalog/'macro_candidates.jsonl')}})
        native = self.root/'native'; native.mkdir(); self.dump(native/'raw.bin', [self.market])
        refs = [{'url': 'https://gamma-api.polymarket.com/markets?id=1&closed=true', 'status': 200, 'file': 'raw.bin', 'sha256': sha256_file(native/'raw.bin')}]
        self.dump(native/'manifest.json', {'requests': refs, 'requests_sha256': canonical_hash(refs)})
        prices = self.root/'prices'; prices.mkdir()
        with sqlite3.connect(prices/'prices.sqlite') as db:
            db.executescript('CREATE TABLE trades(market_id TEXT, block_number INTEGER, numerator TEXT, denominator TEXT);'
                             'CREATE TABLE blocks(block_number INTEGER, unix_time INTEGER);')
            for n, days in enumerate((7, 1)):
                epoch = int((timestamp(RELEASE, 'release')-timedelta(days=days, hours=1)).timestamp())
                db.execute('INSERT INTO blocks VALUES (?,?)', (n, epoch))
                db.execute('INSERT INTO trades VALUES (?,?,?,?)', ('1', n, '3', '4'))
        self.dump(prices/'report.json', {'catalog_report_sha256': sha256_file(catalog/'report.json'),
                                       'artifact_hashes': {'prices.sqlite': sha256_file(prices/'prices.sqlite')}})
        entries = [{'event_id': 'unrelated:1', 'question': 'A different event?'}]
        self.dump(self.root/'heldout.json', {'schema_version': '1', 'source': 'forecastbench', 'entries': entries, 'entries_sha256': canonical_hash(entries)})
        source = self.root/'source'; source.mkdir()
        sources = {'fixture.sol': {'content': '// Synthetic reviewed-source fixture.'}}
        encoded = json.dumps(json.dumps({'sources': sources}))[1:-1]
        page = f"var editor_contractJsonData = '{encoded}'\nDeployed Bytecode<div>0x6000</div>"
        (source/'source.html').write_text(page)
        source_url = 'https://polygonscan.com/address/'+REVIEW_ADAPTER+'#code'
        self.dump(source/'manifest.json', {'requests': [{'url': source_url, 'status': 200, 'request': None,
            'file': 'source.html', 'sha256': sha256_file(source/'source.html')}]})
        self.dump(self.root/'rules.json', {'schema_version': '1', 'review_id': 'pma_uma_2f5e_rules_v1', 'adapter': REVIEW_ADAPTER,
            'source_url': source_url, 'source_page_sha256': sha256_file(source/'source.html'), 'runtime_sha256': sha256_bytes(bytes.fromhex('6000')),
            'source_sha256': {k: sha256_bytes(v['content'].encode()) for k, v in sources.items()},
            'reviewed_properties': ['immutable_initialized_ancillary', 'append_only_creator_updates', 'no_proxy_delegatecall_or_selfdestruct'],
            'trust_model': 'polygonscan_source_verification_plus_single_public_rpc'})
        source_paths = {'catalog': catalog/'report.json', 'enrichment': native/'manifest.json', 'price_store': prices/'report.json',
                        'heldout_index': self.root/'heldout.json', 'rules_policy': self.root/'rules.json', 'rules_source': source/'manifest.json'}
        self.config = self.root/'config.json'
        self.plan = {'schema_version': '1', 'purpose': 'pma_historical_proof_pilot', 'training': False, 'formal_evaluation': False,
            'split': None, 'observation_days_before': [7, 1], 'max_price_age_seconds': 10800, 'workers': 4, 'max_markets': 20,
            'max_rpc_requests_per_market': 160, 'groups': [self.group],
            'sources': {k: {'path': str(p.relative_to(self.root)), 'sha256': sha256_file(p)} for k, p in source_paths.items()}}
        self.dump(self.config, self.plan)
        self.raw = self.root/'capture'
        with patch('foretellmesh.pma_proof_pilot.fetched', side_effect=self.fetch), patch('foretellmesh.polygon_capture.urlopen', side_effect=self.rpc):
            with patch('builtins.print'):
                capture(self.config, self.raw, 'https://polygon.drpc.org')
        self.row = inputs(self.config)[2][0]
        self.rpc_root = self.raw/'rpc'/'1'

    def dump(self, path, obj):
        path.write_text(json.dumps(obj)+'\n')

    def fetch(self, url):
        if '/events/' in url: obj = {'id': '2', 'markets': [self.market]}
        elif '/resolutions?' in url: obj = {'data': [self.state]}
        elif '/clob-markets/' in url: obj = {'c': CONDITION, 't': [{'o': 'Yes', 't': '11'}, {'o': 'No', 't': '22'}]}
        else:
            day = 'May 07, 2025' if '20250507' in url else 'March 19, 2025'
            obj = official_html(day, RATE)
        raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        return {'url': url, 'status': 200, 'sha256': sha256_bytes(raw), 'started_at': '2026-09-19T00:00:00Z',
                'completed_at': '2026-09-19T00:00:01Z', 'server_date': None}, raw

    def block(self, n):
        epoch = int(timestamp('2025-01-01T00:00:00Z', 'init').timestamp()) if n == 100 else int(self.state['last_update_timestamp'])+(n-200)*2
        tx = TX if n == 100 else words(n+1000)
        return {'number': hex(n), 'hash': words(n), 'timestamp': hex(epoch), 'transactions': [tx]}

    def logs(self, n):
        b = self.block(n); common = {'blockNumber': b['number'], 'blockHash': b['hash'], 'blockTimestamp': b['timestamp'],
            'transactionHash': b['transactions'][0], 'transactionIndex': '0x0', 'removed': False}
        if n == 100:
            return [{**common, 'address': REVIEW_ADAPTER, 'topics': [INITIALIZED, QID, words(int(b['timestamp'], 16)), words(int(CREATOR, 16))],
                     'data': words(128, 0, 0, 0)+abi_text(self.ancillary), 'logIndex': '0x0'}]
        if n == 200:
            return [{**common, 'address': REVIEW_ADAPTER, 'topics': [UMA_RESOLVED, QID, words(10**18)], 'data': words(32, 2, 1, 0), 'logIndex': '0x0'},
                    {**common, 'address': REVIEW_OPERATOR, 'topics': [REPORTED, NATIVE], 'data': QID+f'{1:064x}', 'logIndex': '0x1'}]
        if n == 210:
            return [{**common, 'address': CTF, 'topics': [CTF_RESOLVED, CONDITION, words(int(NEG_RISK, 16)), NATIVE],
                     'data': words(2, 64, 2, 1, 0), 'logIndex': '0x0'}]
        return []

    def rpc(self, request, timeout):
        req = json.loads(request.data); method, params = req['method'], req['params']
        if method == 'eth_chainId': result = '0x89'
        elif method == 'eth_getBlockByNumber': result = self.block(300 if params[0] == 'finalized' else int(params[0], 16))
        elif method == 'eth_getTransactionReceipt':
            n = 100 if params[0] == TX else int(params[0], 16)-1000; b = self.block(n)
            result = {'status': '0x1', 'blockNumber': b['number'], 'blockHash': b['hash'], 'transactionHash': params[0],
                      'transactionIndex': '0x0', 'logs': self.logs(n)}
        elif method == 'eth_getLogs': result = [l for l in self.logs(int(params[0]['blockHash'], 16)) if l['address'] == params[0]['address']]
        elif method == 'eth_getCode': result = '0x6000'
        elif method == 'eth_call':
            data = params[0]['data']; n = int(params[1], 16)
            if data.startswith(DENOMINATOR): result = words(int(n >= 210))
            elif data.startswith(NUMERATOR): result = words(int(int(data[-64:], 16) == 0))
            elif data.startswith(GET_UPDATES): result = EMPTY_UPDATES
            elif data.startswith(GET_QUESTION):
                result = words(32, int(self.block(100)['timestamp'], 16), 0, 0, 0, 0, 1, 0, 0, 0, 0, int(CREATOR, 16), 384)+abi_text(self.ancillary)
            else: raise AssertionError(data)
        else: raise AssertionError(method)
        raw = json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}).encode()
        class Response:
            url = request.full_url; status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return raw[:size]
        return Response()

    def audit(self):
        return audit_contract(self.rpc_root, self.row, self.state, '0x6000')

    def mutate_response(self, ref, mutate):
        """Rehash a forged provider response to test semantics beyond file hashing."""
        path = self.rpc_root/ref['file']; obj = json.loads(path.read_text()); mutate(obj); self.dump(path, obj)
        new = {**ref, 'sha256': sha256_file(path)}
        def replace(value):
            if isinstance(value, dict): return new if value == ref else {k: replace(v) for k, v in value.items()}
            if isinstance(value, list): return [replace(v) for v in value]
            return value
        for name in ('manifest.json', 'proof.json'):
            path = self.rpc_root/name; self.dump(path, replace(json.loads(path.read_text())))

    def test_capture_to_replay_is_reproducible_and_labels_are_isolated(self):
        a, b = self.root/'a', self.root/'b'
        report = build(self.config, self.raw, a); second = build(self.config, self.raw, b)
        self.assertEqual(report['artifact_hashes'], second['artifact_hashes'])
        self.assertEqual((report['verified_rule_and_settlement_contracts'], report['proof_complete_observations']), (1, 2))
        self.assertEqual(report['admitted_training_rows'], 0)
        rows = [json.loads(line) for line in (a/'review_inputs.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertEqual(set(row['input']), {'question', 'observation_time', 'evidence', 'market'})
            self.assertNotIn('resolution_time', json.dumps(row['input']))
            self.assertEqual(row['input']['evidence'][0]['published_at'], PRIOR)
        self.assertEqual(self.audit()['resolution_time'], iso(timestamp(RELEASE, 'release')+timedelta(hours=2, seconds=20)))

    def test_changed_current_rules_are_quarantined(self):
        self.row['market']['description'] += ' Updated date.'
        with self.assertRaisesRegex(ValidationError, 'rules differ'): self.audit()

    def replace_ancillary(self, value):
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['initial_receipt'], lambda obj: obj['result']['logs'][0].update(
            data=words(128, 0, 0, 0)+abi_text(value)))
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['question_state'], lambda obj: obj.update(
            result=words(32, int(self.block(100)['timestamp'], 16), 0, 0, 0, 0, 1, 0, 0, 0, 0, int(CREATOR, 16), 384)+abi_text(value)))

    def test_legacy_and_native_id_rule_boundaries_are_bound(self):
        for boundary in (', res_data: ', ' res_data: ', ' market_id: 1 res_data: '):
            with self.subTest(boundary=boundary):
                self.replace_ancillary(self.ancillary.replace(', res_data: ', boundary))
                self.assertEqual(self.audit()['historical_question'], 'q: title: '+QUESTION+', description: '+RULES)

    def test_rule_suffix_cannot_hide_amendments_or_wrong_market_id(self):
        for boundary in (' market_id: 2 res_data: ', ' market_id: 10 res_data: ',
                         ' A changed decision date. res_data: '):
            with self.subTest(boundary=boundary):
                self.replace_ancillary(self.ancillary.replace(', res_data: ', boundary))
                with self.assertRaisesRegex(ValidationError, 'rules differ'): self.audit()

    def test_nonempty_updates_are_rejected_after_rehash(self):
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['creator_updates'], lambda obj: obj.update(result=words(32, 1)))
        with self.assertRaisesRegex(ValidationError, 'nonempty'): self.audit()

    def test_runtime_cannot_be_substituted(self):
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['code'][0], lambda obj: obj.update(result='0x6001'))
        with self.assertRaisesRegex(ValidationError, 'runtime'): self.audit()

    def test_settlement_boundary_request_cannot_be_substituted(self):
        path = self.rpc_root/'proof.json'; proof = json.loads(path.read_text())
        proof['ctf']['previous_denominator'] = proof['ctf']['resolved_denominator']; self.dump(path, proof)
        with self.assertRaisesRegex(ValidationError, 'bind'): self.audit()

    def test_wrong_operator_cannot_supply_native_question_binding(self):
        proof = json.loads((self.rpc_root/'proof.json').read_text()); ref = proof['oracle']['matches'][0]['receipt']
        self.mutate_response(ref, lambda obj: obj['result']['logs'][1].update(address='0x'+'1'*40))
        with self.assertRaisesRegex(ValidationError, 'NegRisk report'): self.audit()

    def test_failed_initial_receipt_fails_closed(self):
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['initial_receipt'], lambda obj: obj['result'].update(status='0x0'))
        with self.assertRaisesRegex(ValidationError, 'failed receipt'): self.audit()

    def test_observation_cutoff_and_official_crosscheck(self):
        proof = self.audit(); prior = {'text': RATE, 'source': self.group['prior_release']['url'], 'published_at': PRIOR}
        official = {'text': RATE, 'published_at': RELEASE}
        args = [self.row, self.group, proof, prior, official, None, '2025-05-06T18:00:00Z', 'fixture']
        observation(*args)
        for bad in (RELEASE, '2024-12-31T00:00:00Z'):
            with self.subTest(bad=bad), self.assertRaises(ValidationError): observation(*args[:6], bad, 'fixture')
        args[3] = {**prior, 'published_at': '2025-05-07T17:00:00Z'}
        with self.assertRaisesRegex(ValidationError, 'evidence'): observation(*args)
        args[3] = prior; args[2] = {**proof, 'outcome': 0}
        with self.assertRaisesRegex(ValidationError, 'official'): observation(*args)

    def test_future_price_or_future_rule_observation_is_rejected(self):
        proof = self.audit(); prior = {'text': RATE, 'source': 'fixture', 'published_at': PRIOR}
        official = {'text': RATE, 'published_at': RELEASE}; at = '2025-05-06T18:00:00Z'
        quote = {'probability': 0.8, 'unix_time': int(timestamp(RELEASE, 'release').timestamp())}
        with self.assertRaisesRegex(ValidationError, 'market'): observation(self.row, self.group, proof, prior, official, quote, at, 'fixture')
        proof['rule_valid_through'] = '2025-05-01T18:00:00Z'
        with self.assertRaisesRegex(ValidationError, 'future observation'): observation(self.row, self.group, proof, prior, official, None, at, 'fixture')

    def test_conflicting_payout_state_cannot_become_a_label(self):
        proof = json.loads((self.rpc_root/'proof.json').read_text())
        self.mutate_response(proof['ctf']['numerators'][0], lambda obj: obj.update(result=words(2)))
        with self.assertRaisesRegex(ValidationError, 'payouts conflict'): self.audit()

    def test_missing_prices_remain_in_coverage_without_admission(self):
        price_root = self.root/'prices'
        with sqlite3.connect(price_root/'prices.sqlite') as db: db.execute('DELETE FROM trades')
        price_report = json.loads((price_root/'report.json').read_text())
        price_report['artifact_hashes']['prices.sqlite'] = sha256_file(price_root/'prices.sqlite'); self.dump(price_root/'report.json', price_report)
        self.plan['sources']['price_store']['sha256'] = sha256_file(price_root/'report.json'); self.dump(self.config, self.plan)
        selection = json.loads((self.raw/'selection.json').read_text()); selection.update(config=self.plan, config_sha256=sha256_file(self.config))
        self.dump(self.raw/'selection.json', selection)
        report = json.loads((self.raw/'report.json').read_text()); report['selection_sha256'] = sha256_file(self.raw/'selection.json'); self.dump(self.raw/'report.json', report)
        report = build(self.config, self.raw, self.root/'missing')
        self.assertEqual((report['planned_observations'], report['verified_observation_labels'], report['proof_complete_observations']), (2, 2, 0))
        self.assertEqual(report['data_blocker_counts'], {'historical_quote_unavailable:no_pre_cutoff_trade': 2})

    def test_incomplete_capture_is_kept_in_denominator(self):
        index_path = self.raw/'index.json'; self.dump(index_path, [{'market_id': '1', 'status': 'capture_incomplete', 'error': 'unavailable'}])
        report = json.loads((self.raw/'report.json').read_text()); report['index_sha256'] = sha256_file(index_path); self.dump(self.raw/'report.json', report)
        report = build(self.config, self.raw, self.root/'incomplete')
        self.assertEqual((report['selected_contracts'], report['planned_observations'], report['proof_complete_observations']), (1, 2, 0))

    def test_group_period_and_off_grid_changes_are_not_silently_accepted(self):
        with self.assertRaises(ValidationError): expected_outcome(QUESTION, '2025-06', 0)
        with self.assertRaises(ValidationError): expected_outcome('Fed decreases interest rates by 25 bps after May 2025 meeting?', '2025-05', -12.5)
        self.assertEqual(expected_outcome('Fed decreases interest rates by 50+ bps after May 2025 meeting?', '2025-05', -50), 1)

    def test_legacy_fomc_titles_preserve_direction_threshold_and_period(self):
        examples = [
            ('Will the Fed decrease interest rates by 50+ bps after its March 2024 meeting?', '2024-03', -50, 1),
            ('No change in Fed raise interest rates after its 2024 March meeting?', '2024-03', 0, 1),
            ('Fed raises interest rates by 25+ bps after 2024 May meeting?', '2024-05', -25, 0),
            ('Fed decreases interest rates by 50 bps after November 2024 meeting?', '2024-11', -75, 0),
            ('Fed decreases interest rates by 75+ bps after December 2024 meeting?', '2024-12', -100, 1),
            ('Will the FED change rates to another level after Nov meeting?', '2024-11', -25, 0),
        ]
        for question, period, change, expected in examples:
            with self.subTest(question=question): self.assertEqual(expected_outcome(question, period, change), expected)
        with self.assertRaises(ValidationError): expected_outcome(examples[-1][0], '2025-11', 0)
        with self.assertRaises(ValidationError): expected_outcome(examples[1][0], '2024-03', 12.5)

    def test_frozen_source_and_capture_coverage_cannot_be_changed(self):
        path = self.root/'native'/'raw.bin'; path.write_text('[]')
        with self.assertRaisesRegex(ValidationError, 'native response'): inputs(self.config)

    def test_raw_rpc_tampering_is_rejected(self):
        path = self.rpc_root/'000.bin'; path.write_text('{}')
        with self.assertRaisesRegex(ValidationError, 'provenance'): self.audit()


if __name__ == '__main__':
    unittest.main()
