from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_historical_market as historical
from foretellmesh.data import sha256_bytes, sha256_file
from foretellmesh.historical_chain_supplement import apply_bundle, build
from foretellmesh.historical_replay_gate import preflight, verify_staging
from foretellmesh.polygon_capture import capture, RpcArchive
from foretellmesh.polygon_proof_audit import EMPTY_UPDATES, GET_QUESTION, GET_UPDATES, INITIALIZED
from foretellmesh.polygon_rules_audit import audit_rules, extract_source
from foretellmesh.polygon_settlement_audit import (
    CTF, CTF_RESOLVED, DENOMINATOR, NUMERATOR, UMA, UMA_RESOLVED, OPERATOR, NEG_RISK, REPORTED,
    audit_settlements, binary_payout,
)
from foretellmesh.polygon_settlement_capture import capture_ctf, capture_oracle
from foretellmesh.schema import ValidationError, parse_record, timestamp

ROOT = Path(__file__).resolve().parents[1]
QID, CONDITION = historical.QID, historical.CONDITION
NATIVE = '0x'+'e'*64
CREATOR = '0x'+'a'*40
REPORT_BLOCK, FINAL_BLOCK, PINNED = 200000, 200010, 300000


def uint(n):return n.to_bytes(32, 'big')
def words(*values):return '0x'+b''.join(uint(n) for n in values).hex()
def bytes_text(s):
    raw = s.encode(); return uint(len(raw))+raw+bytes((-len(raw))%32)


class PolygonSettlementTests(unittest.TestCase):
    """Synthetic chain fixtures exercise the complete capture -> replay path."""
    def setUp(self):
        self.fixture = historical.HistoricalMarketTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups); self.root = self.fixture.root
        self.fixture.market.update(resolvedBy=UMA, negRisk=True)
        self.fixture.state['new_version_q'] = True
        for key in ('payouts', 'resolved_at', 'resolved_block', 'resolution_source'):self.fixture.state.pop(key)
        self.ancillary = ('q: title: '+historical.QUESTION+', description: '+historical.RULES
            +f' Updates made by the question creator via the bulletin board at {UMA} should be considered.,initializer:'+CREATOR[2:])
        self.fetch = self.fixture.fetch
        self.fixture.fetch = lambda url: (200, historical.init_html(self.ancillary, adapter=UMA), None) if 'polygonscan.com' in url else self.fetch(url)
        self.fixture.collect(); self.fixture.build(); self.staging = self.root/'built'
        _, self.candidates = verify_staging(self.staging, self.fixture.capture, self.fixture.index)
        self.initial, self.oracle, self.ctf, self.code, self.explorer = [self.root/n for n in ('initial', 'oracle', 'ctf', 'code', 'explorer')]
        self.plan = self.root/'oracle-plan.json'
        self.plan.write_text(json.dumps({'schema_version': '1', 'questions': [{'question_id': QID, 'condition_id': CONDITION, 'block_number': REPORT_BLOCK}]}))
        with patch('foretellmesh.polygon_capture.urlopen', side_effect=self.respond):
            capture(self.staging, self.initial, 'https://polygon.drpc.org')
            capture_oracle(self.plan, self.oracle, 'https://polygon.drpc.org')
            capture_ctf(self.oracle, self.ctf, 'https://polygon.drpc.org')
            arch = RpcArchive(self.code, 'https://polygon.drpc.org')
            for block in (123456, PINNED):arch.read('eth_getCode', [UMA, hex(block)])
        sources = {'fixture.sol': {'content': '// Synthetic source-verification fixture.'}}
        encoded = json.dumps(json.dumps({'sources': sources}))[1:-1]
        page = f"var editor_contractJsonData = '{encoded}'\nDeployed Bytecode<div>0x6000</div>"
        self.explorer.mkdir(); (self.explorer/'source.html').write_text(page)
        policy = json.loads((ROOT/'configs/polygon_rules_review_v1.json').read_text())
        policy.update(source_page_sha256=sha256_bytes(page.encode()), runtime_sha256=sha256_bytes(bytes.fromhex('6000')),
                      source_sha256={n: sha256_bytes(v['content'].encode()) for n, v in sources.items()})
        self.policy = self.root/'review.json'; self.policy.write_text(json.dumps(policy))
        (self.explorer/'manifest.json').write_text(json.dumps({'kind': 'settlement_research', 'requests': [{
            'url': policy['source_url'], 'request': None, 'status': 200, 'file': 'source.html',
            'sha256': policy['source_page_sha256'], 'started_at': historical.NOW, 'completed_at': historical.NOW}]}))
        self.bundle = self.root/'bundle.json'
        self.bundle.write_text(json.dumps(dict(schema_version='1', oracle_archive='oracle', ctf_archive='ctf',
            initial_archive='initial', code_archive='code', explorer_archive='explorer', rules_policy='review.json')))

    def block(self, n):
        when = ('2026-03-20T12:00:00Z' if n == 123456 else '2026-09-18T08:00:00Z' if n == PINNED else '2026-07-29T20:00:00Z')
        epoch = int(timestamp(when, 'fixture').timestamp())+(0 if n in (123456, PINNED) else (n-REPORT_BLOCK)*2)
        tx = historical.TX if n == 123456 else '0x'+f'{n+1000000:064x}'
        return {'number': hex(n), 'hash': '0x'+f'{n:064x}', 'timestamp': hex(epoch), 'transactions': [tx]}

    def logs(self, n):
        b = self.block(n)
        common = dict(blockNumber=b['number'], blockHash=b['hash'], blockTimestamp=b['timestamp'],
                      transactionHash=b['transactions'][0], transactionIndex='0x0', removed=False)
        if n == 123456:
            data = '0x'+(uint(128)+bytes(96)+bytes_text(self.ancillary)).hex()
            return [dict(common, address=UMA, topics=[INITIALIZED, QID, words(int(timestamp(historical.INIT, 'init').timestamp())), words(int(CREATOR, 16))], data=data, logIndex='0x0')]
        if n == REPORT_BLOCK:
            return [dict(common, address=UMA, topics=[UMA_RESOLVED, QID, words(10**18)], data=words(32, 2, 1, 0), logIndex='0x0'),
                    dict(common, address=OPERATOR, topics=[REPORTED, NATIVE], data=QID+f'{1:064x}', logIndex='0x1')]
        if n == FINAL_BLOCK:
            return [dict(common, address=CTF, topics=[CTF_RESOLVED, CONDITION, words(int(NEG_RISK, 16)), NATIVE], data=words(2, 64, 2, 1, 0), logIndex='0x0')]
        return []

    def respond(self, request, timeout):
        obj = json.loads(request.data); method, params = obj['method'], obj['params']
        if method == 'eth_chainId':result = '0x89'
        elif method == 'eth_getBlockByNumber':result = self.block(PINNED if params[0] == 'finalized' else int(params[0], 16))
        elif method == 'eth_getTransactionReceipt':
            n = 123456 if params[0] == historical.TX else int(params[0], 16)-1000000
            b = self.block(n); result = dict(status='0x1', blockNumber=b['number'], blockHash=b['hash'], transactionHash=params[0], transactionIndex='0x0', logs=self.logs(n))
        elif method == 'eth_getLogs':
            result = [l for l in self.logs(int(params[0]['blockHash'], 16)) if l['address'] == params[0]['address']]
        elif method == 'eth_getCode':result = '0x6000'
        elif method == 'eth_call':
            data = params[0]['data']; n = int(params[1], 16)
            if data.startswith(DENOMINATOR):result = words(int(n >= FINAL_BLOCK))
            elif data.startswith(NUMERATOR):result = words(int(int(data[-64:], 16) == 0))
            elif data.startswith(GET_UPDATES):result = EMPTY_UPDATES
            elif data.startswith(GET_QUESTION):
                head = [32, int(timestamp(historical.INIT, 'init').timestamp()), 0, 0, 0, 0, 1, 0, 0, 0, 0, int(CREATOR, 16), 384]
                result = '0x'+(b''.join(uint(x) for x in head)+bytes_text(self.ancillary)).hex()
            else:raise AssertionError(data)
        else:raise AssertionError(method)
        raw = json.dumps({'jsonrpc': '2.0', 'id': obj['id'], 'result': result}).encode()
        class Response:
            url = request.full_url; status = 200
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def read(self, n):return raw[:n]
        return Response()

    def settlement_audit(self):
        return audit_settlements(self.staging, self.fixture.capture, self.candidates, self.oracle, self.ctf)

    def rules_audit(self):
        return audit_rules(self.staging, self.candidates, self.initial, self.code, self.explorer, self.policy)

    def mutate(self, root, ref, change):
        """Rewrite hashes too: tests must check semantics, not only byte corruption."""
        obj = json.loads((root/ref['file']).read_text()); change(obj['result'])
        (root/ref['file']).write_text(json.dumps(obj)); updated = dict(ref, sha256=sha256_file(root/ref['file']))
        def replace(value):
            if value == ref:return updated
            if isinstance(value, dict):return {k: replace(v) for k, v in value.items()}
            if isinstance(value, list):return [replace(v) for v in value]
            return value
        for name in ('manifest.json', 'index.json'):
            if (root/name).exists():(root/name).write_text(json.dumps(replace(json.loads((root/name).read_text()))))

    def test_final_ctf_time_is_used_and_input_payloads_are_unchanged(self):
        report = self.settlement_audit()
        self.assertEqual(report['contracts'], 1); self.assertEqual(report['observation_labels'], 2)
        self.assertEqual(report['rows'][0]['report_to_settlement_seconds'], 20)
        self.assertEqual({x['resolution_time'] for x in report['labels'].values()}, {'2026-07-29T20:00:20Z'})
        rows, audit = apply_bundle(self.staging, self.fixture.capture, self.candidates, self.bundle)
        self.assertEqual(audit['exact_settled_rows'], 4)
        for old, new in zip(self.candidates, rows):
            self.assertEqual(parse_record(old['record']).forecast_input.to_payload(), parse_record(new['record']).forecast_input.to_payload())
            self.assertIn('semantic_benchmark_review_required', new['blockers'])
            self.assertIn('historical_teacher_target_missing', new['blockers'])
            self.assertFalse(new['ready_for_benchmark'])
        gate = preflight(self.staging, self.fixture.capture, self.fixture.index, ROOT/'configs/historical_replay_admission_v1.json', self.bundle)
        self.assertEqual(gate['eligible_replay_rows'], 0); self.assertEqual(gate['exact_settled_rows'], 4)
        self.assertIsNone(gate['score_metrics']); self.assertEqual(gate['model_calls'], 0)

    def test_offline_rebuild_reproduces_artifact_hashes(self):
        reports = [build(self.staging, self.fixture.capture, self.fixture.index, self.bundle, self.root/name) for name in ('a', 'b')]
        self.assertEqual(reports[0]['artifact_hashes'], reports[1]['artifact_hashes'])
        self.assertEqual(self.rules_audit()['creator_updates'], 0)

    def test_oracle_only_is_not_final_settlement_proof(self):
        (self.ctf/'index.json').write_text('[]')
        with self.assertRaises(ValidationError):self.settlement_audit()

    def test_unknown_or_nonbinary_payouts_are_not_labels(self):
        for values, ctf in [((32, 2, 1, 1), False), ((32, 2, 0, 0), False), ((2, 64, 2, 1, 1), True), ((2, 96, 2, 1, 0), True)]:
            with self.subTest(values=values), self.assertRaises(ValidationError):binary_payout(words(*values), ctf=ctf)

    def test_failed_receipt_is_rejected_even_with_rehashed_archive(self):
        row = json.loads((self.ctf/'index.json').read_text())[0]
        self.mutate(self.ctf, row['matches'][0]['receipt'], lambda r: r.update(status='0x0'))
        with self.assertRaisesRegex(ValidationError, 'failed receipt'):self.settlement_audit()

    def test_block_membership_and_state_boundary_are_checked(self):
        row = json.loads((self.ctf/'index.json').read_text())[0]
        self.mutate(self.ctf, row['block'], lambda r: r.update(transactions=['0x'+'f'*64]))
        with self.assertRaisesRegex(ValidationError, 'not in attested block'):self.settlement_audit()

    def test_substituting_previous_payout_query_cannot_fake_boundary(self):
        rows = json.loads((self.ctf/'index.json').read_text()); rows[0]['previous_denominator'] = rows[0]['resolved_denominator']
        (self.ctf/'index.json').write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValidationError, 'bind'):self.settlement_audit()

    def test_final_condition_identity_and_official_outcome_are_checked(self):
        rows = json.loads((self.ctf/'index.json').read_text()); rows[0]['condition_id'] = QID
        (self.ctf/'index.json').write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValidationError, 'condition binding'):self.settlement_audit()

    def test_neg_risk_request_binding_cannot_be_swapped(self):
        row = json.loads((self.oracle/'index.json').read_text())[0]
        self.mutate(self.oracle, row['matches'][0]['receipt'], lambda r: r['logs'][1].update(data=CONDITION+f'{1:064x}'))
        with self.assertRaisesRegex(ValidationError, 'request/result'):self.settlement_audit()

    def test_observation_after_public_outcome_is_rejected(self):
        for c in self.candidates:
            if c['question_proof']:c['record']['observation_time'] = '2026-07-29T19:59:59Z'
        with self.assertRaisesRegex(ValidationError, 'chronology'):self.settlement_audit()

    def test_source_and_bytecode_review_are_bound(self):
        page = self.explorer/'source.html'; page.write_text(page.read_text()+' changed')
        with self.assertRaisesRegex(ValidationError, 'source page changed'):self.rules_audit()
        with self.assertRaises(ValidationError):extract_source("var editor_contractJsonData = 'execute()'\nDeployed Bytecode<div>0x6000</div>")

    def test_code_response_cannot_be_replaced_after_rehashing(self):
        manifest = json.loads((self.code/'manifest.json').read_text()); ref = manifest['requests'][0]
        path = self.code/ref['file']; obj = json.loads(path.read_text()); obj['result'] = '0x6001'; path.write_text(json.dumps(obj))
        ref['sha256'] = sha256_file(path); (self.code/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, 'bytecode differs'):self.rules_audit()

    def test_current_empty_updates_do_not_prove_a_future_observation(self):
        for c in self.candidates:
            if c['question_proof']:c['record']['observation_time'] = '2026-09-19T00:00:00Z'
        with self.assertRaisesRegex(ValidationError, 'update semantics'):self.rules_audit()

    def test_unbounded_log_queries_are_rejected_before_network(self):
        archive = RpcArchive(self.root/'invalid', 'https://polygon.drpc.org')
        with self.assertRaises(ValidationError):archive.read('eth_getLogs', [{'fromBlock': '0x0', 'toBlock': 'latest'}])


if __name__ == '__main__':unittest.main()
