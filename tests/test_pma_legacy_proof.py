"""Synthetic provider archives: semantic forgeries must fail even after rehashing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_pma_proof_pilot as fixtures
from foretellmesh.pma_legacy import ADAPTER, audit_contract, capture_contract
from foretellmesh.polygon_proof_audit import INITIALIZED, GET_UPDATES, GET_QUESTION, EMPTY_UPDATES
from foretellmesh.polygon_settlement_audit import CTF, CTF_RESOLVED, UMA_RESOLVED, DENOMINATOR, NUMERATOR
from foretellmesh.schema import ValidationError

words, abi_text = fixtures.words, fixtures.abi_text
QID, CONDITION, TX, CREATOR = fixtures.QID, fixtures.CONDITION, fixtures.TX, fixtures.CREATOR

class LegacyProofTests(unittest.TestCase):
    dump = fixtures.PmaProofPilotTests.dump
    block = fixtures.PmaProofPilotTests.block
    mutate_response = fixtures.PmaProofPilotTests.mutate_response

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.rpc_root = Path(tmp.name)/'rpc'
        self.market = {'id': '1', 'question': 'A historical binary event?', 'description': 'Fixed initial rules.',
            'conditionId': CONDITION, 'questionID': QID, 'resolvedBy': ADAPTER, 'negRisk': False, 'outcomes': '["Yes","No"]'}
        self.row = {'market': self.market}
        self.state = {'question_id': QID, 'condition_id': CONDITION, 'status': 'resolved', 'was_disputed': False,
            'extended_review': False, 'was_arbitrated': False, 'transaction_hash': TX,
            'last_update_timestamp': '1746648000', 'price': str(10**18)}
        self.ancillary = 'q: title: '+self.market['question']+', description: '+self.market['description']+', res_data: p1:0, p2:1.,initializer:'+CREATOR[2:]
        with patch('foretellmesh.polygon_capture.urlopen', side_effect=self.rpc):
            capture_contract(self.row, self.state, self.rpc_root, 'https://polygon.drpc.org')

    def logs(self, n):
        b = self.block(n)
        common = {'blockNumber': b['number'], 'blockHash': b['hash'], 'blockTimestamp': b['timestamp'],
                  'transactionHash': b['transactions'][0], 'transactionIndex': '0x0', 'removed': False}
        if n == 100:
            return [{**common, 'address': ADAPTER, 'topics': [INITIALIZED, QID, words(int(b['timestamp'], 16)), words(int(CREATOR, 16))],
                     'data': words(128, 0, 0, 0)+abi_text(self.ancillary), 'logIndex': '0x0'}]
        if n == 210:
            return [{**common, 'address': ADAPTER, 'topics': [UMA_RESOLVED, QID, words(10**18)],
                     'data': words(32, 2, 1, 0), 'logIndex': '0x0'},
                    {**common, 'address': CTF, 'topics': [CTF_RESOLVED, CONDITION, words(int(ADAPTER, 16)), QID],
                     'data': words(2, 64, 2, 1, 0), 'logIndex': '0x1'}]
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
                result = words(32, int(self.block(100)['timestamp'], 16), 0, 0, 0, 1, 0, 0, 0, int(CREATOR, 16), 320)+abi_text(self.ancillary)
            else: raise AssertionError(data)
        else: raise AssertionError(method)
        raw = json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': result}).encode()
        class Response:
            url = request.full_url; status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return raw[:size]
        return Response()

    def audit(self): return audit_contract(self.rpc_root, self.row, self.state, '0x6000')
    def proof(self): return json.loads((self.rpc_root/'proof.json').read_text())

    def test_capture_and_audit_are_deterministic(self):
        a = self.audit(); self.assertEqual(a, self.audit())
        self.assertEqual(a['outcome'], 1); self.assertEqual(a['creator_update_count'], 0)

    def test_current_rule_amendment_rejected(self):
        self.market['description'] += ' Different rule.'
        with self.assertRaisesRegex(ValidationError, 'rules differ'): self.audit()

    def test_rehashed_creator_updates_rejected(self):
        self.mutate_response(self.proof()['creator_updates'], lambda obj: obj.update(result=words(32, 1)))
        with self.assertRaisesRegex(ValidationError, 'creator updates'): self.audit()

    def test_rehashed_reset_and_paused_and_emergency_states_rejected(self):
        ref = self.proof()['question_state']
        initial = json.loads((self.rpc_root/ref['file']).read_text())['result']
        for slot in (4, 6, 7):
            bad = initial[:2+slot*64]+f'{1:064x}'+initial[2+(slot+1)*64:]
            self.mutate_response(self.proof()['question_state'], lambda obj: obj.update(result=bad))
            with self.assertRaisesRegex(ValidationError, 'reset/emergency/state'): self.audit()

    def test_normal_oracle_event_required_in_ctf_receipt(self):
        self.mutate_response(self.proof()['settlement_receipt'], lambda obj: obj['result'].update(logs=obj['result']['logs'][1:]))
        with self.assertRaisesRegex(ValidationError, 'normal oracle'): self.audit()

    def test_rehashed_payout_and_runtime_conflict_rejected(self):
        self.mutate_response(self.proof()['numerators'][0], lambda obj: obj.update(result=words(0)))
        with self.assertRaisesRegex(ValidationError, 'payout state'): self.audit()
        self.mutate_response(self.proof()['code'][0], lambda obj: obj.update(result='0x6001'))
        with self.assertRaisesRegex(ValidationError, 'runtime'): self.audit()

    def test_settlement_previous_block_binding_required(self):
        proof = self.proof(); proof['previous_denominator'] = proof['resolved_denominator']
        self.dump(self.rpc_root/'proof.json', proof)
        with self.assertRaisesRegex(ValidationError, 'bind'): self.audit()

if __name__ == '__main__': unittest.main()
