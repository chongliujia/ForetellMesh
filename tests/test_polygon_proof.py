from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file
from foretellmesh.polygon_capture import capture, RpcArchive
from foretellmesh.polygon_proof_audit import audit, decode_question, EMPTY_UPDATES, GET_UPDATES, INITIALIZED
from foretellmesh.schema import ValidationError

QID = '0x'+'b'*64; TX = '0x'+'c'*64; ADDRESS = '0x'+'d'*40; CREATOR = '0x'+'a'*40
INITIAL = 1700000000


def uint(value):return value.to_bytes(32, 'big')
def text_bytes(value):
    encoded = value.encode(); return uint(len(encoded))+encoded+bytes((-len(encoded))%32)


def question(text='historical question'):
    return '0x'+(uint(32)+b''.join(uint(n) for n in [INITIAL, 0, 0, 0, 0, 1, 0, 0, 0, 0, int(CREATOR, 16), 384])+text_bytes(text)).hex()


class PolygonProofTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.staging = self.root/'staging'; self.staging.mkdir(); self.output = self.root/'rpc'
        proof = {'question_id': QID, 'transaction_hash': TX, 'adapter': ADDRESS, 'ancillary_data': 'historical question',
                 'published_at': datetime.fromtimestamp(INITIAL, timezone.utc).isoformat(), 'block_number': 10}
        (self.staging/'candidates.jsonl').write_text(json.dumps({'question_proof': proof})+'\n')
        (self.staging/'report.json').write_text(json.dumps({'artifact_hashes': {'candidates.jsonl': sha256_file(self.staging/'candidates.jsonl')}}))
        self.receipt = {'status': '0x1', 'blockNumber': '0xa', 'blockHash': '0x'+'e'*64, 'transactionHash': TX,
            'logs': [{'address': ADDRESS, 'topics': [INITIALIZED, QID, '0x'+uint(INITIAL).hex(), '0x'+uint(int(CREATOR, 16)).hex()],
                      'data': '0x'+(uint(128)+bytes(96)+text_bytes('historical question')).hex(), 'removed': False,
                      'blockHash': '0x'+'e'*64, 'transactionHash': TX}]}

    def response(self, request, timeout):
        body = json.loads(request.data); method = body['method']
        if method == 'eth_chainId':result = '0x89'
        elif method == 'eth_getBlockByNumber':result = {'number': '0x14', 'timestamp': hex(INITIAL+1000), 'hash': '0x'+'f'*64}
        elif method == 'eth_getTransactionReceipt':result = deepcopy(self.receipt)
        else:result = EMPTY_UPDATES if body['params'][0]['data'].startswith(GET_UPDATES) else question()
        raw = json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': result}).encode()
        class Response:
            url = request.full_url; status = 200
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def read(self, n):return raw[:n]
        return Response()

    def collect(self):
        with patch('foretellmesh.polygon_capture.urlopen', side_effect=self.response):
            return capture(self.staging, self.output, 'https://polygon.drpc.org')

    def test_capture_and_audit_bind_initial_creator_and_explicit_block(self):
        result = self.collect(); self.assertEqual(result['requests'], 5)
        report = audit(self.output, self.staging)
        self.assertEqual(report['questions'], 1); self.assertEqual(report['samples_promoted'], 0)
        self.assertFalse(report['rows'][0]['exact_settlement_time_proven'])
        self.assertTrue(report['rows'][0]['current_state_resolved'])

    def test_malformed_abi_offsets_padding_flags_and_addresses_fail(self):
        self.assertEqual(decode_question(question())['creator'], CREATOR)
        raw = bytes.fromhex(question()[2:])
        for index, value in [(0, 64), (6, 2), (11, 2**160), (12, 416), (13, 100001)]:
            bad = bytearray(raw); bad[index*32:(index+1)*32] = uint(value)
            with self.assertRaises(ValidationError):decode_question('0x'+bad.hex())
        with self.assertRaises(ValidationError):decode_question(question()[:-2]+'01')
        with self.assertRaises(ValidationError):decode_question('0xabc')

    def test_tampered_raw_bytes_or_index_cannot_be_used_as_proof(self):
        self.collect(); index = json.loads((self.output/'index.json').read_text())
        index[0]['creator'] = ADDRESS; (self.output/'index.json').write_text(json.dumps(index))
        with self.assertRaisesRegex(ValidationError, 'creator'):audit(self.output, self.staging)
        (self.output/'000.bin').write_text('{}')
        with self.assertRaisesRegex(ValidationError, 'provenance'):audit(self.output, self.staging)

    def test_failed_or_removed_initialization_and_write_methods_are_rejected(self):
        self.receipt['status'] = '0x0'
        with self.assertRaises(ValidationError):self.collect()
        self.assertTrue((self.output/'manifest.json').exists())
        archive = RpcArchive(self.root/'empty', 'https://polygon.drpc.org')
        with self.assertRaises(ValidationError):archive.read('eth_sendRawTransaction', ['secret'])
        with self.assertRaises(ValidationError):RpcArchive(self.root/'other', 'https://untrusted.invalid')


if __name__ == '__main__':unittest.main()
