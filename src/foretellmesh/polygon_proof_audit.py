"""Audit supplementary RPC attestations; never promote a historical sample.

ABI: Polymarket/uma-ctf-adapter IUmaCtfAdapter.sol and BulletinBoard.sol.
This checks provider responses, not consensus, deployed bytecode equivalence,
or completeness of settlement logs. Those limitations remain admission gates.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .schema import ValidationError, timestamp

INITIALIZED = '0xeee0897acd6893adcaf2ba5158191b3601098ab6bece35c5d57874340b64c5b7'
GET_UPDATES = '0x555c56fc'
GET_QUESTION = '0x58c039cd'
EMPTY_UPDATES = '0x' + (32).to_bytes(32, 'big').hex() + bytes(32).hex()
RPC_HOSTS = {'https://polygon.drpc.org', 'https://polygon-bor-rpc.publicnode.com'}
METHODS = {'eth_chainId', 'eth_getBlockByNumber', 'eth_getTransactionReceipt', 'eth_call', 'eth_getLogs'}


def abi_bytes(value):
    if not isinstance(value, str) or not re.fullmatch(r'0x(?:[0-9a-fA-F]{64})+', value):
        raise ValidationError('invalid ABI hex')
    return bytes.fromhex(value[2:])


def word(data, offset):
    if offset % 32 or offset < 0 or offset + 32 > len(data):raise ValidationError('ABI word out of bounds')
    return int.from_bytes(data[offset:offset+32], 'big')


def dynamic_text(data, offset):
    n = word(data, offset); start = offset + 32; end = start + n
    if not 0 < n <= 100_000 or len(data) != start + ((n+31)//32)*32 or any(data[end:]):
        raise ValidationError('invalid ABI text length or padding')
    try:return data[start:end].decode('utf-8')
    except UnicodeDecodeError as exc:raise ValidationError('invalid ancillary UTF-8') from exc


def decode_question(value):
    data = abi_bytes(value)
    if word(data, 0) != 32 or word(data, 12*32) != 12*32:
        raise ValidationError('unsupported QuestionData ABI layout')
    for index in (6, 7, 8, 9):
        if word(data, index*32) not in (0, 1):raise ValidationError('invalid question flags')
    creator = word(data, 11*32)
    if creator >= 2**160:raise ValidationError('invalid creator address')
    return {'request_timestamp': word(data, 32), 'creator': f'0x{creator:040x}',
            'resolved': bool(word(data, 6*32)), 'ancillary_data': dynamic_text(data, 13*32)}


def read_rpc_archive(root):
    manifest = strict_json((root/'manifest.json').read_text())
    if manifest.get('kind') != 'polygon_research_only':raise ValidationError('unexpected RPC archive kind')
    refs = {}
    for ref in manifest['requests']:
        path = (root/ref['file']).resolve()
        if not path.is_relative_to(root.resolve()) or ref['file'] in refs:raise ValidationError('unsafe RPC artifact')
        raw = path.read_bytes(); req = ref['request']
        if sha256_bytes(raw) != ref['sha256'] or ref['url'] not in RPC_HOSTS or req['method'] not in METHODS:
            raise ValidationError('RPC artifact provenance changed')
        if timestamp(ref['started_at'], 'request start') > timestamp(ref['completed_at'], 'request end'):
            raise ValidationError('backwards RPC request clock')
        obj = strict_json(raw.decode()) if ref['status'] else None
        if ref['status'] == 200 and (obj.get('id') != req['id'] or obj.get('jsonrpc') != '2.0'):
            raise ValidationError('RPC response identity mismatch')
        refs[ref['file']] = (ref, obj)
    return refs


def audit(root, staging):
    refs = read_rpc_archive(root)
    def get(ref, method, params=None):
        actual, obj = refs.get(ref['file'], (None, None))
        if actual != ref or ref['request']['method'] != method or (params is not None and ref['request']['params'] != params):
            raise ValidationError('RPC index does not bind the requested proof')
        if ref['status'] != 200 or 'error' in obj or 'result' not in obj:raise ValidationError('unsuccessful proof request')
        return obj['result']
    if not any(r['request']['method'] == 'eth_chainId' and r['status'] == 200 and o.get('result') == '0x89' for r, o in refs.values()):
        raise ValidationError('Polygon chain identity not attested')
    # Bind source candidate bytes to their staging report; full raw-source
    # reconstruction is independently required by historical_replay_gate.
    staging_report = strict_json((staging/'report.json').read_text())
    if sha256_file(staging/'candidates.jsonl') != staging_report['artifact_hashes']['candidates.jsonl']:
        raise ValidationError('staging candidates changed')
    candidates = [strict_json(s) for s in (staging/'candidates.jsonl').read_text().splitlines()]
    proofs = {r['question_proof']['question_id']: r['question_proof'] for r in candidates if r['question_proof']}
    index = strict_json((root/'index.json').read_text()); seen = set(); rows = []
    for item in index:
        qid = item['question_id']
        if qid in seen or qid not in proofs:raise ValidationError('duplicate or unrelated question proof')
        seen.add(qid); proof = proofs[qid]
        if item['adapter'] != proof['adapter']:raise ValidationError('adapter mismatch')
        block = get(item['pinned_block'], 'eth_getBlockByNumber')
        if item['pinned_block']['request']['params'] != ['finalized', False]:raise ValidationError('block was not finalized')
        pinned = block['number']; pin_time = int(block['timestamp'], 16)
        receipt = get(item['initialization_receipt'], 'eth_getTransactionReceipt', [proof['transaction_hash']])
        if (receipt['status'] != '0x1' or receipt['transactionHash'] != proof['transaction_hash']
                or int(receipt['blockNumber'], 16) != proof['block_number'] or int(pinned, 16) <= proof['block_number']):
            raise ValidationError('initialization receipt identity or timeline mismatch')
        logs = [l for l in receipt['logs'] if l['address'].lower() == item['adapter'] and len(l['topics']) == 4
                and l['topics'][:2] == [INITIALIZED, qid]]
        if len(logs) != 1:raise ValidationError('ambiguous initialization log')
        log = logs[0]; creator_word = log['topics'][3]
        if (log.get('removed') is not False or log['blockHash'] != receipt['blockHash']
                or log['transactionHash'] != receipt['transactionHash'] or int(creator_word, 16) >= 2**160
                or '0x'+creator_word[-40:] != item['creator']):
            raise ValidationError('initialization creator or receipt log mismatch')
        raw = abi_bytes(log['data'])
        if word(raw, 0) != 128 or dynamic_text(raw, 128) != proof['ancillary_data']:
            raise ValidationError('initial ancillary text changed')
        initialized = int(log['topics'][2], 16)
        if datetime.fromtimestamp(initialized, timezone.utc) != timestamp(proof['published_at'], 'initialization'):
            raise ValidationError('initialization timestamp mismatch')
        state = decode_question(get(item['question_state'], 'eth_call',
            [{'to': item['adapter'], 'data': GET_QUESTION+qid[2:]}, pinned]))
        if state['creator'] != item['creator'] or state['ancillary_data'] != proof['ancillary_data'] or state['request_timestamp'] != initialized:
            raise ValidationError('current question does not match initialization')
        updates = get(item['updates'], 'eth_call',
            [{'to': item['adapter'], 'data': GET_UPDATES+qid[2:]+creator_word[2:]}, pinned])
        if updates != EMPTY_UPDATES:raise ValidationError('nonempty updates require an explicit historical decoder and review')
        for name in ('updates', 'question_state'):
            if pin_time > timestamp(item[name]['completed_at'], 'RPC capture').timestamp() or initialized >= pin_time:
                raise ValidationError('pinned state chronology invalid')
        rows.append({'question_id': qid, 'creator': item['creator'], 'finalized_block': int(pinned, 16),
                     'finalized_block_hash': block['hash'], 'creator_updates_at_pinned_block': 0,
                     'current_state_resolved': state['resolved'], 'exact_settlement_time_proven': False})
    if seen != set(proofs):raise ValidationError('incomplete question coverage')
    return {'status': 'passed_as_supplementary_attestations', 'questions': len(rows), 'rows': rows,
            'rpc_manifest_sha256': sha256_file(root/'manifest.json'), 'index_sha256': sha256_file(root/'index.json'),
            'staging_report_sha256': sha256_file(staging/'report.json'), 'samples_promoted': 0,
            'limitations': ['Single-provider finalized-state attestations, not independent consensus verification.',
                           'Deployed bytecode and full rule-update semantics still require verification.',
                           'Resolved flag does not establish a settlement timestamp or payout.',
                           'Raw failed log requests are retained; no failed request is treated as absence.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'staging', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():raise ValidationError('proof report exists')
    r = audit(a.archive, a.staging); a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json_text(r)); print(json_text({k: r[k] for k in ('status', 'questions', 'samples_promoted')}))


if __name__ == '__main__':main()
