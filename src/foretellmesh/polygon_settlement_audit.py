"""Bind UMA reports to final CTF payouts, with evaluator-only settlement labels.

These are archived public-RPC attestations, not independent consensus proofs.
The oracle report block is deliberately distinct from the CTF settlement block.
Only the reviewed NegRisk binary route is supported; other routes fail closed.
"""
from datetime import datetime, timezone
import re

from .data import sha256_file, strict_json
from .historical_sources import read_archive
from .polygon_proof_audit import abi_bytes, read_rpc_archive, word
from .schema import ValidationError, iso, timestamp

UMA = '0x69c47de9d4d3dad79590d61b9e05918e03775f24'
OPERATOR = '0x661992aebf6becf7ba5abb66f6b0bf62aa7a2e93'
NEG_RISK = '0xd91e80cf2e7be2e162c6513ced06f1dd0da35296'
CTF = '0x4d97dcd97ec945f40cf65f87097ace5ea0476045'
UMA_RESOLVED = '0x566c3fbdd12dd86bb341787f6d531f79fd7ad4ce7e3ae2d15ac0ca1b601af9df'
REPORTED = '0x504306b41b2531b3fd2bc5e1b32dc1fc87501906cfc63c1180e3873af20f0eae'
CTF_RESOLVED = '0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894'
DENOMINATOR = '0xdd34de67'
NUMERATOR = '0x0504c814'


class ProofArchive:
    def __init__(self, root, *, require_chain=True):
        self.root = root
        self.refs = read_rpc_archive(root)
        self.used = []
        if require_chain and not any(
                ref['request']['method'] == 'eth_chainId' and ref['request']['params'] == []
                and ref['status'] == 200 and obj.get('result') == '0x89' and 'error' not in obj
                for ref, obj in self.refs.values()):
            raise ValidationError('Polygon chain identity not attested')

    def get(self, ref, method, params=None):
        actual, obj = self.refs.get(ref['file'], (None, None))
        if (actual != ref or ref['request']['method'] != method
                or (params is not None and ref['request']['params'] != params)):
            raise ValidationError('RPC index does not bind the requested proof')
        if ref['status'] != 200 or 'error' in obj or obj.get('result') is None:
            raise ValidationError('unsuccessful proof request')
        self.used.append(ref)
        return obj['result']

    def index(self):
        rows = strict_json((self.root/'index.json').read_text())
        if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
            raise ValidationError('expected 1 to 20 question proofs')
        result = {r['question_id']: r for r in rows}
        if len(result) != len(rows):raise ValidationError('duplicate question proof')
        return result

    def hashes(self):
        return {name: sha256_file(self.root/name) for name in ('manifest.json', 'index.json')}


def binary_payout(data, *, ctf=False):
    raw = abi_bytes(data)
    head = [2, 64, 2] if ctf else [32, 2]
    if len(raw) != (len(head)+2)*32 or [word(raw, i*32) for i in range(len(head))] != head:
        raise ValidationError('unsupported payout ABI')
    payouts = [word(raw, (len(head)+i)*32) for i in range(2)]
    if payouts not in ([0, 1], [1, 0]):raise ValidationError('non-binary payout')
    return payouts


def receipt_log(log, receipt, block):
    """Check provider block membership and exact log identity, including indices."""
    if receipt.get('status') != '0x1' or log.get('removed') is not False:
        raise ValidationError('failed receipt or removed log')
    if (not isinstance(log.get('topics'), list)
            or any(not isinstance(t, str) or not re.fullmatch(r'0x[0-9a-f]{64}', t) for t in log['topics'])):
        raise ValidationError('invalid event topic encoding')
    for key, expected in (('blockHash', block['hash']), ('blockNumber', block['number']),
                          ('transactionHash', receipt['transactionHash']),
                          ('transactionIndex', receipt['transactionIndex'])):
        if log.get(key) != expected or receipt.get(key) != expected:
            raise ValidationError('receipt/log/block identity mismatch')
    index = int(receipt['transactionIndex'], 16)
    if not 0 <= index < len(block['transactions']) or block['transactions'][index] != receipt['transactionHash']:
        raise ValidationError('transaction not in attested block')
    keys = ('address', 'topics', 'data', 'blockHash', 'blockNumber', 'transactionHash',
            'transactionIndex', 'logIndex', 'removed')
    matches = [item for item in receipt['logs'] if item.get('logIndex') == log.get('logIndex')]
    if len(matches) != 1 or any(matches[0].get(k) != log.get(k) for k in keys):
        raise ValidationError('receipt does not contain exact unique log')
    if log.get('blockTimestamp', block['timestamp']) != block['timestamp']:
        raise ValidationError('log timestamp disagrees with block')


def settlement_event(archive, row, address, query_topics):
    block = archive.get(row['block'], 'eth_getBlockByNumber')
    if row['block']['request']['params'] != [block['number'], False]:
        raise ValidationError('settlement block was not explicitly requested')
    head = archive.get(row['finalized_head'], 'eth_getBlockByNumber', ['finalized', False])
    if (int(head['number'], 16) <= int(block['number'], 16)
            or int(head['timestamp'], 16) < int(block['timestamp'], 16)
            or int(head['timestamp'], 16) > timestamp(row['finalized_head']['completed_at'], 'capture').timestamp()):
        raise ValidationError('settlement not covered by finalized head chronology')
    logs = archive.get(row['log_query'], 'eth_getLogs',
        [{'address': address, 'blockHash': block['hash'], 'topics': query_topics}])
    if len(row['matches']) != 1:raise ValidationError('ambiguous settlement matches')
    item = row['matches'][0]; log = item['log']
    if log not in logs or sum(x == log for x in logs) != 1 or log['address'] != address:
        raise ValidationError('settlement log not in exact block query')
    # Extra logs for the same question/condition are not silently discarded.
    if len([l for l in logs if l['address'] == address and l['topics'][1:2] == query_topics[1:2]]) != 1:
        raise ValidationError('ambiguous settlement event')
    receipt = archive.get(item['receipt'], 'eth_getTransactionReceipt', [log['transactionHash']])
    receipt_log(log, receipt, block)
    for ref in (row['block'], row['log_query'], item['receipt']):
        if int(block['timestamp'], 16) > timestamp(ref['completed_at'], 'capture').timestamp():
            raise ValidationError('settlement after capture')
    return block, log, receipt


def uint_result(archive, ref, params):
    raw = abi_bytes(archive.get(ref, 'eth_call', params))
    if len(raw) != 32:raise ValidationError('noncanonical uint result')
    return word(raw, 0)


def audit_settlements(staging, capture, candidates, oracle_root, ctf_root):
    """Call after verify_staging: candidates and outcome ledger must be rebuilt."""
    oracle, ctf = ProofArchive(oracle_root), ProofArchive(ctf_root)
    oi, ci = oracle.index(), ctf.index()
    selected = [c for c in candidates if c['question_proof']]
    by_qid = {}
    for c in selected:by_qid.setdefault(c['question_proof']['question_id'], []).append(c)
    if set(oi) != set(ci) or set(oi) != set(by_qid):raise ValidationError('incomplete settlement coverage')
    _, sources = read_archive(capture)
    ledger = {r['sample_id']: r for r in map(strict_json, (staging/'outcomes.jsonl').read_text().splitlines())}
    results, labels = [], {}
    for qid in sorted(by_qid):
        group = by_qid[qid]; candidate = group[0]; original = candidate['question_proof']
        mref, raw = sources[candidate['market_ref']['url']]
        if mref != candidate['market_ref']:raise ValidationError('market source not bound')
        markets = [m for m in strict_json(raw.decode())['markets'] if m.get('negRiskRequestID') == qid]
        if len(markets) != 1:raise ValidationError('ambiguous native request mapping')
        market = markets[0]; condition = market['conditionId']; native_qid = market['questionID']
        if (original['adapter'] != UMA or market['resolvedBy'].lower() != UMA or market.get('negRisk') is not True
                or strict_json(market['outcomes']) != ['Yes', 'No']
                or any(c['record']['event_id'] != 'polymarket:'+str(market['id']) for c in group)
                or any('outcome_token_mapping_unverified' in c['blockers'] for c in group)):
            raise ValidationError('unsupported outcome/adapter/native market mapping')
        o, f = oi[qid], ci[qid]
        if o['condition_id'] != condition or f['condition_id'] != condition or f['ctf'] != CTF or f['oracle_block'] != o['block']:
            raise ValidationError('oracle/final condition binding mismatch')
        ob, ol, receipt = settlement_event(oracle, o, UMA, [None, qid])
        if len(ol['topics']) != 3 or ol['topics'][:2] != [UMA_RESOLVED, qid]:
            raise ValidationError('unsupported oracle settlement event')
        payout = binary_payout(ol['data']); outcome = payout[0]
        if int(ol['topics'][2], 16) != outcome*10**18:raise ValidationError('oracle price/payout disagreement')
        reports = [l for l in receipt['logs'] if l['address'] == OPERATOR and l['topics'] == [REPORTED, native_qid]]
        if len(reports) != 1:raise ValidationError('missing unique NegRisk report')
        report = reports[0]; receipt_log(report, receipt, ob)
        if report['data'] != qid+f'{outcome:064x}':raise ValidationError('NegRisk request/result disagreement')
        fb, fl, _ = settlement_event(ctf, f, CTF, [CTF_RESOLVED, condition])
        if fl['topics'] != [CTF_RESOLVED, condition, '0x'+NEG_RISK[2:].rjust(64, '0'), native_qid]:
            raise ValidationError('final CTF identity mismatch')
        if binary_payout(fl['data'], ctf=True) != payout:raise ValidationError('final CTF/oracle payout disagreement')
        n = int(fb['number'], 16)
        if (uint_result(ctf, f['previous_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n-1)]) != 0
                or uint_result(ctf, f['resolved_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n)]) != 1):
            raise ValidationError('final payout boundary not proven')
        if len(f['numerators']) != 2 or [uint_result(ctf, ref,
                [{'to': CTF, 'data': NUMERATOR+condition[2:]+f'{i:064x}'}, hex(n)])
                for i, ref in enumerate(f['numerators'])] != payout:
            raise ValidationError('CTF payout state disagrees with log')
        ot, ft = (datetime.fromtimestamp(int(b['timestamp'], 16), timezone.utc) for b in (ob, fb))
        if not int(ob['number'], 16) < n or not ot <= ft:raise ValidationError('backwards settlement stages')
        available = max((timestamp(ref['completed_at'], 'capture') for ref in oracle.used+ctf.used), default=ft)
        for c in group:
            sid = c['record']['sample_id']; old = ledger[sid]
            if (old['outcome'] != outcome or old['official_outcome'] != outcome
                    or not timestamp(c['question_proof']['published_at'], 'initialization') <= timestamp(c['record']['observation_time'], 'cutoff')
                    < timestamp(c['first_public_result_time'], 'public result') <= ot <= ft <= available):
                raise ValidationError('settlement outcome or historical chronology disagreement')
            labels[sid] = {'outcome': outcome, 'resolution_time': iso(ft), 'available_at': iso(available)}
        results.append({'question_id': qid, 'condition_id': condition, 'native_question_id': native_qid,
            'outcome': outcome, 'payouts_yes_no': payout, 'oracle_report_time': iso(ot), 'resolution_time': iso(ft),
            'report_to_settlement_seconds': (ft-ot).total_seconds(),
            'settlement_block': n, 'settlement_block_hash': fb['hash'], 'transaction_hash': fl['transactionHash']})
    return {'kind': 'polygon_final_settlement_audit', 'status': 'passed_provider_attestations',
            'contracts': len(results), 'observation_labels': len(labels), 'rows': results, 'labels': labels,
            'oracle_archive_hashes': oracle.hashes(), 'ctf_archive_hashes': ctf.hashes(),
            'trust_scope': 'Public Polygon RPC block, receipt and state attestations; not a consensus/light-client proof.'}
