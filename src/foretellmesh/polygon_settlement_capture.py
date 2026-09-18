"""Bounded read-only recapture of located UMA reports and final CTF payouts.

Discovery hints are not settlement timestamps. The offline audit derives times
from receipt-bound blocks and independently checks market/condition identities.
"""
import argparse
from pathlib import Path
import re

from .data import sha256_file, strict_json
from .evaluation import json_text
from .polygon_capture import RpcArchive
from .polygon_proof_audit import RPC_HOSTS
from .polygon_settlement_audit import (
    CTF, CTF_RESOLVED, DENOMINATOR, NUMERATOR, UMA, UMA_RESOLVED,
    ProofArchive, binary_payout, settlement_event,
)
from .schema import ValidationError


class CachedCapture(RpcArchive):
    def __init__(self, output, endpoint):
        super().__init__(output, endpoint)
        self.cache = {}

    def read(self, method, params):
        key = json_text([method, params])
        if key not in self.cache:
            if len(self.refs) >= 1500:raise ValidationError('settlement request budget exhausted')
            self.cache[key] = super().read(method, params)
        return self.cache[key]


def start(output, endpoint):
    archive = CachedCapture(output, endpoint)
    chain, _ = archive.read('eth_chainId', [])
    if chain != '0x89':raise ValidationError('wrong chain')
    head, ref = archive.read('eth_getBlockByNumber', ['finalized', False])
    return archive, head, ref


def capture_oracle(plan_path, output, endpoint):
    plan = strict_json(plan_path.read_text())
    rows = plan['questions']
    if (plan.get('schema_version') != '1' or not isinstance(rows, list) or not 1 <= len(rows) <= 20
            or len({r['question_id'] for r in rows}) != len(rows)):
        raise ValidationError('invalid bounded oracle block plan')
    for row in rows:
        if (any(not re.fullmatch(r'0x[0-9a-f]{64}', row[k]) for k in ('question_id', 'condition_id'))
                or type(row['block_number']) is not int or row['block_number'] <= 0):
            raise ValidationError('invalid oracle block locator')
    archive, head, head_ref = start(output, endpoint); index = []
    for row in rows:
        if row['block_number'] >= int(head['number'], 16):raise ValidationError('oracle block not finalized')
        block, block_ref = archive.read('eth_getBlockByNumber', [hex(row['block_number']), False])
        logs, log_ref = archive.read('eth_getLogs', [{'address': UMA, 'blockHash': block['hash'], 'topics': [None, row['question_id']]}])
        if len(logs) != 1 or logs[0]['topics'][:2] != [UMA_RESOLVED, row['question_id']]:
            raise ValidationError('no unique supported oracle report at locator')
        binary_payout(logs[0]['data'])
        _, receipt_ref = archive.read('eth_getTransactionReceipt', [logs[0]['transactionHash']])
        index.append({'question_id': row['question_id'], 'condition_id': row['condition_id'],
                      'block': block_ref, 'log_query': log_ref, 'matches': [{'log': logs[0], 'receipt': receipt_ref}],
                      'finalized_head': head_ref})
        (output/'index.json').write_text(json_text(index))
    (output/'collection.json').write_text(json_text({'plan_sha256': sha256_file(plan_path), 'kind': 'located_oracle_recapture'}))
    return {'contracts': len(index), 'requests': len(archive.refs)}


def capture_ctf(oracle_root, output, endpoint):
    oracle = ProofArchive(oracle_root); rows = oracle.index()
    archive, head, head_ref = start(output, endpoint); index = []
    for qid, row in sorted(rows.items()):
        block, _, _ = settlement_event(oracle, row, UMA, [None, qid])
        first = int(block['number'], 16); condition = row['condition_id']
        def denominator(n):
            value, ref = archive.read('eth_call', [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n)])
            return int(value, 16), ref
        if denominator(first)[0]:raise ValidationError('CTF already resolved at oracle block; historical lower bound required')
        lo, hi = first, first+16
        for _ in range(13):
            if hi >= int(head['number'], 16):raise ValidationError('search would exceed finalized head')
            if denominator(hi)[0]:break
            hi = first+2*(hi-first)
        else:raise ValidationError('no payout within bounded settlement interval')
        while hi-lo > 1:
            middle = (lo+hi)//2
            if denominator(middle)[0]:hi = middle
            else:lo = middle
        _, before = denominator(lo); after, resolved = denominator(hi)
        if after != 1:raise ValidationError('unsupported payout denominator')
        final, final_ref = archive.read('eth_getBlockByNumber', [hex(hi), False])
        logs, log_ref = archive.read('eth_getLogs', [{'address': CTF, 'blockHash': final['hash'], 'topics': [CTF_RESOLVED, condition]}])
        if len(logs) != 1:raise ValidationError('no unique final CTF resolution')
        binary_payout(logs[0]['data'], ctf=True)
        _, receipt_ref = archive.read('eth_getTransactionReceipt', [logs[0]['transactionHash']])
        numerators = [archive.read('eth_call', [{'to': CTF, 'data': NUMERATOR+condition[2:]+f'{i:064x}'}, hex(hi)])[1] for i in (0, 1)]
        index.append({'question_id': qid, 'condition_id': condition, 'ctf': CTF,
                      'oracle_block': row['block'], 'block': final_ref, 'log_query': log_ref,
                      'matches': [{'log': logs[0], 'receipt': receipt_ref}],
                      'previous_denominator': before, 'resolved_denominator': resolved,
                      'numerators': numerators, 'finalized_head': head_ref})
        (output/'index.json').write_text(json_text(index))
    (output/'collection.json').write_text(json_text({'oracle_archive_hashes': oracle.hashes(), 'kind': 'bounded_ctf_settlement_capture'}))
    return {'contracts': len(index), 'requests': len(archive.refs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--oracle-plan', type=Path); mode.add_argument('--oracle-archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoint', choices=sorted(RPC_HOSTS), default='https://polygon.drpc.org')
    args = parser.parse_args()
    result = (capture_oracle(args.oracle_plan, args.output, args.endpoint) if args.oracle_plan
              else capture_ctf(args.oracle_archive, args.output, args.endpoint))
    print(json_text(result))


if __name__ == '__main__':main()
