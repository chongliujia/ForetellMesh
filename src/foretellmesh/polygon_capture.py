"""Bounded read-only capture of creator updates for already staged questions.

No wallet, transaction submission, arbitrary ABI calls, or automatic admission.
Settlement log discovery is separate; a current resolved flag is not a label.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .polygon_proof_audit import GET_QUESTION, GET_UPDATES, INITIALIZED, RPC_HOSTS
from .schema import ValidationError


class RpcArchive:
    def __init__(self, output, endpoint):
        if endpoint not in RPC_HOSTS:raise ValidationError('unsupported public Polygon RPC')
        if output.exists():raise ValidationError('RPC capture exists')
        output.mkdir(parents=True); self.output = output; self.endpoint = endpoint; self.refs = []

    def read(self, method, params):
        if method not in {'eth_chainId', 'eth_getBlockByNumber', 'eth_getTransactionReceipt', 'eth_call', 'eth_getLogs', 'eth_getCode'}:
            raise ValidationError('unsupported read-only RPC method')
        if method == 'eth_getLogs' and (len(params) != 1 or not isinstance(params[0], dict)
                or set(params[0]) != {'address', 'blockHash', 'topics'}):
            raise ValidationError('log capture requires one explicit blockHash, address and topics')
        body = {'jsonrpc': '2.0', 'id': len(self.refs)+1, 'method': method, 'params': params}
        started = datetime.now(timezone.utc).isoformat()
        request = Request(self.endpoint, data=json_text(body).encode(), headers={
            'Content-Type': 'application/json', 'User-Agent': 'ForetellMesh/0.1 historical-research'})
        try:
            with urlopen(request, timeout=25) as response:
                if response.url != self.endpoint:raise ValidationError('unexpected RPC redirect')
                raw, status = response.read(4_000_001), response.status
        except HTTPError as exc:raw, status = exc.read(4_000_001), exc.code
        except (URLError, TimeoutError, ConnectionError) as exc:raw, status = str(exc).encode(), 0
        if len(raw) > 4_000_000:raise ValidationError('RPC response exceeds bound')
        name = f'{len(self.refs):03d}.bin'; (self.output/name).write_bytes(raw)
        ref = {'url': self.endpoint, 'request': body, 'started_at': started,
               'completed_at': datetime.now(timezone.utc).isoformat(), 'status': status,
               'file': name, 'sha256': sha256_bytes(raw)}
        self.refs.append(ref)
        (self.output/'manifest.json').write_text(json_text({'schema_version': '1', 'kind': 'polygon_research_only', 'requests': self.refs}))
        obj = strict_json(raw.decode()) if status == 200 else None
        if not obj or obj.get('jsonrpc') != '2.0' or obj.get('id') != body['id'] or 'result' not in obj:
            raise ValidationError('RPC request failed; response retained in archive')
        return obj['result'], ref


def capture(staging, output, endpoint):
    report = strict_json((staging/'report.json').read_text())
    if sha256_file(staging/'candidates.jsonl') != report['artifact_hashes']['candidates.jsonl']:
        raise ValidationError('staging candidates changed')
    rows = [strict_json(s) for s in (staging/'candidates.jsonl').read_text().splitlines()]
    proofs = {r['question_proof']['question_id']: r['question_proof'] for r in rows if r['question_proof']}
    if not 1 <= len(proofs) <= 20:raise ValidationError('expected 1 to 20 staged questions')
    archive = RpcArchive(output, endpoint)
    chain, _ = archive.read('eth_chainId', [])
    if chain != '0x89':raise ValidationError('wrong chain')
    block, block_ref = archive.read('eth_getBlockByNumber', ['finalized', False]); index = []
    for qid, proof in sorted(proofs.items()):
        receipt, receipt_ref = archive.read('eth_getTransactionReceipt', [proof['transaction_hash']])
        logs = [l for l in receipt['logs'] if l['address'].lower() == proof['adapter'] and len(l['topics']) == 4
                and l['topics'][:2] == [INITIALIZED, qid]]
        if len(logs) != 1 or receipt['status'] != '0x1':raise ValidationError('initialization not uniquely proved')
        creator = logs[0]['topics'][3]
        _, updates_ref = archive.read('eth_call', [{'to': proof['adapter'], 'data': GET_UPDATES+qid[2:]+creator[2:]}, block['number']])
        _, state_ref = archive.read('eth_call', [{'to': proof['adapter'], 'data': GET_QUESTION+qid[2:]}, block['number']])
        index.append({'question_id': qid, 'adapter': proof['adapter'], 'creator': '0x'+creator[-40:],
                      'initialization_receipt': receipt_ref, 'updates': updates_ref,
                      'question_state': state_ref, 'pinned_block': block_ref})
        (output/'index.json').write_text(json_text(index))
    return {'questions': len(index), 'requests': len(archive.refs), 'samples_promoted': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staging', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', choices=sorted(RPC_HOSTS), default='https://polygon.drpc.org')
    a = p.parse_args(); print(json_text(capture(a.staging, a.output, a.endpoint)))


if __name__ == '__main__':main()
