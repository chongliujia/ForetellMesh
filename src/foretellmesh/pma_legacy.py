"""Explicitly reviewed legacy UMA adapter: capture and verify direct CTF settlement."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .macro_history_capture import fetched
from .pma_enrichment import token_pairs
from .pma_proof_pilot import require, read_jsonl, read_http
from .polygon_proof_audit import INITIALIZED, GET_QUESTION, GET_UPDATES, EMPTY_UPDATES, abi_bytes, word, dynamic_text
from .polygon_settlement_audit import ProofArchive, CTF, CTF_RESOLVED, DENOMINATOR, NUMERATOR, UMA_RESOLVED, receipt_log, binary_payout, uint_result
from .polygon_settlement_capture import CachedCapture
from .polygon_rules_audit import extract_source
from .schema import timestamp, iso
from .synthetic_sft import canonical_hash
from .historical_sources import normalize

ADAPTER = '0x6a9d222616c90fca5754cd1333cfd9b7fb6a4f74'


def decode_legacy_question(value):
    raw = abi_bytes(value)
    require(word(raw, 0) == 32 and word(raw, 10*32) == 10*32, 'unsupported legacy question ABI')
    require(all(word(raw, i*32) in (0, 1) for i in (5, 6, 7)), 'invalid legacy flags')
    creator = word(raw, 9*32)
    require(creator < 2**160, 'invalid creator')
    return {'request_timestamp': word(raw, 32), 'creator': f'0x{creator:040x}',
        'resolved': bool(word(raw, 5*32)), 'paused': bool(word(raw, 6*32)), 'reset': bool(word(raw, 7*32)),
        'emergency_time': word(raw, 4*32), 'ancillary_data': dynamic_text(raw, 11*32)}


def inputs(config_path):
    c = strict_json(config_path.read_text())
    require(c['kind'] == 'pma_legacy_fomc_training_extension_v1' and c['partition'] == 'train'
            and c['train_before'] == '2025-01-01T00:00:00Z' and not c['automatic_training']
            and 1 <= len(c['groups']) <= 8, 'unsupported legacy scope')
    paths = {k: (config_path.parent / v['path']).resolve() for k, v in c['sources'].items()}
    for k, p in paths.items():
        require(sha256_file(p) == c['sources'][k]['sha256'], 'legacy source changed: ' + k)
    manifest = strict_json(paths['native'].read_text()); catalog = strict_json(paths['catalog'].read_text())
    cp = paths['catalog'].parent / 'macro_candidates.jsonl'
    require(sha256_file(cp) == catalog['artifact_hashes']['macro_candidates.jsonl'], 'candidate catalog changed')
    candidates = {r['market_id']: r for r in read_jsonl(cp)}
    groupmap = {g['native_event_id']: g for g in c['groups']}
    require(len(groupmap) == len(c['groups']), 'duplicate group')
    selected = []
    for ref in manifest['requests']:
        p = (paths['native'].parent/ref['file']).resolve()
        require(p.is_relative_to(paths['native'].parent) and sha256_file(p) == ref['sha256'] and ref['status'] == 200,
                'native capture changed')
        for m in strict_json(p.read_text()):
            matches = [e['id'] for e in m.get('events', []) if e['id'] in groupmap]
            if not matches:
                continue
            require(len(matches) == 1, 'ambiguous parent')
            g = groupmap[matches[0]]; r = candidates[m['id']]
            require(m['resolvedBy'].lower() == ADAPTER and not m.get('negRisk') and not m.get('negRiskRequestID')
                    and m['questionID'] and m['conditionId'] == r['condition_id'] and m['question'] == r['question']
                    and token_pairs(m['outcomes'], m['clobTokenIds']) == r['tokens'] and not r['benchmark_matches'],
                    'legacy identity or reservation mismatch')
            require(timestamp(g['prior_release']['published_at'], 'prior') < timestamp(g['release']['published_at'], 'release')
                    < timestamp(c['train_before'], 'cutoff'), 'event outside training chronology')
            selected.append({'market_id': m['id'], 'event_group_id': g['event_group_id'], 'native_event_id': matches[0],
                             'market': m, 'candidate': r})
    selected.sort(key=lambda r: int(r['market_id']))
    require(1 <= len(selected) <= 32 and {r['native_event_id'] for r in selected} == set(groupmap), 'incomplete legacy selection')
    projection = [{k: r[k] for k in ('market_id', 'event_group_id', 'native_event_id')} |
                  {k: r['market'][k] for k in ('question', 'description', 'conditionId', 'questionID')} for r in selected]
    require(c['selection_sha256'] == canonical_hash(projection), 'selection review changed')
    return c, paths, selected


def urls(c, selected):
    values = []
    for g in c['groups']:
        values.extend(['https://gamma-api.polymarket.com/events/' + g['native_event_id'],
                       'https://data-api.polymarket.com/v2/resolutions?event_id=' + g['native_event_id'],
                       g['prior_release']['url'], g['release']['url']])
    values.extend('https://clob.polymarket.com/clob-markets/' + r['market']['conditionId'] for r in selected)
    return list(dict.fromkeys(values))


def capture_contract(row, state, output, endpoint):
    rpc = CachedCapture(output, endpoint); m = row['market']; qid = m['questionID']; condition = m['conditionId']
    require(state['question_id'] == qid and state['condition_id'] == condition, 'state identity differs')
    chain, _ = rpc.read('eth_chainId', []); require(chain == '0x89', 'wrong chain')
    head, hr = rpc.read('eth_getBlockByNumber', ['finalized', False])
    receipt, ir = rpc.read('eth_getTransactionReceipt', [state['transaction_hash']])
    require(receipt and receipt['status'] == '0x1', 'missing initialization receipt')
    logs = [l for l in receipt['logs'] if l['address'] == ADAPTER and l['topics'][:2] == [INITIALIZED, qid]]
    require(len(logs) == 1 and len(logs[0]['topics']) == 4, 'no unique legacy initialization')
    initial, ib = rpc.read('eth_getBlockByNumber', [receipt['blockNumber'], False])
    creator = logs[0]['topics'][3]
    _, qr = rpc.read('eth_call', [{'to': ADAPTER, 'data': GET_QUESTION + qid[2:]}, head['number']])
    _, ur = rpc.read('eth_call', [{'to': ADAPTER, 'data': GET_UPDATES + qid[2:] + creator[2:]}, head['number']])
    codes = [rpc.read('eth_getCode', [ADAPTER, n])[1] for n in (initial['number'], head['number'])]
    def denominator(n):
        v, ref = rpc.read('eth_call', [{'to': CTF, 'data': DENOMINATOR + condition[2:]}, hex(n)])
        require(len(rpc.refs) <= 96, 'legacy RPC budget exceeded')
        return int(v, 16), ref
    lo, hi = int(initial['number'], 16), int(head['number'], 16)
    require(denominator(lo)[0] == 0 and denominator(hi)[0] == 1, 'unsupported initial/final payout')
    while hi - lo > 1:
        mid = (lo + hi)//2
        if denominator(mid)[0]: hi = mid
        else: lo = mid
    _, before = denominator(lo); _, after = denominator(hi)
    block, br = rpc.read('eth_getBlockByNumber', [hex(hi), False])
    logs, lr = rpc.read('eth_getLogs', [{'address': CTF, 'blockHash': block['hash'], 'topics': [CTF_RESOLVED, condition]}])
    require(len(logs) == 1, 'no unique CTF settlement')
    _, sr = rpc.read('eth_getTransactionReceipt', [logs[0]['transactionHash']])
    nums = [rpc.read('eth_call', [{'to': CTF, 'data': NUMERATOR + condition[2:] + f'{i:064x}'}, hex(hi)])[1] for i in (0, 1)]
    proof = {'market_id': m['id'], 'question_id': qid, 'condition_id': condition,
             'head': hr, 'initial_receipt': ir, 'initial_block': ib, 'question_state': qr, 'creator_updates': ur,
             'code': codes, 'settlement_block': br, 'settlement_logs': lr, 'settlement_receipt': sr,
             'previous_denominator': before, 'resolved_denominator': after, 'numerators': nums}
    (output/'proof.json').write_text(json_text(proof))
    return {'market_id': m['id'], 'status': 'captured', 'requests': len(rpc.refs),
            'proof_sha256': sha256_file(output/'proof.json'), 'manifest_sha256': sha256_file(output/'manifest.json')}


def capture(config_path, output, endpoint='https://polygon.drpc.org', reuse=None):
    c, _, selected = inputs(config_path); require(not output.exists(), 'legacy capture exists')
    output.mkdir(parents=True); refs = []; previous = {}; parent_hash = None
    (output/'selection.json').write_text(json_text({'config_sha256': sha256_file(config_path), 'market_ids': [r['market_id'] for r in selected]}))
    if reuse is not None:
        old = strict_json((reuse/'report.json').read_text()); parent_hash = sha256_file(reuse/'report.json')
        require(old['config_sha256'] == sha256_file(config_path) and sha256_file(reuse/'index.json') == old['index_sha256']
                and sha256_file(reuse/'http/manifest.json') == old['http_manifest_sha256']
                and (reuse/'selection.json').read_bytes() == (output/'selection.json').read_bytes(), 'reuse bindings differ')
        rm, _ = read_http(reuse/'http'); refs = rm['requests']; shutil.copytree(reuse/'http', output/'http')
        previous = {r['market_id']: r for r in strict_json((reuse/'index.json').read_text())}
        require(set(previous) == {r['market_id'] for r in selected}, 'reuse coverage differs')
    else:
        (output/'http').mkdir()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ref, raw in pool.map(fetched, urls(c, selected)):
                ref['file'] = f'{len(refs):03d}.bin'; (output/'http'/ref['file']).write_bytes(raw); refs.append(ref)
                (output/'http/manifest.json').write_text(json_text({'requests': refs, 'requests_sha256': canonical_hash(refs)}))
    _, http = read_http(output/'http')
    def collect(row):
        try:
            old = previous.get(row['market_id'])
            if old and old['status'] == 'captured':
                root = reuse/'rpc'/row['market_id']
                require(sha256_file(root/'proof.json') == old['proof_sha256'] and sha256_file(root/'manifest.json') == old['manifest_sha256'],
                        'reused contract changed')
                ProofArchive(root)
                shutil.copytree(root, output/'rpc'/row['market_id'])
                return old
            ref, raw = http['https://data-api.polymarket.com/v2/resolutions?event_id=' + row['native_event_id']]
            require(ref['status'] == 200, 'resolution API failed')
            states = [s for s in strict_json(raw.decode())['data'] if s['condition_id'] == row['market']['conditionId']]
            require(len(states) == 1, 'ambiguous resolution state')
            return capture_contract(row, states[0], output/'rpc'/row['market_id'], endpoint)
        except (ValueError, KeyError, TypeError) as e:
            return {'market_id': row['market_id'], 'status': 'failed', 'error': str(e)}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(collect, selected):
            results.append(r); (output/'index.json').write_text(json_text(results)); print(json_text(r), flush=True)
    report = {'config_sha256': sha256_file(config_path), 'selected': len(selected),
        'captured': sum(r['status'] == 'captured' for r in results), 'http_requests': len(refs),
        'index_sha256': sha256_file(output/'index.json'), 'http_manifest_sha256': sha256_file(output/'http/manifest.json'),
        'selection_sha256': sha256_file(output/'selection.json'), 'reused_capture_report_sha256': parent_hash,
        'reused_contracts': sum(r['status'] == 'captured' for r in previous.values()),
        'new_http_requests': 0 if reuse else len(refs)}
    (output/'report.json').write_text(json_text(report)); return report


def reviewed_runtime(paths):
    policy = strict_json(paths['rules_policy'].read_text()); manifest = strict_json(paths['rules_source'].read_text())
    require(policy['adapter'] == ADAPTER and policy['review_id'] == 'pma_uma_6a9d_direct_ctf_rules_v1', 'wrong legacy source review')
    refs = [r for r in manifest['requests'] if r['url'] == policy['source_url']]
    require(len(refs) == 1 and refs[0]['status'] == 200, 'missing verified source')
    p = (paths['rules_source'].parent/refs[0]['file']).resolve()
    require(p.is_relative_to(paths['rules_source'].parent) and sha256_file(p) == refs[0]['sha256'] == policy['source_page_sha256'], 'source changed')
    sources, runtime = extract_source(p.read_text())
    require({k: sha256_bytes(v['content'].encode()) for k, v in sources.items()} == policy['source_sha256']
            and sha256_bytes(runtime) == policy['runtime_sha256'], 'source/runtime review mismatch')
    return '0x' + runtime.hex()


def audit_contract(root, row, state, runtime):
    a = ProofArchive(root); p = strict_json((root/'proof.json').read_text()); m = row['market']
    qid, condition = m['questionID'], m['conditionId']
    require(m['resolvedBy'].lower() == ADAPTER and not m.get('negRisk') and strict_json(m['outcomes']) == ['Yes', 'No']
            and p['market_id'] == m['id'] and p['question_id'] == qid and p['condition_id'] == condition
            and state['question_id'] == qid and state['condition_id'] == condition, 'legacy proof identity conflict')
    require(state['status'] == 'resolved' and state['was_disputed'] is False and state['extended_review'] is False
            and state.get('was_arbitrated', False) is False, 'unsupported resolution lifecycle')
    head = a.get(p['head'], 'eth_getBlockByNumber', ['finalized', False])
    initial = a.get(p['initial_block'], 'eth_getBlockByNumber')
    require(p['initial_block']['request']['params'] == [initial['number'], False], 'initial block identity')
    receipt = a.get(p['initial_receipt'], 'eth_getTransactionReceipt', [state['transaction_hash']])
    logs = [l for l in receipt['logs'] if l['address'] == ADAPTER and l['topics'][:2] == [INITIALIZED, qid]]
    require(len(logs) == 1 and len(logs[0]['topics']) == 4, 'ambiguous initialization')
    log = logs[0]; receipt_log(log, receipt, initial); creator = log['topics'][3]
    initialized = int(initial['timestamp'], 16)
    require(int(log['topics'][2], 16) == initialized and int(creator, 16) < 2**160, 'invalid initialization')
    raw = abi_bytes(log['data']); require(word(raw, 0) == 128, 'initialized text ABI differs')
    ancillary = dynamic_text(raw, 128)
    historical = 'q: title: ' + m['question'] + ', description: ' + m['description']
    # Match the complete content-bound native rule to immutable initialization.
    # A later edited description never becomes a historical input by truncation.
    boundaries = (', res_data: ', ' res_data: ', f' market_id: {m["id"]} res_data: ')
    require(any(normalize(ancillary).startswith(normalize(historical) + b) for b in boundaries)
            and ancillary.endswith(',initializer:' + creator[-40:]), 'native rules differ from legacy initialization')
    current = decode_legacy_question(a.get(p['question_state'], 'eth_call', [{'to': ADAPTER, 'data': GET_QUESTION+qid[2:]}, head['number']]))
    require(current['creator'] == '0x'+creator[-40:] and current['ancillary_data'] == ancillary
            and current['request_timestamp'] == initialized and current['resolved'] and not current['reset']
            and not current['paused'] and current['emergency_time'] == 0, 'legacy reset/emergency/state mismatch')
    require(a.get(p['creator_updates'], 'eth_call', [{'to': ADAPTER, 'data': GET_UPDATES+qid[2:]+creator[2:]}, head['number']]) == EMPTY_UPDATES,
            'creator updates require separate historical review')
    require(len(p['code']) == 2, 'missing runtime coverage')
    for ref, n in zip(p['code'], (initial['number'], head['number'])):
        require(a.get(ref, 'eth_getCode', [ADAPTER, n]) == runtime, 'unreviewed runtime')
    final = a.get(p['settlement_block'], 'eth_getBlockByNumber')
    require(p['settlement_block']['request']['params'] == [final['number'], False], 'settlement block identity')
    logs = a.get(p['settlement_logs'], 'eth_getLogs', [{'address': CTF, 'blockHash': final['hash'], 'topics': [CTF_RESOLVED, condition]}])
    require(len(logs) == 1, 'ambiguous CTF settlement')
    log = logs[0]; receipt = a.get(p['settlement_receipt'], 'eth_getTransactionReceipt', [log['transactionHash']])
    receipt_log(log, receipt, final)
    require(log['topics'] == [CTF_RESOLVED, condition, '0x'+ADAPTER[2:].rjust(64, '0'), qid], 'direct CTF mapping differs')
    payouts = binary_payout(log['data'], ctf=True); outcome = payouts[0]
    oracle = [l for l in receipt['logs'] if l['address'] == ADAPTER and l['topics'][:2] == [UMA_RESOLVED, qid]]
    require(len(oracle) == 1 and oracle[0]['topics'] == [UMA_RESOLVED, qid, '0x'+f'{outcome*10**18:064x}']
            and binary_payout(oracle[0]['data']) == payouts, 'normal oracle resolution missing/conflicting')
    receipt_log(oracle[0], receipt, final)
    n = int(final['number'], 16)
    require(uint_result(a, p['previous_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n-1)]) == 0
            and uint_result(a, p['resolved_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n)]) == 1,
            'final payout boundary differs')
    require(len(p['numerators']) == 2 and [uint_result(a, ref, [{'to': CTF, 'data': NUMERATOR+condition[2:]+f'{i:064x}'}, hex(n)])
            for i, ref in enumerate(p['numerators'])] == payouts, 'CTF payout state conflict')
    require(int(state['price']) == outcome*10**18 and int(initial['number'], 16) < n < int(head['number'], 16)
            and initialized < int(final['timestamp'], 16) <= int(head['timestamp'], 16), 'payout/time conflict')
    endpoints = {r['url'] for r in a.used}
    chains = {r['url'] for r, obj in a.refs.values() if r['status'] == 200 and r['request']['method'] == 'eth_chainId'
              and r['request']['params'] == [] and obj.get('result') == '0x89' and 'error' not in obj}
    require(len(endpoints) == 1 and endpoints <= chains, 'unbound chain/provider')
    require(all(timestamp(ref['completed_at'], 'capture').timestamp() >= int(head['timestamp'], 16)
                for ref in (p['question_state'], p['creator_updates'], *p['code'])), 'state capture before pin')
    from .pma_proof_replay import utc_seconds
    return {'market_id': m['id'], 'historical_question': historical, 'initialized_at': utc_seconds(initialized),
        'rule_valid_through': utc_seconds(int(head['timestamp'], 16)), 'creator_update_count': 0,
        'ancillary_sha256': sha256_bytes(ancillary.encode()), 'runtime_sha256': sha256_bytes(bytes.fromhex(runtime[2:])),
        'outcome': outcome, 'oracle_report_time': utc_seconds(int(final['timestamp'], 16)),
        'resolution_time': utc_seconds(int(final['timestamp'], 16)), 'settlement_block': n,
        'available_at': max((r['completed_at'] for r in a.used), key=lambda t: timestamp(t, 'capture')),
        'proof_sha256': sha256_file(root/'proof.json'), 'manifest_sha256': sha256_file(root/'manifest.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--endpoint', default='https://polygon.drpc.org')
    p.add_argument('--reuse', type=Path)
    a = p.parse_args(); print(json_text(capture(a.config, a.output, a.endpoint, a.reuse)))
