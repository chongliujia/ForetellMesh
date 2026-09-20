"""Bounded PMA historical proof pilot; source capture and offline replay.

All selected contracts stay in the denominator. This module does not authorize
training or benchmark use: semantic overlap review and cohort freezing follow.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .macro_history_capture import fetched
from .pma_enrichment import token_pairs
from .polygon_proof_audit import INITIALIZED, GET_QUESTION, GET_UPDATES
from .polygon_settlement_capture import CachedCapture
from .polygon_settlement_audit import CTF, CTF_RESOLVED, DENOMINATOR, NUMERATOR, UMA_RESOLVED
from .schema import ValidationError, timestamp
from .synthetic_sft import canonical_hash


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines() if line]


def inputs(config_path):
    config = strict_json(config_path.read_text())
    is_bls = config.get('purpose') == 'pma_bls_historical_proof_pilot'
    require(config.get('schema_version') == '1' and config.get('purpose') in ('pma_historical_proof_pilot', 'pma_bls_historical_proof_pilot')
            and config.get('training') is False and config.get('formal_evaluation') is False
            and config.get('split') is None and config.get('observation_days_before') == [7, 1]
            and config.get('max_price_age_seconds') == 10800 and config.get('workers') == 4
            and config.get('max_markets') == 20 and config.get('max_rpc_requests_per_market') == 160,
            'unsupported proof pilot policy')
    paths = {key: (config_path.parent/ref['path']).resolve() for key, ref in config['sources'].items()}
    expected_sources = {'catalog', 'enrichment', 'price_store', 'heldout_index', 'rules_policy', 'rules_source'}
    if is_bls: expected_sources |= {'bls_archive', 'bls_policy'}
    require(set(paths) == expected_sources,
            'incomplete source binding')
    for key, path in paths.items():
        require(sha256_file(path) == config['sources'][key]['sha256'], 'frozen source changed: '+key)
    groups = config['groups']
    require(1 <= len(groups) <= 5 and len({g['event_group_id'] for g in groups}) == len(groups)
            and len({g['native_event_id'] for g in groups}) == len(groups), 'duplicate or unbounded groups')
    for group in groups:
        family = group.get('family') if is_bls else 'fomc'
        require(family in (('cpi_yoy', 'unemployment_u3') if is_bls else ('fomc',)), 'unsupported release family')
        require(re.fullmatch(r'\d{4}-\d{2}', group['period'])
                and group['event_group_id'] == 'macro:us:'+family+':'+group['period']
                and group['native_event_id'].isdigit() and group['group_rationale'], 'invalid macro group')
        for source in (group['release'], group['prior_release']):
            pattern = (r'https://www\.bls\.gov/news\.release/archives/'+('cpi' if family == 'cpi_yoy' else 'empsit')+r'_\d{8}\.htm'
                       if is_bls else r'https://www\.federalreserve\.gov/newsevents/pressreleases/monetary\d{8}a\.htm')
            require(re.fullmatch(pattern, source['url']),
                    'unsupported release URL')
            timestamp(source['published_at'], 'release')
        require((is_bls or group['release']['published_at'].startswith(group['period']))
                and timestamp(group['prior_release']['published_at'], 'prior') < timestamp(group['release']['published_at'], 'release'),
                'invalid release order')
    catalog = strict_json(paths['catalog'].read_text())
    candidates_path = paths['catalog'].parent/'macro_candidates.jsonl'
    require(sha256_file(candidates_path) == catalog['artifact_hashes']['macro_candidates.jsonl'], 'catalog candidates changed')
    candidates = {row['market_id']: row for row in read_jsonl(candidates_path)}
    manifest = strict_json(paths['enrichment'].read_text())
    require(manifest['requests_sha256'] == canonical_hash(manifest['requests']), 'native manifest changed')
    by_event = {g['native_event_id']: g for g in groups}
    selected = []
    for ref in manifest['requests']:
        path = (paths['enrichment'].parent/ref['file']).resolve()
        require(path.is_relative_to(paths['enrichment'].parent) and sha256_file(path) == ref['sha256'], 'native response changed')
        require(ref['status'] == 200, 'incomplete native enrichment')
        for market in strict_json(path.read_text()):
            matches = [e['id'] for e in market.get('events', []) if e['id'] in by_event]
            if not matches:
                continue
            require(len(matches) == 1 and market['id'] in candidates, 'ambiguous selected identity')
            candidate = candidates[market['id']]
            require(market['conditionId'] == candidate['condition_id'] and market['question'] == candidate['question']
                    and token_pairs(market['outcomes'], market['clobTokenIds']) == candidate['tokens'], 'PMA/native identity mismatch')
            selected.append({'market_id': market['id'], 'event_group_id': by_event[matches[0]]['event_group_id'],
                             'native_event_id': matches[0], 'market': market, 'candidate': candidate})
    selected.sort(key=lambda r: int(r['market_id']))
    require(1 <= len(selected) <= config['max_markets'] and len({r['market_id'] for r in selected}) == len(selected)
            and {r['native_event_id'] for r in selected} == set(by_event), 'pilot selection incomplete or unbounded')
    return config, paths, selected


def http_urls(config, selected):
    urls = []
    for group in config['groups']:
        urls += ['https://gamma-api.polymarket.com/events/'+group['native_event_id'],
                 'https://data-api.polymarket.com/v2/resolutions?event_id='+group['native_event_id']]
        if config['purpose'] == 'pma_historical_proof_pilot':
            urls += [group['release']['url'], group['prior_release']['url']]
    urls += ['https://clob.polymarket.com/clob-markets/'+r['market']['conditionId'] for r in selected]
    return list(dict.fromkeys(urls))


def read_http(root):
    manifest = strict_json((root/'manifest.json').read_text())
    require(manifest['requests_sha256'] == canonical_hash(manifest['requests']), 'HTTP manifest changed')
    result = {}
    for ref in manifest['requests']:
        path = (root/ref['file']).resolve()
        require(path.is_relative_to(root.resolve()) and sha256_file(path) == ref['sha256']
                and ref['url'] not in result, 'HTTP artifact or identity changed')
        require(timestamp(ref['started_at'], 'start') <= timestamp(ref['completed_at'], 'end'), 'HTTP clock backwards')
        result[ref['url']] = (ref, path.read_bytes())
    return manifest, result


class PilotRpc(CachedCapture):
    def read(self, method, params):
        require(len(self.refs) < 160 or json_text([method, params]) in self.cache, 'pilot RPC budget exhausted')
        return super().read(method, params)


def capture_contract(row, state, output, endpoint):
    """Use API time only to locate a report; labels come from receipt-bound blocks."""
    rpc = PilotRpc(output, endpoint)
    chain, _ = rpc.read('eth_chainId', [])
    require(chain == '0x89', 'wrong chain')
    head, head_ref = rpc.read('eth_getBlockByNumber', ['finalized', False])
    market = row['market']; qid = market['negRiskRequestID']; adapter = market['resolvedBy'].lower()
    require(state['condition_id'] == market['conditionId'] and state['question_id'] == qid, 'API request/condition mismatch')
    receipt, receipt_ref = rpc.read('eth_getTransactionReceipt', [state['transaction_hash']])
    require(receipt and receipt.get('status') == '0x1', 'initialization receipt missing or failed')
    logs = [l for l in receipt['logs'] if l['address'] == adapter and l['topics'][:2] == [INITIALIZED, qid]]
    require(len(logs) == 1 and len(logs[0]['topics']) == 4, 'no unique initialized question')
    initial, initial_ref = rpc.read('eth_getBlockByNumber', [receipt['blockNumber'], False])
    creator = logs[0]['topics'][3]
    _, state_ref = rpc.read('eth_call', [{'to': adapter, 'data': GET_QUESTION+qid[2:]}, head['number']])
    _, updates_ref = rpc.read('eth_call', [{'to': adapter, 'data': GET_UPDATES+qid[2:]+creator[2:]}, head['number']])
    code_refs = [rpc.read('eth_getCode', [adapter, n])[1] for n in (initial['number'], head['number'])]
    index = {'market_id': row['market_id'], 'question_id': qid, 'condition_id': market['conditionId'],
             'initial_receipt': receipt_ref, 'initial_block': initial_ref, 'finalized_head': head_ref,
             'question_state': state_ref, 'creator_updates': updates_ref, 'code': code_refs}
    (output/'proof.json').write_text(json_text(index))
    hint = int(state['last_update_timestamp'])
    require(int(initial['timestamp'], 16) < hint < int(head['timestamp'], 16), 'report locator outside lifetime')
    lo, hi = int(initial['number'], 16), int(head['number'], 16)
    while hi-lo > 1:
        mid = (lo+hi)//2
        b, _ = rpc.read('eth_getBlockByNumber', [hex(mid), False])
        if int(b['timestamp'], 16) < hint: lo = mid
        else: hi = mid
    found = []
    for n in (hi-1, hi, hi+1):
        b, bref = rpc.read('eth_getBlockByNumber', [hex(n), False])
        values, lref = rpc.read('eth_getLogs', [{'address': adapter, 'blockHash': b['hash'], 'topics': [None, qid]}])
        if values:
            require(len(values) == 1 and values[0]['topics'][:2] == [UMA_RESOLVED, qid], 'unsupported/ambiguous oracle event')
            _, rref = rpc.read('eth_getTransactionReceipt', [values[0]['transactionHash']])
            found.append({'question_id': qid, 'condition_id': market['conditionId'], 'block': bref,
                          'log_query': lref, 'matches': [{'log': values[0], 'receipt': rref}], 'finalized_head': head_ref})
    require(len(found) == 1, 'oracle locator has no unique report')
    index['oracle'] = found[0]
    condition = market['conditionId']
    def denominator(n):
        value, ref = rpc.read('eth_call', [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n)])
        return int(value, 16), ref
    first = int(found[0]['block']['request']['params'][0], 16)
    require(denominator(first)[0] == 0, 'CTF already settled at oracle block; unsupported route')
    lo, hi = first, int(head['number'], 16)
    require(denominator(hi)[0] == 1, 'no supported finalized binary payout')
    while hi-lo > 1:
        mid = (lo+hi)//2
        if denominator(mid)[0]: hi = mid
        else: lo = mid
    _, before = denominator(lo); _, after = denominator(hi)
    block, bref = rpc.read('eth_getBlockByNumber', [hex(hi), False])
    values, lref = rpc.read('eth_getLogs', [{'address': CTF, 'blockHash': block['hash'], 'topics': [CTF_RESOLVED, condition]}])
    require(len(values) == 1, 'no unique CTF settlement')
    _, rref = rpc.read('eth_getTransactionReceipt', [values[0]['transactionHash']])
    numerators = [rpc.read('eth_call', [{'to': CTF, 'data': NUMERATOR+condition[2:]+f'{i:064x}'}, hex(hi)])[1] for i in (0, 1)]
    index['ctf'] = {'question_id': qid, 'condition_id': condition, 'block': bref, 'log_query': lref,
                    'matches': [{'log': values[0], 'receipt': rref}], 'finalized_head': head_ref,
                    'previous_denominator': before, 'resolved_denominator': after, 'numerators': numerators}
    (output/'proof.json').write_text(json_text(index))
    return {'market_id': row['market_id'], 'status': 'captured_not_audited', 'requests': len(rpc.refs),
            'proof_sha256': sha256_file(output/'proof.json'), 'manifest_sha256': sha256_file(output/'manifest.json')}


def capture(config_path, output, endpoint):
    config, _, selected = inputs(config_path)
    require(not output.exists(), 'proof capture already exists')
    output.mkdir(parents=True); (output/'http').mkdir(); (output/'rpc').mkdir()
    (output/'selection.json').write_text(json_text({'config': config, 'config_sha256': sha256_file(config_path),
        'market_ids': [r['market_id'] for r in selected]}))
    refs = []
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for ref, raw in pool.map(fetched, http_urls(config, selected)):
            ref['file'] = f'{len(refs):03d}.bin'; (output/'http'/ref['file']).write_bytes(raw); refs.append(ref)
            (output/'http'/'manifest.json').write_text(json_text({'requests': refs, 'requests_sha256': canonical_hash(refs)}))
    _, http = read_http(output/'http')
    def collect(row):
        root = output/'rpc'/row['market_id']
        try:
            ref, raw = http['https://data-api.polymarket.com/v2/resolutions?event_id='+row['native_event_id']]
            require(ref['status'] == 200, 'resolution API failed')
            states = [s for s in strict_json(raw.decode())['data'] if s['condition_id'] == row['market']['conditionId']]
            require(len(states) == 1, 'missing/duplicate API condition')
            return capture_contract(row, states[0], root, endpoint)
        except (ValidationError, KeyError, ValueError, TypeError) as exc:
            return {'market_id': row['market_id'], 'status': 'capture_incomplete', 'error': str(exc)}
    results = []
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for result in pool.map(collect, selected):
            results.append(result)
            (output/'index.json').write_text(json_text(results))
            print(json_text(result), flush=True)
    report = {'selected_contracts': len(selected), 'captured_contracts': sum(r['status'] == 'captured_not_audited' for r in results),
              'http_requests': len(refs), 'http_failures': sum(r['status'] != 200 for r in refs),
              'selection_sha256': sha256_file(output/'selection.json'), 'index_sha256': sha256_file(output/'index.json'),
              'http_manifest_sha256': sha256_file(output/'http'/'manifest.json'), 'samples_admitted': 0}
    (output/'report.json').write_text(json_text(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['capture', 'build'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--endpoint', default='https://polygon.drpc.org')
    args = parser.parse_args()
    if args.command == 'capture':
        result = capture(args.config, args.output, args.endpoint)
    else:
        from .pma_proof_replay import build
        result = build(args.config, args.capture, args.output)
    print(json_text(result))


if __name__ == '__main__':
    main()
