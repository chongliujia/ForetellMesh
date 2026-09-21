"""Bounded topic-neutral historical rule/settlement proof collection.

Reuse the two already reviewed on-chain adapters, without macro outcome rules.
All selected failures remain in the report; proofs alone do not assign a split.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .pma_enrichment import token_pairs
from .pma_legacy import (ADAPTER, capture_contract as capture_legacy,
                         audit_contract as audit_legacy, reviewed_runtime as runtime_legacy)
from .pma_proof_pilot import capture_contract as capture_negrisk
from .pma_proof_replay import (REVIEW_ADAPTER, audit_contract as audit_negrisk,
                              reviewed_runtime as runtime_negrisk)
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_learning import require


def runtime(config_path, loader):
    c = strict_json(config_path.read_text()); refs = c['sources']
    paths = {k: (config_path.parent/refs[k]['path']).resolve() for k in ('rules_policy', 'rules_source')}
    for k, path in paths.items():
        require(sha256_file(path) == refs[k]['sha256'], 'reviewed adapter source changed')
    loaded = loader(paths)
    # NegRisk returns (review policy, runtime); the legacy loader returns runtime.
    value = loaded[1] if isinstance(loaded, tuple) and len(loaded) == 2 else loaded
    require(isinstance(value, str) and value.startswith('0x'), 'invalid reviewed runtime type')
    return value


def prepare(selection, native, limit):
    require(type(limit) is int and 1 <= limit <= 128, 'invalid proof budget')
    scope = strict_json((selection/'report.json').read_text())
    require(scope['selection_sha256'] == sha256_file(selection/'markets.jsonl'), 'selection changed')
    manifest = strict_json((native/'manifest.json').read_text())
    require(manifest['selection_report_sha256'] == sha256_file(selection/'report.json')
            and manifest['selection_sha256'] == scope['selection_sha256']
            and manifest['requests_sha256'] == canonical_hash(manifest['requests']), 'native capture binding changed')
    markets = {}; states = {}
    for ref in manifest['requests']:
        path = (native/ref['file']).resolve()
        require(path.is_relative_to(native.resolve()) and sha256_file(path) == ref['sha256'], 'native response changed')
        if ref['status'] != 200: continue
        response = strict_json(path.read_text())
        if ref['url'].startswith('https://gamma-api.polymarket.com/markets?'):
            for m in response:
                require(m['id'] not in markets, 'duplicate native market'); markets[m['id']] = m
        elif ref['url'].startswith('https://data-api.polymarket.com/v2/resolutions?event_id='):
            for state in response['data']:
                key = state['condition_id']
                require(key not in states or states[key] == state, 'conflicting resolution locator')
                states[key] = state
    # Bound the already frozen hash order, not successful prices or outcomes.
    candidates = [strict_json(s) for s in (selection/'markets.jsonl').read_text().splitlines()][:limit]
    return candidates, markets, states


def capture(selection, native, output, limit, endpoint, reuse=None):
    require(not output.exists(), 'team proof output exists')
    candidates, markets, states = prepare(selection, native, limit)
    runtimes = {ADAPTER: runtime(Path('configs/pma_legacy_fomc_extension_v1.json'), runtime_legacy),
                REVIEW_ADAPTER: runtime(Path('configs/pma_proof_pilot_v1.json'), runtime_negrisk)}
    output.mkdir(parents=True)
    reuse_hash = None
    if reuse is not None:
        old = strict_json((reuse/'plan.json').read_text()); prior = strict_json((reuse/'report.json').read_text())
        require(prior['plan_sha256'] == sha256_file(reuse/'plan.json')
                and prior['proofs_sha256'] == sha256_file(reuse/'proofs.jsonl')
                and old['selection_report_sha256'] == sha256_file(selection/'report.json')
                and old['native_manifest_sha256'] == sha256_file(native/'manifest.json')
                and old['market_ids'] == [r['market_id'] for r in candidates][:len(old['market_ids'])]
                and old['endpoint'] == endpoint, 'reusable proof source differs')
        shutil.copytree(reuse/'rpc', output/'rpc'); reuse_hash = sha256_file(reuse/'report.json')
    (output/'plan.json').write_text(json_text({'kind': 'topic_neutral_chain_proof_v1',
        'selection_report_sha256': sha256_file(selection/'report.json'),
        'native_manifest_sha256': sha256_file(native/'manifest.json'), 'limit': limit, 'endpoint': endpoint,
        'market_ids': [r['market_id'] for r in candidates], 'training': False,
        'runtime_hashes': {a: canonical_hash(v) for a, v in runtimes.items()}, 'reuse_report_sha256': reuse_hash}))
    def one(candidate):
        mid = candidate['market_id']; root = output/'rpc'/mid
        try:
            require(mid in markets, 'missing native market'); market = markets[mid]
            require(market['conditionId'] == candidate['condition_id']
                    and token_pairs(market['outcomes'], market['clobTokenIds']) == candidate['tokens'], 'identity mismatch')
            require(market['conditionId'] in states, 'missing resolution locator')
            state = states[market['conditionId']]; adapter = market['resolvedBy'].lower()
            require(adapter in runtimes, 'unreviewed adapter')
            row = {'market_id': mid, 'market': market, 'candidate': candidate}
            collect, check = (capture_legacy, audit_legacy) if adapter == ADAPTER else (capture_negrisk, audit_negrisk)
            if not (root/'proof.json').exists(): collect(row, state, root, endpoint)
            proof = check(root, row, state, runtimes[adapter])
            return {'market_id': mid, 'status': 'chain_proof_verified', 'proof': proof,
                    'native_event_ids': sorted(e['id'] for e in market.get('events', [])),
                    'ready_for_training': False, 'ready_for_scoring': False}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return {'market_id': mid, 'status': 'proof_incomplete', 'error': str(exc),
                    'ready_for_training': False, 'ready_for_scoring': False}
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(one, candidates):
            results.append(row); (output/'proofs.jsonl').write_text(jsonl(results))
            print(row['market_id']+': '+row['status']+(' '+row['error'] if 'error' in row else ''), flush=True)
    report = {'kind': 'topic_neutral_chain_proof_v1', 'selected': len(candidates),
              'verified': sum(r['status'] == 'chain_proof_verified' for r in results),
              'plan_sha256': sha256_file(output/'plan.json'), 'proofs_sha256': sha256_file(output/'proofs.jsonl'),
              'admitted_training_markets': 0, 'admitted_evaluation_markets': 0,
              'limitations': ['Single-provider receipt and runtime verification, not a consensus proof.',
                  'Economic result comes from the reviewed contract payout; no separate official event interpretation claimed.',
                  'Semantic benchmark clearance, full event grouping and chronological partitions remain required.']}
    (output/'report.json').write_text(json_text(report)); return report


def audit(selection, native, output):
    plan = strict_json((output/'plan.json').read_text()); report = strict_json((output/'report.json').read_text())
    require(plan['selection_report_sha256'] == sha256_file(selection/'report.json')
            and plan['native_manifest_sha256'] == sha256_file(native/'manifest.json')
            and report['plan_sha256'] == sha256_file(output/'plan.json')
            and report['proofs_sha256'] == sha256_file(output/'proofs.jsonl'), 'proof run binding changed')
    candidates, markets, states = prepare(selection, native, plan['limit'])
    results = [strict_json(s) for s in (output/'proofs.jsonl').read_text().splitlines()]
    require([r['market_id'] for r in results] == plan['market_ids'] == [r['market_id'] for r in candidates],
            'proof population differs')
    runtimes = {ADAPTER: runtime(Path('configs/pma_legacy_fomc_extension_v1.json'), runtime_legacy),
                REVIEW_ADAPTER: runtime(Path('configs/pma_proof_pilot_v1.json'), runtime_negrisk)}
    require(plan['runtime_hashes'] == {a: canonical_hash(v) for a, v in runtimes.items()}, 'reviewed runtime changed')
    count = 0
    for candidate, result in zip(candidates, results):
        require(result['ready_for_training'] is False and result['ready_for_scoring'] is False, 'proof is not admission')
        if result['status'] != 'chain_proof_verified': continue
        mid = candidate['market_id']; market = markets[mid]; adapter = market['resolvedBy'].lower()
        checker = audit_legacy if adapter == ADAPTER else audit_negrisk
        proof = checker(output/'rpc'/mid, {'market_id': mid, 'market': market, 'candidate': candidate},
                        states[market['conditionId']], runtimes[adapter])
        require(proof == result['proof'], 'proof offline replay differs'); count += 1
    require(count == report['verified'] and len(results) == report['selected'], 'proof counts differ')
    value = {'status': 'passed', 'verified_replayed': count, 'failed_retained': len(results)-count,
             'report_sha256': sha256_file(output/'report.json'), 'new_network_requests': 0,
             'admitted_training_markets': 0}
    (output/'audit.json').write_text(json_text(value)); return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'native', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--limit', type=int, default=8); p.add_argument('--endpoint', default='https://polygon.drpc.org')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--reuse', type=Path)
    a = p.parse_args()
    print(json_text(audit(a.selection, a.native, a.output) if a.audit else
                    capture(a.selection, a.native, a.output, a.limit, a.endpoint, a.reuse)))
