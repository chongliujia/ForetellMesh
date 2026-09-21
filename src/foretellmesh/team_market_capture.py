"""Capture native identity/parent metadata for topic-neutral discovery candidates.

No current API field becomes historical model input or a training label here.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

from .data import sha256_file, strict_json
from .evaluation import json_text
from .macro_history_capture import fetched
from .pma_enrichment import token_pairs
from .schema import ValidationError
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def capture(selection, output):
    if output.exists(): raise ValidationError('team identity capture exists')
    selected = strict_json((selection/'report.json').read_text())
    if (selected['kind'] != 'team_market_discovery_selection_v1'
            or selected['selection_sha256'] != sha256_file(selection/'markets.jsonl')):
        raise ValidationError('discovery selection changed')
    rows = [strict_json(s) for s in (selection/'markets.jsonl').read_text().splitlines()]
    if not 1 <= len(rows) <= 2000 or any(not r['market_id'].isdigit() for r in rows):
        raise ValidationError('unbounded/invalid market identity request')
    output.mkdir(parents=True); (output/'http').mkdir(); refs = []; native = {}; errors = {}
    def save():
        (output/'manifest.json').write_text(json_text({'kind': 'team_discovery_native_capture_v1',
            'selection_report_sha256': sha256_file(selection/'report.json'),
            'selection_sha256': selected['selection_sha256'], 'requests': refs,
            'requests_sha256': canonical_hash(refs)}))
    def collect(urls):
        with ThreadPoolExecutor(max_workers=4) as pool:
            for ref, raw in pool.map(fetched, urls):
                ref['file'] = f'http/{len(refs):04d}.bin'; (output/ref['file']).write_bytes(raw); refs.append(ref); save()
                yield ref, raw
    # The detail endpoint omits native parent events. Explicitly request both
    # closed states from the list endpoint; closed status is NOT a selection.
    urls = ['https://gamma-api.polymarket.com/markets?'+urlencode(
        [('id', r['market_id']) for r in rows[start:start+100]]+[('closed', closed), ('limit', 100)])
        for start in range(0, len(rows), 100) for closed in ('true', 'false')]
    responses = {}
    for ref, raw in collect(urls):
        if ref['status'] != 200: continue
        batch = strict_json(raw.decode())
        if not isinstance(batch, list): raise ValidationError('native list response required')
        for market in batch:
            if market['id'] not in {r['market_id'] for r in rows} or market['id'] in responses:
                raise ValidationError('unrequested or duplicate native identity')
            responses[market['id']] = market
    for row in rows:
        mid = row['market_id']
        try:
            if mid not in responses: raise ValidationError('native market not returned')
            market = responses[mid]
            if (market['id'] != mid or market['conditionId'] != row['condition_id']
                    or token_pairs(market['outcomes'], market['clobTokenIds']) != row['tokens']):
                raise ValidationError('native identity/token mapping mismatch')
            native[mid] = market
        except (ValueError, KeyError, TypeError) as exc:
            errors[mid] = str(exc)
    parents = sorted({e['id'] for m in native.values() for e in m.get('events', [])})
    if any(not isinstance(p, str) or not p.isdigit() for p in parents):
        raise ValidationError('invalid parent event identity')
    urls = ['https://data-api.polymarket.com/v2/resolutions?event_id='+eid for eid in parents]
    for _ in collect(urls): pass
    review = []
    for row in rows:
        market = native.get(row['market_id'], {})
        review.append({'market_id': row['market_id'], 'native_identity_matches': bool(market),
                      'error': errors.get(row['market_id']),
                      'native_event_ids': sorted(e['id'] for e in market.get('events', [])),
                      'adapter_for_review': market.get('resolvedBy'), 'neg_risk_for_review': market.get('negRisk'),
                      'question_unchanged_from_archive': market.get('question') == row['question'],
                      'ready_for_training': False, 'ready_for_scoring': False})
    (output/'identity_review.jsonl').write_text(jsonl(review))
    report = {'kind': 'team_discovery_identity_review_v1', 'status': 'captured_not_historical_admission',
              'selected_markets': len(rows), 'identity_matches': len(native), 'errors': errors,
              'native_parent_events': len(parents), 'http_requests': len(refs),
              'http_failures': sum(r['status'] != 200 for r in refs),
              'adapters': dict(Counter(str(r['adapter_for_review']) for r in review)),
              'manifest_sha256': sha256_file(output/'manifest.json'),
              'identity_review_sha256': sha256_file(output/'identity_review.jsonl'),
              'admitted_training_markets': 0, 'model_calls': 0,
              'limitations': ['Current native responses are review metadata, never historical evidence.',
                  'Parent IDs alone do not establish complete semantic grouping or benchmark clearance.',
                  'Resolution API responses are locators for proof collection, not independently verified payout labels.']}
    (output/'report.json').write_text(json_text(report)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json_text(capture(a.selection, a.output)))
