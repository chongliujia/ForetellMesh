"""Inventory existing PMA prices and admission gaps without opening heldout inputs/labels."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

from .data import sha256_file, strict_json
from .evaluation import json_text
from .pma_proof_pilot import require, read_jsonl
from .synthetic_sft import canonical_hash


def parent_family(title):
    text = title.lower()
    if re.search(r'fed interest rates|fed decision in', text):
        return 'scheduled_fomc_candidate'
    if re.search(r'what .*say|mention|tie at|says ', text):
        return 'speech_or_mention'
    if re.search(r'inflation|cpi', text):
        return 'inflation_needs_country_and_period_review'
    if 'unemployment' in text:
        return 'employment_needs_period_review'
    return 'other_macro_or_keyword_match'


def dispositions(markets, coverage, membership, cutoff):
    """Native-parent protection propagates across all siblings; no automatic admission."""
    owners = defaultdict(set)
    for mid, split in membership.items():
        if mid in markets:
            for event in markets[mid].get('events', []):
                owners[event['id']].add(split)
    output = []
    for mid, m in sorted(markets.items(), key=lambda x: int(x[0])):
        parents = m.get('events', [])
        protected = sorted({s for e in parents for s in owners[e['id']] if s != 'train'})
        c = coverage[mid]; end = m.get('endDate') or ''
        if mid in membership:
            status = 'existing_' + membership[mid]
        elif protected:
            status = 'protected_parent_sibling'
        elif c['benchmark_matches']:
            status = 'reserved_benchmark_identity'
        elif not c['time_joined_trades']:
            status = 'no_reconstructed_prices'
        elif not end or end >= cutoff:
            status = 'date_unknown_or_outside_early_scope'
        else:
            status = 'early_candidate_needs_admission'
        output.append({'market_id': mid, 'native_event_ids': sorted(e['id'] for e in parents),
            'parent_titles_for_review_only': [e['title'] for e in parents],
            'families_for_review_only': sorted({parent_family(e['title']) for e in parents}),
            'end_date_hint_only': end, 'adapter': (m.get('resolvedBy') or '').lower(),
            'time_joined_trades': c['time_joined_trades'], 'status': status,
            'protected_splits': protected, 'benchmark_matches': c['benchmark_matches'],
            'ready_for_training': False})
    return output


def build(config_path, output):
    require(not output.exists(), 'training inventory exists')
    c = strict_json(config_path.read_text())
    require(c['kind'] == 'pma_existing_archive_training_inventory_v1' and c['train_before'] == '2025-01-01T00:00:00Z',
            'unsupported inventory policy')
    paths = {k: (config_path.parent / r['path']).resolve() for k, r in c['sources'].items()}
    for k, p in paths.items():
        require(sha256_file(p) == c['sources'][k]['sha256'], 'inventory source changed: ' + k)
    tr = strict_json(paths['prices'].read_text()); dr = strict_json(paths['dataset'].read_text())
    coverage_path = paths['prices'].parent / 'market_coverage.jsonl'
    member_path = paths['dataset'].parent / 'membership.jsonl'
    require(sha256_file(coverage_path) == tr['artifact_hashes']['market_coverage.jsonl'], 'coverage changed')
    require(sha256_file(member_path) == dr['artifact_hashes']['membership.jsonl'], 'membership changed')
    coverage = {r['market_id']: r for r in read_jsonl(coverage_path)}
    membership = {}
    for r in read_jsonl(member_path):
        mid = r['event_id'].removeprefix('polymarket:')
        require(membership.get(mid, r['split']) == r['split'], 'cross-split market')
        membership[mid] = r['split']
    manifest = strict_json(paths['native'].read_text()); markets = {}
    require(manifest['requests_sha256'] == canonical_hash(manifest['requests']), 'native manifest changed')
    for ref in manifest['requests']:
        p = (paths['native'].parent / ref['file']).resolve()
        require(p.is_relative_to(paths['native'].parent) and sha256_file(p) == ref['sha256'] and ref['status'] == 200,
                'native response changed')
        for row in strict_json(p.read_text()):
            require(row['id'] not in markets, 'duplicate native market')
            # Project away snapshot outcome, price, volume and resolution fields.
            markets[row['id']] = {k: row.get(k) for k in ('id', 'endDate', 'resolvedBy')}
            markets[row['id']]['events'] = [{k: e[k] for k in ('id', 'title')} for e in row.get('events', [])]
    require(set(markets) == set(coverage), 'native/coverage population mismatch')
    rr = dispositions(markets, coverage, membership, c['train_before'])
    early = [r for r in rr if r['status'] == 'early_candidate_needs_admission']
    parents = defaultdict(list)
    for row in early:
        for eid, title in zip(row['native_event_ids'], row['parent_titles_for_review_only']):
            parents[eid].append(row['market_id'])
    output.mkdir(parents=True)
    (output/'markets.jsonl').write_text(''.join(json_text(r).replace('\n', '')+'\n' for r in rr))
    report = {'kind': c['kind'], 'status': 'inventoried_not_admitted', 'config_sha256': sha256_file(config_path),
        'source_hashes': {k: sha256_file(p) for k, p in paths.items()},
        'coverage_sha256': sha256_file(coverage_path), 'membership_sha256': sha256_file(member_path),
        'counts': {'markets': len(rr), 'with_prices': sum(r['time_joined_trades'] > 0 for r in rr),
                   'by_status': dict(Counter(r['status'] for r in rr)),
                   'early_by_adapter': dict(Counter(r['adapter'] for r in early)),
                   'early_native_parents': len(parents)},
        'early_parent_markets': dict(parents), 'admitted_new_training_markets': 0,
        'artifact_hashes': {'markets.jsonl': sha256_file(output/'markets.jsonl')},
        'script_sha256': sha256_file(Path(__file__)), 'heldout_inputs_opened': False, 'heldout_labels_opened': False,
        'limitations': ['Native dates/titles are retrospective discovery hints, not historical evidence or final event grouping.',
            'Automatic parent protection is only a minimum; semantic overlap review and prior reservation checks precede admission.',
            'Candidate status does not waive rule, settlement, identity or price-time checks.']}
    (output/'report.json').write_text(json_text(report))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json_text(build(a.config, a.output)['counts']))
