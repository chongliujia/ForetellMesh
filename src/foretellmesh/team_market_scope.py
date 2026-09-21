"""Topic-neutral discovery inventory. Retrospective metadata is never admission."""
import argparse
from collections import Counter
from pathlib import Path

from .data import sha256_file, strict_json
from .evaluation import json_text
from .schema import ValidationError, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def select_candidates(markets, membership, config):
    if (config['kind'] != 'team_market_discovery_v1' or type(config['limit']) is not int
            or not 1 <= config['limit'] <= 2000 or type(config['seed']) is not int):
        raise ValidationError('invalid team discovery policy')
    start = timestamp(config['created_from'], 'scope start')
    end = timestamp(config['created_before'], 'scope end')
    if start >= end:
        raise ValidationError('invalid discovery interval')
    # Exclude ALL previously partitioned identities, including old training.
    # New candidates still require sibling/semantic review before admission.
    reserved = {r['event_id'].removeprefix('polymarket:') for r in membership}
    reasons = Counter(); candidates = []; seen = set()
    for row in markets:
        mid = row['market_id']
        if mid in seen:
            raise ValidationError('duplicate discovery market')
        seen.add(mid)
        if mid in reserved or row['benchmark_matches']:
            reasons['reserved_identity'] += 1; continue
        if not row['tokens']:
            reasons['unsupported_token_mapping'] += 1; continue
        if row['created_at'] is None:
            reasons['missing_creation_hint'] += 1; continue
        try:
            created = timestamp(row['created_at'], 'creation hint')
        except ValidationError:
            reasons['invalid_creation_hint'] += 1; continue
        if not start <= created < end:
            reasons['outside_creation_hint_window'] += 1; continue
        candidates.append(row)
    ranked = sorted(candidates, key=lambda r: (canonical_hash([config['seed'], r['market_id']]), r['market_id']))
    selected = []
    for row in ranked[:config['limit']]:
        selected.append({**row, 'ready_for_training': False, 'ready_for_scoring': False,
                         'discovery_only': True, 'historical_features_admitted': False})
    return selected, {'catalog_markets': len(seen), 'eligible_discovery_candidates': len(candidates),
                      'selected': len(selected), 'exclusions': dict(sorted(reasons.items()))}


def build(config_path, catalog, membership_path, output):
    if output.exists():
        raise ValidationError('discovery output exists')
    config = strict_json(config_path.read_text()); report = strict_json((catalog/'report.json').read_text())
    if (sha256_file(catalog/'report.json') != config['catalog_report_sha256']
            or sha256_file(catalog/'markets.jsonl') != report['artifact_hashes']['markets.jsonl']
            or sha256_file(membership_path) != config['membership_sha256']):
        raise ValidationError('discovery source changed')
    membership = [strict_json(s) for s in membership_path.read_text().splitlines()]
    with (catalog/'markets.jsonl').open() as stream:
        selected, counts = select_candidates((strict_json(s) for s in stream), membership, config)
    if not selected:
        raise ValidationError('empty discovery scope')
    output.mkdir(parents=True)
    (output/'markets.jsonl').write_text(jsonl(selected))
    result = {'kind': 'team_market_discovery_selection_v1', 'status': 'discovery_not_admitted',
              'config': config, 'config_sha256': sha256_file(config_path),
              'catalog_report_sha256': sha256_file(catalog/'report.json'),
              'selection_sha256': sha256_file(output/'markets.jsonl'), 'counts': counts,
              'code_sha256': sha256_file(Path(__file__)), 'admitted_training_markets': 0,
              'limitations': ['No keyword, category, final price, volume or closed-status selection.',
                  'Creation dates and text are retrospective discovery hints, not historical features.',
                  'Exact reservation filtering is not event-sibling or semantic clearance.',
                  'Supported Yes/No tokens only; unsupported outcome labels remain outside this adapter.',
                  'Selected markets, including missing prices, remain in the coverage denominator.']}
    (output/'report.json').write_text(json_text(result))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'catalog', 'membership', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    print(json_text(build(a.config, a.catalog, a.membership, a.output)))
