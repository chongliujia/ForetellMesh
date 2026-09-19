"""Inventory native PMA Parquet shards and preserve benchmark reservations.

This scans all extracted shard metadata and market rows, without loading the
trade corpus into memory. Snapshot terminal prices are isolated as hints, never
converted into resolution labels or historical forecast features.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

from .benchmark_review import read_index
from .data import question_key, sha256_file, strict_json
from .evaluation import json_text
from .polymarket_dataset import mapping
from .schema import ValidationError, iso


MACRO = re.compile(r'\b(fed|fomc|inflation|cpi|unemployment)\b', re.I)


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')


def extraction_entries(root: Path):
    manifest = strict_json((root/'manifest.json').read_text())
    if (manifest.get('schema_version') != '1' or manifest.get('kind') != 'pma_selective_extraction'
            or sha256_file(root/'members.jsonl') != manifest['member_ledger_sha256']):
        raise ValidationError('invalid PMA extraction manifest or ledger')
    selected = [strict_json(line) for line in (root/'members.jsonl').read_text().splitlines() if line]
    selected = [entry for entry in selected if entry['selected']]
    if selected != manifest['selected']:raise ValidationError('extraction ledger and selected entries disagree')
    seen = set()
    for entry in selected:
        path = (root/entry['file']).resolve()
        if (not path.is_relative_to(root.resolve()) or entry['file'] in seen
                or entry['name'] != 'data/'+entry['file'] or not entry['selected']
                or path.stat().st_size != entry['bytes']):raise ValidationError('unbound extracted shard')
        seen.add(entry['file'])
    return manifest, selected


def exclusion_lookup(index_path, inventory_path):
    index = read_index(index_path); aliases, questions = defaultdict(set), defaultdict(set)
    for item in index['entries']:
        aliases[item['event_id'].lower()].add(item['event_id'])
        for q in (item['question'], item['question'].split('\n', 1)[0]):questions[question_key(q)].add(item['event_id'])
    inventory = strict_json(inventory_path.read_text())
    for group in inventory['groups']:
        for event in group['native_events']:
            if event['platform'] != 'polymarket':continue
            for market in event['contracts']:
                for alias in market['aliases']:aliases[alias.lower()].add(group['event_group_id'])
    return aliases, questions


def market_record(row, ref, row_number, aliases, questions):
    blockers = ['historical_question_rules_required', 'event_group_review_required', 'semantic_dedup_review_required',
                'final_settlement_proof_required']
    tokens = None
    try:
        tokens, _ = mapping({'id': row['id'], 'conditionId': row['condition_id'],
                             'outcomes': row['outcomes'], 'clobTokenIds': row['clob_token_ids']})
    except (ValidationError, KeyError):blockers.append('invalid_or_missing_binary_token_mapping')
    def time_field(key):
        value = row[key]
        if value is None:return None
        if not isinstance(value, datetime) or value.tzinfo is None:
            blockers.append(key+'_timezone_unproven'); return str(value)
        return iso(value)
    created, end = time_field('created_at'), time_field('end_date')
    ids = ['polymarket:'+str(row['id']), 'polymarket:'+str(row['condition_id']).lower()]
    overlaps = set().union(*(aliases.get(key, set()) for key in ids), questions.get(question_key(row['question']), set()))
    if overlaps:blockers.append('reserved_benchmark_identity_or_exact_question')
    if row['closed'] is not True:blockers.append('snapshot_not_closed')
    return {'market_id': str(row['id']), 'condition_id': row['condition_id'], 'question': row['question'],
            'created_at': created, 'end_date_hint': end, 'tokens': tokens, 'snapshot_closed': row['closed'],
            'metadata_fetched_at_raw': str(row['_fetched_at']),
            'source_ref': {'file': ref['file'], 'sha256': ref['sha256'], 'row': row_number},
            'benchmark_matches': sorted(overlaps), 'blockers': sorted(blockers),
            'event_group_id': None, 'split': None, 'ready_for_training': False, 'ready_for_scoring': False}


def shard_info(path, entry):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path); schema = parquet.schema_arrow
    summary = {**entry, 'rows': parquet.metadata.num_rows, 'row_groups': parquet.metadata.num_row_groups,
               'schema': {field.name: str(field.type) for field in schema}, 'column_bounds': {}}
    for key in ('block_number', 'timestamp', 'created_at', 'end_date', '_fetched_at'):
        if key not in schema.names:continue
        idx = schema.names.index(key); values = []
        for group in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(group).column(idx).statistics
            if stats and stats.has_min_max:values.append((stats.min, stats.max))
        if values:
            low, high = min(x[0] for x in values), max(x[1] for x in values)
            summary['column_bounds'][key] = {'min': low if isinstance(low, int) else str(low),
                                              'max': high if isinstance(high, int) else str(high)}
    return parquet, summary


def build(extraction: Path, index_path: Path, inventory_path: Path, output: Path):
    if output.exists():raise ValidationError('PMA catalog output exists')
    manifest, entries = extraction_entries(extraction)
    aliases, questions = exclusion_lookup(index_path, inventory_path)
    output.mkdir(parents=True); counts, years, macro_years = Counter(), Counter(), Counter()
    schema_variants = defaultdict(Counter); seen_ids = set(); token_owners = {}; collisions = set()
    columns = ['id', 'condition_id', 'question', 'outcomes', 'outcome_prices', 'clob_token_ids',
               'closed', 'created_at', 'end_date', '_fetched_at']
    names = ['markets.jsonl', 'macro_candidates.jsonl', 'snapshot_outcome_hints.jsonl', 'shards.jsonl']
    streams = {name: (output/name).open('w') for name in names}
    try:
        for n, entry in enumerate(sorted(entries, key=lambda e: e['file'])):
            path = extraction/entry['file']
            if sha256_file(path) != entry['sha256']:raise ValidationError('native PMA shard hash changed')
            parquet, summary = shard_info(path, entry); table = entry['file'].split('/')[1]
            counts[table+'_shards'] += 1; counts[table+'_rows'] += summary['rows']
            schema_variants[table][json.dumps(summary['schema'], sort_keys=True)] += 1
            write_row(streams['shards.jsonl'], summary)
            if table != 'markets':continue
            missing = set(columns)-set(parquet.schema_arrow.names)
            if missing:raise ValidationError('PMA market schema missing '+str(sorted(missing)))
            row_number = 0
            for batch in parquet.iter_batches(batch_size=4096, columns=columns):
                for row in batch.to_pylist():
                    if row['id'] in seen_ids:raise ValidationError('duplicate PMA native market ID requires explicit snapshot policy')
                    seen_ids.add(row['id']); result = market_record(row, entry, row_number, aliases, questions); row_number += 1
                    for token in (result['tokens'] or {}).values():
                        if token in token_owners and token_owners[token] != row['id']:collisions.add(token)
                        token_owners[token] = row['id']
                    year = result['end_date_hint'][:4] if result['end_date_hint'] else 'missing'; years[year] += 1
                    counts['binary_token_mapped_markets'] += result['tokens'] is not None
                    counts['closed_snapshot_markets'] += result['snapshot_closed'] is True
                    counts['exact_reserved_markets'] += bool(result['benchmark_matches'])
                    write_row(streams['markets.jsonl'], result)
                    if result['snapshot_closed'] is True and MACRO.search(result['question']):
                        counts['closed_macro_keyword_candidates'] += 1; macro_years[year] += 1
                        write_row(streams['macro_candidates.jsonl'], result)
                        write_row(streams['snapshot_outcome_hints.jsonl'], {'market_id': result['market_id'],
                            'outcomes_raw': row['outcomes'], 'snapshot_prices_raw': row['outcome_prices'],
                            'source_ref': result['source_ref'], 'is_resolution_label': False})
            if n % 1000 == 0:print(f'Cataloged {n+1}/{len(entries)} shards', flush=True)
    finally:
        for stream in streams.values():stream.close()
    report = {'schema_version': '1', 'kind': 'pma_polymarket_catalog', 'status': 'inventory_not_admitted',
              'dataset_version': 'sha256:'+manifest['archive_sha256'],
              'extraction_manifest_sha256': sha256_file(extraction/'manifest.json'),
              'heldout_index_sha256': sha256_file(index_path), 'reserved_inventory_sha256': sha256_file(inventory_path),
              'counts': dict(sorted(counts.items())), 'market_end_hint_year_counts': dict(sorted(years.items())),
              'closed_macro_keyword_end_hint_year_counts': dict(sorted(macro_years.items())),
              'token_collision_count': len(collisions),
              'schema_variants': {table: [{'fields': json.loads(k), 'shards': v} for k, v in variants.items()]
                                  for table, variants in sorted(schema_variants.items())},
              'artifact_hashes': {name: sha256_file(output/name) for name in names},
              'model_calls': 0, 'admitted_training_rows': 0, 'admitted_evaluation_rows': 0,
              'limitations': ['Keyword candidates include non-US releases and mention markets; no automatic country or event grouping.',
                             'End dates are scheduling hints, not official release or resolution timestamps.',
                             'Market snapshots were fetched after most historical observations; current text/prices are not historical features.',
                             'Closed flags and terminal snapshot prices are not verified payout labels.',
                             'Exact benchmark matches are exclusions; non-matches are not semantic clearance.',
                             'Full native trade row counts do not certify deduplication or complete chain coverage.']}
    (output/'report.json').write_text(json_text(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('extraction', 'heldout-index', 'reserved-inventory', 'output'):parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args(); print(json_text(build(args.extraction, args.heldout_index, args.reserved_inventory, args.output)))


if __name__ == '__main__':main()
