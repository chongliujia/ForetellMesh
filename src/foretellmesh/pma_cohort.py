"""Freeze a small, reviewed PMA research dataset from replayed raw proofs.

Research partitions are distinct from formal multi-agent benchmark admission.
This exporter never launches training and never manufactures SFT probabilities.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from .benchmark_review import read_index, exact_overlaps
from .data import sha256_file, strict_json, question_key
from .evaluation import json_text
from .metrics import score_predictions
from .pma_proof_pilot import inputs, require, read_jsonl
from .pma_proof_replay import build as replay
from .schema import parse_record, timestamp
from .synthetic_sft import canonical_hash

PARTITIONS = ('train', 'validation', 'test')


def split_for(release_time, policy):
    require(set(policy) == {'train_release_before', 'validation_release_before', 'test_release_before',
                           'unit', 'label_before_next_split_observation'}
            and policy['unit'] in ('scheduled_fomc_release', 'scheduled_us_macro_release')
            and policy['label_before_next_split_observation'] is True,
            'unsupported chronological policy')
    bounds = [timestamp(policy[k+'_release_before'], k+' cutoff') for k in PARTITIONS]
    require(bounds[0] < bounds[1] < bounds[2], 'split cutoffs not strictly chronological')
    point = timestamp(release_time, 'release time')
    for name, bound in zip(PARTITIONS, bounds):
        if point < bound:
            return name
    raise ValueError('event outside frozen release window')


def cohort_inputs(config_path):
    config = strict_json(config_path.read_text())
    macro = config.get('purpose') == 'pma_macro_chronological_research_dataset'
    require(config.get('schema_version') == '1' and config.get('purpose') in (
                'pma_fomc_chronological_research_dataset', 'pma_macro_chronological_research_dataset')
            and config.get('minimum_formal_evaluation_groups') == 20 and config.get('formal_evaluation') is False
            and config.get('automatic_training') is False and 1 <= len(config['batches']) <= (16 if macro else 8), 'unsupported cohort policy')
    require(config['split_policy']['unit'] == ('scheduled_us_macro_release' if macro else 'scheduled_fomc_release'),
            'split unit differs from cohort scope')
    split_for('2000-01-01T00:00:00Z', config['split_policy'])
    index_path = (config_path.parent/config['benchmark_index']['path']).resolve()
    require(sha256_file(index_path) == config['benchmark_index']['sha256'], 'benchmark index changed')
    index = read_index(index_path)
    batches, projection, groups, source_bindings = [], [], {}, {}
    for batch in config['batches']:
        path = (config_path.parent/batch['config']).resolve()
        require(sha256_file(path) == batch['config_sha256'], 'batch config changed')
        plan, paths, selected = inputs(path)
        require(macro or plan['purpose'] == 'pma_historical_proof_pilot', 'BLS batch outside FOMC-only policy')
        hashes = {k: sha256_file(v) for k, v in paths.items()}
        require(hashes['heldout_index'] == config['benchmark_index']['sha256'], 'mixed benchmark versions')
        require(all(k not in source_bindings or source_bindings[k] == v for k, v in hashes.items()), 'mixed upstream source versions')
        source_bindings.update(hashes)
        for group in plan['groups']:
            gid = group['event_group_id']
            require(gid not in groups, 'event group repeated across capture batches')
            groups[gid] = {**group, 'planned_split': split_for(group['release']['published_at'], config['split_policy'])}
        for row in selected:
            projection.append({'market_id': row['market_id'], 'event_group_id': row['event_group_id'],
                'native_event_id': row['native_event_id'],
                'condition_id': row['market']['conditionId'], 'question': row['market']['question'],
                'description': row['market']['description'], 'tokens': row['candidate']['tokens']})
        batches.append({'config': path, 'capture': (config_path.parent/batch['capture']).resolve(),
                        'plan': plan, 'paths': paths, 'selected': selected})
    projection.sort(key=lambda r: int(r['market_id']))
    require(len({p['market_id'] for p in projection}) == len(projection) <= (200 if macro else 100)
            and len({p['condition_id'] for p in projection}) == len(projection)
            and len(groups) <= (48 if macro else 24), 'duplicate or unbounded cohort identities')
    return config, index, batches, projection, groups


def review_scope(review_path, config_path, config, index, projection, groups):
    """Validate a content-bound explicit review; lexical non-match is not review."""
    review = strict_json(review_path.read_text())
    review_id = 'pma_macro_scope_review_v1' if config['purpose'] == 'pma_macro_chronological_research_dataset' else 'pma_fomc_scope_review_v1'
    require(review.get('schema_version') == '1' and review.get('review_id') == review_id
            and review.get('cohort_config_sha256') == sha256_file(config_path)
            and review.get('cohort_projection_sha256') == canonical_hash(projection)
            and review.get('benchmark_index_sha256') == config['benchmark_index']['sha256']
            and review.get('benchmark_entries_sha256') == canonical_hash(index['entries']), 'review scope/content changed')
    require(review.get('unselected_parent_policy') == 'exclude_all_from_this_release'
            and review.get('rationale') and review.get('historical_knowledge_policy'), 'incomplete scope review')
    reviewed = {g['event_group_id']: g for g in review['groups']}
    require(len(reviewed) == len(review['groups']) and set(reviewed) == set(groups), 'group review does not cover cohort')
    index_ids = {e['event_id'] for e in index['entries']}
    for g in reviewed.values():
        require(g['disposition'] in ('release_in_planned_split', 'benchmark_related_reserve', 'exclude')
                and g['rationale'] and set(g['related_benchmark_entry_ids']) <= index_ids, 'unsupported group review')
        if g['disposition'] == 'benchmark_related_reserve':
            require(g['related_benchmark_entry_ids'], 'unbound benchmark reservation')
    # Mechanical exact matches override any manually supplied release disposition.
    exact = defaultdict(set)
    for row in projection:
        matches = exact_overlaps(['polymarket:'+row['market_id'], 'polymarket:'+row['condition_id']],
                                 [row['question']], index)
        exact[row['event_group_id']].update(matches)
    return reviewed, {g: sorted(ids) for g, ids in exact.items() if ids}


def validate_partitions(partitions):
    owners, events, questions, samples = {}, {}, {}, set()
    for split in PARTITIONS:
        for row in partitions[split]:
            record = parse_record(row)
            require(record.label is not None and record.forecast_input.market is not None, 'release row lacks label or market')
            require(record.sample_id not in samples, 'duplicate released sample')
            samples.add(record.sample_id)
            for table, key in ((owners, record.event_group_id), (events, record.event_id),
                               (questions, question_key(row['question']))):
                require(key not in table or table[key] == split, 'event or equivalent question crosses partitions')
                table[key] = split
    for i, earlier in enumerate(PARTITIONS[:-1]):
        later = [r for name in PARTITIONS[i+1:] for r in partitions[name]]
        if not partitions[earlier] or not later:
            continue
        earliest = min(timestamp(r['observation_time'], 'later observation') for r in later)
        require(max(timestamp(r['observation_time'], 'earlier observation') for r in partitions[earlier]) < earliest,
                'observation times cross partitions')
        require(max(timestamp(r['label']['resolution_time'], 'earlier settlement') for r in partitions[earlier]) < earliest,
                'earlier labels settle after a later partition starts')


def development_baselines(partitions):
    """Only train/validation diagnostics; do not inspect frozen test targets here."""
    train_groups = defaultdict(list)
    for row in partitions['train']:
        train_groups[row['event_group_id']].append(row['label']['outcome'])
    base_rate = (sum(sum(v)/len(v) for v in train_groups.values())/len(train_groups)) if train_groups else None
    report = {'kind': 'development_data_diagnostics', 'test_scored': False,
              'training_event_weighted_base_rate': base_rate, 'partitions': {}}
    for split in ('train', 'validation'):
        rows = partitions[split]; outcomes = [r['label']['outcome'] for r in rows]
        arms = {'constant_0_5': [0.5]*len(rows), 'market': [r['market']['probability'] for r in rows]}
        if split == 'validation' and base_rate is not None:
            arms['training_base_rate'] = [base_rate]*len(rows)
        scores = {}
        for name, predictions in arms.items():
            scores[name] = score_predictions(predictions, outcomes, ece_bins=10)
            per_group = defaultdict(list)
            for row, p in zip(rows, predictions):
                per_group[row['event_group_id']].append((p, row['label']['outcome']))
            group_scores = [score_predictions([p for p, _ in pairs], [y for _, y in pairs]) for pairs in per_group.values()]
            for metric in ('brier', 'log_loss'):
                scores[name]['event_weighted_'+metric] = sum(s[metric] for s in group_scores)/len(group_scores) if group_scores else None
        report['partitions'][split] = {'rows': len(rows), 'event_groups': len({r['event_group_id'] for r in rows}),
            'scores': scores, 'ece_weighting': 'observation_weighted_descriptive_only'}
    return report


def write_rows(path, rows):
    path.write_text(''.join(json_text(row).replace('\n', '')+'\n' for row in rows))


def preserve_parent(partitions, parent, expected_hash):
    """An additive release must retain frozen inputs, labels and split roles."""
    require(sha256_file(parent/'report.json') == expected_hash, 'parent release report changed')
    report = strict_json((parent/'report.json').read_text())
    count = 0
    for split in PARTITIONS:
        name = f'partitions/{split}.records.jsonl'
        require(sha256_file(parent/name) == report['artifact_hashes'][name], 'parent records changed')
        current = {r['sample_id']: r for r in partitions[split]}
        for old in read_jsonl(parent/name):
            require(old['sample_id'] in current, 'parent sample removed or reassigned')
            new = current[old['sample_id']]
            ignore = {'dataset_source', 'dataset_version'}
            require({k: v for k, v in old.items() if k not in ignore} == {k: v for k, v in new.items() if k not in ignore},
                    'frozen parent input or label changed')
            count += 1
    return count


def build(config_path, review_path, output):
    config, index, batches, projection, groups = cohort_inputs(config_path)
    macro = config['purpose'] == 'pma_macro_chronological_research_dataset'
    reviewed, exact = review_scope(review_path, config_path, config, index, projection, groups)
    require(not output.exists(), 'cohort output already exists')
    output.mkdir(parents=True); (output/'audit').mkdir(); (output/'partitions').mkdir()
    candidates, contracts, reports = [], {}, []
    # Regenerate from raw archives; never trust edited ready flags or reports.
    for i, batch in enumerate(batches):
        root = output/'audit'/f'{i:02d}'
        report = replay(batch['config'], batch['capture'], root)
        reports.append({'config_sha256': sha256_file(batch['config']),
                        'capture_report_sha256': sha256_file(batch['capture']/'report.json'), 'replay_report_sha256': sha256_file(root/'report.json')})
        candidates.extend(read_jsonl(root/'candidates.jsonl'))
        for contract in read_jsonl(root/'contracts.jsonl'):
            require(contract['market_id'] not in contracts, 'duplicate replayed contract')
            contracts[contract['market_id']] = contract
    blocked_groups = set(exact)
    blocked_groups.update(c['event_group_id'] for c in contracts.values() if c['existing_reservations'])
    code = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    version = 'sha256:'+canonical_hash({'config': sha256_file(config_path), 'review': sha256_file(review_path),
                                      'batches': reports, 'code': code})
    partitions = {name: [] for name in PARTITIONS}; exclusions, membership = [], []; seen_questions = {}
    for c in sorted(candidates, key=lambda c: c['sample_id']):
        gid = c['event_group_id']; split = groups[gid]['planned_split']; reasons = list(c['data_blockers'])
        disposition = reviewed[gid]['disposition']
        if disposition != 'release_in_planned_split': reasons.append(disposition)
        if gid in blocked_groups: reasons.append('reserved_or_exact_benchmark_overlap_in_event_group')
        record = c['record']
        if record is None: reasons.append('no_verified_record')
        if not reasons:
            parsed = parse_record(record)
            require(c['proof_complete'] and parsed.label is not None and parsed.forecast_input.market is not None, 'replay proof flag inconsistent')
            key = (question_key(record['question']), record['observation_time'])
            if key in seen_questions:
                other = seen_questions[key]
                require(other['event_group_id'] == gid and other['label'] == record['label'], 'duplicate question has inconsistent identity or label')
                reasons.append('duplicate_equivalent_observation')
            else:
                seen_questions[key] = record
        if reasons:
            exclusions.append({'sample_id': c['sample_id'], 'event_group_id': gid, 'planned_split': split, 'reasons': reasons})
            continue
        released = deepcopy(record); released['dataset_source'] = 'foretellmesh_pma_macro_research' if macro else 'foretellmesh_pma_fomc_research'
        released['dataset_version'] = version; partitions[split].append(released)
        membership.append({'sample_id': c['sample_id'], 'event_id': released['event_id'], 'event_group_id': gid,
                           'split': split, 'dataset_version': version})
    validate_partitions(partitions)
    parent_rows = None
    if macro:
        parent_ref = config['parent_release']
        parent_rows = preserve_parent(partitions, (config_path.parent/parent_ref['path']).resolve(), parent_ref['report_sha256'])
    for split, rows in partitions.items():
        write_rows(output/'partitions'/f'{split}.records.jsonl', rows)
        write_rows(output/'partitions'/f'{split}.inputs.jsonl',
                   [{'sample_id': r['sample_id'], 'input': parse_record(r).forecast_input.to_payload()} for r in rows])
        write_rows(output/'partitions'/f'{split}.labels.jsonl', [{'sample_id': r['sample_id'], 'label': r['label']} for r in rows])
    write_rows(output/'membership.jsonl', membership); write_rows(output/'exclusions.jsonl', exclusions)
    group_rows = []
    for gid in sorted(groups):
        planned = sum(c['event_group_id'] == gid for c in candidates)
        released = sum(r['event_group_id'] == gid for r in membership)
        group_rows.append({**groups[gid], **reviewed[gid], 'exact_benchmark_matches': exact.get(gid, []),
            'selected_contracts': sum(p['event_group_id'] == gid for p in projection),
            'planned_rows': planned, 'released_rows': released, 'excluded_rows': planned-released,
            'release_coverage': released/planned if planned else None})
    write_rows(output/'event_groups.jsonl', group_rows)
    # Bind the full candidate universe. No unselected annual/cumulative/combo
    # parent can be imported just because a monthly sibling passed this review.
    catalog_root = batches[0]['paths']['catalog'].parent
    selected_ids = {r['market_id'] for r in projection}
    scope = [{'market_id': r['market_id'], 'condition_id': r['condition_id'],
              'status': 'selected_subject_to_sample_gates' if r['market_id'] in selected_ids else 'not_authorized_by_this_release'}
             for r in read_jsonl(catalog_root/'macro_candidates.jsonl')]
    write_rows(output/'market_scope.jsonl', scope)
    (output/'development_baselines.json').write_text(json_text(development_baselines(partitions)))
    hashes = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()}
    complete_partitions = all(partitions.values())
    released_groups = len({r['event_group_id'] for r in membership})
    heldout_groups = len({r['event_group_id'] for r in partitions['test']})
    formal_blockers = ['single_platform_not_the_existing_dual_platform_comparison', 'model_run_manifest_not_frozen']
    if heldout_groups < config['minimum_formal_evaluation_groups']:
        formal_blockers.insert(0, 'fewer_than_20_distinct_heldout_events')
    result = {'schema_version': '1', 'kind': 'pma_macro_research_release' if macro else 'pma_fomc_research_release',
        'status': 'frozen_research_partitions' if complete_partitions else 'incomplete_research_partitions',
        'has_nonempty_train_validation_test': complete_partitions,
        'preserved_parent_rows': parent_rows,
        'dataset_version': version, 'config_sha256': sha256_file(config_path), 'review_sha256': sha256_file(review_path),
        'selected_contracts': len(projection), 'planned_event_groups': len(groups), 'planned_observations': len(candidates),
        'verified_contracts': sum(c['proof'] is not None for c in contracts.values()),
        'proof_complete_observations_before_scope_review': sum(c['proof_complete'] for c in candidates),
        'released_observations': len(membership), 'excluded_observations': len(exclusions),
        'released_event_groups': released_groups,
        'heldout_event_groups': heldout_groups,
        'partition_counts': {name: {'observations': len(rows), 'contracts': len({r['event_id'] for r in rows}),
            'event_groups': len({r['event_group_id'] for r in rows})} for name, rows in partitions.items()},
        'exclusion_reason_counts': dict(Counter(reason for row in exclusions for reason in row['reasons'])),
        'allowed_use': {'train': 'outcome-supervised forecasting / RL prototypes; task-capable checkpoint required',
                        'validation': 'development diagnostics', 'test': 'heldout; no tuning or training'},
        'ready_for_sft': False, 'training_started': False, 'model_calls': 0,
        'formal_evaluation_admitted': False,
        'formal_evaluation_blockers': formal_blockers,
        'artifact_hashes': hashes, 'code_hashes': code,
        'limitations': ['Retrospective single-domain research dataset; not an independent prospective benchmark.',
            'Correlated brackets/observation times are not independent events; shared macro drivers remain across meetings.',
            'Provider/explorer and official historical publication attestations; not independent consensus or contemporaneous local snapshots.',
            'Chronological boundaries use public chain settlement time, not a claim that this training was executed at that historical time.',
            'Coverage failures remain in exclusions; no forecast-quality gain is claimed. Test baseline scores were not computed.']}
    require(sum(x['observations'] for x in result['partition_counts'].values())+len(exclusions) == len(candidates), 'coverage denominator lost')
    (output/'report.json').write_text(json_text(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'review', 'output'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); result = build(args.config, args.review, args.output)
    print(json_text({k: result[k] for k in ('status', 'selected_contracts', 'released_observations', 'partition_counts', 'exclusion_reason_counts')}))


if __name__ == '__main__': main()
