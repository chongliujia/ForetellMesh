"""Replay a bounded, documented cross-benchmark review for a frozen cohort.

Native aliases and exact text matches are mechanical checks. Semantic decisions
are explicit reviewed annotations bound to the complete index and cohort; a
lexical non-match is never treated as a semantic clearance.
"""
from collections import Counter
from copy import deepcopy
import re

from .data import question_key, sha256_file, strict_json
from .historical_sources import read_archive
from .schema import ValidationError, fields, parse_record
from .synthetic_sft import canonical_hash


def read_index(path):
    index = strict_json(path.read_text())
    if (index.get('schema_version') != '1' or index.get('source') != 'forecastbench'
            or not index.get('entries') or canonical_hash(index['entries']) != index.get('entries_sha256')
            or len({e['event_id'] for e in index['entries']}) != len(index['entries'])):
        raise ValidationError('invalid heldout review index')
    return index


def native_aliases(candidates, capture):
    _, sources = read_archive(capture)
    aliases = {}
    for c in candidates:
        event = c['record']['event_id']
        if event in aliases:continue
        ids = {event}
        if event.startswith('polymarket:'):
            ref, raw = sources[c['market_ref']['url']]
            if ref != c['market_ref']:raise ValidationError('unbound native identity source')
            markets = [m for m in strict_json(raw.decode())['markets'] if 'polymarket:'+str(m['id']) == event]
            if len(markets) != 1:raise ValidationError('ambiguous native market identity')
            condition = markets[0]['conditionId']
            if not re.fullmatch(r'0x[0-9a-fA-F]{64}', condition):raise ValidationError('invalid native condition')
            ids.add('polymarket:'+condition.lower())
        aliases[event] = sorted(ids)
    return aliases


def cohort_projection(candidates, aliases):
    return sorted([{'sample_id': c['record']['sample_id'], 'event_group_id': c['record']['event_group_id'],
                    'aliases': aliases[c['record']['event_id']],
                    'input': parse_record(c['record']).forecast_input.to_payload(),
                    'original_question': c['original_question']} for c in candidates], key=lambda r: r['sample_id'])


def exact_overlaps(aliases, questions, index):
    ids = {a.lower() for a in aliases}; texts = {question_key(q) for q in questions}
    return sorted({e['event_id'] for e in index['entries']
                   if e['event_id'].lower() in ids or question_key(e['question']) in texts
                   or question_key(e['question'].split('\n', 1)[0]) in texts})


def apply_review(candidates, capture, heldout_index, review_path):
    index = read_index(heldout_index); policy = strict_json(review_path.read_text())
    fields(policy, {'schema_version', 'review_id', 'heldout_index_sha256', 'cohort_projection_sha256',
                    'review_scope', 'groups', 'related_entry_ids', 'related_entries_rationale',
                    'allowed_use', 'training', 'prompt_tuning', 'reward_tuning', 'checkpoint_selection'}, 'benchmark review')
    if (policy['schema_version'] != '1' or policy['allowed_use'] != 'heldout_replay'
            or any(policy[k] is not False for k in ('training', 'prompt_tuning', 'reward_tuning', 'checkpoint_selection'))
            or policy['heldout_index_sha256'] != sha256_file(heldout_index)
            or not all(isinstance(policy[k], str) and policy[k].strip()
                       for k in ('review_id', 'review_scope', 'related_entries_rationale'))):
        raise ValidationError('review use or heldout index changed')
    aliases = native_aliases(candidates, capture); projected = cohort_projection(candidates, aliases)
    if canonical_hash(projected) != policy['cohort_projection_sha256']:
        raise ValidationError('reviewed cohort inputs or native identities changed')
    groups = {g['event_group_id']: g for g in policy['groups']}
    if len(groups) != len(policy['groups']) or set(groups) != {c['record']['event_group_id'] for c in candidates}:
        raise ValidationError('review does not cover exact cohort groups')
    related = policy['related_entry_ids']; index_ids = {e['event_id'] for e in index['entries']}
    if not isinstance(related, list) or len(set(related)) != len(related) or set(related)-index_ids:
        raise ValidationError('unbound related benchmark entry')
    for g in groups.values():
        fields(g, {'event_group_id', 'disposition', 'rationale'}, 'group review')
        if g['disposition'] not in ('evaluation_only', 'exclude') or not g['rationale'].strip():
            raise ValidationError('unsupported semantic disposition')
    matches = {}
    for c in candidates:
        record = c['record']
        found = exact_overlaps(aliases[record['event_id']],
            [c['original_question'], c['original_question'].split('\n', 1)[0], record['question']], index)
        if found:matches.setdefault(record['event_group_id'], set()).update(found)
    rows = deepcopy(candidates)
    for c in rows:
        group = c['record']['event_group_id']; reasons = set(c['blockers'])
        reasons.discard('semantic_benchmark_review_required')
        reasons.add('evaluation_only_reserved')
        if group in matches:reasons.add('heldout_same_event_group_overlap')
        if groups[group]['disposition'] == 'exclude':reasons.add('semantic_review_excluded_group')
        c['blockers'] = sorted(reasons)
        c['ready_for_sft'] = False; c['ready_for_benchmark'] = False
    report = {'kind': 'frozen_benchmark_scope_review', 'review_id': policy['review_id'],
              'review_sha256': sha256_file(review_path), 'heldout_index_sha256': sha256_file(heldout_index),
              'cohort_projection_sha256': policy['cohort_projection_sha256'],
              'index_entries': len(index['entries']), 'reviewed_rows': len(rows), 'reviewed_groups': len(groups),
              'exact_overlap_groups': {g: sorted(ids) for g, ids in sorted(matches.items())},
              'related_entry_ids': related, 'related_entry_count': len(related),
              'usage': {k: policy[k] for k in ('allowed_use', 'training', 'prompt_tuning', 'reward_tuning', 'checkpoint_selection')},
              'blocker_counts': dict(Counter(b for c in rows for b in c['blockers'])),
              'model_input_payloads_unchanged': projected == cohort_projection(rows, aliases),
              'limitations': ['Review is specific to this frozen index and these groups, not a universal semantic deduplicator.',
                             'Shared macro drivers are recorded conservatively; different releases are not assumed statistically independent.',
                             'This cohort and its linked benchmark entries are reserved for evaluation; no training or development tuning.']}
    if not report['model_input_payloads_unchanged']:raise ValidationError('review changed model inputs')
    version = 'sha256:'+canonical_hash({'parent_versions': sorted({c['record']['dataset_version'] for c in candidates}), 'review': report})
    for c in rows:c['record']['dataset_version'] = version
    report['dataset_version'] = version
    return rows, report
