"""Fixed, label-independent sampling for the information-state experiment."""
from collections import Counter, defaultdict
from itertools import product
import random

from .agent_baseline_data import input_context
from .agent_runtime import validate_agent_output
from .research_tool_counterfactual import CONDITIONS
from .schema import ValidationError, fields
from .synthetic_sft import canonical_hash

TRAIN_POLICY = 'legacy_plus_paired_information_v1'
EVALUATION_POLICY = 'counterfactual_validation_pairs_v1'


def validate_curriculum(policy: dict) -> dict:
    fields(policy, {'policy', 'selection_seed', 'groups_per_family'}, 'curriculum')
    if (policy['policy'] != TRAIN_POLICY or type(policy['selection_seed']) is not int
            or not 0 <= policy['selection_seed'] < 2**32
            or type(policy['groups_per_family']) is not int or policy['groups_per_family'] != 8):
        raise ValidationError('unsupported curriculum')
    return policy


def information_groups(rows: list[dict], partition: str) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    identities = set()
    for row in rows:
        if row['partition'] != partition or row['sample_id'] in identities:
            raise ValidationError('curriculum partition or identity mismatch')
        identities.add(row['sample_id'])
        case = row['case']
        if case['kind'] == 'information_state':
            if case['family'] not in CONDITIONS:raise ValidationError('unknown curriculum family')
            groups[case['family']][row['event_group_id']].append(row)
    if set(groups) != set(CONDITIONS):raise ValidationError('missing curriculum family')
    for family, members in groups.items():
        expected = set(product(CONDITIONS[family], ('calculation', 'forecast'), ('field_names', 'aliases'), ('en', 'zh'), ('research', 'risk')))
        for values in members.values():
            actual = {tuple(row['case'][k] for k in ('condition', 'scope', 'format', 'language', 'role')) for row in values}
            if len(values) != len(expected) or actual != expected:
                raise ValidationError('incomplete counterfactual group')
    return groups


def select_training_rows(rows: list[dict], policy: dict | None = None) -> tuple[list[dict], dict | None]:
    if policy is None:return rows, None
    validate_curriculum(policy)
    groups = information_groups(rows, 'train')
    if any(len(members) != policy['groups_per_family'] for members in groups.values()):
        raise ValidationError('curriculum requires eight training groups per family')
    if any(row['case']['kind'] not in ('legacy', 'information_state') for row in rows):
        raise ValidationError('curriculum requires the v3 source bundle')
    chosen = [row for row in rows if row['case']['kind'] == 'legacy']
    if not chosen:raise ValidationError('curriculum must retain legacy tasks')
    contexts = list(product(('en', 'zh'), ('research', 'risk'), ('field_names', 'aliases')))
    for family, members in sorted(groups.items()):
        shuffled = list(contexts)
        random.Random(canonical_hash({'seed': policy['selection_seed'], 'family': family})).shuffle(shuffled)
        for group, (language, role, fmt) in zip(sorted(members), shuffled):
            # Four disclosures x both scopes. Each family's eight groups cover
            # every language/role/naming context exactly once, without repeats.
            chosen.extend(row for row in members[group] if (row['language'], row['request']['agent'], row['case']['format']) == (language, role, fmt))
    chosen.sort(key=lambda row: row['sample_id'])
    new = [row for row in chosen if row['case']['kind'] == 'information_state']
    if len(new) != 256:raise ValidationError('unexpected curriculum size')
    selection = {'policy': policy, 'source_examples': len(rows), 'selected_examples': len(chosen),
        'legacy_examples': len(chosen) - len(new), 'information_examples': len(new),
        'event_groups': len({row['event_group_id'] for row in chosen}),
        'tasks': dict(Counter(row['task'] for row in chosen)),
        'sample_ids': [row['sample_id'] for row in chosen]}
    return chosen, selection


def select_information_validation(rows: list[dict], policy: str) -> list[dict]:
    if policy != EVALUATION_POLICY:raise ValidationError('unsupported information evaluation policy')
    groups = information_groups(rows, 'validation')
    chosen = []
    for family, members in sorted(groups.items()):
        if len(members) < 2:raise ValidationError('information evaluation needs two groups per family')
        for index, group in enumerate(sorted(members)[:2]):
            for row in members[group]:
                role = row['request']['agent']
                fmt = 'aliases' if (role == 'risk') != bool(index) else 'field_names'
                if row['language'] == ('en', 'zh')[index] and row['case']['format'] == fmt:
                    chosen.append(row)
    chosen.sort(key=lambda row: row['sample_id'])
    if len(chosen) != 128:raise ValidationError('unexpected information evaluation size')
    return chosen


def judge_information(row: dict, output: dict | None) -> dict:
    if output is None:
        return {'schema_valid': False, 'vocabulary_valid': False, 'unknowns_correct': False, 'task_correct': False}
    role = row['request']['agent']
    validate_agent_output(role, output, input_context(row['request']['input']))
    actual, expected = set(output['unknowns']), set(row['target']['unknowns'])
    allowed = set(row['oracle']['output_names'].values())
    grounded = (all(set(output[key]) == set(row['target'][key]) for key in ('evidence_ids', 'counter_evidence_ids'))
                if role == 'research' else output['risks'] == [])
    return {'schema_valid': True, 'vocabulary_valid': actual <= allowed, 'unknowns_correct': actual == expected,
            'task_correct': actual == expected and grounded, 'grounding_correct': grounded,
            'known_as_unknown': sorted(actual & (allowed - expected)), 'missing_unknowns': sorted(expected - actual),
            'out_of_vocabulary': sorted(actual - allowed)}


def summarize_information(tasks: list[dict], outputs: dict) -> dict:
    indexed = {row['sample_id']: row for row in tasks}
    if not tasks or len(indexed) != len(tasks) or set(indexed) != set(outputs):
        raise ValidationError('information evaluation identity mismatch')
    details = {sid: judge_information(row, outputs[sid]) for sid, row in indexed.items()}
    def counts(ids):
        return {'examples': len(ids), **{key: sum(details[sid][key] for sid in ids)
            for key in ('schema_valid', 'vocabulary_valid', 'unknowns_correct', 'task_correct')},
            'known_as_unknown_examples': sum(bool(details[sid].get('known_as_unknown')) for sid in ids),
            'missing_unknown_examples': sum(bool(details[sid].get('missing_unknowns')) for sid in ids)}
    def grouped(keys):
        groups = defaultdict(list)
        for sid, row in indexed.items():
            groups[tuple(row['event_group_id'] if key == 'group' else row['case'][key] for key in keys)].append(sid)
        return groups
    scope_pairs = grouped(('group', 'language', 'role', 'format', 'condition'))
    disclosure_sets = grouped(('group', 'language', 'role', 'format', 'scope'))
    if any(len(v) != 2 for v in scope_pairs.values()) or any(len(v) != 4 for v in disclosure_sets.values()):
        raise ValidationError('information evaluation requires complete paired scopes and disclosure sets')
    report = {**counts(list(indexed)), 'details': details,
        'event_groups': len({r['event_group_id'] for r in tasks}),
        'scope_pairs': {'total': len(scope_pairs), 'all_correct': sum(all(details[i]['task_correct'] for i in ids) for ids in scope_pairs.values())},
        'disclosure_sets': {'total': len(disclosure_sets), 'all_correct': sum(all(details[i]['task_correct'] for i in ids) for ids in disclosure_sets.values())}}
    for name in ('family', 'condition', 'role', 'language', 'format', 'scope'):
        report['by_' + name] = {value: counts([sid for sid, r in indexed.items() if r['case'][name] == value])
                               for value in sorted({r['case'][name] for r in tasks})}
    return report
