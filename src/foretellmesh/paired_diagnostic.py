"""Full-factorial, answer-invariant rewrites of a frozen development diagnostic.

This cohort is for diagnosing existing checkpoints, never for training. Its
anchor is byte-identical to the source probe; all eight variants keep the same
allowed names, numeric evidence, observation time, role and unknown-name target.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import tempfile

from .agent_baseline_data import input_context
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .schema import ValidationError, fields
from .sft_data import jsonl
from .uncertainty_diagnostic import judge_uncertainty, read_uncertainty_cohort

VERSION = 'paired_information_diagnostic_v1'
FACTORS = {'wording': ('original', 'audit'), 'formula': ('implicit', 'explicit'),
           'presentation': ('inline', 'json')}
FORMULAS = {
    'mixture': ('Equivalent identity: success_probability = weight*high_rate + (1-weight)*low_rate.',
                '等价恒等式：success_probability = weight*high_rate + (1-weight)*low_rate。'),
    'bayes': ('Equivalent identity: posterior_probability = prior*sensitivity / (prior*sensitivity + (1-prior)*false_positive_rate).',
              '等价恒等式：posterior_probability = prior*sensitivity / (prior*sensitivity + (1-prior)*false_positive_rate)。'),
    'complement': ('Equivalent identity: hit_probability + miss_probability = 1.',
                   '等价恒等式：hit_probability + miss_probability = 1。'),
    'sampling': ('Equivalent identity: empirical_frequency = observed success count / observed historical draw count.',
                 '等价恒等式：empirical_frequency = 已观察到的成功次数 / 已观察到的历史抽样次数。'),
}
AUDIT = (
    'Synthetic information audit. Check only whether the listed quantities are exactly determined by the evidence. '
    'Return your role JSON; unknowns must contain only the corresponding name strings for undetermined quantities, '
    'in any order, or [] when none. Do not output calculations or extra fields. Field meanings: ',
    '合成信息审核。只检查下面列出的量是否能由证据精确确定。按角色的 JSON 格式回答；unknowns 只填写无法确定的量对应的名称字符串，顺序不限，没有则返回 []。不要输出计算数值或额外字段。字段含义：',
)


def load_paired_config(path: Path) -> dict:
    c = fields(strict_json(path.read_text()), {'schema_version', 'dataset_version', 'partition',
        'source_policy', 'factors', 'synthetic_only', 'training_allowed'}, 'paired diagnostic config')
    fixed = {'schema_version': '1', 'dataset_version': VERSION, 'partition': 'validation',
             'source_policy': 'all_frozen_uncertainty_v1_rows', 'factors': {k: list(v) for k, v in FACTORS.items()},
             'synthetic_only': True, 'training_allowed': False}
    if any(type(c[k]) is not type(v) or c[k] != v for k, v in fixed.items()):
        raise ValidationError('unsupported paired diagnostic policy')
    return c


def paired_rows(inputs: list[dict], judges: list[dict]) -> tuple[list[dict], list[dict]]:
    labels = {j['sample_id']: j for j in judges}
    if len(labels) != len(judges) or len(inputs) != len(labels) or {r['sample_id'] for r in inputs} != set(labels):
        raise ValidationError('paired source identities mismatch')
    result_inputs, result_judges = [], []
    for original in inputs:
        label = labels[original['sample_id']]
        zh = label['language'] == 'zh'
        marker = '字段含义：' if zh else 'Field meanings: '
        question = original['input']['question']
        if question.count(marker) != 1:raise ValidationError('source field definitions are ambiguous')
        prefix, definitions = question.split(marker)
        items = [part.split(': ', 1) for part in definitions.split('; ')]
        if any(len(item) != 2 for item in items) or sorted(k for k, _ in items) != label['allowed_fields']:
            raise ValidationError('source field definitions mismatch')
        for levels in product(*FACTORS.values()):
            factors = dict(zip(FACTORS, levels))
            sid = original['sample_id'] + ':paired:' + ':'.join(levels)
            payload = deepcopy(original['input'])
            heading = prefix + marker if factors['wording'] == 'original' else AUDIT[zh]
            listing = definitions if factors['presentation'] == 'inline' else json.dumps(
                [{'name': k, 'meaning': v} for k, v in items], ensure_ascii=False, separators=(',', ':'))
            payload['question'] = heading + listing
            if factors['formula'] == 'explicit':
                payload['evidence'][0]['text'] += ' ' + FORMULAS[label['family']][zh]
            input_context(payload)
            result_inputs.append({'sample_id': sid, 'role': original['role'], 'input': payload})
            result_judges.append({**deepcopy(label), 'sample_id': sid, 'parent_sample_id': original['sample_id'],
                                  'factors': factors})
    return result_inputs, result_judges


def build_paired_cohort(source: Path, config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError('paired cohort already exists')
    load_paired_config(config_path)
    manifest, original, labels = read_uncertainty_cohort(source)
    inputs, judges = paired_rows(original, labels)
    artifacts = {'config.json': config_path.read_text(), 'inputs.jsonl': jsonl(inputs), 'judge.jsonl': jsonl(judges)}
    artifacts.update({'source/' + name: (source/name).read_text()
                      for name in ('config.json', 'manifest.json', 'inputs.jsonl', 'judge.jsonl')})
    report = {'schema_version': '1', 'kind': VERSION, 'partition': 'validation',
              'frozen_at': datetime.now(timezone.utc).isoformat(), 'code': code_provenance(),
              'examples': len(inputs), 'parent_examples': len(original), 'event_groups': manifest['event_groups'],
              'source_manifest_sha256': sha256_file(source/'manifest.json'),
              'artifact_hashes': {k: sha256_bytes(v.encode()) for k, v in artifacts.items()},
              'limitations': ['Reused development scenarios; not an unseen-event or final-test benchmark.',
                  'Eight variants per parent share targets; 256 records represent only four independent numeric scenarios.',
                  'Wording repeats existing role constraints; it is a specified prompt intervention, not all possible paraphrases.',
                  'Explicit formulas restate mathematical identities without supplying missing parameters or realized outcomes.',
                  'Presentation changes only the field glossary, not evidence format or field names.',
                  'No training, probability-score claim, automatic prompt selection or checkpoint promotion.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.paired-', dir=output.parent) as tmp:
        stage = Path(tmp)/'cohort'; (stage/'source').mkdir(parents=True)
        for name, content in artifacts.items():(stage/name).write_text(content)
        (stage/'manifest.json').write_text(json_text(report)); stage.rename(output)
    return report


def read_paired_cohort(path: Path) -> tuple[dict, list[dict], list[dict]]:
    report = strict_json((path/'manifest.json').read_text())
    if report.get('kind') != VERSION or report.get('partition') != 'validation' or report.get('schema_version') != '1':
        raise ValidationError('invalid paired cohort')
    names = {'config.json', 'inputs.jsonl', 'judge.jsonl', 'source/config.json', 'source/manifest.json',
             'source/inputs.jsonl', 'source/judge.jsonl'}
    if set(report['artifact_hashes']) != names:raise ValidationError('paired artifact set mismatch')
    for name, digest in report['artifact_hashes'].items():
        if sha256_file(path/name) != digest:raise ValidationError('paired artifact hash mismatch')
    load_paired_config(path/'config.json')
    source, original, labels = read_uncertainty_cohort(path/'source')
    inputs, judges = paired_rows(original, labels)
    for name, expected in (('inputs.jsonl', inputs), ('judge.jsonl', judges)):
        if [strict_json(s) for s in (path/name).read_text().splitlines()] != expected:
            raise ValidationError('paired replay mismatch')
    if (report['examples'] != len(inputs) or report['parent_examples'] != len(original)
            or report['event_groups'] != source['event_groups']
            or report['source_manifest_sha256'] != sha256_file(path/'source/manifest.json')):
        raise ValidationError('paired metadata mismatch')
    return report, inputs, judges


def summarize_paired(inputs: list[dict], judges: list[dict], results: list[dict],
                     arms: tuple[str, ...] = ('base', 'previous', 'candidate')) -> dict:
    rows = {r['sample_id']: r for r in inputs}; labels = {r['sample_id']: r for r in judges}
    if (not rows or len(rows) != len(inputs) or len(labels) != len(judges) or set(rows) != set(labels)
            or arms != ('base', 'previous', 'candidate')):
        raise ValidationError('invalid paired scoring identities')
    parents = defaultdict(list)
    for sid, label in labels.items():parents[label['parent_sample_id']].append(sid)
    expected_levels = set(product(*FACTORS.values()))
    for ids in parents.values():
        if len(ids) != 8 or {tuple(labels[s]['factors'][k] for k in FACTORS) for s in ids} != expected_levels:
            raise ValidationError('incomplete paired factor grid')
        reference = labels[ids[0]]
        for sid in ids:
            if any(labels[sid][key] != reference[key] for key in ('allowed_fields', 'known_fields', 'unknown_fields',
                                                                 'role', 'language', 'condition', 'event_group_id')):
                raise ValidationError('answers or identities changed within a pair')
    indexed = {}
    for record in results:
        key = record['arm'], record['sample_id']
        if key in indexed or key[0] not in arms or key[1] not in rows:
            raise ValidationError('duplicate or unknown paired result')
        indexed[key] = record
    if set(indexed) != {(arm, sid) for arm in arms for sid in rows}:
        raise ValidationError('incomplete paired results')
    report = {'examples_per_arm': len(rows), 'parent_examples': len(parents),
              'independent_event_groups': len({j['event_group_id'] for j in judges}), 'arms': {}}
    for arm in arms:
        details = {sid: judge_uncertainty(row, labels[sid], indexed[arm, sid]['output']) for sid, row in rows.items()}
        for sid, grade in details.items():
            output = indexed[arm, sid]['output']
            grounded = False
            if output is not None:
                grounded = (set(output['evidence_ids']) == {e['evidence_id'] for e in rows[sid]['input']['evidence']}
                            and not output['counter_evidence_ids']) if rows[sid]['role'] == 'research' else not output['risks']
            grade['grounding_correct'] = bool(grounded)
            grade['role_contract_correct'] = grade['correct'] and bool(grounded)
        def counts(ids):
            return {'examples': len(ids), **{k: sum(details[s][k] for s in ids) for k in
                ('correct', 'schema_valid', 'vocabulary_valid', 'grounding_correct', 'role_contract_correct')},
                'known_as_unknown_examples': sum(bool(details[s].get('known_as_unknown')) for s in ids),
                'missing_unknown_examples': sum(bool(details[s].get('missing_unknowns')) for s in ids)}
        factors = {}
        for factor, levels in FACTORS.items():
            pair_ids = defaultdict(dict)
            for sid, label in labels.items():
                key = (label['parent_sample_id'], *(label['factors'][k] for k in FACTORS if k != factor))
                pair_ids[key][label['factors'][factor]] = sid
            def contrast(pairs):
                states = [(details[p[levels[0]]]['correct'], details[p[levels[1]]]['correct']) for p in pairs]
                return {'pairs': len(states), 'both_correct': states.count((True, True)),
                        'both_wrong': states.count((False, False)), 'improved': states.count((False, True)),
                        'regressed': states.count((True, False)),
                        'net_correct_change': states.count((False, True)) - states.count((True, False))}
            factors[factor] = {'direction': list(levels), **contrast(list(pair_ids.values())),
                'levels': {level: counts([sid for sid, j in labels.items() if j['factors'][factor] == level]) for level in levels},
                'by_condition': {c: contrast([p for p in pair_ids.values() if labels[p[levels[0]]]['condition'] == c])
                                 for c in sorted({j['condition'] for j in judges})}}
        cells = {':'.join(levels): counts([sid for sid, j in labels.items()
                 if tuple(j['factors'][k] for k in FACTORS) == levels]) for levels in product(*FACTORS.values())}
        report['arms'][arm] = {**counts(list(rows)), 'details': details, 'cells': cells, 'factors': factors,
            'all_variants_correct': sum(all(details[s]['correct'] for s in ids) for ids in parents.values()),
            'all_variants_schema_valid': sum(all(details[s]['schema_valid'] for s in ids) for ids in parents.values()),
            **{'by_' + key: {v: counts([sid for sid, j in labels.items() if j[key] == v])
                            for v in sorted({j[key] for j in judges})} for key in ('condition', 'language', 'role', 'event_group_id')}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'config', 'output'):parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    print(json_text(build_paired_cohort(a.source, a.config, a.output)))


if __name__ == '__main__':main()
