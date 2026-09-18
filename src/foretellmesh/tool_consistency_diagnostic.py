"""New numerical groups, matched inputs, and independent diagnostic labels."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .schema import ValidationError, fields
from .sft_data import jsonl
from .tool_assisted_diagnostic import (exclusions, numeric_configs, rows_for_configs, fingerprint,
                                       read as read_previous, load_config as load_base_config)
from .tool_consistency import SOURCE
from .tool_consistency import requested_fields
from .agent_baseline_data import input_context
from .quant_state import build_quant_state

VERSION = 'synthetic_tool_consistency_diagnostic_v1'
VARIANTS = ('tool',)


def materialize(path):
    c = fields(strict_json(path.read_text()), {'schema_version', 'base_config', 'previous_cohort', 'seed'}, 'consistency cohort')
    if c['schema_version'] != '1' or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32:
        raise ValidationError('invalid consistency cohort config')
    base = load_base_config(c['base_config']); base['seed'] = c['seed']
    excluded = exclusions(base)
    previous = Path(c['previous_cohort']); _, old, _ = read_previous(previous)
    blocked = set(excluded['signatures'])
    for row in old:
        for evidence in row['variants']['tool']['evidence']:
            if evidence['source'] == 'structured://probability-model/v1':
                spec = strict_json(evidence['text']); blocked.add(fingerprint(spec['model'], spec['values']))
    excluded['signatures'] = sorted(blocked)
    excluded['sources'].update(base_config_sha256=sha256_file(Path(c['base_config'])),
                              previous_cohort_manifest_sha256=sha256_file(previous/'manifest.json'))
    configs = numeric_configs(base, excluded); inputs, judges = rows_for_configs(configs)
    for row in inputs:
        payload = deepcopy(row['variants']['tool'])
        # Scope comes from the question's requested vocabulary, never its label.
        names = scope_names(payload['question'])
        if not names:raise ValidationError('question vocabulary absent')
        evidence = deepcopy(payload['evidence'][0])
        evidence.update(evidence_id='requested_scope', source=SOURCE,
                        text=json_text({'schema_version': '1', 'requested_fields': names}).strip())
        payload['evidence'].append(evidence)
        context = input_context(payload); requested_fields(context, build_quant_state(context))
        row['variants'] = {'tool': payload}
    return c, excluded, configs, inputs, judges


def scope_names(question):
    marker = 'Field meanings: ' if 'Field meanings: ' in question else '字段含义：'
    if question.count(marker) != 1:raise ValidationError('ambiguous question vocabulary')
    definitions = question.split(marker)[1].split('; ')
    names = []
    for definition in definitions:
        match = re.fullmatch(r'([a-z_]+): .+', definition)
        if not match:raise ValidationError('invalid question field definition')
        names.append(match[1])
    if len(set(names)) != len(names):raise ValidationError('duplicate question fields')
    return sorted(names)


def build(path, output):
    if output.exists():raise ValidationError('consistency cohort exists')
    c, excluded, configs, inputs, judges = materialize(path)
    artifacts = {'config.json': path.read_text(), 'exclusions.json': json_text(excluded),
                 'numeric_configs.json': json_text(configs), 'inputs.jsonl': jsonl(inputs), 'judge.jsonl': jsonl(judges)}
    report = {'schema_version': '1', 'kind': VERSION, 'partition': 'validation',
              'frozen_at': datetime.now(timezone.utc).isoformat(), 'role_tasks': len(inputs),
              'event_groups': len({j['event_group_id'] for j in judges}),
              'artifact_hashes': {k: sha256_bytes(v.encode()) for k, v in artifacts.items()},
              'limitations': ['Synthetic development classification; no real forecasting or training claim.',
                             'New numerical groups, reused task templates; rows share groups.',
                             'Input includes an explicit scope contract in both control and guarded arms.',
                             'Runtime guard checks a subset of consistency, not the full uncertainty answer.']}
    output.mkdir(parents=True)
    for name, value in artifacts.items():(output/name).write_text(value)
    (output/'manifest.json').write_text(json_text(report)); return report


def read(path):
    report = strict_json((path/'manifest.json').read_text())
    if report.get('kind') != VERSION or report.get('partition') != 'validation':raise ValidationError('invalid cohort')
    c, excluded, configs, inputs, judges = materialize(path/'config.json')
    expected = {'config.json': c, 'exclusions.json': excluded, 'numeric_configs.json': configs,
                'inputs.jsonl': inputs, 'judge.jsonl': judges}
    if set(report['artifact_hashes']) != set(expected):raise ValidationError('artifact set changed')
    for name, value in expected.items():
        raw = (path/name).read_text()
        actual = [strict_json(s) for s in raw.splitlines()] if name.endswith('.jsonl') else strict_json(raw)
        if actual != value or sha256_file(path/name) != report['artifact_hashes'][name]:raise ValidationError('cohort replay changed')
    if report['role_tasks'] != len(inputs) or report['event_groups'] != len({j['event_group_id'] for j in judges}):
        raise ValidationError('cohort counts changed')
    return report, inputs, judges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(json_text(build(a.config, a.output)))


if __name__ == '__main__':main()
