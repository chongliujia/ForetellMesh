"""Replayable synthetic capability tasks; no market outcomes or web evidence."""
from collections import Counter, defaultdict
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile

from .agent_baseline_data import expected_tool, input_context
from .agent_runtime import OUTPUT_PROTOCOLS, PROMPTS, validate_agent_output
from .config import load_config
from .data import load_records, sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .probability_tools import execute_probability_tool
from .schema import ValidationError, fields, iso, timestamp
from .sft_data import jsonl, validate_target
from .splits import split_records
from .synthetic_sft import SOURCE, canonical_hash, render_case

VERSION = 'synthetic_research_tool_v1'
PARTITIONS = ('train', 'validation', 'test')


def load_dataset_config(path: Path) -> dict:
    c = fields(strict_json(path.read_text()), {'schema_version', 'dataset_version', 'seed', 'output_protocol',
        'tool_families', 'evidence_groups', 'observation_times', 'languages', 'evaluation_groups_per_family', 'synthetic_only'}, 'research/tool dataset config')
    if (c['schema_version'] != '1' or c['dataset_version'] != VERSION or c['synthetic_only'] is not True
            or c['output_protocol'] != 'plain_json_v1' or c['tool_families'] != ['mixture', 'bayes_signal']
            or c['languages'] != ['en', 'zh'] or type(c['seed']) is not int
            or c['evaluation_groups_per_family'] != 2 or type(c['evaluation_groups_per_family']) is not int):
        raise ValidationError('unsupported research/tool dataset policy')
    fields(c['evidence_groups'], set(PARTITIONS), 'group counts')
    fields(c['observation_times'], set(PARTITIONS), 'observation times')
    for n in c['evidence_groups'].values():
        if type(n) is not int or not 3 <= n <= 100:
            raise ValidationError('invalid evidence group count')
    times = [timestamp(c['observation_times'][p], p) for p in PARTITIONS]
    if not times[0] < times[1] < times[2]:
        raise ValidationError('observation times must increase')
    return c


def make_request(role: str, payload: dict, *, repair_error: str | None = None) -> dict:
    input_context(payload)
    request = {'agent': role, 'adapter': None, 'instruction': PROMPTS[role] + OUTPUT_PROTOCOLS['plain_json_v1'],
               'input': payload, 'upstream': {}}
    if repair_error is not None:
        request['repair'] = {'validation_error': repair_error,
                             'instruction': 'Return a complete corrected JSON object matching the requested schema.'}
    return request


def render_tool_task(case: dict) -> tuple[str, str, dict, dict, dict]:
    fields(case, {'kind', 'source_case', 'repair'}, 'tool case')
    if case['kind'] != 'tool' or type(case['repair']) is not bool:
        raise ValidationError('invalid tool case')
    record, _ = render_case(case['source_case'])
    if case['source_case']['family'] not in ('mixture', 'bayes_signal'):
        raise ValidationError('unsupported capability tool family')
    payload = {k: record[k] for k in ('question', 'observation_time', 'market')}
    payload['evidence'] = [e for e in record['evidence'] if e['evidence_id'] == 'setup']
    target = expected_tool(case['source_case'])
    error = None
    if case['repair']:
        # Feedback is only a validator error, matching AgentRunner's real repair
        # interface. Neither the desired call nor a computed answer is exposed.
        if target['name'] == 'weighted_probability':
            invalid = {'name': target['name'], 'arguments': {'probabilities': [.5, .5], 'weights': [1, 1]}}
        else:
            invalid = {'name': target['name'], 'arguments': {'prior': 0, 'sensitivity': 0, 'false_positive_rate': 0}}
        try:
            execute_probability_tool(invalid)
        except ValidationError as exc:
            error = str(exc)
        if error is None:
            raise ValidationError('repair fixture must fail the real validator')
    return (record['event_group_id'], case['source_case']['language'], make_request('quant', payload, repair_error=error),
            target, {'family': case['source_case']['family'], 'expected_result': execute_probability_tool(target)['result']})


def render_evidence_task(case: dict) -> tuple[str, str, dict, dict, dict]:
    fields(case, {'kind', 'parameters', 'language', 'observation_time', 'role'}, 'evidence case')
    p = fields(case['parameters'], {'serial', 'threshold', 'provisional', 'pending', 'risk_present'}, 'evidence parameters')
    if (case['kind'] != 'evidence' or case['language'] not in ('en', 'zh') or case['role'] not in ('research', 'risk')
            or any(type(p[k]) is not int or p[k] < 1 for k in ('serial', 'threshold', 'provisional', 'pending'))
            or type(p['risk_present']) is not bool):
        raise ValidationError('invalid evidence case')
    group = 'evidence:' + canonical_hash(p)[:24]
    zh = case['language'] == 'zh'
    observed = timestamp(case['observation_time'], 'observation_time')
    due = iso(observed + timedelta(days=1))
    project, other = f'P{p["serial"]}', f'Q{p["serial"]}'
    question = (f'合成预测合约：项目 {project} 在 {due} 前通过独立审核的产量是否至少为 {p["threshold"]} 单位？' if zh else
                f'Synthetic forecast contract: Will project {project} have at least {p["threshold"]} independently verified units by {due}?')
    ids = {key: 'e' + canonical_hash({'group': group, 'key': key})[:6] for key in ('rule', 'report', 'audit', 'noise', 'opinion')}
    rule = (f'合成规则：仅独立审核通过的 {project} 产量计入门槛 {p["threshold"]}；当前结果未知。' if zh else
            f'Synthetic rule: Only independently verified units from {project} count toward {p["threshold"]}; the outcome is unknown.')
    reported = (f'项目 {project} 的运营记录：暂报产量 {p["provisional"]} 单位，尚非最终审核结果。' if zh else
                f'Operator record for {project}: {p["provisional"]} provisional units, not a final audited result.')
    risk = (f'{project} 有 {p["pending"]} 单位等待复核，可能错过截止时间。' if zh else
            f'{project} has {p["pending"]} units awaiting recheck that may miss the deadline.')
    audit = risk if p['risk_present'] else (f'审核记录：{project} 当前未报告复核积压。' if zh else
                                           f'Audit record: No recheck backlog currently reported for {project}.')
    unknown = '截止时的最终审核产量尚未知。' if zh else 'The final verified count at the deadline is unknown.'
    noise = (f'另一个项目 {other} 的正式记录：设备故障导致停产；这不是 {project} 的记录。' if zh else
             f'Official record for a different project {other}: equipment failure halted production; this is not a record for {project}.')
    opinion = (f'未经核实的论坛猜测：{project} 一定会遭遇技术故障；没有记录支持这一说法。' if zh else
               f'Unverified forum conjecture: {project} will certainly suffer technical failures; no records support this claim.')
    texts = {'rule': rule, 'report': reported, 'audit': audit, 'noise': noise, 'opinion': opinion}
    evidence = [{'evidence_id': ids[key], 'text': text, 'source': f'synthetic://{key}',
                 'published_at': iso(observed - timedelta(hours=1)), 'available_at': iso(observed - timedelta(minutes=30))}
                for key, text in texts.items()]
    random.Random(int(canonical_hash({'group': group, 'language': case['language']})[:16], 16)).shuffle(evidence)
    payload = {'question': question, 'observation_time': iso(observed), 'evidence': evidence, 'market': None}
    if case['role'] == 'research':
        target = {'evidence_ids': sorted([ids['rule'], ids['report']] + ([] if p['risk_present'] else [ids['audit']])),
                  'counter_evidence_ids': [ids['audit']] if p['risk_present'] else [],
                  'unknowns': [unknown], 'observation_time': iso(observed)}
    else:
        target = {'risks': [{'scenario': risk, 'evidence_ids': [ids['audit']]}] if p['risk_present'] else [],
                  'unknowns': [unknown], 'observation_time': iso(observed)}
    return group, case['language'], make_request(case['role'], payload), target, {
        'family': 'evidence_selection', 'forbidden_evidence_ids': [ids['noise'], ids['opinion']], 'risk_present': p['risk_present']}


def render_task(case: dict, partition: str) -> dict:
    if partition not in PARTITIONS:
        raise ValidationError('unknown partition')
    group, language, request, target, oracle = render_tool_task(case) if case.get('kind') == 'tool' else render_evidence_task(case)
    validate_agent_output(request['agent'], target, input_context(request['input']))
    task = request['agent'] + ('_repair' if 'repair' in request else '')
    return {'sample_id': f'{group}:{language}:{task}', 'event_group_id': group, 'family': oracle['family'],
            'language': language, 'task': task, 'partition': partition, 'case': case,
            'case_sha256': canonical_hash(case), 'request': request, 'target': target, 'oracle': oracle}


def validate_rows(partitions: dict[str, list[dict]], config: dict) -> None:
    owners, ids = {}, set()
    for partition, rows in partitions.items():
        for row in rows:
            if render_task(row['case'], partition) != row:
                raise ValidationError('task differs from deterministic case replay')
            if row['request']['input']['observation_time'] != config['observation_times'][partition]:
                raise ValidationError('task observation time disagrees with partition')
            group = row['event_group_id']
            if owners.setdefault(group, partition) != partition or row['sample_id'] in ids:
                raise ValidationError('cross-partition event group or duplicate identity')
            ids.add(row['sample_id'])
    # Require both languages and all task variants of each event to stay together.
    for rows in partitions.values():
        groups = defaultdict(list)
        for row in rows:groups[row['event_group_id']].append(row)
        for members in groups.values():
            tasks = {'quant', 'quant_repair'} if members[0]['case']['kind'] == 'tool' else {'research', 'risk'}
            if len(members) != 4 or {(r['language'], r['task']) for r in members} != {(lang, task) for lang in ('en', 'zh') for task in tasks}:
                raise ValidationError('incomplete bilingual event/task group')


def select_validation(rows: list[dict]) -> list[str]:
    """Freeze two groups per family, including both evidence risk conditions."""
    groups = defaultdict(dict)
    for row in rows:
        groups[row['family']].setdefault(row['event_group_id'], []).append(row)
    selected = []
    for family, members in sorted(groups.items()):
        if family == 'evidence_selection':
            for flag in (False, True):
                candidates = sorted(g for g, values in members.items() if values[0]['oracle']['risk_present'] is flag)
                if not candidates:raise ValidationError('missing risk condition in validation')
                selected.extend(members[candidates[0]])
        else:
            if len(members) < 2:raise ValidationError('insufficient tool validation groups')
            for group in sorted(members)[:2]:selected.extend(members[group])
    return sorted(r['sample_id'] for r in selected)


def build_research_tool_data(raw_dataset: Path, split_config: Path, config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError('research/tool output already exists')
    config = load_dataset_config(config_path)
    policy, policy_hash = load_config(split_config)
    manifest = strict_json((raw_dataset / 'manifest.json').read_text())
    records, digest = load_records(raw_dataset / 'records.jsonl')
    if (manifest.get('kind') != 'synthetic_schema_warmup' or manifest.get('historical_data') is not False
            or digest != manifest['records_sha256'] or sha256_file(raw_dataset / 'targets.jsonl') != manifest['targets_sha256']
            or sha256_file(raw_dataset / 'config.json') != manifest['config_sha256']):
        raise ValidationError('source archive hash/type mismatch')
    targets = {}
    for line in (raw_dataset / 'targets.jsonl').read_text().splitlines():
        target = strict_json(line)
        if target['sample_id'] in targets:raise ValidationError('duplicate source target')
        targets[target['sample_id']] = target
    if set(targets) != {r.sample_id for r in records}:raise ValidationError('source identities differ')
    split = split_records(records, policy)
    partitions = {p: [] for p in PARTITIONS}
    for partition, members in split.partitions.items():
        for record in members:
            if record.dataset_source != SOURCE:raise ValidationError('only synthetic warmup source is permitted')
            target = targets[record.sample_id]
            validate_target(target, record, policy.sources[SOURCE], raw_dataset)
            source_case = strict_json((raw_dataset / target['provenance']['artifact']).read_text())
            from .schema import parse_record
            if parse_record(render_case(source_case)[0]) != record:
                raise ValidationError('source differs from oracle replay')
            if source_case['family'] in config['tool_families']:
                for repair in (False, True):
                    partitions[partition].append(render_task({'kind': 'tool', 'source_case': source_case, 'repair': repair}, partition))
    rng = random.Random(config['seed'])
    serial = 100
    for partition in PARTITIONS:
        for index in range(config['evidence_groups'][partition]):
            serial += rng.randint(1, 20)
            threshold = rng.randint(30, 200)
            params = {'serial': serial, 'threshold': threshold, 'provisional': threshold + rng.randint(2, 25),
                      'pending': rng.randint(1, 30), 'risk_present': index % 3 != 0}
            for language in config['languages']:
                for role in ('research', 'risk'):
                    case = {'kind': 'evidence', 'parameters': params, 'language': language,
                            'observation_time': config['observation_times'][partition], 'role': role}
                    partitions[partition].append(render_task(case, partition))
        partitions[partition].sort(key=lambda r: r['sample_id'])
    validate_rows(partitions, config)
    selection = select_validation(partitions['validation'])
    artifacts = {p + '.jsonl': jsonl(rows) for p, rows in partitions.items()}
    artifacts.update({'config.json': json_text(config), 'evaluation_ids.json': json_text(selection)})
    report = {'schema_version': '1', 'kind': 'synthetic_research_tool_bundle', 'dataset_version': VERSION,
              'config_sha256': sha256_file(config_path), 'source_manifest_sha256': sha256_file(raw_dataset / 'manifest.json'),
              'source_records_sha256': digest, 'split_config_sha256': policy_hash,
              'split_version': policy.split_version + ':research_tool_v1',
              'artifact_hashes': {name: sha256_bytes(contents.encode()) for name, contents in artifacts.items()},
              'counts': {p: {'examples': len(rows), 'event_groups': len({r['event_group_id'] for r in rows}),
                             'tasks': dict(Counter(r['task'] for r in rows))} for p, rows in partitions.items()},
              'validation_evaluation_examples': len(selection), 'code': code_provenance(),
              'limitations': ['Synthetic behavior warmup, not historical prediction-market training or forecasting evaluation.',
                             'Tool parameter groups inherit original warmup partitions; language and repair variants share groups.',
                             'Evidence targets follow a synthetic extractive annotation policy; quote-match is not a general factual-grounding judge.',
                             'Final test is generated and integrity-checked, not used for training or model selection.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.research-tool-', dir=output.parent) as temp:
        stage = Path(temp) / 'bundle'; stage.mkdir()
        for name, content in artifacts.items():(stage / name).write_text(content)
        (stage / 'manifest.json').write_text(json_text(report))
        stage.rename(output)
    return report


def read_research_tool_data(path: Path) -> tuple[dict, dict, dict]:
    manifest = strict_json((path / 'manifest.json').read_text())
    if manifest.get('dataset_version') == 'synthetic_research_tool_v3':
        from .research_tool_counterfactual import read_counterfactual_data
        return read_counterfactual_data(path)
    if manifest.get('dataset_version') == 'synthetic_research_tool_v2':
        from .research_tool_augmentation import read_augmented_data
        return read_augmented_data(path)
    names = {p + '.jsonl' for p in PARTITIONS} | {'config.json', 'evaluation_ids.json'}
    if (manifest.get('schema_version') != '1' or manifest.get('kind') != 'synthetic_research_tool_bundle'
            or manifest.get('dataset_version') != VERSION or set(manifest['artifact_hashes']) != names):
        raise ValidationError('invalid research/tool bundle')
    for name, digest in manifest['artifact_hashes'].items():
        if sha256_file(path / name) != digest:raise ValidationError('research/tool artifact hash mismatch')
    config = load_dataset_config(path / 'config.json')
    partitions = {p: [strict_json(s) for s in (path / (p + '.jsonl')).read_text().splitlines()] for p in PARTITIONS}
    validate_rows(partitions, config)
    for p, rows in partitions.items():
        expected = {'examples': len(rows), 'event_groups': len({r['event_group_id'] for r in rows}),
                    'tasks': dict(Counter(r['task'] for r in rows))}
        if expected != manifest['counts'][p]:raise ValidationError('research/tool count mismatch')
    selected = strict_json((path / 'evaluation_ids.json').read_text())
    if selected != select_validation(partitions['validation']) or len(selected) != manifest['validation_evaluation_examples']:
        raise ValidationError('evaluation cohort selection changed')
    return manifest, partitions, config
