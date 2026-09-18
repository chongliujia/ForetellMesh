"""Matched prose/structured/tool development diagnostic on excluded numerical groups.

Targets are derived from disclosure and requested scope, independently of the
runtime calculator. Existing test partitions are read for exclusion identities
only: no test prompts are selected for tokenization, generation or scoring.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import random
import tempfile

from .agent_baseline_data import input_context
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .quant_state import SOURCE, validate_spec
from .research_tool_counterfactual import information_state, sample_parameters, CONDITIONS as OLD_CONDITIONS
from .research_tool_data import read_research_tool_data
from .schema import ValidationError, fields, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash, render_case
from .uncertainty_diagnostic import CONDITIONS, VERSION as PROBE_VERSION, render_probe, load_uncertainty_config

VERSION = 'synthetic_tool_assisted_diagnostic_v1'
VARIANTS = ('prose', 'structured', 'tool')


def load_config(path):
    c = fields(strict_json(Path(path).read_text()), {'schema_version', 'dataset_version', 'partition', 'seed',
        'groups_per_family', 'observation_time', 'candidate_bundle', 'original_raw', 'previous_probe_config'}, 'tool diagnostic config')
    if (c['schema_version'] != '1' or c['dataset_version'] != VERSION or c['partition'] != 'validation'
            or type(c['seed']) is not int or not 0 <= c['seed'] < 2**32 or type(c['groups_per_family']) is not int
            or not 1 <= c['groups_per_family'] <= 20):raise ValidationError('unsupported diagnostic config')
    timestamp(c['observation_time'], 'diagnostic time')
    return c


def specification(c, condition):
    family = condition.split('_')[0]; p = c[family]
    if family == 'mixture':
        v = {'high_rate': p['high'], 'low_rate': p['low']}
        if condition == 'mixture_known':v['weight'] = p['weight']
    elif family == 'bayes':
        v = {k: p[k] for k in ('sensitivity', 'false_positive_rate')}
        if condition == 'bayes_known':v['prior'] = p['prior']
    elif family == 'complement':v = dict(p)
    else:
        v = {k: p[k] for k in ('successes', 'trials')}
        if condition == 'sampling_population':v['population_probability'] = p['population_probability']
    return validate_spec({'schema_version': '1', 'model': family, 'values': v})


def fingerprint(model, values):
    return canonical_hash({'model': model, 'values': values})


def exclusions(c):
    bundle, raw, probe = (Path(c[k]) for k in ('candidate_bundle', 'original_raw', 'previous_probe_config'))
    _, parts, _ = read_research_tool_data(bundle)
    blocked = set()
    for rows in parts.values():
        for row in rows:
            case = row['case']
            if case['kind'] != 'information_state':continue
            family = case['family']
            for condition in OLD_CONDITIONS[family]:
                given, _, _, _ = information_state(family, case['parameters'], condition)
                if given:blocked.add(fingerprint(family, given))
    manifest = strict_json((raw/'manifest.json').read_text())
    if sha256_file(raw/'targets.jsonl') != manifest['targets_sha256']:raise ValidationError('raw source changed')
    proof_hashes = {}
    for line in (raw/'targets.jsonl').read_text().splitlines():
        target = strict_json(line); provenance = target['provenance']; path = raw/provenance['artifact']
        if not path.resolve().is_relative_to(raw.resolve()):raise ValidationError('unsafe proof path')
        digest = sha256_file(path)
        if digest != provenance['artifact_sha256']:raise ValidationError('raw proof changed')
        proof_hashes[provenance['artifact']] = digest
        case = strict_json(path.read_text()); render_case(case)
        f, p = case['family'], case['parameters']
        if f == 'mixture':
            v = {'high_rate': p['high']/100, 'low_rate': p['low']/100, 'weight': p['weight']/100}
            blocked.add(fingerprint('mixture', v)); blocked.add(fingerprint('mixture', {k:v[k] for k in ('high_rate','low_rate')}))
        elif f == 'bayes_signal':
            v = {'prior':p['prior']/100, 'sensitivity':p['sensitivity']/100, 'false_positive_rate':p['false_positive']/100}
            blocked.add(fingerprint('bayes',v));blocked.add(fingerprint('bayes',{k:v[k] for k in ('sensitivity','false_positive_rate')}))
        elif f == 'complement':blocked.add(fingerprint('complement', {'hit_probability':p['a']/p['b']}))
    old = load_uncertainty_config(probe)
    for condition in CONDITIONS:
        spec = specification(old, condition); blocked.add(fingerprint(spec['model'], spec['values']))
    return {'signatures': sorted(blocked), 'sources': {
        'candidate_bundle_manifest_sha256': sha256_file(bundle/'manifest.json'),
        'original_raw_manifest_sha256': sha256_file(raw/'manifest.json'),
        'original_proofs_sha256': canonical_hash(proof_hashes), 'previous_probe_config_sha256': sha256_file(probe)}}


def numeric_configs(c, excluded):
    rng = random.Random(c['seed']); blocked = set(excluded['signatures']); configs = []
    for _ in range(c['groups_per_family']):
        probe = {'schema_version':'1', 'dataset_version':PROBE_VERSION, 'partition':'validation',
                 'observation_time':c['observation_time'], 'languages':['en','zh'], 'roles':['research','risk'], 'synthetic_only':True}
        for family in ('mixture','bayes','complement','sampling'):
            for attempt in range(10000):
                p = sample_parameters(rng, family)
                if family == 'mixture':p = {'high':p['high_rate'], 'low':p['low_rate'], 'weight':p['weight']}
                if family == 'complement':p = {'hit_probability':p['hit_probability']}
                probe[family] = p
                specs = [specification(probe, cond) for cond in CONDITIONS if cond.startswith(family+'_')]
                signatures = {fingerprint(s['model'],s['values']) for s in specs}
                if not blocked & signatures:break
            else:raise ValidationError('cannot find disjoint numerical group')
            blocked.update(signatures)
        configs.append(probe)
    return configs


def rows_for_configs(configs):
    inputs, judges = [], []
    for c in configs:
        for condition in CONDITIONS:
            for language in c['languages']:
                for role in c['roles']:
                    row, judge = render_probe(c, condition, language, role)
                    spec = specification(c, condition)
                    variants = {}
                    for variant in VARIANTS:
                        payload = deepcopy(row['input'])
                        if variant != 'prose':
                            e = deepcopy(payload['evidence'][0]); e.update(evidence_id='structured_model', source=SOURCE, text=json_text(spec).strip())
                            payload['evidence'].append(e)
                        input_context(payload); variants[variant] = payload
                    inputs.append({'sample_id': row['sample_id'], 'role':role, 'variants':variants})
                    judges.append(judge)
    return inputs, judges


def build(config_path, output):
    if output.exists():raise ValidationError('tool diagnostic already exists')
    c = load_config(config_path); excluded = exclusions(c)
    configs = numeric_configs(c, excluded); inputs, judges = rows_for_configs(configs)
    artifacts = {'config.json':Path(config_path).read_text(), 'exclusions.json':json_text(excluded),
        'numeric_configs.json':json_text(configs), 'inputs.jsonl':jsonl(inputs), 'judge.jsonl':jsonl(judges)}
    report = {'schema_version':'1','kind':VERSION,'partition':'validation','frozen_at':datetime.now(timezone.utc).isoformat(),
        'role_tasks':len(inputs), 'event_groups':len({j['event_group_id'] for j in judges}), 'variants':list(VARIANTS),
        'artifact_hashes':{k:sha256_bytes(v.encode()) for k,v in artifacts.items()},
        'limitations':['Synthetic development diagnostic; no training, final-test evaluation or real forecasting claim.',
                      'Numerical groups are new; task templates and field names are reused.',
                      'Structured facts are supplied explicitly, not extracted from arbitrary text by a model.',
                      'Role/language/condition variants share groups; row counts are not independent events.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.tool-diagnostic-') as tmp:
        stage=Path(tmp)/'cohort';stage.mkdir()
        for name,content in artifacts.items():(stage/name).write_text(content)
        (stage/'manifest.json').write_text(json_text(report));stage.rename(output)
    return report


def read(path):
    report=strict_json((path/'manifest.json').read_text())
    if report.get('kind')!=VERSION or report.get('partition')!='validation':raise ValidationError('invalid tool diagnostic')
    names={'config.json','exclusions.json','numeric_configs.json','inputs.jsonl','judge.jsonl'}
    if set(report['artifact_hashes'])!=names:raise ValidationError('tool diagnostic artifacts changed')
    for name,digest in report['artifact_hashes'].items():
        if sha256_file(path/name)!=digest:raise ValidationError('tool diagnostic hash changed')
    c=load_config(path/'config.json'); excluded=exclusions(c); configs=numeric_configs(c,excluded)
    inputs,judges=rows_for_configs(configs)
    for name,expected in [('exclusions.json',excluded),('numeric_configs.json',configs),('inputs.jsonl',inputs),('judge.jsonl',judges)]:
        actual=[strict_json(s) for s in (path/name).read_text().splitlines()] if name.endswith('.jsonl') else strict_json((path/name).read_text())
        if actual!=expected:raise ValidationError('tool diagnostic replay mismatch')
    if report['role_tasks']!=len(inputs) or report['event_groups']!=len({j['event_group_id'] for j in judges}) or report['variants']!=list(VARIANTS):
        raise ValidationError('tool diagnostic counts changed')
    return report,inputs,judges


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json_text(build(a.config,a.output)))

if __name__=='__main__':main()
