"""Replayable v2 behavior data: supplied probabilities versus unresolved outcomes.

All new cases inherit the existing synthetic source's event/time partitions.
The separate finite-vocabulary uncertainty diagnostic is never a training source.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import tempfile

from .agent_baseline_data import input_context
from .agent_runtime import agent_instruction, validate_agent_output
from .config import load_config
from .data import load_records, sha256_bytes, sha256_file, strict_json
from .evaluation import code_provenance, json_text
from .research_tool_data import PARTITIONS, read_research_tool_data, render_task, select_validation
from .schema import ValidationError, fields, parse_record
from .sft_data import jsonl, validate_target
from .splits import split_records
from .synthetic_sft import FAMILIES, SOURCE, canonical_hash, render_case

VERSION = 'synthetic_research_tool_v2'
PROTOCOL = 'grounded_json_v3'


def load_augmentation_config(path: Path) -> dict:
    c=fields(strict_json(path.read_text()), {'schema_version','dataset_version','output_protocol','math_groups_per_family',
        'parameter_audit_groups_per_family','selection','evaluation_selection','synthetic_only'},'augmentation config')
    fixed={'schema_version':'1','dataset_version':VERSION,'output_protocol':PROTOCOL,
           'selection':'lexicographically_first_source_groups_with_inherited_splits',
           'evaluation_selection':'unchanged_legacy_v1_role_cohort','synthetic_only':True}
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):raise ValidationError('unsupported augmentation policy')
    for key in ('math_groups_per_family','parameter_audit_groups_per_family'):
        fields(c[key],set(PARTITIONS),'augmentation counts')
        if any(type(n) is not int or not 1<=n<=8 for n in c[key].values()):raise ValidationError('invalid augmentation count')
    if any(c['parameter_audit_groups_per_family'][p]>c['math_groups_per_family'][p] for p in PARTITIONS):
        raise ValidationError('parameter audit must be a subset of math event groups')
    return c


def replay_augmented_task(case: dict, partition: str) -> dict:
    if partition not in PARTITIONS:raise ValidationError('invalid augmented partition')
    if case.get('kind')=='legacy':
        fields(case,{'kind','source_case'},'legacy wrapper')
        row=render_task(case['source_case'],partition)
        row['case']=case;row['case_sha256']=canonical_hash(case)
        row['request']['instruction']=agent_instruction(row['request']['agent'],PROTOCOL)
        return row
    fields(case,{'kind','source_case','role','view'},'math role case')
    if case['kind']!='math_role' or case['role'] not in ('research','risk') or case['view'] not in ('forecast','parameters'):
        raise ValidationError('invalid math role case')
    record,forecast=render_case(case['source_case'])
    payload=parse_record(record).forecast_input.to_payload()
    role,view,language=case['role'],case['view'],case['source_case']['language']
    if view=='parameters':
        payload['question']+= ('\n本任务只审核按上述模型计算所需的输入是否齐全，不审核尚未观察到的随机结果或潜在状态。' if language=='zh' else
                              '\nThis task audits only whether the inputs needed for the stated model calculation are supplied, not the unobserved random outcome or latent state.')
    # Unknowns describe unobserved outcomes/latent parameters, never the
    # explicitly supplied or computed conditional/predictive probabilities.
    unknowns=forecast['unknowns'] if view=='forecast' else []
    target={'unknowns':unknowns,'observation_time':payload['observation_time']}
    if role=='research':
        target.update(evidence_ids=forecast['key_evidence'],counter_evidence_ids=forecast['counter_evidence'])
    else:target['risks']=[]
    validate_agent_output(role,target,input_context(payload))
    group=record['event_group_id']
    return {'sample_id':f'{group}:{language}:{role}:math_{view}','event_group_id':group,
            'family':'math_'+case['source_case']['family'],'language':language,'task':f'{role}_math_{view}',
            'partition':partition,'case':case,'case_sha256':canonical_hash(case),
            'request':{'agent':role,'adapter':None,'instruction':agent_instruction(role,PROTOCOL),
                       'input':payload,'upstream':{}},'target':target,
            'oracle':{'family':'math_'+case['source_case']['family'],'forbidden_evidence_ids':[],
                      'unknowns_contract':'unobserved_outcome_or_latent_rate_for_forecast_missing_calculation_inputs_for_parameters'}}


def validate_augmented_rows(parts: dict, source_splits: dict, times: dict) -> None:
    fields(parts,set(PARTITIONS),'augmented partitions')
    owners,identities={},set()
    for partition,rows in parts.items():
        grouped=defaultdict(list)
        for row in rows:
            if row!=replay_augmented_task(row['case'],partition):raise ValidationError('augmented task replay mismatch')
            group=row['event_group_id']
            if (owners.setdefault(group,partition)!=partition or row['sample_id'] in identities
                    or source_splits.get(group)!=partition or row['request']['input']['observation_time']!=times[partition]):
                raise ValidationError('augmented identity, source split or timestamp mismatch')
            identities.add(row['sample_id']);grouped[group].append(row)
        for members in grouped.values():
            tasks={r['task'] for r in members}
            if {(r['language'],r['task']) for r in members}!={(lang,task) for lang in ('en','zh') for task in tasks}:
                raise ValidationError('incomplete bilingual augmented group')
            for left,right in (('research','risk'),('quant','quant_repair'),('research_math_forecast','risk_math_forecast'),
                               ('research_math_parameters','risk_math_parameters')):
                if (left in tasks)!=(right in tasks):raise ValidationError('incomplete role pair')


def counts(parts):
    return {p:{'examples':len(rows),'event_groups':len({r['event_group_id'] for r in rows}),
               'tasks':dict(Counter(r['task'] for r in rows))} for p,rows in parts.items()}


def build_augmented_data(legacy_bundle: Path, raw_dataset: Path, split_config: Path, config_path: Path, output: Path) -> dict:
    if output.exists():raise ValidationError('augmented dataset already exists')
    config=load_augmentation_config(config_path)
    legacy_manifest,legacy,legacy_config=read_research_tool_data(legacy_bundle)
    if legacy_manifest['dataset_version']!='synthetic_research_tool_v1':raise ValidationError('augmentation requires v1 source')
    policy,policy_hash=load_config(split_config)
    source_manifest=strict_json((raw_dataset/'manifest.json').read_text())
    records,records_hash=load_records(raw_dataset/'records.jsonl')
    if (source_manifest.get('kind')!='synthetic_schema_warmup' or source_manifest.get('historical_data') is not False
            or sha256_file(raw_dataset/'manifest.json')!=legacy_manifest['source_manifest_sha256']
            or records_hash!=source_manifest['records_sha256'] or records_hash!=legacy_manifest['source_records_sha256']
            or sha256_file(raw_dataset/'targets.jsonl')!=source_manifest['targets_sha256']
            or sha256_file(raw_dataset/'config.json')!=source_manifest['config_sha256']
            or policy_hash!=legacy_manifest['split_config_sha256']):
        raise ValidationError('augmentation source provenance mismatch')
    target_rows=[strict_json(s) for s in (raw_dataset/'targets.jsonl').read_text().splitlines()]
    targets={r['sample_id']:r for r in target_rows}
    if len(targets)!=len(target_rows) or set(targets)!={r.sample_id for r in records}:raise ValidationError('source target identities mismatch')
    parts={p:[replay_augmented_task({'kind':'legacy','source_case':r['case']},p) for r in rows] for p,rows in legacy.items()}
    source_splits={r['event_group_id']:p for p,rows in legacy.items() for r in rows}
    for partition,members in split_records(records,policy).partitions.items():
        by_family=defaultdict(lambda:defaultdict(list))
        for record in members:
            if record.dataset_source!=SOURCE:raise ValidationError('augmentation permits only synthetic oracle source')
            target=targets[record.sample_id]
            validate_target(target,record,policy.sources[SOURCE],raw_dataset)
            case=strict_json((raw_dataset/target['provenance']['artifact']).read_text())
            if parse_record(render_case(case)[0])!=record:raise ValidationError('source case replay mismatch')
            by_family[case['family']][record.event_group_id].append(case)
            if source_splits.setdefault(record.event_group_id,partition)!=partition:raise ValidationError('source split conflict')
        if set(by_family)!=set(FAMILIES):raise ValidationError('missing synthetic math family')
        for family,groups in sorted(by_family.items()):
            take=config['math_groups_per_family'][partition]
            if len(groups)<take:raise ValidationError('insufficient source groups')
            for index,group in enumerate(sorted(groups)[:take]):
                cases=groups[group]
                if len(cases)!=2 or {c['language'] for c in cases}!={'en','zh'}:raise ValidationError('incomplete source group')
                views=['forecast']+(['parameters'] if index<config['parameter_audit_groups_per_family'][partition] else [])
                for case in cases:
                    for view in views:
                        for role in ('research','risk'):
                            parts[partition].append(replay_augmented_task({'kind':'math_role','source_case':case,'role':role,'view':view},partition))
        parts[partition].sort(key=lambda r:r['sample_id'])
    times=legacy_config['observation_times']
    validate_augmented_rows(parts,source_splits,times)
    selected=select_validation(legacy['validation'])
    artifacts={p+'.jsonl':jsonl(rows) for p,rows in parts.items()}
    artifacts.update({'config.json':config_path.read_text(),'evaluation_ids.json':json_text(selected),
                      'source_splits.json':json_text(source_splits),'observation_times.json':json_text(times)})
    report={'schema_version':'1','kind':'synthetic_research_tool_bundle','dataset_version':VERSION,
            'config_sha256':sha256_file(config_path),'legacy_manifest_sha256':sha256_file(legacy_bundle/'manifest.json'),
            'source_manifest_sha256':sha256_file(raw_dataset/'manifest.json'),'source_records_sha256':records_hash,
            'split_config_sha256':policy_hash,'split_version':policy.split_version+':research_tool_v2',
            'artifact_hashes':{k:sha256_bytes(v.encode()) for k,v in artifacts.items()},'counts':counts(parts),
            'validation_evaluation_examples':len(selected),'code':code_provenance(),
            'limitations':legacy_manifest['limitations']+[
                'Additional research/risk targets distinguish specified model probabilities from unobserved draws and latent rates.',
                'Parameter-audit variants address only inputs required for the stated calculation, not future realization.',
                'All math variants inherit source event partitions; the separate uncertainty diagnostic is not used for training.',
                'Original role evaluation IDs remain fixed; synthetic templates are shared across splits.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.research-tool-v2-',dir=output.parent) as tmp:
        stage=Path(tmp)/'bundle';stage.mkdir()
        for name,content in artifacts.items():(stage/name).write_text(content)
        (stage/'manifest.json').write_text(json_text(report));stage.rename(output)
    return report


def read_augmented_data(path: Path) -> tuple[dict,dict,dict]:
    manifest=strict_json((path/'manifest.json').read_text())
    names={p+'.jsonl' for p in PARTITIONS}|{'config.json','evaluation_ids.json','source_splits.json','observation_times.json'}
    if (manifest.get('schema_version')!='1' or manifest.get('kind')!='synthetic_research_tool_bundle'
            or manifest.get('dataset_version')!=VERSION or set(manifest['artifact_hashes'])!=names):
        raise ValidationError('invalid augmented research/tool bundle')
    for name,digest in manifest['artifact_hashes'].items():
        if sha256_file(path/name)!=digest:raise ValidationError('augmented artifact hash mismatch')
    config=load_augmentation_config(path/'config.json')
    if sha256_file(path/'config.json')!=manifest['config_sha256']:raise ValidationError('augmented config hash mismatch')
    parts={p:[strict_json(s) for s in (path/(p+'.jsonl')).read_text().splitlines()] for p in PARTITIONS}
    source_splits=strict_json((path/'source_splits.json').read_text())
    times=strict_json((path/'observation_times.json').read_text())
    validate_augmented_rows(parts,source_splits,times)
    selected=strict_json((path/'evaluation_ids.json').read_text())
    legacy=[r for r in parts['validation'] if r['case']['kind']=='legacy']
    if (manifest['counts']!=counts(parts) or selected!=select_validation(legacy)
            or len(selected)!=manifest['validation_evaluation_examples']):raise ValidationError('augmented counts/selection mismatch')
    for partition,rows in parts.items():
        for view,key in (('forecast','math_groups_per_family'),('parameters','parameter_audit_groups_per_family')):
            for family in FAMILIES:
                groups={r['event_group_id'] for r in rows if r['case']['kind']=='math_role'
                        and r['case']['view']==view and r['case']['source_case']['family']==family}
                if len(groups)!=config[key][partition]:raise ValidationError('augmented family count mismatch')
    return manifest,parts,config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-bundle','raw-dataset','split-config','config','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    print(json_text(build_augmented_data(a.legacy_bundle,a.raw_dataset,a.split_config,a.config,a.output)))


if __name__=='__main__':main()
