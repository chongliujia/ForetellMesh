"""Bounded native identity enrichment for PMA macro candidates.

Selection uses only closed status and a frozen title keyword predicate. Current
API descriptions are review material, never retroactively available evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from urllib.parse import urlencode

from .data import sha256_file, strict_json
from .evaluation import json_text
from .macro_history_capture import fetched
from .pma_catalog import MACRO, write_row
from .schema import ValidationError
from .synthetic_sft import canonical_hash


def selection(markets_root):
    import pyarrow.parquet as pq
    selected, sources = [], []
    for path in sorted(markets_root.glob('*.parquet')):
        source = {'file': path.name, 'sha256': sha256_file(path)}; sources.append(source)
        position = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,
                columns=['id','condition_id','question','outcomes','clob_token_ids','closed']):
            for row in batch.to_pylist():
                if row['closed'] is True and MACRO.search(row['question']):
                    selected.append({**row, 'source_ref': {**source, 'row': position}})
                position += 1
    selected.sort(key=lambda row: int(row['id']))
    if not sources or not 1 <= len(selected) <= 2000 or len({r['id'] for r in selected}) != len(selected):
        raise ValidationError('invalid or unbounded PMA enrichment selection')
    return {'selection_rule': 'closed snapshot and title regex '+MACRO.pattern, 'sources': sources, 'rows': selected}


def urls_for(rows):
    return ['https://gamma-api.polymarket.com/markets?'+urlencode(
        [('id', row['id']) for row in rows[start:start+100]]+[('closed','true'),('limit',100)])
        for start in range(0,len(rows),100)]


def token_pairs(outcomes, tokens):
    outcomes = strict_json(outcomes) if isinstance(outcomes,str) else outcomes
    tokens = strict_json(tokens) if isinstance(tokens,str) else tokens
    if (not isinstance(outcomes,list) or len(outcomes)!=2 or any(not isinstance(o,str) for o in outcomes)
            or len(set(outcomes))!=2 or not isinstance(tokens,list) or len(tokens)!=2
            or any(not isinstance(t,str) or not re.fullmatch(r'[0-9]+',t) for t in tokens) or len(set(tokens))!=2):
        raise ValidationError('binary outcome token mapping unavailable')
    return dict(zip(outcomes,tokens))


def capture(markets_root, output):
    plan = selection(markets_root)
    if output.exists():raise ValidationError('PMA enrichment archive exists')
    output.mkdir(parents=True); (output/'raw').mkdir(); (output/'selection.json').write_text(json_text(plan))
    refs=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ref,raw in pool.map(fetched,urls_for(plan['rows'])):
            ref['file']=f'raw/{len(refs):03d}.bin';(output/ref['file']).write_bytes(raw);refs.append(ref)
            (output/'manifest.json').write_text(json_text({'schema_version':'1','kind':'pma_native_enrichment',
                'selection_sha256':sha256_file(output/'selection.json'),'requests':refs,'requests_sha256':canonical_hash(refs)}))
    return {'selected_markets':len(plan['rows']),'requests':len(refs),'failures':sum(r['status']!=200 for r in refs)}


def build(archive, output):
    if output.exists():raise ValidationError('PMA enrichment output exists')
    manifest=strict_json((archive/'manifest.json').read_text());plan=strict_json((archive/'selection.json').read_text())
    if (manifest.get('kind')!='pma_native_enrichment' or manifest['selection_sha256']!=sha256_file(archive/'selection.json')
            or manifest['requests_sha256']!=canonical_hash(manifest['requests'])
            or [r['url'] for r in manifest['requests']]!=urls_for(plan['rows'])):
        raise ValidationError('PMA enrichment selection or coverage changed')
    responses={}
    for ref in manifest['requests']:
        path=(archive/ref['file']).resolve()
        if not path.is_relative_to(archive.resolve()) or sha256_file(path)!=ref['sha256']:raise ValidationError('native response changed')
        if ref['status']!=200:continue
        batch=strict_json(path.read_text())
        if not isinstance(batch,list):raise ValidationError('invalid native market batch')
        for row in batch:
            if row['id'] in responses:raise ValidationError('duplicate native market in requested batches')
            responses[row['id']] = (ref,row)
    if responses.keys()-{r['id'] for r in plan['rows']}:raise ValidationError('native query returned unrequested markets')
    output.mkdir(parents=True); matched=0;events=set();missing=0;yes_no=0;unmapped=0;unsupported=0
    with (output/'identities.jsonl').open('w') as stream:
        for source in plan['rows']:
            blockers=['current_description_is_review_only','historical_rule_version_required','semantic_event_group_review_required']
            native=responses.get(source['id']);identity_ok=False;token_ok=False;parent_ids=[];supported=False
            if native:
                ref,row=native;identity_ok=row['conditionId']==source['condition_id'];token_ok=False
                try:
                    tokens=token_pairs(row['outcomes'],row.get('clobTokenIds'))
                    original=token_pairs(source['outcomes'],source['clob_token_ids']);token_ok=tokens==original
                    supported=set(original)=={'Yes','No'}
                    if not supported:unsupported+=1;blockers.append('unsupported_non_yes_no_outcome_labels')
                    if not token_ok:blockers.append('native_outcome_token_pairs_disagree')
                except (ValidationError,KeyError):unmapped+=1;blockers.append('binary_token_mapping_unavailable')
                parent_ids=sorted({event['id'] for event in row.get('events',[])})
                if not identity_ok:blockers.append('native_condition_identity_disagrees')
                if source['question']!=row['question']:blockers.append('current_question_differs_from_archive')
                if not parent_ids:blockers.append('native_parent_event_missing')
                matched+=identity_ok and token_ok;yes_no+=identity_ok and token_ok and supported;events.update(parent_ids)
            else:
                missing+=1;ref=None;row={};blockers.append('native_market_missing')
            write_row(stream,{'market_id':source['id'],'source_ref':source['source_ref'],'native_ref':ref,
                'condition_identity_match':identity_ok,'token_mapping_match':token_ok,'native_event_ids':parent_ids,
                'supported_yes_no_outcomes':supported,
                'description_for_review_only':row.get('description'),'resolution_source_for_review_only':row.get('resolutionSource'),
                'question_from_archive':source['question'],'blockers':sorted(blockers),'ready_for_training':False,'ready_for_scoring':False})
    report={'schema_version':'1','kind':'pma_identity_enrichment_audit','capture_manifest_sha256':sha256_file(archive/'manifest.json'),
        'selection_sha256':sha256_file(archive/'selection.json'),'selected_markets':len(plan['rows']),'returned_markets':len(responses),
        'identity_and_token_matches':matched,'missing_markets':missing,'distinct_native_parent_events':len(events),
        'supported_yes_no_matches':yes_no,'markets_missing_token_mapping':unmapped,'unsupported_outcome_label_markets':unsupported,
        'identities_sha256':sha256_file(output/'identities.jsonl'),'model_calls':0,
        'limitations':['Native parents are a minimum grouping unit; correlated events across different parents still require merging.',
                       'Current descriptions and snapshots cannot establish historical information availability.']}
    (output/'report.json').write_text(json_text(report));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['capture','build'])
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json_text(capture(args.input,args.output) if args.command=='capture' else build(args.input,args.output)))


if __name__=='__main__':main()
