"""Recheck only archived task-tag failures; never rewrite or rerun a trajectory."""
import argparse
from pathlib import Path
from foretellmesh.agent_runtime import decode_agent_response
from foretellmesh.data import strict_json,sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.team_research_handoff import validate_derived_task,validate_reflection
from foretellmesh.team_learning import require
from foretellmesh.synthetic_sft import canonical_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'tag audit output exists')
    report=strict_json((args.source/'report.json').read_text())
    require(sha256_file(args.source/'calls.json')==report['artifact_hashes']['calls.json'],'calls changed')
    calls=strict_json((args.source/'calls.json').read_text())
    cat=strict_json((args.source/'catalogue.json').read_text())
    require(sha256_file(args.source/'catalogue.json')==report['artifact_hashes']['catalogue.json'],'catalogue changed')
    rows=[]
    for index,call in enumerate(calls):
        if call['error']!='new_contracts must add at least one different market ID':continue
        request=call['request'];previous=request['input']['previous_task'];value,_=decode_agent_response(call['output'],'single_json_fence')
        error=None;derived=None
        try:
            if request['agent']=='research_coordinator':derived=validate_derived_task(value,cat,previous)
            else:derived=validate_reflection(value,request['upstream']['feedback'],cat,previous,derive_task_change=True)['next_task']
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        rows.append({'call_index':index,'raw_call_sha256':canonical_hash(call),'derived_task':derived,'error':error})
    result={'kind':'recorded_task_metadata_compatibility_check_v1','source_report_sha256':sha256_file(args.source/'report.json'),
        'script_sha256':sha256_file(Path(__file__)),'validator_source_sha256':sha256_file(Path('src/foretellmesh/team_research_handoff.py')),
        'original_task_tag_failures':len(rows),'compatible_after_derivation':sum(r['error'] is None for r in rows),
        'checks':rows,'new_model_calls':0,'original_trajectory_unchanged':True,
        'limitation':'Only recorded task validation is reprocessed. Later requests would change, so this is not a counterfactual trajectory or new research result.'}
    args.output.mkdir(parents=True);(args.output/'report.json').write_text(json_text(result))
    print(json_text({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':main()
