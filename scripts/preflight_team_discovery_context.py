"""Bounded context sizing for a valid two-page browsing path; no model calls."""
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
from foretellmesh.team_discovery_experiment import prepare
from foretellmesh.team_discovery import STUDY, compact_coverage
from foretellmesh.data import strict_json,sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.peft_runtime import render_agent_prompt
from foretellmesh.team_learning import require

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','model-manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'context preflight exists')
    cfg=strict_json(a.config.read_text());jobs,cat,feed=prepare(cfg)
    tok=AutoTokenizer.from_pretrained(strict_json(a.model_manifest.read_text())['snapshot'],local_files_only=True)
    rows=[]
    for j in jobs:
        c=j['context'];targets={m['market_id'] for m in c['markets']}
        searches=[cat.search({'text':'','limit':6,'offset':offset},c['observation_time'],targets) for offset in (0,6)]
        pool={r['market_id']:r for s in searches for r in s['records']}
        request={'agent':'investigation','instruction':STUDY+' Allowed IDs: '+json.dumps(sorted(set(pool)|targets)),
            'input':c,'upstream':{'retrieved_markets':list(pool.values()),
                'search_coverage':[compact_coverage(s.get('coverage',{})) for s in searches],'account':{}},'adapter':None}
        raw={**request,'upstream':{**request['upstream'],'search_coverage':[s.get('coverage',{}) for s in searches]}}
        rows.append({'episode_id':c['episode_id'],'table_prompt_tokens':len(tok.encode(render_agent_prompt(request))),
            'raw_prompt_tokens':len(tok.encode(render_agent_prompt(raw)))})
    result={'kind':'two_page_browse_context_preflight_v1','config_sha256':sha256_file(a.config),
        'rows':rows,'max_table_prompt_plus_output_tokens':max(r['table_prompt_tokens'] for r in rows)+cfg['max_new_tokens'],
        'max_raw_prompt_plus_output_tokens':max(r['raw_prompt_tokens'] for r in rows)+cfg['max_new_tokens'],
        'runtime_strict_budget_still_enforced':True,'new_model_calls':0,
        'limitations':['Valid two-page browse path only, not an exhaustive bound on arbitrary queries or repair responses.',
            'Account is empty in this sizing check; runtime accounts and repair responses also consume tokens.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json_text(result))
    print(json_text({k:v for k,v in result.items() if k!='rows'}))
