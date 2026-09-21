"""Measure data expansion on identical old learning observation times, without labels."""
import argparse
from pathlib import Path
from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.team_discovery_experiment import prepare
from foretellmesh.team_learning import run_query, require


def profile(jobs, catalogue, feed, age):
    rows=[]
    for job in jobs:
        context=job['context']; at=context['observation_time']
        targets={r['market_id'] for r in context['markets']}; visible=catalogue.visible(at)
        require(targets <= set(visible),'reference target absent from expanded learning catalogue')
        pairs={tuple(sorted((a,b))) for a in targets for b in visible if a!=b}
        windows={}
        for days in (7,14,30):
            sampled=[]
            for mids in sorted(pairs):
                result=run_query({'tool':'pair_changes','market_ids':list(mids),'lookback_days':days},feed,at,age)
                sampled.append(result)
            windows[str(days)]={'candidate_pairs':len(pairs),
                'pairs_with_any_daily_change':sum(r['paired_daily_changes']>0 for r in sampled),
                'pairs_with_computable_correlation':sum(r['pearson_change_correlation'] is not None for r in sampled),
                'maximum_paired_daily_changes':max((r['paired_daily_changes'] for r in sampled),default=0)}
        rows.append({'episode_id':context['episode_id'],'as_of':at,'visible_contracts':len(visible),'windows_days':windows})
    return {'catalogue_contracts':len(catalogue.rows),'reference_observations':len(rows),'rows':rows,
        'observations_with_computable_30d_pair':sum(r['windows_days']['30']['pairs_with_computable_correlation']>0 for r in rows)}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--expanded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'coverage audit exists')
    first=strict_json(a.original.read_text()); second=strict_json(a.expanded.read_text())
    jobs,old,old_feed=prepare(first); _,new,new_feed=prepare(second)
    require(first['execution_policy']==second['execution_policy'],'different coverage freshness policy')
    age=first['execution_policy']['max_quote_age_seconds']
    result={'kind':'matched_asof_learning_coverage_audit_v1',
        'config_hashes':{'original':sha256_file(a.original),'expanded':sha256_file(a.expanded)},
        'original':profile(jobs,old,old_feed,age),'expanded':profile(jobs,new,new_feed,age),
        'effectiveness_verified':False,'labels_used_for_coverage':False,'heldouts_opened':False,
        'interpretation':'Data availability on identical learning clocks, not evidence of a profitable relationship.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json_text(result))
    print(json_text({k:{n:v for n,v in result[k].items() if n!='rows'} for k in ('original','expanded')}))
