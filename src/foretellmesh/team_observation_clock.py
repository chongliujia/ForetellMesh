"""Causal input-only observation clocks for research coverage diagnostics.

This does not change the executable-method evaluator or trading cadence. Neither
outcomes nor future settlement times are accepted by the schedule generator.
"""
import argparse
from collections import Counter
from datetime import timedelta
from pathlib import Path

from .data import strict_json, sha256_file
from .evaluation import json_text
from .schema import timestamp, iso
from .synthetic_sft import canonical_hash
from .team_executable_methods import fresh_price
from .team_learning import require


def input_schedule(binding, catalogue, feed, as_of, *, mode, refs=(('target',0),),
                   max_age_seconds=10800, max_points=120, spacing_days=1):
    require(mode in ('daily_90', 'trade_clock'), 'unknown observation clock')
    require(type(max_points) is int and 1 <= max_points <= 120, 'invalid observation cap')
    require(type(spacing_days) is int and 1 <= spacing_days <= 14, 'invalid observation spacing')
    require(type(max_age_seconds) is int and 1 <= max_age_seconds <= 10800, 'invalid freshness')
    require(isinstance(binding, dict) and set(binding)=={'target','peer'}, 'invalid clock binding')
    refs = set(refs) | {('target',0)}
    for role, lag in refs:
        require(role in ('target','peer') and type(lag) is int and 0 <= lag <= 30, 'invalid clock input')
        require(binding[role] in catalogue, 'clock input has unknown/unbound contract')
    target = binding['target']; cutoff = timestamp(as_of,'as of')
    initialized = {mid: timestamp(catalogue[mid]['initialized_at'],'init') for mid in set(binding.values()) if mid is not None}
    if mode == 'daily_90':
        start = initialized[target].replace(hour=0,minute=0,second=0,microsecond=0)+timedelta(days=1)
        candidates = (start+timedelta(days=n) for n in range(0,90,spacing_days))
    else:
        candidates = (row[0] for row in feed.data.get(target,[]))
    due = initialized[target]; points = []; failures = Counter(); checked = 0
    for at in candidates:
        if at > cutoff: break
        if at < due: continue
        checked += 1; errors = []; evidence = []
        for role, lag in sorted(refs):
            mid = binding[role]; read_at = at-timedelta(days=lag)
            if read_at < initialized[mid]:
                errors.append(role+':before_initialization'); continue
            row, error = fresh_price(feed,mid,read_at,max_age_seconds)
            if error: errors.append(role+':'+error)
            else: evidence.append({**row,'role':role,'lag_days':lag})
        failures.update(errors)
        if errors: continue
        points.append({'observation_time':iso(at),'evidence':evidence})
        # Advance only after a usable input state, determined by information
        # available now. Never choose times using future targets or best returns.
        due = at+timedelta(days=spacing_days)
        if len(points) >= max_points: break
    result = {'mode':mode,'binding':binding,'as_of':as_of,'refs':sorted(refs),'spacing_days':spacing_days,
        'max_points':max_points,'max_quote_age_seconds':max_age_seconds,'candidate_states_checked':checked,
        'failures':dict(failures),'points':points,'schedule_uses_labels':False}
    result['schedule_sha256']=canonical_hash(result);return result


def live_points(schedule, binding, settlements):
    """Private retrospective audit, separate from schedule selection and models.

    A schedule may encounter a late print after resolution. Censor it here;
    never replace it by a hand-picked earlier point or reveal future times.
    """
    return [p for p in schedule['points'] if all(
        timestamp(p['observation_time'],'observation')-timedelta(days=lag)
        < timestamp(settlements[binding[role]],'settlement') for role,lag in schedule['refs'])]


def build(config):
    from .team_research_loop import prepare
    base,_,cat,feed,labels,protocol,_,inventory=prepare(config)
    settlements={k:v['resolution_time'] for k,v in labels.items()};rows=[]
    for target in sorted(cat):
        for peer in [None]+[k for k in sorted(cat) if k!=target]:
            binding={'target':target,'peer':peer};refs=[('target',0)]+([('peer',0)] if peer else [])
            record={'binding':binding,'cross_event_group':bool(peer and cat[target]['event_group_id']!=cat[peer]['event_group_id'])}
            for mode in ('daily_90','trade_clock'):
                result=input_schedule(binding,cat,feed,protocol['as_of'],mode=mode,refs=refs)
                live=live_points(result,binding,settlements)
                record[mode]={'scheduled':len(result['points']),'live_points':len(live),
                    'censored_after_resolution':len(result['points'])-len(live),
                    'schedule_sha256':result['schedule_sha256'],'live_times_sha256':canonical_hash([p['observation_time'] for p in live]),
                    'first_live_observation':live[0]['observation_time'] if live else None,
                    'last_live_observation':live[-1]['observation_time'] if live else None,
                    'candidate_states_checked':result['candidate_states_checked'],'failures':result['failures']}
            rows.append(record)
    summary={}
    for mode in ('daily_90','trade_clock'):
        singles=[r for r in rows if r['binding']['peer'] is None];pairs=[r for r in rows if r['binding']['peer'] is not None]
        summary[mode]={'single_contracts_with_input':sum(r[mode]['live_points']>0 for r in singles),
            'single_contracts_with_8_inputs':sum(r[mode]['live_points']>=8 for r in singles),
            'directed_pairs_with_input':sum(r[mode]['live_points']>0 for r in pairs),
            'cross_group_directed_pairs_with_input':sum(r['cross_event_group'] and r[mode]['live_points']>0 for r in pairs),
            'cross_group_directed_pairs_with_8_inputs':sum(r['cross_event_group'] and r[mode]['live_points']>=8 for r in pairs)}
    return {'kind':'training_observation_clock_diagnostic_v1','inventory':inventory,'protocol':protocol,
        'price_store_report_sha256':base['price_store_report_sha256'],'source_sha256':sha256_file(Path(__file__)),
        'summary':summary,'bindings':rows,'model_calls':0,'development_opened':False,'final_test_opened':False,
        'training_evaluator_changed':False,'net_profit_evaluated':False,
        'limitations':['Full-span first eligible trade per rolling day versus first 90 days UTC midnight; several sampling properties change together.',
            'Trade-clock points are conditional on input liquidity/availability; future target coverage may still fail.',
            'Pair counts are directed; peers are not restricted to their own first 90 days. This differs from the earlier undirected intersection inventory.',
            'At most 120 earliest eligible points; no outcome or future-time selection. Resolution censoring occurs only after scheduling.',
            'Coverage is not predictive efficacy or a requirement to trade daily. Actual historic prints do not prove executable quotes.']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    require(not args.output.exists(),'clock diagnostic output exists')
    report=build(strict_json(args.config.read_text()));report['config_sha256']=sha256_file(args.config)
    args.output.mkdir(parents=True);(args.output/'report.json').write_text(json_text(report))
    print(json_text(report['summary']))

if __name__=='__main__':main()
