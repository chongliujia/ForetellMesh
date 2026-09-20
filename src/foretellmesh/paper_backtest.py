"""Audited USD 100 development replay from offline Base multi-agent forecasts."""
import argparse
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import math
from pathlib import Path
import shutil
import statistics

from .data import sha256_file,strict_json
from .evaluation import code_provenance,json_text
from .market_development import read_inputs,rows
from .metrics import score_predictions
from .offline_base import audit as audit_base
from .paper_trading import Signal,Settlement,TradePrintFeed,decimal,simulate,validate_policy
from .schema import ValidationError,fields,parse_record,timestamp
from .sft_data import jsonl


def load_config(path):
    c=strict_json(path.read_text())
    fixed={'schema_version':'1','initial_cash':'100','execution_model':'next_post_decision_block_mean_full_fill_assumption',
           'exit_policy':'hold_to_verified_settlement','partition':'validation','tuning_on_final_test':False,'scenario_fees_are_hypothetical':True}
    required=set(fixed)|{'max_trade_usd','max_event_usd','max_portfolio_usd','min_trade_usd','min_edge',
                         'max_quote_age_seconds','fill_window_seconds','scenarios'}
    fields(c,required,'paper backtest config')
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):raise ValidationError('unsupported paper backtest scope')
    if not isinstance(c['scenarios'],dict) or not c['scenarios']:raise ValidationError('cost scenarios required')
    policies={}
    for name,costs in c['scenarios'].items():
        if not isinstance(name,str) or not name.replace('_','').isalnum():raise ValidationError('invalid scenario name')
        fields(costs,{'entry_price_premium','fee_fraction'},'scenario costs')
        p={k:c[k] for k in ('initial_cash','max_trade_usd','max_event_usd','max_portfolio_usd','min_trade_usd',
                            'min_edge','max_quote_age_seconds','fill_window_seconds')}
        p.update(costs);validate_policy(p);policies[name]=p
    return c,policies


def prepare(forecast_run,dataset,trade_store,config_path):
    c,policies=load_config(config_path)
    if sha256_file(config_path)!=sha256_file(forecast_run/'paper_config.json'):
        raise ValidationError('trading rules changed after forecast plan was frozen')
    audited=audit_base(forecast_run)
    fc=strict_json((forecast_run/'config.json').read_text())
    manifest,inputs,membership=read_inputs(dataset,fc)
    indexed={r['sample_id']:r for r in inputs}
    generated=rows(forecast_run/'results.jsonl');frozen=rows(forecast_run/'inputs.jsonl')
    if (len(frozen)!=len(inputs) or {r['sample_id']:r for r in frozen}!=indexed
            or len(generated)!=len(inputs) or {r['sample_id'] for r in generated}!=set(indexed)):
        raise ValidationError('forecast and paper dataset populations differ')
    labels=rows(dataset/'partitions/validation.labels.jsonl')
    by_label={r['sample_id']:r['label'] for r in labels}
    if len(by_label)!=len(labels) or set(by_label)!=set(indexed):raise ValidationError('paper label identity mismatch')
    train=rows(dataset/'partitions/train.labels.jsonl')
    if len({r['sample_id'] for r in train})!=len(train) or {r['sample_id'] for r in train}!={s for s,m in membership.items() if m['split']=='train'}:
        raise ValidationError('invalid training-only baseline labels')
    train_groups=defaultdict(list)
    from .schema import binary_outcome
    for r in train:train_groups[membership[r['sample_id']]['event_group_id']].append(binary_outcome(r['label']['outcome']))
    prior=statistics.fmean(statistics.fmean(v) for v in train_groups.values())
    settlements={};signals={name:[] for name in ('base_multi_agent','market_implied','training_event_prior','cash')}
    outcomes=[];probabilities={name:[] for name in signals if name!='cash'}
    for r in generated:
        sid=r['sample_id'];row=indexed[sid];m=membership[sid];payload=row['input']
        record=parse_record({**payload,'sample_id':sid,'event_id':m['event_id'],'event_group_id':m['event_group_id'],
                             'dataset_source':'paper_replay','dataset_version':m['dataset_version'],'label':by_label[sid]})
        if not m['event_id'].startswith('polymarket:'):raise ValidationError('unsupported paper platform')
        market=m['event_id'].split(':',1)[1]
        if not market.isdigit():raise ValidationError('invalid paper market identity')
        if payload['market'] is None:raise ValidationError('missing historical decision reference')
        settle=Settlement(market,record.label.resolution_time,record.label.outcome)
        if market in settlements and settlements[market]!=settle:raise ValidationError('conflicting market settlements')
        settlements[market]=settle
        seconds=r['seconds']
        if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds<0:raise ValidationError('invalid model latency')
        p=r['result']['prediction']['probability'] if r['result']['status']=='completed' else None
        pp={'base_multi_agent':p,'market_implied':payload['market']['probability'],'training_event_prior':prior,'cash':None}
        for name,value in pp.items():
            signals[name].append(Signal(sid,market,m['event_group_id'],record.forecast_input.observation_time,
                record.forecast_input.observation_time+timedelta(seconds=math.ceil(seconds)),
                timestamp(payload['market']['observed_at'],'historical price time'),decimal(payload['market']['probability']),
                None if value is None else decimal(value)))
            if name!='cash':probabilities[name].append(value)
        outcomes.append(record.label.outcome)
    trade_report=strict_json((trade_store/'report.json').read_text())
    counts=trade_report.get('counts',{})
    if (trade_report.get('kind')!='pma_historical_trade_store'
            or counts.get('unique_valid_selected_trades')!=counts.get('trades_with_block_time')
            or counts.get('required_distinct_blocks')!=counts.get('matched_block_times')
            or not counts.get('unique_valid_selected_trades')):
        raise ValidationError('incomplete native trade time provenance')
    digest=sha256_file(trade_store/'prices.sqlite')
    if digest!=trade_report['artifact_hashes']['prices.sqlite']:raise ValidationError('historical trade store changed')
    bindings={'forecast_report_sha256':audited['report_sha256'],'dataset_report_sha256':sha256_file(dataset/'report.json'),
              'trade_report_sha256':sha256_file(trade_store/'report.json'),'trade_store_sha256':digest,
              'config_sha256':sha256_file(config_path)}
    score={name:score_predictions(p,outcomes) for name,p in probabilities.items()}
    return c,policies,signals,list(settlements.values()),bindings,score,prior


def calculate(signals,settlements,trade_store,policies):
    feed=TradePrintFeed(trade_store/'prices.sqlite')
    try:
        # Bind decision references to this exact trade store before simulating fills.
        for s in signals['base_multi_agent']:
            quote=feed.latest(s.market_id,s.observation_time)
            if quote is None or quote[1]!=s.quote_time or abs(quote[0]-s.market_probability)>decimal('0.000000000001'):
                raise ValidationError('decision reference does not match historical price store')
        return {scenario:{name:simulate(ss,settlements,feed,policy) for name,ss in signals.items()}
                for scenario,policy in policies.items()}
    finally:feed.close()


def run(forecast_run,dataset,trade_store,config_path,output):
    if output.exists():raise ValidationError('paper output already exists')
    c,policies,signals,settlements,bindings,scores,prior=prepare(forecast_run,dataset,trade_store,config_path)
    output.mkdir(parents=True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(config_path,output/'config.json')
    plan={'kind':'usd100_base_historical_simulation','frozen_at':datetime.now(timezone.utc).isoformat(),'bindings':bindings,
          'source_paths':{k:str(v.resolve()) for k,v in {'forecast_run':forecast_run,'dataset':dataset,'trade_store':trade_store}.items()},
          'code':code_provenance(),'artifact_hashes':{str(p.relative_to(output)):sha256_file(p) for p in output.rglob('*') if p.is_file()},
          'policies':policies,'initial_cash_usd':'100','training':False,'real_orders_sent':0,'final_test_scored':False}
    (output/'plan.json').write_text(json_text(plan))
    results=calculate(signals,settlements,trade_store,policies)
    report={'status':'completed','kind':plan['kind'],'plan_sha256':sha256_file(output/'plan.json'),
            'observation_count':len(signals['base_multi_agent']),'event_groups':len({s.event_group_id for s in signals['base_multi_agent']}),
            'market_count':len(settlements),'forecast_scores':scores,'training_event_prior':prior,'scenarios':{},
            'simulation_only':True,'real_orders_sent':0,'training':False,'final_test_scored':False,
            'limitations':['Development validation, not an untouched profitability benchmark.',
                'Prices are historical trade prints, not executable quotes. No historical book depth or queue is modeled.',
                'Full hypothetical fill at the first post-decision block mean, subject to the predeclared price limit; no fill guarantee.',
                'No token-lot, minimum-order-size, market impact, reward/rebate, or gas reconstruction; USD 2 cap is a scenario choice.',
                'Fee fractions and price premiums are hypothetical cost stress scenarios, not reconstructed historical platform fees.',
                'Equity uses historical print liquidation proxies sampled at replay events; stale marks are reported. Intraperiod drawdown may be larger.',
                'Archived label available_at is capture time; historical settlement replay uses the separately audited on-chain resolution_time.',
                'Historical outcomes may be present in foundation pretraining; no prospective profitability claim.',
                'Baselines use the same decision latency and risk limits; no hyperparameters were optimized on these results.']}
    artifacts={}
    for scenario,arms in results.items():
        report['scenarios'][scenario]={}
        for arm,result in arms.items():
            dest=output/scenario;dest.mkdir(exist_ok=True)
            for field in ('ledger','equity_curve'):
                path=dest/f'{arm}.{field}.jsonl';path.write_text(jsonl(result[field]));artifacts[str(path.relative_to(output))]=sha256_file(path)
            report['scenarios'][scenario][arm]={k:v for k,v in result.items() if k not in ('ledger','equity_curve')}
    report.update(artifact_hashes=artifacts,finished_at=datetime.now(timezone.utc).isoformat())
    (output/'report.json').write_text(json_text(report))
    return report


def audit(root):
    report=strict_json((root/'report.json').read_text());plan=strict_json((root/'plan.json').read_text())
    if report['status']!='completed' or report['plan_sha256']!=sha256_file(root/'plan.json'):raise ValidationError('paper plan/report changed')
    for name,digest in {**plan['artifact_hashes'],**report['artifact_hashes']}.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name)!=digest:raise ValidationError('paper artifact changed')
    _,policies,signals,settlements,bindings,scores,prior=prepare(
        **{k:Path(v) for k,v in plan['source_paths'].items()},config_path=root/'config.json')
    if bindings!=plan['bindings'] or policies!=plan['policies'] or scores!=report['forecast_scores'] or prior!=report['training_event_prior']:
        raise ValidationError('paper inputs or forecast metrics changed')
    rebuilt=calculate(signals,settlements,Path(plan['source_paths']['trade_store']),policies)
    if set(rebuilt)!=set(report['scenarios']):raise ValidationError('paper scenarios changed')
    for scenario,arms in rebuilt.items():
        if set(arms)!=set(report['scenarios'][scenario]):raise ValidationError('paper strategies changed')
        for arm,value in arms.items():
            if {k:v for k,v in value.items() if k not in ('ledger','equity_curve')}!=report['scenarios'][scenario][arm]:
                raise ValidationError('paper metrics do not reproduce')
            for field in ('ledger','equity_curve'):
                if rows(root/scenario/f'{arm}.{field}.jsonl')!=value[field]:raise ValidationError('paper ledger does not reproduce')
    return {'status':'passed','report_sha256':sha256_file(root/'report.json'),'ledgers_exactly_reproduced':True,
            'initial_cash_usd':'100','real_orders_sent':0,'final_test_scored':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run')
    for name in ('forecast-run','dataset','trade-store','config','output'):r.add_argument('--'+name,type=Path,required=True)
    a=sub.add_parser('audit');a.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.command=='audit':print(json_text(audit(args.run)))
    else:
        kw=vars(args);kw.pop('command');kw['config_path']=kw.pop('config');r=run(**kw)
        print(json_text({'status':r['status'],'observations':r['observation_count'],'scenarios':r['scenarios']}))


if __name__=='__main__':main()
