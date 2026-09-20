"""Train a small masked PPO allocation policy, freeze it, then evaluate development data."""
import argparse
from collections import defaultdict
from importlib.metadata import version
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

from .data import sha256_file,strict_json
from .evaluation import code_provenance,json_text
from .market_development import rows
from .schema import ValidationError,iso
from .sft_data import jsonl
from .trading_rl_data import prepare,settlements


ENV_KEYS={'schema_version','initial_cash','step_seconds','lookback_hours','minimum_history_points','quote_max_age_seconds','std_floor',
    'fill_window_seconds','cadence_seconds','cadence_lead_seconds','max_holding_seconds','weekly_holding_seconds','cooldown_seconds',
    'max_entries_per_day','max_event_usd','max_portfolio_usd','max_positions','candidate_slots','minimum_partial_exit_usd',
    'entry_price_tolerance','stop_price_points','min_yes_price','max_yes_price','entry_z','max_24h_change','min_round_trip_edge'}


def load_config(path):
    c=strict_json(path.read_text())
    fixed={'schema_version':'1','train_partition':'train','evaluation_partition':'validation','final_test_scored':False,
        'real_orders_sent':0,'foundation_model':'Qwen/Qwen3-8B-Base','foundation_model_updated':False,'llm_calls':0,
        'signal_source':'causal_price_features_only','reward_version':'net_liquidation_equity_delta_usd_v1',
        'checkpoint_selection':'last_fixed_step_for_every_seed_no_validation_selection','training_scenario':'cost_assumption',
        'fixed_controls':['corrected_mean_reversion','weekly_cost_only']}
    if set(c)!=set(fixed)|{'run_name','dataset_report_sha256','seeds','primary_seed','ppo','environment','scenarios','libraries'}:
        raise ValidationError('RL configuration fields differ')
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):raise ValidationError('unsupported RL scope')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']) or any(type(s) is not int or not 0<=s<2**32 for s in c['seeds']):raise ValidationError('invalid seeds')
    if c['primary_seed'] not in c['seeds']:raise ValidationError('primary seed must be prespecified')
    p=c['environment']
    if (set(p)!=ENV_KEYS or p['schema_version']!='1' or p['initial_cash']!='100' or p['candidate_slots']!=4 or p['max_positions']!=4
        or p['cadence_seconds']!=604800 or not 0<p['cadence_lead_seconds']<p['cadence_seconds']
        or not 0<p['fill_window_seconds']<p['step_seconds'] or not 1<=p['minimum_history_points']<=p['lookback_hours']
        or not 0<float(p['max_event_usd'])<=float(p['max_portfolio_usd'])<=100):raise ValidationError('invalid risk/environment policy')
    h=c['ppo']
    if (set(h)!={'total_timesteps','n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef','net_arch','device','torch_threads'}
        or h['device']!='cpu' or h['torch_threads']!=1 or h['n_steps']<2 or h['batch_size']<2
        or h['n_steps']%h['batch_size'] or h['total_timesteps']%h['n_steps'] or h['total_timesteps']<h['n_steps']
        or not 0<h['gamma']<=1 or not 0<h['learning_rate']<1):raise ValidationError('invalid PPO budget')
    if set(c['scenarios'])!={'cost_assumption','cost_stress','frictionless_reference'}:raise ValidationError('missing execution scenarios')
    for costs in c['scenarios'].values():
        if set(costs)!={'entry_price_premium','fee_fraction'} or any(not 0<=float(v)<1 for v in costs.values()):raise ValidationError('invalid costs')
    return c


def runtime(c):
    import torch
    observed={name:version(name) for name in c['libraries']}
    if observed!=c['libraries']:raise ValidationError('RL dependency versions differ: '+str(observed))
    torch.set_num_threads(c['ppo']['torch_threads']);torch.use_deterministic_algorithms(True)
    return {'python':sys.version,'libraries':observed,'device':'cpu','cuda_used':False,'torch_threads':torch.get_num_threads()}


def artifacts(root):
    return {str(p.relative_to(root)):sha256_file(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def freeze_data(root,data):
    (root/f'{data.partition}.catalog.jsonl').write_text(jsonl(data.catalog))
    (root/f'{data.partition}.features.jsonl').write_text(jsonl(data.feature_rows()))
    return {'partition':data.partition,'start':iso(data.start),'end':iso(data.end),'market_count':len(data.windows),
        'event_groups':len({m.event_group_id for m in data.windows}),'observations':len(data.sample_markets),
        'decision_steps':len(data.ticks)-1,'native_rows':sum(n for bars in data.feed.data.values() for _,_,n in bars),
        'bindings':data.bindings,'catalog_sha256':sha256_file(root/f'{data.partition}.catalog.jsonl'),
        'features_sha256':sha256_file(root/f'{data.partition}.features.jsonl')}


def train_models(root,data,labels,c):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from .trading_rl_env import AllocationEnv
    report={};h=c['ppo']
    for seed in c['seeds']:
        dest=root/f'seed_{seed}';dest.mkdir();episodes=[];started=time.perf_counter()
        env=AllocationEnv(data,labels,c['environment'],c['scenarios'][c['training_scenario']])
        model=MaskablePPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0,
            policy_kwargs={'net_arch':h['net_arch']},**{k:h[k] for k in ('n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef')})
        model.set_logger(configure(str(dest),['csv']))
        model.save(dest/'initial.zip')
        initial_parameters={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
        class Capture(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'episode_summary' in info:episodes.append({'timesteps':self.num_timesteps,**info['episode_summary']})
                if self.num_timesteps%8192==0:print(f'PPO seed {seed}: {self.num_timesteps}/{h["total_timesteps"]} steps, {len(episodes)} completed training episodes',flush=True)
                return True
        model.learn(total_timesteps=h['total_timesteps'],callback=Capture())
        model.save(dest/'trained.zip');(dest/'episodes.jsonl').write_text(jsonl(episodes))
        delta=sum(float(torch.sum((v.detach()-initial_parameters[k])**2)) for k,v in model.policy.state_dict().items())**.5
        optimizer_metrics={k:float(v) if math.isfinite(float(v)) else None
                           for k,v in model.logger.name_to_value.items() if k.startswith('train/')}
        model.logger.close()
        report[str(seed)]={'actual_timesteps':model.num_timesteps,'episodes':len(episodes),
            'seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.policy.parameters()),
            'parameter_update_l2':delta,'last_optimizer_metrics':optimizer_metrics,
            'initial_sha256':sha256_file(dest/'initial.zip'),'trained_sha256':sha256_file(dest/'trained.zip'),
            'last_training_episode':episodes[-1] if episodes else None}
        env.close()
        print(f'PPO seed {seed} frozen; {report[str(seed)]["seconds"]:.1f}s',flush=True)
    return report


def arms(c):
    return c['fixed_controls']+[f'{stage}_{seed}' for seed in c['seeds'] for stage in ('initial','ppo')]


def evaluate_one(data,labels,c,costs,*,model=None,weekly_only=False):
    from .trading_rl_env import AllocationEnv
    env=AllocationEnv(data,labels,c['environment'],costs,record=True);obs,_=env.reset();started=time.perf_counter()
    while not env.done:
        if model is None:action=env.fixed_action(weekly_only=weekly_only)
        else:action=int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
        obs,_,_,_,_=env.step(action)
    result={'metrics':env.summary(),'ledger':env.ledger,'decisions':env.decisions,'equity_curve':env.curve,
            'seconds':time.perf_counter()-started}
    env.close();return result


def evaluate(root,data,labels,c,*,reproduce=False):
    from sb3_contrib import MaskablePPO
    models={}
    for seed in c['seeds']:
        for stage,filename in [('initial','initial'),('ppo','trained')]:
            models[f'{stage}_{seed}']=MaskablePPO.load(root/f'seed_{seed}'/f'{filename}.zip',device='cpu')
    results={};timings={}
    for scenario,costs in c['scenarios'].items():
        results[scenario]={};timings[scenario]={};dest=root/'evaluation'/scenario
        if not reproduce:dest.mkdir(parents=True)
        for arm in arms(c):
            value=evaluate_one(data,labels,c,costs,model=models.get(arm),weekly_only=arm=='weekly_cost_only')
            results[scenario][arm]=value['metrics'];timings[scenario][arm]=value['seconds']
            for field in ('ledger','decisions','equity_curve'):
                path=dest/f'{arm}.{field}.jsonl'
                if reproduce:
                    if rows(path)!=value[field]:raise ValidationError('RL replay differs: '+str(path))
                else:path.write_text(jsonl(value[field]))
        print(f'{"Reproduced" if reproduce else "Evaluated"} {scenario}: {len(results[scenario])} arms',flush=True)
    return results,timings


def aggregates(result,c):
    out={}
    for scenario,values in result.items():
        pnl=[float(values[f'ppo_{s}']['net_pnl']) for s in c['seeds']]
        out[scenario]={'ppo_mean_net_pnl':statistics.fmean(pnl),'ppo_std_net_pnl':statistics.pstdev(pnl),
            'ppo_min_net_pnl':min(pnl),'ppo_max_net_pnl':max(pnl),
            'ppo_cadence_passes':sum(values[f'ppo_{s}']['policy_requirements_met'] for s in c['seeds']),
            'seed_count':len(c['seeds']),'primary_seed':c['primary_seed']}
    return out


def run(dataset,trade_store,config_path,output):
    if output.exists():raise ValidationError('output already exists')
    c=load_config(config_path);resources=runtime(c);started=time.perf_counter()
    train=prepare(dataset,trade_store,'train',c)
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    training_data=freeze_data(output,train)
    plan={'config':c,'resources':resources,'code':code_provenance(),'training_data':training_data,
          'source_paths':{'dataset':str(dataset.resolve()),'trade_store':str(trade_store.resolve())},'artifact_hashes':artifacts(output)}
    (output/'plan.json').write_text(json_text(plan))
    print(f'Training only: {training_data["market_count"]} markets / {training_data["event_groups"]} groups / {training_data["decision_steps"]} hourly steps',flush=True)
    train_labels=settlements(dataset,train)
    trained=train_models(output,train,train_labels,c)
    # All seeds/checkpoints are frozen before validation labels or prices are opened.
    training_report={'status':'completed','plan_sha256':sha256_file(output/'plan.json'),'seeds':trained,
                     'artifact_hashes':artifacts(output),'validation_used_for_training':False}
    (output/'training_report.json').write_text(json_text(training_report))
    valid=prepare(dataset,trade_store,'validation',c)
    if ({m.event_group_id for m in train.windows}&{m.event_group_id for m in valid.windows}
        or {m.market_id for m in train.windows}&{m.market_id for m in valid.windows}
        or train.end>=valid.start or max(s.time for s in train_labels)>=valid.start):raise ValidationError('training/evaluation boundary leakage')
    validation_data=freeze_data(output,valid)
    evaluation_plan={'training_report_sha256':sha256_file(output/'training_report.json'),'validation_data':validation_data,
                     'arms':arms(c),'scenarios':list(c['scenarios'])}
    (output/'evaluation_plan.json').write_text(json_text(evaluation_plan))
    result,timings=evaluate(output,valid,settlements(dataset,valid),c)
    report={'status':'completed','plan_sha256':sha256_file(output/'plan.json'),
        'training_report_sha256':sha256_file(output/'training_report.json'),'evaluation_plan_sha256':sha256_file(output/'evaluation_plan.json'),
        'training_data':training_data,'validation_data':validation_data,'seeds':trained,'scenarios':result,'aggregates':aggregates(result,c),
        'evaluation_seconds':timings,'elapsed_seconds':time.perf_counter()-started,
        'training_performed':'small_masked_ppo_allocation_network','foundation_model_updated':False,'llm_calls':0,
        'final_test_scored':False,'real_orders_sent':0,'default_promotion':False,
        'limitations':['Only six admitted training event groups; repeated hours/episodes do not increase independent event coverage.',
            'Development validation has been inspected in earlier strategy work. This is not a new untouched test.',
            'Retrospective labels were archived in 2026; this is not a claim that the dataset or policy existed before the replay dates.',
            'Costs are unchanged hypothetical stress scenarios, not reconstructed historical bid/ask/fees. No depth, queue, partial exchange fills or minimum lot proof.',
            'No tail-side flip for weekly entry; no synthetic zero-price exits. Missing executable references can cause cadence violations.',
            'PPO and corrected controls share candidate truncation, position caps, risk exits and the cadence action guard. Guard activity is not independent learned skill.',
            'First RL input is deterministic historical quant features, not LLM-generated research or game-theory analyses.',
            'Terminal rewards include audited settlement of residual positions after the policy calendar; period-end equity is reported separately.'],
        'artifact_hashes':artifacts(output)}
    (output/'report.json').write_text(json_text(report));return report


def audit(root):
    c=load_config(root/'config.json');runtime(c);r=strict_json((root/'report.json').read_text())
    p=strict_json((root/'plan.json').read_text());tr=strict_json((root/'training_report.json').read_text());ep=strict_json((root/'evaluation_plan.json').read_text())
    if (r['status']!='completed' or r['plan_sha256']!=sha256_file(root/'plan.json')
        or r['training_report_sha256']!=sha256_file(root/'training_report.json')
        or r['evaluation_plan_sha256']!=sha256_file(root/'evaluation_plan.json')
        or ep['training_report_sha256']!=r['training_report_sha256'] or tr['plan_sha256']!=r['plan_sha256'] or c!=p['config']):
        raise ValidationError('training/evaluation freeze changed')
    for name,digest in r['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(root/name)!=digest:raise ValidationError('RL artifact changed: '+name)
    dataset=Path(p['source_paths']['dataset']);store=Path(p['source_paths']['trade_store'])
    datasets={}
    for partition in ('train','validation'):
        d=prepare(dataset,store,partition,c);expected=p['training_data'] if partition=='train' else ep['validation_data']
        if (d.bindings!=expected['bindings'] or d.catalog!=rows(root/f'{partition}.catalog.jsonl')
            or d.feature_rows()!=rows(root/f'{partition}.features.jsonl') or iso(d.start)!=expected['start'] or iso(d.end)!=expected['end']):
            raise ValidationError('partition features do not reproduce')
        datasets[partition]=d
    labels=settlements(dataset,datasets['train'])
    if max(s.time for s in labels)>=datasets['validation'].start:raise ValidationError('future training labels')
    result,_=evaluate(root,datasets['validation'],settlements(dataset,datasets['validation']),c,reproduce=True)
    if result!=r['scenarios'] or aggregates(result,c)!=r['aggregates']:raise ValidationError('RL metrics differ')
    return {'status':'passed','report_sha256':sha256_file(root/'report.json'),'features_reproduced':True,
        'checkpoint_actions_reproduced':True,'ledgers_and_rewards_reproduced':True,'all_seeds_reported':True,
        'validation_used_for_gradients':False,'final_test_scored':False,'real_orders_sent':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('run')
    for name in ('dataset','trade-store','config','output'):a.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('audit').add_argument('--run',type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='run':args['config_path']=args.pop('config');result=run(**args)
    else:result=audit(args['run']);(args['run']/'audit.json').write_text(json_text(result))
    print(json_text(result))


if __name__=='__main__':main()
