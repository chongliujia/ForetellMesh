"""Frozen Base fact-gate comparison plus a bounded, reused-development paper pair."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
import shutil

from .data import strict_json, sha256_file
from .evaluation import json_text
from .lora_probe import verify_model_manifest
from .peft_runtime import SharedPeftExecutor, PeftTextBackend
from .schema import timestamp
from .team_learning import require
from .team_lifecycle import LifecycleRunner, run_window
from .team_lifecycle_experiment import prepare as prepare_source, RecordedBackend, AuditRunner
from .team_memory_experiment import prepare as prepare_development
from .team_method_memory import validate_memory
from .team_outcome_learning import OutcomeBackend, binary_prompt, binary_probability
from .team_rsi_experiment import now
from .team_trade_experience import learn_available
from .team_experience_gate import facts_for, CRITIC, validate_critique, freeze_gated_memory
from .synthetic_sft import canonical_hash


def prepare(config):
    require(config['kind']=='team_fact_gate_comparison_v1' and config['automatic_fine_tuning'] is False
        and config['final_test_opened'] is False and config['development_is_fresh'] is False,'invalid gate comparison protocol')
    for name in ('source_config','development_config'):
        require(sha256_file(Path(config[name]))==config[name+'_sha256'],'source configuration changed')
    base,_,cat,_,cases=prepare_source(strict_json(Path(config['source_config']).read_text()))
    _,jobs,_,eval_cat,feed,_=prepare_development(strict_json(Path(config['development_config']).read_text()))
    old=Path(config['legacy_gate_run']);require(sha256_file(old/'report.json')==config['legacy_report_sha256'],'legacy report changed')
    old_report=strict_json((old/'report.json').read_text())
    for name in ('reference_reflections.json','reference_trade_experiences.json'):
        require(sha256_file(old/name)==old_report['artifact_hashes'][name],'legacy reflection source changed')
    require(cases==strict_json((old/'reference_trade_experiences.json').read_text()),'experience population changed')
    source_groups={r['event_group_id'] for r in cat.rows.values()};target_groups={r['event_group_id'] for r in eval_cat.rows.values()}
    require(not source_groups&target_groups,'memory source overlaps diagnostic catalogue')
    job=jobs[0]  # Frozen chronology, never select by fill count or return.
    require(max(timestamp(c['exit']['time'],'source') for c in cases)<timestamp(job['context']['observation_time'],'target'),
        'source events later than diagnostic target')
    require(type(config['max_reviews']) is int and 1<=config['max_reviews']<=4 and config['review_window_days']==14,'unbounded pair')
    return base,cat,cases,strict_json((old/'reference_reflections.json').read_text()),job,eval_cat,feed


def check_legacy(runner, cases, old, cutoff):
    by_id={r['experience_id']:r for r in old};results=[]
    for case in cases:
        candidate=by_id[case['experience_id']]['output'];verdict=None;error=None
        try:
            verdict=runner.structured('trade_reflection_critic',CRITIC,
                {'phase':'retrospective','observation_time':cutoff,'facts':facts_for(case)},
                {'candidate':candidate},lambda v:validate_critique(v,case['experience_id']))
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        results.append({'experience_id':case['experience_id'],'legacy_format_gate_would_pass':candidate is not None,
            'critique':verdict,'error':error,'admitted_to_current_memory':False,
            'reason':'Legacy records have no ledger-bound checks or the new explicit test protocol.'})
    return results


def run(config_path, output):
    config=strict_json(config_path.read_text());require(not output.exists() and output.name==config['run_name'],'output exists/name differs')
    base,cat,cases,old,job,eval_cat,feed=prepare(config)
    source=strict_json(Path(config['source_config']).read_text());manifest=Path(source['reference_run'])/'model_manifest.json'
    path,_=verify_model_manifest(manifest,base)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'BF16 GPU unavailable')
    output.mkdir(parents=True);shutil.copyfile(config_path,output/'config.json');shutil.copyfile(manifest,output/'model_manifest.json')
    shutil.copytree(Path(__file__).parent,output/'source_snapshot/foretellmesh',ignore=shutil.ignore_patterns('__pycache__'))
    for name,data in [('cases.json',cases),('diagnostic_job.json',job)]: (output/name).write_text(json_text(data))
    plan={'created_at':now(),'source_hashes':{str(p.relative_to(output)):sha256_file(p) for p in (output/'source_snapshot').rglob('*.py')},
        'config_sha256':sha256_file(config_path),'cases_sha256':sha256_file(output/'cases.json'),
        'job_sha256':sha256_file(output/'diagnostic_job.json'),'selection':'all nine trades; chronological first previously used development job',
        'development_is_fresh':False,'profitability_generalization_claim_allowed':False,
        'memory_selection':'first at most three gate-admitted lessons by entry time; never select by PnL',
        'arms':['control','gated_memory'],'max_reviews':config['max_reviews'],'window_days':config['review_window_days']}
    (output/'plan.json').write_text(json_text(plan))
    report={'kind':config['kind'],'status':'running','started_at':now(),'plan_sha256':sha256_file(output/'plan.json'),
        'base_model':base['model'],'model_revision':base['model_revision'],'seed':base['seed'],
        'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
        'packages':{p:version(p) for p in ('torch','transformers','peft','langgraph')},
        'fine_tuning_triggered':False,'effectiveness_verified':False,'final_test_opened':False,'real_orders_sent':0,
        'development_is_fresh':False}
    def save(): (output/'report.json').write_text(json_text(report))
    save()
    try:
        set_seed(base['seed']);torch.backends.cuda.matmul.allow_tf32=False
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',use_safetensors=True)
        backend=OutcomeBackend(PeftTextBackend(SharedPeftExecutor(model),tok,max_context_tokens=12288,
            max_new_tokens=config['max_new_tokens'],sampling=None,stop_on_json_object=True),max_scoring_tokens=6144)
        runner=LifecycleRunner(backend,cat,config['timing']);cutoff=now();report['feedback_cutoff']=cutoff
        for case in cases:
            learn_available(runner,[case],cutoff,enabled=True)
            row=runner.trade_reflections[case['experience_id']]
            (output/'reviews.json').write_text(json_text(list(runner.trade_reflections.values())))
            (output/'review_calls.json').write_text(json_text(runner.calls))
            print('Gate',case['market_id'],'admitted',row['admitted_to_exploratory_memory'],'error',row['error'],flush=True)
        rows=list(runner.trade_reflections.values())
        memory=freeze_gated_memory(rows,cases,{r['event_group_id'] for r in cat.rows.values()},now())
        validate_memory(memory,job['context'],{r['event_group_id'] for r in eval_cat.rows.values()})
        (output/'memory.json').write_text(json_text(memory))
        (output/'memory_freeze.json').write_text(json_text({'frozen_at':now(),'memory_sha256':sha256_file(output/'memory.json'),
            'review_sha256':sha256_file(output/'reviews.json'),'evaluation_calls_so_far':0,'independent_validation':False}))
        old_runner=LifecycleRunner(backend,cat,config['timing'])
        legacy_checks=check_legacy(old_runner,cases,old,cutoff)
        (output/'legacy_checks.json').write_text(json_text(legacy_checks));(output/'legacy_check_calls.json').write_text(json_text(old_runner.calls))
        print('Memory frozen:',len(memory['lessons']),'candidate procedures; legacy checks completed',flush=True)
        windows={}
        if memory['lessons']:
            for arm in ('control','gated_memory'):
                set_seed(base['seed']);backend.readouts=[]
                runner=LifecycleRunner(backend,eval_cat,config['timing'],memory if arm=='gated_memory' else None)
                window=run_window(job,feed,base['execution_policy'],runner,
                    end=timestamp(job['context']['observation_time'],'start')+timedelta(days=config['review_window_days']),
                    max_reviews=config['max_reviews'],learning=False)
                windows[arm]=window;(output/(arm+'.json')).write_text(json_text(window))
                (output/(arm+'.readouts.json')).write_text(json_text(backend.readouts))
                print('Arm',arm,'reviews',window['review_count'],'valid',sum(e['decision'] is not None for e in window['episodes']),
                    'entries',window['observation_account']['entry_count'],'exits',window['observation_account']['early_exit_count'],flush=True)
        require(all(not p.requires_grad for p in model.parameters()),'weights unfrozen')
        allcalls=strict_json((output/'review_calls.json').read_text())+old_runner.calls+[
            c for w in windows.values() for e in w['episodes'] for c in e['calls']]
        report.update(status='completed',finished_at=now(),base_model_loads=1,all_parameters_frozen=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            experience_count=len(cases),result_classes=dict(Counter(c['result']['result_class'] for c in cases)),
            schema_valid_proposals=sum(r['output'] is not None for r in rows),
            critic_completed=sum(r['critique'] is not None for r in rows),
            admitted_for_exploration=sum(r['admitted_to_exploratory_memory'] for r in rows),
            frozen_lesson_count=len(memory['lessons']),
            legacy_checked=len(legacy_checks),legacy_fact_rejections=sum(bool(r['critique']) and not r['critique']['facts_consistent'] for r in legacy_checks),
            model_calls=len(allcalls),invalid_calls=sum(c['error'] is not None for c in allcalls),
            model_call_seconds=sum(c['seconds'] for c in allcalls),
            paired_diagnostic_performed=bool(windows),paired_skip_reason=None if windows else 'No admitted lesson; do not invent an intervention.',
            arms={name:{'valid_decisions':sum(e['decision'] is not None for e in w['episodes']),
                'review_count':w['review_count'],'entries':w['observation_account']['entry_count'],
                'early_exits':w['observation_account']['early_exit_count'],'snapshot':w['observation_account']['snapshot'],
                'terminal_without_further_reviews':{k:w['terminal_without_further_reviews'][k] for k in
                    ('final_cash','net_pnl','fees_paid','sampled_equity_proxy_max_drawdown_usd')}} for name,w in windows.items()},
            limitations=['Same frozen Base provides proposals and criticism; critic agreement does not certify semantic truth.',
                'Previously used development events: this paired diagnostic cannot establish independent effectiveness.',
                'Adaptive review schedules and measured generation latency may differ across arms; each has its own continuous $100 account.',
                'No forced trading, training, strategy promotion, new final-test exposure or executable-book liquidity claim.'])
        report['artifact_hashes']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!='report.json'};save()
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=now());save();raise
    return report


def audit(output):
    report=strict_json((output/'report.json').read_text());config=strict_json((output/'config.json').read_text())
    plan=strict_json((output/'plan.json').read_text());require(report['status']=='completed','unfinished comparison')
    require(sha256_file(output/'plan.json')==report['plan_sha256'],'plan changed')
    for name,digest in {**plan['source_hashes'],**report['artifact_hashes']}.items():
        p=(output/name).resolve();require(p.is_relative_to(output.resolve()) and sha256_file(p)==digest,'artifact changed')
    base,cat,cases,old,job,eval_cat,feed=prepare(config)
    require(cases==strict_json((output/'cases.json').read_text()) and job==strict_json((output/'diagnostic_job.json').read_text()),'inputs changed')
    calls=strict_json((output/'review_calls.json').read_text());backend=RecordedBackend(calls)
    runner=AuditRunner(backend,cat,config['timing']);learn_available(runner,cases,report['feedback_cutoff'],enabled=True)
    rows=list(runner.trade_reflections.values());require(rows==strict_json((output/'reviews.json').read_text()) and backend.index==len(calls),'gate replay differs')
    memory=strict_json((output/'memory.json').read_text())
    require(memory==freeze_gated_memory(rows,cases,{r['event_group_id'] for r in cat.rows.values()},memory['built_at']),'frozen memory differs')
    calls=strict_json((output/'legacy_check_calls.json').read_text());backend=RecordedBackend(calls)
    runner=AuditRunner(backend,cat,config['timing'])
    require(check_legacy(runner,cases,old,report['feedback_cutoff'])==strict_json((output/'legacy_checks.json').read_text())
        and backend.index==len(calls),'legacy check differs')
    for arm in report['arms']:
        window=strict_json((output/(arm+'.json')).read_text())
        calls=[c for e in window['episodes'] for c in e['calls']]+window['closing_reflection_calls']
        backend=RecordedBackend(calls);runner=AuditRunner(backend,eval_cat,config['timing'],memory if arm=='gated_memory' else None)
        rebuilt=run_window(job,feed,base['execution_policy'],runner,
            end=timestamp(job['context']['observation_time'],'start')+timedelta(days=config['review_window_days']),
            max_reviews=config['max_reviews'],learning=False,recorded_latencies=[e['decision_seconds'] for e in window['episodes']])
        require(rebuilt==window and backend.index==len(calls),'paired replay differs')
        for row in strict_json((output/(arm+'.readouts.json')).read_text()):
            request=next(c['request'] for c in calls if canonical_hash(c['request'])==row['request_sha256'])
            require(binary_prompt(request,row['market_id'])==row['prompt'] and
                abs(binary_probability(row['logits_no_yes'])-row['probability'])<1e-12,'forecast readout differs')
    result={'status':'passed','report_sha256':sha256_file(output/'report.json'),'trades_replayed':len(cases),
        'arms_replayed':len(report['arms']),'model_calls_replayed':report['model_calls'],'new_model_calls':0,'gpu_required':False}
    (output/'audit.json').write_text(json_text(result));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run','audit'))
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json_text(run(args.config,args.output) if args.action=='run' else audit(args.output)))


if __name__=='__main__':main()
