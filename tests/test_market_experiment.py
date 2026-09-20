from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.capabilities import load_capabilities
from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.market_experiment import ARMS, prepare, jobs, load_config, evaluate, audit
from foretellmesh.offline_base import make_runner
from foretellmesh.schema import ValidationError
from foretellmesh.sft_data import jsonl
from foretellmesh.synthetic_sft import canonical_hash
import test_paper_backtest as fixtures

ROOT=Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.BacktestTests();f.setUp();self.addCleanup(f.doCleanups);self.fixture=f
        self.dataset=f.fixture.root;self.c=load_config(ROOT/'configs/market_expert_experiment_v1.json')
        self.c['dataset_report_sha256']=f.fixture.config['dataset_report_sha256']
        self.base_audit=patch('foretellmesh.market_experiment.audit_base',return_value={'report_sha256':'scripted_fixture'})
        self.base_audit.start();self.addCleanup(self.base_audit.stop)
        self.paths={'dataset':self.dataset,'trade_store':f.trades,'baseline':f.forecasts}
        self.agents,_=load_capabilities(ROOT/'configs/market_expert_agents_v1.json')
        self.output=self.dataset/'experts'

    def fake_run(self,fail=False):
        raw,enriched,proof,binding=prepare(**self.paths,c=self.c);root=self.output;root.mkdir()
        for p in proof:p['seconds']=.01
        (root/'config.json').write_text(json_text(self.c));(root/'agent_config.json').write_text(json_text(self.agents))
        (root/'paper_config.json').write_bytes((ROOT/'configs/paper_trading_100usd_v1.json').read_bytes())
        for name,value in [('inputs.jsonl',enriched),('original_inputs.jsonl',raw),('feature_audits.jsonl',proof)]:
            (root/name).write_text(jsonl(value))
        plan={'frozen_at':'2026-01-01T00:00:00Z','config':self.c,'jobs':jobs(enriched),'bindings':binding,
            'source_paths':{k:str(v) for k,v in self.paths.items()},
            'artifact_hashes':{p.name:sha256_file(p) for p in root.iterdir()}}
        (root/'plan.json').write_text(json_text(plan));results=[]
        for job in plan['jobs']:
            row=next(r for r in enriched if r['sample_id']==job['sample_id']);calls=[]
            class Backend:
                available_adapters=set()
                def generate(b,request):
                    role=request['agent'];v={'unknowns':[],'observation_time':request['input']['observation_time']}
                    if role=='research':v.update(evidence_ids=['e'],counter_evidence_ids=[])
                    elif role=='market_quant':v.update(signals=[],market_view='no_independent_edge')
                    elif role=='game_theory':v.update(mechanism='information_aggregation',assumption='If price aggregates information.',implication='No demonstrated edge.',evidence_ids=[],market_view='no_independent_edge')
                    else:v.update(event=request['input']['question'],probability=.4,confidence='low',base_rate=None,key_evidence=['e'],counter_evidence=[])
                    if fail and job['arm']=='quant' and job['sample_id']==raw[0]['sample_id'] and role=='market_quant':v={}
                    text=json_text(v)
                    calls.append({'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':text,
                        'error':None,'error_type':None,'usage':{'input_tokens':100,'output_tokens':50,'output_reached_token_limit':False}})
                    return text
            result=make_runner(self.agents,Backend(),self.c).run(input_context(row['input']),workflow=ARMS[job['arm']])
            results.append({**job,'input_sha256':canonical_hash(row['input']),'calls':calls,'result':result,'seconds':.1,
                'peak_allocated_bytes':100,'peak_reserved_bytes':200})
        (root/'results.jsonl').write_text(jsonl(results))
        (root/'generation_report.json').write_text(json_text({'status':'completed','plan_sha256':sha256_file(root/'plan.json'),
            'results_sha256':sha256_file(root/'results.jsonl'),'loaded_adapters':[],'all_parameters_frozen':True,'base_model_loads':1,
            'started_at':'2026-01-01T00:00:01Z','generation_started_at':'2026-01-01T00:00:02Z','finished_at':'2026-01-01T00:01:00Z',
            'planned_jobs':len(results),'completed_jobs':len(results)}))
        return results

    def test_preparation_never_parses_labels_or_test_inputs(self):
        original=Path.read_text
        def guard(path,*args,**kwargs):
            self.assertNotIn('labels',path.name);self.assertNotIn('test.',path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):raw,enriched,proof,_=prepare(**self.paths,c=self.c)
        self.assertEqual(len(raw),2);self.assertEqual(len(enriched),2)
        self.assertEqual({r['sample_id'] for r in raw},{'v0','v2'})
        self.assertEqual(len(proof),2)

    def test_end_to_end_raw_replay_features_ledgers_and_failure_coverage(self):
        self.fake_run(fail=True);r=evaluate(self.output)
        self.assertEqual(r['observations'],2);self.assertEqual(r['event_groups'],2);self.assertEqual(r['common_valid_count'],1)
        self.assertEqual(r['forecast_scores']['quant']['coverage'],.5)
        self.assertEqual(r['scenarios']['cost_assumption']['cash']['final_cash'],'100')
        self.assertGreater(r['scenarios']['cost_assumption']['quant_game']['filled_trades'],0)
        self.assertEqual(audit(self.output)['status'],'passed')
        p=self.output/'evaluation/cost_assumption/quant_game.ledger.jsonl';p.write_text(p.read_text()+'{}\n')
        with self.assertRaises(ValidationError):audit(self.output)

    def test_configuration_and_raw_outputs_cannot_change_after_freeze(self):
        self.fake_run();p=self.output/'paper_config.json';original=p.read_bytes();p.write_text('{}')
        with self.assertRaises(ValidationError):evaluate(self.output)
        p.write_bytes(original)
        p=self.output/'results.jsonl';r=rows(p);r[0]['calls'][0]['output']='{}';p.write_text(jsonl(r))
        with self.assertRaises(ValidationError):evaluate(self.output)

    def test_source_price_mutation_detected_before_metrics(self):
        self.fake_run();p=self.fixture.trades/'report.json';p.write_text('{}')
        with self.assertRaises(ValidationError):evaluate(self.output)


if __name__=='__main__':unittest.main()
