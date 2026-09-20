from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.forecast_diagnostic import (FILES, candidate_summary, load_config, make_jobs,
    read_data, replay, score, select_train, visible_input)
from foretellmesh.market_development import runner
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
import test_market_development as fixtures

ROOT = Path(__file__).resolve().parents[1]


class ForecastDiagnosticTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.MarketDevelopmentTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root; self.agents = fixture.agents
        train = [{'sample_id': 'train', 'input': deepcopy(fixture.inputs[0]['input'])}]
        fixture.write('partitions/train.inputs.jsonl', train)
        self.config = load_config(ROOT/'configs/forecast_diagnostic_v1.json')
        self.freeze()

    def freeze(self):
        (self.root/'report.json').write_text(json_text({'dataset_version': 'fixture',
            'artifact_hashes': {name: sha256_file(self.root/name) for name in FILES}}))
        self.config['dataset_report_sha256'] = sha256_file(self.root/'report.json')
        self.manifest, self.inputs, self.membership = read_data(self.root, self.config)

    def fake_results(self, invalid=None):
        result = []
        by_id = {r['sample_id']:r for part in self.inputs.values() for r in part}
        for job in make_jobs(self.inputs, self.membership, self.config):
            calls = []
            class Backend:
                available_adapters = set()
                def generate(backend, request):
                    value = {'event':request['input']['question'], 'probability':.3 if job['price']=='visible' else .5,
                             'confidence':'low', 'base_rate':None, 'key_evidence':['e'], 'counter_evidence':[],
                             'unknowns':[], 'observation_time':request['input']['observation_time']}
                    output = 'invalid' if (job['phase'],job['sample_id'])==invalid else json_text(value)
                    calls.append({'request':deepcopy(request),'request_sha256':canonical_hash(request),'output':output,
                                  'error_type':None,'error':None,'usage':None})
                    return output
            value=runner(self.agents,Backend(),self.config).run(input_context(visible_input(by_id[job['sample_id']],job['price'])),
                                                              workflow='single_forecast',mode='base')
            result.append({**job,'result':value,'calls':calls})
        return result

    def test_labels_not_parsed_test_not_opened_and_only_market_changes(self):
        original=Path.read_text
        def guard(path,*args,**kwargs):
            self.assertNotIn('test.',path.name);self.assertNotIn('labels',path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):read_data(self.root,self.config)
        for row in self.inputs['validation']:
            hidden=visible_input(row,'hidden');self.assertIsNone(hidden['market'])
            self.assertEqual({k:v for k,v in hidden.items() if k!='market'},
                             {k:v for k,v in row['input'].items() if k!='market'})
            self.assertIsNotNone(row['input']['market'])

    def test_selection_is_label_blind_order_stable_and_seeds_are_paired(self):
        rows=[{'sample_id':str(i),'input':self.inputs['train'][0]['input']} for i in range(8)]
        membership={str(i):{'split':'train','event_group_id':'g'+str(i//4),'event_id':'c'+str(i//2)} for i in range(8)}
        a=select_train(rows,membership,self.config);b=select_train(list(reversed(rows)),membership,self.config)
        self.assertEqual(a,b);self.assertEqual(len(a),4)
        self.assertEqual(len({membership[r['sample_id']]['event_id'] for r in a}),4)
        jobs=[j for j in make_jobs(self.inputs,self.membership,self.config) if j['phase']=='train_sampling']
        self.assertEqual(len(jobs),8)
        for i in range(0,len(jobs),2):
            self.assertEqual(jobs[i]['seed'],jobs[i+1]['seed'])
            self.assertNotEqual(jobs[i]['price'],jobs[i+1]['price'])
        self.assertEqual(len({j['seed'] for j in jobs}),4)
        membership['0']['split']='test'
        with self.assertRaises(ValidationError):select_train(rows,membership,self.config)

    def test_input_leakage_hash_and_split_rejected(self):
        row=deepcopy(self.inputs['train'][0]);row['input']['outcome']=1
        self.fixture.write('partitions/train.inputs.jsonl',[row])
        with self.assertRaises(ValidationError):read_data(self.root,self.config)
        with self.assertRaises(ValidationError):self.freeze()

    def test_raw_replay_rejects_price_adapter_response_and_seed_tampering(self):
        rr=self.fake_results();replay(self.inputs,self.membership,rr,self.agents,self.config)
        for mutation in ('price','adapter','output','seed'):
            changed=deepcopy(rr)
            if mutation=='price':changed[0]['calls'][0]['request']['input']['market']=None
            elif mutation=='adapter':changed[0]['calls'][0]['request']['adapter']='research_tool_lora'
            elif mutation=='output':changed[0]['calls'][0]['output']='{}'
            else:changed[0]['seed']+=1
            with self.assertRaises(ValidationError):replay(self.inputs,self.membership,changed,self.agents,self.config)
        with self.assertRaises(ValidationError):replay(self.inputs,self.membership,rr[:-1],self.agents,self.config)

    def test_missing_predictions_remain_in_denominator_common_scores_match(self):
        rr=self.fake_results(invalid=('price_ablation','v0'))
        cached=[r for r in self.fixture.fake_results() if r['arm']=='single_base']
        s=score(self.root,self.inputs,self.membership,rr,cached,self.config)
        a=s['price_ablation']
        self.assertEqual(a['full']['hidden']['coverage'],2/3)
        self.assertEqual(a['common_sample_ids'],['v1','v2'])
        self.assertAlmostEqual(a['common']['market']['brier'],(.8**2+.2**2)/2)
        self.assertEqual(a['full']['hidden']['event_mean']['brier'],.25)
        self.assertEqual(s['training_candidates']['visible']['groups_with_valid_reward_variation'],0)
        with self.assertRaises(ValidationError):score(self.root,self.inputs,self.membership,rr+rr[:1],cached,self.config)

    def test_reward_floor_cannot_masquerade_as_forecast_diversity(self):
        summary=candidate_summary([.2,.2,None,.2],1,self.config)
        self.assertEqual(summary['probability_spread']['range'],0)
        self.assertEqual(summary['valid_reward_spread']['range'],0)
        self.assertTrue(summary['format_only_reward_variation'])
        self.assertAlmostEqual(summary['all_reward_spread']['range'],.36)
        summary=candidate_summary([.1,.3,.1,.3],0,self.config)
        self.assertAlmostEqual(summary['valid_reward_spread']['range'],.08)
        self.assertFalse(summary['format_only_reward_variation'])
        summary=candidate_summary([None]*4,0,self.config)
        self.assertIsNone(summary['valid_reward_spread']['range']);self.assertEqual(summary['all_reward_spread']['range'],0)
        with self.assertRaises(ValidationError):candidate_summary([1.5],0,self.config)

    def test_config_rejects_test_partition_training_and_reward_changes(self):
        for k,v in [('sampling_partition','validation'),('validation_partition','test'),('training',True),('invalid_reward',0.0)]:
            c=deepcopy(self.config);c[k]=v;path=self.root/'config.json';path.write_text(json_text(c))
            with self.assertRaises(ValidationError):load_config(path)


if __name__=='__main__':unittest.main()
