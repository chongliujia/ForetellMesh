from copy import deepcopy
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file,strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError,iso
from foretellmesh.trading_rl import load_config,run,audit
from foretellmesh.trading_rl_data import prepare,settlements
import test_reversion_experiment as fixtures

ROOT=Path(__file__).resolve().parents[1]
HAS_RL=all(find_spec(n) is not None for n in ('torch','gymnasium','stable_baselines3','sb3_contrib'))


class AllocationDataTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.ReversionExperimentTests();f.setUp();self.addCleanup(f.doCleanups);self.fixture=f;self.root=f.f.root;self.store=f.store
        self.c=strict_json((ROOT/'configs/allocation_ppo_v1.json').read_text())
        self.c['seeds']=[7];self.c['ppo'].update(total_timesteps=64,n_steps=32,batch_size=16,n_epochs=1,net_arch=[8,8])
        self.t=f.start-timedelta(days=31)
        rr=[];mm=[m for m in f.f.membership if m['split']!='train'];pp=deepcopy(f.proofs)
        for i in range(3):
            sid='train'+str(i);mid=str(200+i);r=deepcopy(f.f.inputs[i]);r['sample_id']=sid
            r['input']['question']='Training contract '+str(i)+'?';r['input']['observation_time']=iso(self.t+timedelta(days=i*7))
            r['input']['market'].update(observed_at=r['input']['observation_time'],available_at=r['input']['observation_time'])
            for e in r['input']['evidence']:e.update(published_at='2024-12-01T00:00:00Z',available_at='2024-12-01T00:00:00Z')
            rr.append(r);mm.append({'sample_id':sid,'event_id':'polymarket:'+mid,'event_group_id':'train_group'+str(i//2),'split':'train','dataset_version':'fixture'})
            pp.append({'market_id':mid,'event_group_id':mm[-1]['event_group_id'],'proof_errors':[],
                'proof':{'initialized_at':iso(self.t-timedelta(days=4)),'rule_valid_through':iso(self.t+timedelta(days=20)),
                         'historical_question':r['input']['question'],'creator_update_count':0}})
        f.f.write('membership.jsonl',mm);f.f.write('partitions/train.inputs.jsonl',rr);f.f.write('audit/00/contracts.jsonl',pp)
        f.f.write('partitions/train.labels.jsonl',[{'sample_id':r['sample_id'],'label':{'outcome':1,
            'resolution_time':iso(self.t+timedelta(days=16)),'available_at':iso(self.t+timedelta(days=16))}} for r in rr])
        self.freeze()
        con=sqlite3.connect(self.store/'prices.sqlite');block=100000
        for h in range(-76,14*24+1):
            for second in (0,60):
                block+=1;ts=int((self.t+timedelta(hours=h,seconds=second)).timestamp())
                con.execute('INSERT INTO blocks VALUES (?,?,?,?)',(block,ts,'train_fixture',block))
                for mid in ('200','201','202'):
                    con.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)',(mid+':'+str(block),0,block,mid,'1','2','fixture','train_fixture',block))
        n=con.execute('SELECT COUNT(*) FROM trades').fetchone()[0];b=con.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
        con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
        (self.store/'report.json').write_text(json_text({'kind':'pma_historical_trade_store',
            'counts':{'unique_valid_selected_trades':n,'trades_with_block_time':n,'required_distinct_blocks':b,'matched_block_times':b},
            'artifact_hashes':{'prices.sqlite':sha256_file(self.store/'prices.sqlite')}}))

    def freeze(self):
        files=['membership.jsonl','partitions/train.inputs.jsonl','partitions/train.labels.jsonl',
            'partitions/validation.inputs.jsonl','partitions/validation.labels.jsonl','audit/00/contracts.jsonl']
        (self.root/'report.json').write_text(json_text({'dataset_version':'fixture','artifact_hashes':{n:sha256_file(self.root/n) for n in files}}))
        self.c['dataset_report_sha256']=sha256_file(self.root/'report.json');self.config=self.root/'rl.json';self.config.write_text(json_text(self.c))

    def test_train_features_never_parse_other_inputs_or_any_label_partition(self):
        original=Path.read_text
        def guard(path,*args,**kwargs):
            self.assertNotIn('labels',path.name);self.assertNotIn('validation.inputs',path.name);self.assertNotIn('test.',path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):d=prepare(self.root,self.store,'train',self.c)
        self.assertEqual({m.market_id for m in d.windows},{'200','201','202'})
        self.assertEqual(len(settlements(self.root,d)),3)
        with self.assertRaises(ValidationError):prepare(self.root,self.store,'test',self.c)

    def test_event_leakage_and_duplicate_training_labels_rejected(self):
        mm=[strict_json(x) for x in (self.root/'membership.jsonl').read_text().splitlines()]
        mm[-1]['event_group_id']='g0';self.fixture.f.write('membership.jsonl',mm);self.freeze()
        with self.assertRaises(ValidationError):prepare(self.root,self.store,'train',self.c)
        mm[-1]['event_group_id']='train_group1';self.fixture.f.write('membership.jsonl',mm)
        ll=[strict_json(x) for x in (self.root/'partitions/train.labels.jsonl').read_text().splitlines()]
        self.fixture.f.write('partitions/train.labels.jsonl',ll+[ll[0]]);self.freeze()
        d=prepare(self.root,self.store,'train',self.c)
        with self.assertRaises(ValidationError):settlements(self.root,d)

    def test_scope_and_step_budget_validation(self):
        for field,value in [('train_partition','validation'),('final_test_scored',True),('foundation_model_updated',True),('real_orders_sent',1)]:
            c=deepcopy(self.c);c[field]=value;self.config.write_text(json_text(c))
            with self.assertRaises(ValidationError):load_config(self.config)

    @unittest.skipUnless(HAS_RL,'optional masked PPO dependencies unavailable')
    def test_real_ppo_update_frozen_before_validation_replay_and_tamper(self):
        out=self.root/'run';original=Path.read_text
        def guard(path,*args,**kwargs):
            if path.name in ('validation.inputs.jsonl','validation.labels.jsonl'):
                self.assertTrue((out/'training_report.json').exists())
            self.assertNotIn('test.',path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):r=run(self.root,self.store,self.config,out)
        self.assertEqual(r['seeds']['7']['actual_timesteps'],64)
        self.assertGreater(r['seeds']['7']['parameter_update_l2'],0)
        self.assertNotEqual(r['seeds']['7']['initial_sha256'],r['seeds']['7']['trained_sha256'])
        self.assertEqual(set(r['scenarios']['cost_assumption']),{'corrected_mean_reversion','weekly_cost_only','initial_7','ppo_7'})
        self.assertEqual(audit(out)['status'],'passed')
        path=out/'evaluation/cost_assumption/ppo_7.ledger.jsonl';path.write_text(path.read_text()+'{}\n')
        with self.assertRaises(ValidationError):audit(out)


if __name__=='__main__':unittest.main()
