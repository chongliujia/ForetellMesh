from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.paper_trading import TradePrintFeed
from foretellmesh.pma_trades import prepare_db
from foretellmesh.reversion_experiment import config, prepare, materialize, run, audit
from foretellmesh.schema import ValidationError, timestamp, iso
from foretellmesh.sft_data import jsonl
import test_market_development as fixtures

ROOT=Path(__file__).resolve().parents[1]


class ReversionExperimentTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.MarketDevelopmentTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        self.start=timestamp(f.inputs[0]['input']['observation_time'],'fixture')
        proofs=[]
        for i,row in enumerate(f.inputs):
            t=self.start+timedelta(days=i*7)
            row['input']['observation_time']=iso(t)
            row['input']['market']={'probability':.5,'observed_at':iso(t),'available_at':iso(t)}
            f.membership[i]['event_id']='polymarket:'+str(100+i)
            proofs.append({'market_id':str(100+i),'event_group_id':f.membership[i]['event_group_id'],'proof_errors':[],
                'proof':{'initialized_at':iso(self.start-timedelta(days=4)),
                    'rule_valid_through':iso(self.start+timedelta(days=20)),
                    'historical_question':row['input']['question'],'creator_update_count':0,
                    'outcome':i%2,'resolution_time':'future outcome must not enter policy'}})
        self.proofs=proofs
        f.write('partitions/validation.inputs.jsonl',f.inputs);f.write('membership.jsonl',f.membership)
        f.write('partitions/validation.labels.jsonl',[{'sample_id':r['sample_id'],'label':{
            'outcome':0,'resolution_time':iso(self.start+timedelta(days=16)),
            'available_at':iso(self.start+timedelta(days=16))}} for r in f.inputs])
        f.write('audit/00/contracts.jsonl',proofs);f.freeze()
        self.cfg=strict_json((ROOT/'configs/mean_reversion_weekly_v1.json').read_text())
        self.refresh()
        self.store=f.root/'trades';self.store.mkdir()
        con=sqlite3.connect(self.store/'prices.sqlite');prepare_db(con);n=0
        for h in range(-76,14*24+1):
            for seconds in (0,60):
                n+=1;ts=int((self.start+timedelta(hours=h,seconds=seconds)).timestamp())
                con.execute('INSERT INTO blocks VALUES (?,?,?,?)',(n,ts,'fixture',n))
                for mid in ('100','101','102'):
                    con.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)',(mid+':'+str(n),0,n,mid,'1','2','fixture','fixture',n))
        con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
        (self.store/'report.json').write_text(json_text({'kind':'pma_historical_trade_store',
            'counts':{'unique_valid_selected_trades':n*3,'trades_with_block_time':n*3,
                      'required_distinct_blocks':n,'matched_block_times':n},
            'artifact_hashes':{'prices.sqlite':sha256_file(self.store/'prices.sqlite')}}))

    def refresh(self):
        path=self.f.root/'report.json';r=strict_json(path.read_text())
        r['artifact_hashes']['audit/00/contracts.jsonl']=sha256_file(self.f.root/'audit/00/contracts.jsonl')
        path.write_text(json_text(r));self.cfg['dataset_report_sha256']=sha256_file(path)
        self.config_path=self.f.root/'reversion.json';self.config_path.write_text(json_text(self.cfg))

    def test_reproduce_nine_runs_and_detect_ledger_tampering(self):
        root=self.f.root/'result'
        result=run(self.f.root,self.store,self.config_path,root)
        self.assertEqual(result['market_count'],3)
        for arms in result['scenarios'].values():
            self.assertTrue(arms['mean_reversion_weekly']['cadence']['compliant'])
            self.assertEqual(arms['mean_reversion_only']['entry_count'],0)
        self.assertEqual(audit(root)['status'],'passed')
        ledger=root/'cost_assumption/mean_reversion_weekly.ledger.jsonl'
        ledger.write_text(ledger.read_text()+'{}\n')
        with self.assertRaises(ValidationError):audit(root)

    def test_no_label_or_test_parsing_during_materialization(self):
        original=Path.read_text
        def guard(path,*args,**kwargs):
            self.assertNotIn('labels',path.name);self.assertNotIn('test.',path.name)
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            before=materialize(self.f.root,self.store,self.cfg)
        self.proofs[0]['proof']['outcome']=1-self.proofs[0]['proof']['outcome']
        self.f.write('audit/00/contracts.jsonl',self.proofs);self.refresh()
        after=materialize(self.f.root,self.store,self.cfg)
        self.assertEqual(before[0:4],after[0:4]);self.assertEqual(before[-1],after[-1])

    def test_native_bars_match_existing_asof_and_execution_price_feed(self):
        native=HistoricalBars.from_store(self.store/'prices.sqlite',['102'],self.start-timedelta(days=3),self.start+timedelta(days=1))
        other=TradePrintFeed(self.store/'prices.sqlite');self.addCleanup(other.close)
        for minute in (0,1,15,60):
            t=self.start+timedelta(minutes=minute)
            self.assertEqual(native.latest('102',t),other.latest('102',t))
            self.assertEqual(native.next('102',t,t+timedelta(seconds=300)),other.next('102',t,t+timedelta(seconds=300)))

    def test_changed_proof_rules_and_trade_store_rejected(self):
        self.proofs[0]['proof']['historical_question']='Changed contract'
        self.f.write('audit/00/contracts.jsonl',self.proofs)
        with self.assertRaises(ValidationError):prepare(self.f.root,self.store,self.cfg)
        self.refresh()
        with self.assertRaises(ValidationError):prepare(self.f.root,self.store,self.cfg)

    def test_scope_cannot_be_silently_promoted(self):
        for field,value in [('partition','test'),('training',True),('real_orders_sent',1)]:
            c=deepcopy(self.cfg);c[field]=value;self.config_path.write_text(json_text(c))
            with self.assertRaises(ValidationError):config(self.config_path)


if __name__=='__main__':unittest.main()
