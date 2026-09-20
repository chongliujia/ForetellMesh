from copy import deepcopy
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file,strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.paper_backtest import run,audit,load_config
from foretellmesh.pma_trades import prepare_db
from foretellmesh.schema import ValidationError,timestamp
from foretellmesh.sft_data import jsonl
import test_market_development as fixtures

ROOT=Path(__file__).resolve().parents[1]


class BacktestTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.MarketDevelopmentTests();f.setUp();self.addCleanup(f.doCleanups);self.fixture=f
        for i,m in enumerate(f.membership[:3]):m['event_id']='polymarket:'+str(100+i)
        f.write('membership.jsonl',f.membership);f.freeze()
        self.forecasts=f.root/'forecasts';self.forecasts.mkdir()
        self.config=self.forecasts/'paper_config.json';self.config.write_bytes((ROOT/'configs/paper_trading_100usd_v1.json').read_bytes())
        fc=strict_json((ROOT/'configs/offline_base_forecasts_v1.json').read_text());fc['dataset_report_sha256']=f.config['dataset_report_sha256']
        (self.forecasts/'config.json').write_text(json_text(fc))
        (self.forecasts/'inputs.jsonl').write_text(jsonl(f.inputs))
        (self.forecasts/'results.jsonl').write_text(jsonl([r for r in f.fake_results() if r['arm']=='multi_base']))
        self.trades=f.root/'trades';self.trades.mkdir()
        con=sqlite3.connect(self.trades/'prices.sqlite');prepare_db(con)
        ts=int(timestamp(f.inputs[0]['input']['observation_time'],'fixture').timestamp())
        for block,t in [(1,ts),(2,ts+2)]:
            con.execute('INSERT INTO blocks VALUES (?,?,?,?)',(block,t,'synthetic',block))
            for market in ('100','101','102'):
                con.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)',(market+str(block),0,block,market,'1','5','fixture','synthetic',block))
        con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
        report={'kind':'pma_historical_trade_store','counts':{'unique_valid_selected_trades':6,'trades_with_block_time':6,
                  'required_distinct_blocks':2,'matched_block_times':2},'artifact_hashes':{'prices.sqlite':sha256_file(self.trades/'prices.sqlite')}}
        (self.trades/'report.json').write_text(json_text(report))
        # Model raw replay is independently covered; this fixture substitutes only that prior audit.
        self.mock=patch('foretellmesh.paper_backtest.audit_base',return_value={'report_sha256':'fixture'});self.mock.start();self.addCleanup(self.mock.stop)

    def execute(self):
        return run(self.forecasts,self.fixture.root,self.trades,self.config,self.fixture.root/'paper')

    def test_end_to_end_ledger_replay_and_tamper_detection(self):
        r=self.execute()
        self.assertEqual(r['observation_count'],3);self.assertEqual(r['event_groups'],2)
        self.assertEqual(r['scenarios']['cost_assumption']['cash']['final_cash'],'100')
        self.assertGreater(r['scenarios']['cost_assumption']['base_multi_agent']['filled_trades'],0)
        root=self.fixture.root/'paper';self.assertEqual(audit(root)['status'],'passed')
        path=root/'cost_assumption/base_multi_agent.ledger.jsonl';path.write_text(path.read_text()+'{}\n')
        with self.assertRaises(ValidationError):audit(root)

    def test_cost_parameters_cannot_be_changed_after_model_generation(self):
        cfg=strict_json(self.config.read_text());cfg['min_edge']='.001'
        other=self.fixture.root/'other.json';other.write_text(json_text(cfg))
        with self.assertRaises(ValidationError):run(self.forecasts,self.fixture.root,self.trades,other,self.fixture.root/'paper')

    def test_price_source_must_match_decision_references(self):
        con=sqlite3.connect(self.trades/'prices.sqlite');con.execute("UPDATE trades SET numerator='2' WHERE block_number=1");con.commit();con.close()
        path=self.trades/'report.json';report=strict_json(path.read_text());report['artifact_hashes']['prices.sqlite']=sha256_file(self.trades/'prices.sqlite');path.write_text(json_text(report))
        with self.assertRaises(ValidationError):self.execute()

    def test_no_final_test_or_live_execution_configuration(self):
        c=strict_json(self.config.read_text())
        for k,v in [('partition','test'),('scenario_fees_are_hypothetical',False),('execution_model','live')]:
            bad=deepcopy(c);bad[k]=v;p=self.fixture.root/'bad.json';p.write_text(json_text(bad))
            with self.assertRaises(ValidationError):load_config(p)


if __name__=='__main__':unittest.main()
