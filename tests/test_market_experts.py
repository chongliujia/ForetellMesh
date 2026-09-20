from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner, validate_agent_output
from foretellmesh.capabilities import load_capabilities, route_plan, MARKET_WORKFLOWS
from foretellmesh.langgraph_runtime import LangGraphRunner
from foretellmesh.market_experts import build_features, enrich, FEATURE_ID, SOURCE
from foretellmesh.market_development import without_timing
from foretellmesh.market_experiment import select, load_config
from foretellmesh.paper_trading import TradePrintFeed
from foretellmesh.pma_trades import prepare_db
from foretellmesh.schema import ValidationError, timestamp

ROOT=Path(__file__).resolve().parents[1]


class ExpertTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'prices.sqlite';con=sqlite3.connect(self.path);prepare_db(con)
        self.t=timestamp('2025-02-01T00:00:00Z','fixture');ts=int(self.t.timestamp())
        for i,(offset,n) in enumerate([(-86400,2),(-3600,3),(-1800,4),(0,6),(1,9)],1):
            con.execute('INSERT INTO blocks VALUES (?,?,?,?)',(i,ts+offset,'fixture',i))
            con.execute('INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)',(str(i),0,i,'100',str(n),'10','fixture','fixture',i))
        con.commit();con.execute('PRAGMA wal_checkpoint(TRUNCATE)');con.close()
        self.feed=TradePrintFeed(self.path);self.addCleanup(self.feed.close)
        self.feature,self.proof=build_features(self.feed,'100',self.t)
        fixture=json.loads((ROOT/'examples/agent_workflow_fixture_v1.json').read_text())
        p=fixture['input'];p['observation_time']='2025-02-01T00:00:00Z'
        p['market']={'probability':.6,'observed_at':p['observation_time'],'available_at':p['observation_time']}
        self.row=enrich({'sample_id':'v','input':p},self.feature);self.context=input_context(self.row['input'])
        t=p['observation_time'];self.responses=fixture['responses']
        for values in self.responses.values():
            for r in values:r['observation_time']=t
        self.responses.update(market_quant=[{'signals':[{'feature':'change_24h','interpretation':'The observed price rose.',
            'evidence_ids':[FEATURE_ID]}],'market_view':'no_independent_edge','unknowns':[],'observation_time':t}],
            game_theory=[{'mechanism':'information_aggregation','assumption':'If supplied price aggregates public information.',
            'implication':'No demonstrated independent edge.','evidence_ids':[FEATURE_ID],
            'market_view':'no_independent_edge','unknowns':[],'observation_time':t}])
        self.agents,_=load_capabilities(ROOT/'configs/market_expert_agents_v1.json')

    def test_features_exact_values_window_cutoffs_and_missing_anchor(self):
        f=self.feature['features'];self.assertAlmostEqual(f['change_24h'],.4);self.assertAlmostEqual(f['change_1h'],.3)
        self.assertEqual(f['trade_count_24h'],3);self.assertEqual(f['block_count_24h'],3)
        self.assertEqual(f['trade_count_1h'],2);self.assertAlmostEqual(f['price_range_1h'],.2)
        self.assertIsNone(f['change_7d']);self.assertEqual(self.proof['source_rows'],4)
        self.assertEqual(self.proof['max_source_time'],'2025-02-01T00:00:00Z')
        self.assertNotIn('volume',f);self.assertNotIn('outcome',json.dumps(self.feature))

    def test_future_prices_never_change_features_or_provenance(self):
        con=sqlite3.connect(self.path);con.execute("UPDATE trades SET numerator='1' WHERE block_number=5");con.commit();con.close()
        feed=TradePrintFeed(self.path)
        try:self.assertEqual(build_features(feed,'100',self.t),(self.feature,self.proof))
        finally:feed.close()

    def test_stale_reference_and_duplicate_feature_source_rejected(self):
        with self.assertRaises(ValidationError):build_features(self.feed,'100',self.t+timedelta(hours=4))
        with self.assertRaises(ValidationError):enrich(self.row,self.feature)
        bad=deepcopy(self.row);f=json.loads(bad['input']['evidence'][-1]['text']);f['as_of']='2025-02-02T00:00:00Z'
        bad['input']['evidence'][-1]['text']=json.dumps(f)
        with self.assertRaises(ValidationError):LangGraphRunner(self.agents,ScriptedBackend({})).run(input_context(bad['input']),workflow='market_control')

    def test_all_four_routes_equal_serial_and_have_same_forecast_input(self):
        forecasts=[]
        for workflow in MARKET_WORKFLOWS:
            a=ScriptedBackend(self.responses);b=ScriptedBackend(self.responses)
            x=AgentRunner(self.agents,a).run(self.context,workflow=workflow)
            y=LangGraphRunner(self.agents,b).run(self.context,workflow=workflow)
            self.assertEqual(x['status'],'completed');self.assertEqual(without_timing(x),without_timing(y));self.assertEqual(a.calls,b.calls)
            self.assertEqual([r['agent'] for r in a.calls],list(MARKET_WORKFLOWS[workflow]))
            self.assertTrue(all(r['adapter'] is None for r in a.calls))
            forecast=a.calls[-1];forecasts.append(forecast['input']);self.assertIn(FEATURE_ID,{e['evidence_id'] for e in forecast['input']['evidence']})
            self.assertEqual(set(forecast['upstream']),set(MARKET_WORKFLOWS[workflow][:-1]))
        self.assertTrue(all(p==self.context.to_payload() for p in forecasts))

    def test_expert_failure_repairs_once_then_blocks_forecast(self):
        rr=deepcopy(self.responses);rr['market_quant']=[{},{}];b=ScriptedBackend(rr)
        r=LangGraphRunner(self.agents,b).run(self.context,workflow='market_quant_game_forecast')
        self.assertEqual(r['status'],'failed');self.assertEqual(r['stage'],'market_quant');self.assertIsNone(r['prediction'])
        self.assertEqual([x['agent'] for x in b.calls],['research','market_quant','market_quant']);self.assertIn('repair',b.calls[-1])
        rr['market_quant']=[{},self.responses['market_quant'][0]];b=ScriptedBackend(rr)
        r=LangGraphRunner(self.agents,b).run(self.context,workflow='market_quant_forecast');self.assertEqual(r['status'],'completed')

    def test_unknown_references_fabricated_metrics_and_hypothesis_contract(self):
        q=self.responses['market_quant'][0];g=self.responses['game_theory'][0]
        changes=[]
        for key in ['nonexistent_volume','change_7d']:
            x=deepcopy(q);x['signals'][0]['feature']=key;changes.append(('market_quant',x))
        x=deepcopy(q);x['signals'][0]['evidence_ids']=['invented'];changes.append(('market_quant',x))
        x=deepcopy(g);x.pop('assumption');changes.append(('game_theory',x))
        x=deepcopy(g);x['mechanism']='insider_positions';changes.append(('game_theory',x))
        x=deepcopy(g);x['observation_time']='2025-02-02T00:00:00Z';changes.append(('game_theory',x))
        x=deepcopy(g);x['probability']=.9;changes.append(('game_theory',x))
        for role,x in changes:
            with self.subTest(role=role,value=x),self.assertRaises(ValidationError):validate_agent_output(role,x,self.context)

    def test_role_configuration_is_explicit_and_base_only(self):
        old,_=load_capabilities(ROOT/'configs/capability_agents_v1.json')
        with self.assertRaises(ValidationError):route_plan(old,'market_quant_forecast')
        with self.assertRaises(ValidationError):route_plan(self.agents,'market_quant_forecast',mode='capability')

    def test_selection_uses_first_time_then_nearest_half_and_is_order_invariant(self):
        rr=[deepcopy(self.row) for _ in range(4)]
        for i,r in enumerate(rr):r['sample_id']=str(i)
        rr[0]['input']['market']['probability']=.1;rr[1]['input']['market']['probability']=.4
        rr[2]['input']['observation_time']='2025-02-02T00:00:00Z';rr[2]['input']['market']['probability']=.5
        mm={r['sample_id']:{'split':'validation','event_group_id':'a' if i<3 else 'b'} for i,r in enumerate(rr)}
        self.assertEqual([r['sample_id'] for r in select(rr,mm)],['1','3'])
        self.assertEqual(select(rr,mm),select(rr[::-1],mm))

    def test_experiment_does_not_allow_training_test_or_policy_search(self):
        config=load_config(ROOT/'configs/market_expert_experiment_v1.json')
        for key,value in [('training',True),('partition','test'),('load_adapters',True),('do_sample',True),('selection','best_pnl')]:
            c=deepcopy(config);c[key]=value;p=Path(self.tmp.name)/'c.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValidationError):load_config(p)


if __name__=='__main__':unittest.main()
