from copy import deepcopy
from datetime import timedelta
import json
import unittest

from foretellmesh import team_multi_input as graph
from foretellmesh import team_variable_contracts as typed
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import ValidationError,iso,timestamp
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.team_research_handoff import run_loop
from test_team_executable_methods import fixture,AT
from test_team_lifecycle import TIMING


def setup(targets=('a','b','c')):
    cat,feed,labels,protocol,_=fixture()
    for key,row in cat.items():row.update(market_id=key,question='Question '+key)
    variables=[{'variable_id':'v'+k,'market_id':k,'definition':'Historical Yes share price of '+k,
        'quantity':typed.SOURCE['quantity'],'unit':typed.SOURCE['unit'],'lag_days':0} for k in cat]
    relation={'hypothesis':'Test joint information from three contract prices.','forecast_target':'future_yes_price',
        'horizon_days':1,'variables':variables,'predictions':[{'target':k,'input_ids':['va','vb','vc']} for k in targets]}
    registry=[{**typed.SOURCE,'artifact_sha256':'a'*64,'source_report_sha256':'b'*64}]
    mapping={'bindings':[{'variable_id':v['variable_id'],'source_id':typed.SOURCE['source_id'],'missing_reason':None} for v in variables]}
    calc={'expressions':[{'target':k,'expression':"(v('va')+v('vb')+v('vc'))/3+0.01"} for k in targets],
          'min_observations':8,'min_improvement':.00005,'reason':None}
    proposal={'plan':{'relation':relation,'mapping':mapping,'calculation':calc},'no_plan_reason':None}
    return cat,feed,labels,protocol,registry,proposal


class Backend:
    def __init__(self,proposal):self.proposal=deepcopy(proposal);self.requests=[]
    def generate(self,request):
        self.requests.append(deepcopy(request));role=request['agent'];p=self.proposal['plan']
        if role=='research_coordinator':value={'market_ids':['a','b','c'],'objective':'Test joint contract inputs.'}
        elif role=='relation_researcher':value={'relation':p['relation'],'reason':None}
        elif role=='method_data_member':value=p['mapping']
        elif role=='method_quant_member':value=p['calculation']
        elif role=='research_learning_member':
            value={'feedback_sha256':request['upstream']['feedback']['feedback_sha256'],'next_focus':'Consider revised variables.',
                'data_requests':[],'next_task':{'market_ids':['a','b','c'],'objective':'Consider revised variables.'}}
        else:
            r=request['upstream']['result'];value={k:r[k] for k in ('result_sha256','status','eligible_for_independent_validation')}
            value['limitation']='Synthetic training screen, not independent efficacy or profit.'
        return json.dumps(value)


class MultiInputTests(unittest.TestCase):
    def test_one_target_two_peers_is_accepted_and_scored_once_per_time(self):
        cat,feed,labels,protocol,registry,p=setup(('a',))
        graph.validate_relation({'relation':p['plan']['relation'],'reason':None},cat)
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        self.assertEqual(result['valid_observations'],20)
        self.assertEqual(result['target_event_groups'],1)
        self.assertEqual(result['status'],'insufficient_data')
        self.assertEqual(len({(r['target'],r['observation_time']) for r in result['observations']}),20)

    def test_duplicate_target_must_combine_inputs_not_multiply_predictions(self):
        cat,_,_,_,registry,p=setup(('a',));p['plan']['relation']['predictions']*=2
        with self.assertRaisesRegex(ValidationError,'ONE input_ids'):
            graph.validate_plan(p,cat,registry)

    def test_every_variable_names_admitted_contract_and_causal_lag(self):
        cat,_,_,_,_,p=setup();relation=p['plan']['relation']
        for changes in [{'market_id':'heldout'},{'market_id':123},{'lag_days':-1},{'lag_days':31},{'lag_days':True}]:
            bad=deepcopy(relation);bad['variables'][0].update(changes)
            with self.assertRaises(ValidationError):graph.validate_relation({'relation':bad,'reason':None},cat)
        bad=deepcopy(relation);bad['variables'][1].update(market_id='a')
        with self.assertRaisesRegex(ValidationError,'duplicate measurement'):graph.validate_relation({'relation':bad,'reason':None},cat)

    def test_undefined_unconsumed_and_duplicate_variable_refs_are_rejected(self):
        cat,_,_,_,_,p=setup();relation=p['plan']['relation']
        for refs in [['unknown'],['va','va'],[]]:
            bad=deepcopy(relation);bad['predictions'][0]['input_ids']=refs
            with self.assertRaises(ValidationError):graph.validate_relation({'relation':bad,'reason':None},cat)
        bad=deepcopy(relation)
        for pred in bad['predictions']:pred['input_ids']=['va']
        with self.assertRaisesRegex(ValidationError,'all be used'):graph.validate_relation({'relation':bad,'reason':None},cat)

    def test_external_measurement_cannot_bind_to_share_price(self):
        cat,feed,labels,protocol,registry,p=setup();v=p['plan']['relation']['variables'][1];v.update(quantity='movie_revenue',unit='USD')
        with self.assertRaisesRegex(ValidationError,'source quantity/unit mismatch'):
            graph.execute_plan(p,cat,registry,feed,labels,protocol)

    def test_missing_source_stops_before_feed_or_labels(self):
        cat,_,_,protocol,registry,p=setup();p['plan']['relation']['variables'][1].update(quantity='movie_revenue',unit='USD')
        p['plan']['mapping']['bindings'][1].update(source_id=None,missing_reason='No historical revenue observations.')
        class Forbidden:
            def latest(self,*a):raise AssertionError('price proxy read')
            def __getitem__(self,k):raise AssertionError('outcome read')
        runner=DiscoveryRunner(Backend(p),None)
        result=graph.run_workflow(runner,{'variable_sources':registry},cat,Forbidden(),Forbidden(),protocol,lambda p:self.fail('frozen missing data'))
        self.assertEqual(result['status'],'unavailable_variables');self.assertIsNone(result['result'])
        self.assertEqual(result['data_check']['unavailable_variables'][0]['variable']['market_id'],'b')

    def test_expression_requires_exact_declared_inputs_and_safe_syntax(self):
        _,_,_,_,_,p=setup();relation=p['plan']['relation'];calc=p['plan']['calculation']
        for expression in ["v('va')", "v('va')+v('vb')+v('other')"]:
            bad=deepcopy(calc);bad['expressions'][0]['expression']=expression
            with self.assertRaises(ValidationError):graph.validate_calculation(bad,relation)
        for expression in ["__import__('os').system('x')","p('target',0)","v('va').real","v('va')**2","v(variable)","v('va',1)","[v('va')][0]","v('va',x=1)"]:
            with self.assertRaises(ValidationError):graph.compile_variables(expression)

    def test_values_use_separate_contracts_and_lags(self):
        cat,_,_,_,_,p=setup(('a',));p['plan']['relation']['variables'][1]['lag_days']=1
        feed=HistoricalBars({'a':[(AT+timedelta(days=2),.2,1)],'b':[(AT+timedelta(days=1),.3,1),(AT+timedelta(days=2),.9,1)],
                             'c':[(AT+timedelta(days=2),.4,1)]})
        forecast,evidence,error=graph.forecast_at(p['plan'],p['plan']['relation']['predictions'][0],cat,feed,AT+timedelta(days=2),10800)
        self.assertIsNone(error);self.assertAlmostEqual(forecast['prediction'],.31)
        ev=next(e for e in evidence if e['variable_id']=='vb')
        self.assertEqual(ev['market_id'],'b');self.assertEqual(ev['source_time'],iso(AT+timedelta(days=1)))

    def test_future_or_stale_values_never_backfill_a_variable(self):
        cat,_,_,_,_,p=setup(('a',));pred=p['plan']['relation']['predictions'][0]
        for hours in (-4,1):
            feed=HistoricalBars({'a':[(AT+timedelta(days=2),.4,1)],'b':[(AT+timedelta(days=2,hours=hours),.4,1)],'c':[(AT+timedelta(days=2),.4,1)]})
            forecast,_,error=graph.forecast_at(p['plan'],pred,cat,feed,AT+timedelta(days=2),10800)
            self.assertIsNone(forecast);self.assertIn('vb:',error)
        class Future:
            def latest(self,mid,at):return .4,at+timedelta(seconds=1)
        with self.assertRaisesRegex(ValidationError,'future/invalid'):graph.forecast_at(p['plan'],pred,cat,Future(),AT+timedelta(days=2),10800)

    def test_division_and_bounds_fail_without_silent_clipping(self):
        cat,feed,_,_,_,p=setup(('a',));pred=p['plan']['relation']['predictions'][0]
        for expression in ["v('va')/(v('vb')-v('vc'))","v('va')+v('vb')+v('vc')+2"]:
            p['plan']['calculation']['expressions'][0]['expression']=expression
            forecast,_,error=graph.forecast_at(p['plan'],pred,cat,feed,AT+timedelta(days=2),10800)
            self.assertIsNone(forecast);self.assertTrue(error)

    def test_event_outcome_once_per_target_not_per_input_and_feedback_cutoff(self):
        cat,feed,labels,protocol,registry,p=setup();p['plan']['relation'].update(forecast_target='resolves_yes',horizon_days=None)
        p['plan']['calculation']['min_observations']=3
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        self.assertEqual(result['observations_attempted'],3);self.assertEqual(result['metric'],'brier')
        labels['a']['available_at']=iso(AT+timedelta(days=100))
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        self.assertEqual(result['valid_observations'],2);self.assertEqual(result['failures']['target:feedback_not_available'],1)

    def test_grouped_scoring_cannot_turn_correlated_contracts_into_independence(self):
        cat,feed,labels,protocol,registry,p=setup()
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        self.assertEqual(result['status'],'training_screen_passed');self.assertEqual(result['valid_observations'],60)
        self.assertAlmostEqual(result['mean_group_improvement'],.0001)
        for row in cat.values():row['event_group_id']='shared'
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        self.assertEqual(result['target_event_groups'],1);self.assertFalse(result['eligible_for_independent_validation'])

    def test_fingerprint_ignores_aliases_but_preserves_wiring(self):
        _,_,_,_,_,p=setup();relation=p['plan']['relation'];changed=deepcopy(relation)
        changed['variables'][0]['variable_id']='renamed';changed['hypothesis']='Reworded'
        for pred in changed['predictions']:pred['input_ids']=['renamed' if k=='va' else k for k in pred['input_ids']]
        changed['predictions'].reverse();changed['variables'].reverse()
        self.assertEqual(graph.relation_fingerprint(relation),graph.relation_fingerprint(changed))
        changed['variables'][0]['lag_days']=1
        self.assertNotEqual(graph.relation_fingerprint(relation),graph.relation_fingerprint(changed))

    def test_low_level_forecast_cannot_bypass_missing_source_gate(self):
        cat,feed,_,_,_,p=setup(('a',));p['plan']['mapping']['bindings'][1].update(source_id=None,missing_reason='Missing')
        class Forbidden:
            def latest(self,*a):raise AssertionError('missing input was read')
        with self.assertRaisesRegex(ValidationError,'unbound variable'):
            graph.forecast_at(p['plan'],p['plan']['relation']['predictions'][0],cat,Forbidden(),AT+timedelta(days=2),10800)

    def test_horizon_stride_and_resolution_boundaries_are_preserved(self):
        cat,feed,labels,protocol,registry,p=setup(('a',));p['plan']['relation']['horizon_days']=7
        labels['a']['resolution_time']=iso(AT+timedelta(days=16))
        result=graph.execute_plan(p,cat,registry,feed,labels,protocol)
        times=[timestamp(r['observation_time'],'observation') for r in result['observations']]
        self.assertEqual(times,[AT+timedelta(days=1),AT+timedelta(days=8)])
        self.assertTrue(all(t+timedelta(days=7)<timestamp(labels['a']['resolution_time'],'resolution') for t in times))

    def test_named_kind_translates_only_the_key_and_never_guesses_from_market_id(self):
        cat,_,_,_,_,p=setup();relation=deepcopy(p['plan']['relation'])
        relation['prediction_kind']=relation.pop('forecast_target')
        value={'relation':relation,'reason':None};original=deepcopy(value)
        normalized=graph.validate_named_relation(value,cat)
        self.assertEqual(value,original);self.assertEqual(normalized['relation'],p['plan']['relation'])
        relation['prediction_kind']='a'
        with self.assertRaisesRegex(ValidationError,'never a market ID'):graph.validate_named_relation(value,cat)
        relation['prediction_kind']='future_yes_price';relation['forecast_target']='future_yes_price'
        with self.assertRaises(ValidationError):graph.validate_named_relation(value,cat)

    def test_named_kind_graph_replays_raw_output_and_canonical_freeze(self):
        cat,feed,labels,protocol,registry,p=setup()
        class NamedBackend(Backend):
            def generate(self,request):
                raw=super().generate(request)
                if request['agent']=='relation_researcher':
                    value=json.loads(raw);r=value['relation'];r['prediction_kind']=r.pop('forecast_target');return json.dumps(value)
                return raw
        runner=DiscoveryRunner(NamedBackend(p),None);frozen=[]
        context={'phase':'offline_adaptive_training_research','variable_sources':registry}
        first=run_loop(runner,context,cat,feed,labels,protocol,1,lambda n,p:frozen.append(deepcopy(p)),lambda r:None,
                       derive_task_change=True,multi_input=True,named_prediction_kind=True)
        self.assertEqual(first[0]['workflow']['status'],'training_screen_passed');self.assertEqual(frozen,[p])
        raw=next(c['output'] for c in runner.calls if c['request']['agent']=='relation_researcher')
        self.assertIn('prediction_kind',json.loads(raw)['relation'])
        backend=RecordedBackend(runner.calls);audited=AuditRunner(backend,None,TIMING)
        second=run_loop(audited,context,cat,feed,labels,protocol,1,lambda n,p:None,lambda r:None,
                        derive_task_change=True,multi_input=True,named_prediction_kind=True)
        self.assertEqual(first,second)

    def test_full_team_loop_freezes_before_scores_and_replays_calls(self):
        cat,feed,labels,protocol,registry,p=setup();backend=Backend(p);runner=DiscoveryRunner(backend,None);freezes={}
        class GuardedLabels(dict):
            def __getitem__(self,key):
                if not freezes:raise AssertionError('scoring before freeze')
                return super().__getitem__(key)
        def freeze(n,proposal):freezes[n]=(deepcopy(proposal),canonical_hash(runner.calls))
        context={'phase':'offline_adaptive_training_research','variable_sources':registry}
        first=run_loop(runner,context,cat,feed,GuardedLabels(labels),protocol,2,freeze,lambda r:None,derive_task_change=True,multi_input=True)
        self.assertEqual(first[0]['workflow']['status'],'training_screen_passed')
        self.assertEqual(first[1]['workflow']['status'],'duplicate_relation')
        self.assertEqual(first[1]['task_origin'],'preceding_reflection');self.assertEqual(len(freezes),1)
        recorded=RecordedBackend(runner.calls);audited=AuditRunner(recorded,None,TIMING)
        def check(n,proposal):self.assertEqual((proposal,canonical_hash(audited.calls)),freezes[n])
        second=run_loop(audited,context,cat,feed,labels,protocol,2,check,lambda r:None,derive_task_change=True,multi_input=True)
        self.assertEqual(first,second);self.assertEqual(recorded.index,len(runner.calls))
        for request in backend.requests:
            self.assertNotIn('labels',request['input'])
            if request['agent']=='relation_researcher':self.assertNotIn('variable_sources',request['input'])

if __name__=='__main__':unittest.main()
