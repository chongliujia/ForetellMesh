from copy import deepcopy
from datetime import timedelta
import json
import unittest

from foretellmesh import team_data_catalogue as data
from foretellmesh import team_variable_contracts as typed
from foretellmesh.team_research_handoff import run_loop,summary
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import timestamp,iso
from test_team_task_negotiation import Backend,reduced_proposal
from test_team_lifecycle import TIMING

class CatalogueBackend(Backend):
    def __init__(self,p,invalid=False):super().__init__(p);self.bad_query=invalid
    def generate(self,request):
        if request['agent']=='research_data_catalogue':
            self.requests.append(deepcopy(request))
            if self.bad_query:return '{}'
            return json.dumps({'source_ids':[typed.SOURCE['source_id']], 'market_ids':['a','b'],
                'as_of':request['input']['research_cutoff'],'reason':'Inspect declared quantities and input times.'})
        return super().generate(request)

class CatalogueTests(unittest.TestCase):
    def test_query_filters_initialization_and_future_without_price_statistics(self):
        cat,_,_,protocol,registry,_=reduced_proposal()
        start=timestamp(cat['a']['initialized_at'],'init');end=timestamp(protocol['as_of'],'end')
        q={'source_ids':[typed.SOURCE['source_id']],'market_ids':['a'],'as_of':protocol['as_of'],'reason':'Inspect coverage.'}
        rows=[(start-timedelta(seconds=1),.13,4),(start,.21,2),(end,.42,3),(end+timedelta(seconds=1),.97,9)]
        a=data.query_catalogue(q,cat,registry,HistoricalBars({'a':rows}),protocol['as_of'])
        b=data.query_catalogue(q,cat,registry,HistoricalBars({'a':[(start,.88,2),(end,.19,3)]}),protocol['as_of'])
        self.assertEqual(a,b);m=a['markets'][0]
        self.assertEqual(m['historical_trade_rows'],5);self.assertEqual(m['historical_blocks'],2)
        self.assertEqual(m['first_source_time'],iso(start));self.assertEqual(m['last_source_time'],iso(end))
        self.assertFalse(a['prices_exposed']);self.assertFalse(a['outcome_labels_exposed'])

    def test_missing_coverage_is_explicit_and_preinit_query_fails(self):
        cat,_,_,protocol,registry,_=reduced_proposal();q={'source_ids':[typed.SOURCE['source_id']],
            'market_ids':['a'],'as_of':protocol['as_of'],'reason':'Inspect coverage.'}
        out=data.query_catalogue(q,cat,registry,HistoricalBars({}),protocol['as_of'])
        self.assertEqual(out['markets'][0]['historical_blocks'],0);self.assertIsNone(out['markets'][0]['first_source_time'])
        q['as_of']=iso(timestamp(cat['a']['initialized_at'],'init')-timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError,'not initialized'):data.query_catalogue(q,cat,registry,HistoricalBars({}),protocol['as_of'])

    def test_unknown_ids_sources_future_and_arbitrary_arguments_rejected(self):
        cat,_,_,protocol,registry,_=reduced_proposal();base={'source_ids':[typed.SOURCE['source_id']],
            'market_ids':['a'],'as_of':protocol['as_of'],'reason':'Inspect.'}
        for changes in ({'market_ids':['heldout']},{'market_ids':['a','a']},{'source_ids':['revenue']},
                        {'market_ids':[1]},{'as_of':'2099-01-01T00:00:00Z'},{'outcome':1}):
            q={**base,**changes}
            with self.assertRaises(ValueError):data.validate_query(q,cat,registry,protocol['as_of'])

    def run_loop(self,runner):
        cat,feed,labels,protocol,registry,_=reduced_proposal()
        return run_loop(runner,{'phase':'training','variable_sources':registry},cat,feed,labels,protocol,1,
            lambda *args:None,lambda a:None,derive_task_change=True,multi_input=True,semantic_gate=True,
            semantic_advisory=True,task_negotiation=True,data_discovery=True)

    def test_query_precedes_design_and_metadata_reaches_team_and_replays(self):
        *_,p=reduced_proposal();b=CatalogueBackend(p);r=DiscoveryRunner(b,None);attempts=self.run_loop(r)
        self.assertEqual(b.requests[0]['agent'],'research_data_catalogue')
        self.assertEqual(summary(attempts)['data_catalogue']['queries_completed'],1)
        self.assertIsNotNone(attempts[0]['workflow']['result'])
        received=[]
        for req in b.requests:
            if req['agent'] in ('research_task_assessor','task_negotiation_coordinator','relation_researcher'):
                received.append(req['agent']);self.assertIn('data_catalogue',req['input'])
                self.assertEqual(req['input']['data_catalogue'],attempts[0]['data_catalogue']['result'])
            if req['agent']=='relation_researcher':
                self.assertFalse(req['input']['source_availability_not_provided'])
                self.assertTrue(req['input']['source_values_not_provided'])
                self.assertEqual(req['input']['data_catalogue']['sources'][0]['quantity'],typed.SOURCE['quantity'])
        self.assertEqual(len(received),3)
        rb=RecordedBackend(r.calls);ar=AuditRunner(rb,None,TIMING)
        self.assertEqual(self.run_loop(ar),attempts);self.assertEqual(ar.calls,r.calls)

    def test_failed_query_consumes_attempt_and_prevents_design(self):
        *_,p=reduced_proposal();b=CatalogueBackend(p,True);attempts=self.run_loop(DiscoveryRunner(b,None))
        self.assertEqual(attempts[0]['workflow']['status'],'data_catalogue_failed')
        self.assertEqual(sum(r['agent']=='research_data_catalogue' for r in b.requests),2)
        self.assertNotIn('relation_researcher',[r['agent'] for r in b.requests])
        self.assertIsNone(attempts[0]['workflow']['result'])

    def test_old_research_context_keeps_prior_source_visibility(self):
        out=typed.research_context({'variable_sources':[]});self.assertTrue(out['source_availability_not_provided'])
        self.assertNotIn('variable_sources',out);self.assertNotIn('data_catalogue',out)

if __name__=='__main__':unittest.main()
