from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh import team_review_boundary as boundary
from foretellmesh import team_multi_input as graph
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.team_research_handoff import run_loop
from foretellmesh.synthetic_sft import canonical_hash
from test_team_semantic_consistency import SemanticBackend,Forbidden
from test_team_multi_input import setup
from test_team_lifecycle import TIMING

class BoundaryTests(unittest.TestCase):
    def test_all_review_opinions_and_invalid_review_cannot_veto_empirical_screen(self):
        for verdicts in (['mismatch'],['uncertain'],['invalid','invalid'],['aligned']):
            cat,feed,labels,protocol,registry,p=setup(('a',));b=SemanticBackend(p,verdicts)
            r=DiscoveryRunner(b,None);frozen=[]
            class Labels(dict):
                def __getitem__(self,k):
                    self_test.assertTrue(frozen,'target labels accessed before method frozen')
                    return super().__getitem__(k)
            self_test=self
            result=graph.run_workflow(r,{'variable_sources':registry},cat,feed,Labels(labels),protocol,frozen.append,
                semantic_gate=True,semantic_advisory=True,probe_qualified=False)
            self.assertIsNotNone(result['result']);self.assertEqual(result['relation']['relation'],p['plan']['relation'])
            self.assertEqual(len(frozen),1);self.assertEqual(sum(q['agent']=='relation_researcher' for q in b.requests),1)
            self.assertEqual(result['review_boundary']['model_review_authority'],'advisory_only')
            for q in b.requests:
                if q['agent'] in ('method_data_member','method_quant_member'):
                    self.assertNotIn('semantic_reviews',q['input'])
                    self.assertEqual(q['input']['execution_contract']['relation_sha256'],canonical_hash(p['plan']['relation']))

    def test_card_distinguishes_future_price_from_event_probability(self):
        cat,_,_,protocol,_,p=setup(('a',));rel=p['plan']['relation']
        card=boundary.execution_contract(rel,cat,protocol)
        self.assertEqual(card['targets'][0]['output_unit'],'USD_per_YES_share')
        rel.update(forecast_target='resolves_yes',horizon_days=None)
        card=boundary.execution_contract(rel,cat,protocol)
        self.assertEqual(card['targets'][0]['output_quantity'],'event_resolution_probability')
        self.assertIsNone(card['targets'][0]['horizon_days']);self.assertFalse(card['narrative_alignment_verified'])
        rel['horizon_days']=7
        with self.assertRaisesRegex(ValueError,'target/horizon'):boundary.execution_contract(rel,cat,protocol)

    def test_wrong_target_lag_unit_and_initialization_still_block_before_review(self):
        for change in ('target','lag','unit','initialization'):
            cat,feed,labels,protocol,registry,p=setup(('a',))
            if change=='target':p['plan']['relation']['predictions'][0]['target']='heldout'
            if change=='lag':p['plan']['relation']['variables'][0]['lag_days']=-1
            if change=='unit':p['plan']['relation']['variables'][0]['unit']='percent'
            if change=='initialization':cat['a']['initialized_at']='2099-01-01T00:00:00Z'
            b=SemanticBackend(p);r=DiscoveryRunner(b,None)
            result=graph.run_workflow(r,{'variable_sources':registry},cat,Forbidden(),Forbidden(),protocol,
                lambda p:self.fail('invalid contract frozen'),semantic_gate=True,semantic_advisory=True)
            self.assertTrue(result['status'].endswith('_failed'));self.assertEqual(b.reviews,0)
            self.assertNotIn('method_data_member',[q['agent'] for q in b.requests])

    def test_price_proxy_and_missing_sources_still_block_before_values(self):
        for missing in (False,True):
            cat,_,_,protocol,registry,p=setup(('a',));p['plan']['relation']['variables'][0].update(quantity='box_office_revenue',unit='USD')
            if missing:p['plan']['mapping']['bindings'][0].update(source_id=None,missing_reason='No historical revenue source.')
            result=graph.run_workflow(DiscoveryRunner(SemanticBackend(p,['mismatch']),None),{'variable_sources':registry},cat,
                Forbidden(),Forbidden(),protocol,lambda p:self.fail('unsupported variable frozen'),semantic_gate=True,semantic_advisory=True)
            self.assertEqual(result['status'],'unavailable_variables' if missing else 'data_selection_failed')
            self.assertIsNone(result['result'])

    def test_feedback_marks_opinions_unverified_and_loop_replays_exactly(self):
        cat,feed,labels,protocol,registry,p=setup(('a',));b=SemanticBackend(p,['mismatch']);r=DiscoveryRunner(b,None)
        def run(r):return run_loop(r,{'phase':'training','variable_sources':registry},cat,feed,labels,protocol,1,
            lambda *args:None,lambda a:None,derive_task_change=True,multi_input=True,semantic_gate=True,semantic_advisory=True)
        attempts=run(r);self.assertTrue(attempts[0]['feedback']['semantic_consistency_is_unverified_advice'])
        request=next(q for q in b.requests if q['agent']=='research_learning_member')
        self.assertIn('UNVERIFIED advisory opinions',request['instruction'])
        replay=RecordedBackend(r.calls);ar=AuditRunner(replay,None,TIMING)
        self.assertEqual(run(ar),attempts);self.assertEqual(ar.calls,r.calls)

    def test_archived_bad_review_replays_without_authority_or_price_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=boundary.audit_reference(Path('configs/team_review_boundary_v1.json'),
                Path('runs/qwen3_8b_team_task_negotiation_v1'),
                'a37b8452d8aa86e8d07e4b8008dc287ce12eb53056139c90ce6be9ca6b031729',Path(tmp)/'audit')
        self.assertEqual(result['review']['status'],'revision_required')
        self.assertIsNone(result['advisory_result']['stop_reason'])
        self.assertEqual(result['deterministic_missing_source_check']['status'],'unavailable_variables')
        self.assertEqual(result['model_reviews_replayed'],2)

if __name__=='__main__':unittest.main()
