from copy import deepcopy
import json
from pathlib import Path
import unittest

from foretellmesh import team_task_negotiation as negotiation
from foretellmesh.team_research_handoff import run_loop,summary
from foretellmesh.team_research_loop import prepare
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.synthetic_sft import canonical_hash
from test_team_multi_input import setup
from test_team_semantic_consistency import SemanticBackend,Forbidden
from test_team_lifecycle import TIMING

TASK={'market_ids':['a','b','c'],'objective':'Original test','change':'new_contracts'}

def assessment(ids=('a','b')):
    return {'contract_assessments':[{'market_id':k,'disposition':'retain' if k in ids else 'release',
        'reason':'An explicit test input.' if k in ids else 'No role stated in this test.'} for k in TASK['market_ids']],
        'proposed_task':{'market_ids':list(ids),'objective':'Test whether b price informs a future price.'} if ids else None,
        'reason':'Clarify the test; do not assume predictive advantage.'}

class Backend(SemanticBackend):
    def __init__(self,p,decision='accept_proposal',invalid=False):super().__init__(p);self.decision=decision;self.invalid=invalid
    def generate(self,req):
        if req['agent'] in ('research_task_assessor','task_negotiation_coordinator'):
            self.requests.append(deepcopy(req))
            if self.invalid:return '{}'
            return json.dumps(assessment() if req['agent']=='research_task_assessor' else {'decision':self.decision,'reason':'Explicitly recorded decision.'})
        return super().generate(req)


def reduced_proposal():
    cat,feed,labels,protocol,registry,p=setup(('a',))
    p['plan']['relation']['variables']=p['plan']['relation']['variables'][:2]
    p['plan']['relation']['predictions'][0]['input_ids']=['va','vb']
    p['plan']['mapping']['bindings']=p['plan']['mapping']['bindings'][:2]
    p['plan']['calculation']['expressions'][0]['expression']="(v('va')+v('vb'))/2"
    return cat,feed,labels,protocol,registry,p

class NegotiationTests(unittest.TestCase):
    def test_assessment_requires_exact_accounting_and_admitted_ids(self):
        cat,*_=setup();good=assessment();self.assertEqual(negotiation.validate_assessment(good,TASK,cat),good)
        bads=[]
        b=deepcopy(good);b['contract_assessments'].pop();bads.append(b)
        b=deepcopy(good);b['contract_assessments'][1]['market_id']='a';bads.append(b)
        b=deepcopy(good);b['contract_assessments'][2]['disposition']='retain';bads.append(b)
        b=deepcopy(good);b['proposed_task']['market_ids']=['heldout'];bads.append(b)
        b=deepcopy(good);b['proposed_task']['market_ids']=[1];bads.append(b)
        for b in bads:
            with self.assertRaises(ValueError):negotiation.validate_assessment(b,TASK,cat)
        self.assertIsNone(negotiation.validate_assessment(assessment(()),TASK,cat)['proposed_task'])
        with self.assertRaisesRegex(ValueError,'absent proposal'):
            negotiation.validate_decision({'decision':'accept_proposal','reason':'x'},assessment(()))

    def test_coordinator_owns_effective_task_and_can_add_contract(self):
        cat,_,_,_,_,p=setup();original=deepcopy(TASK);original['market_ids']=['a','b']
        proposed=assessment(('a','b','c'));proposed['contract_assessments']=proposed['contract_assessments'][:2]
        class Add(Backend):
            def generate(self,req):
                if req['agent']=='research_task_assessor':return json.dumps(proposed)
                return super().generate(req)
        result=negotiation.negotiate(DiscoveryRunner(Add(p),None),original,cat,[],{})
        self.assertEqual(result['added_market_ids'],['c']);self.assertEqual(original['market_ids'],['a','b'])
        keep=negotiation.negotiate(DiscoveryRunner(Backend(p,'retain_original'),None),TASK,cat,[],{})
        self.assertEqual(keep['effective_task'],TASK);self.assertEqual(keep['removed_market_ids'],[])

    def run_loop(self,backend,forbidden=False):
        cat,feed,labels,protocol,registry,_=reduced_proposal()
        runner=backend if isinstance(backend,AuditRunner) else DiscoveryRunner(backend,None)
        context={'phase':'training','variable_sources':registry,'semantic_probe_qualified':True}
        freezes=[]
        attempts=run_loop(runner,context,cat,Forbidden() if forbidden else feed,Forbidden() if forbidden else labels,
            protocol,1,lambda n,p:freezes.append(p),lambda r:None,derive_task_change=True,multi_input=True,
            semantic_gate=True,task_negotiation=True)
        return runner,attempts,freezes

    def test_accepted_task_is_used_with_full_rules_and_exact_replay(self):
        *_,p=reduced_proposal();b=Backend(p);r,attempts,frozen=self.run_loop(b)
        a=attempts[0];self.assertEqual(a['active_task']['market_ids'],['a','b'])
        self.assertEqual(a['task_negotiation']['removed_market_ids'],['c']);self.assertIsNotNone(a['workflow']['result'])
        self.assertEqual(len(frozen),1);self.assertEqual(summary(attempts)['task_negotiation']['effective_tasks_changed'],1)
        for req in b.requests:
            if req['agent'] in ('research_task_assessor','task_negotiation_coordinator'):
                self.assertEqual(set(req['input']),{'phase','original_task','catalogue','index'})
            if req['agent']=='relation_researcher':
                self.assertEqual({c['market_id'] for c in req['input']['catalogue']},{'a','b'})
                self.assertIn('task_negotiation',req['input'])
        rb=RecordedBackend(r.calls);ar=AuditRunner(rb,None,TIMING)
        _,replayed,_=self.run_loop(ar);self.assertEqual(replayed,attempts);self.assertEqual(rb.index,len(r.calls))
        self.assertEqual(canonical_hash(ar.calls),canonical_hash(r.calls))

    def test_retain_original_does_not_silently_allow_researcher_subset(self):
        *_,p=reduced_proposal();_,attempts,freeze=self.run_loop(Backend(p,'retain_original'),True)
        self.assertEqual(attempts[0]['workflow']['status'],'relation_failed');self.assertEqual(freeze,[])
        self.assertIn('ALL and ONLY',attempts[0]['workflow']['stop_reason'])

    def test_abstain_and_invalid_assessment_block_all_research_data_and_labels(self):
        for decision,invalid,status in [('abstain',False,'task_negotiation_abstained'),('accept_proposal',True,'task_negotiation_failed')]:
            *_,p=reduced_proposal();b=Backend(p,decision,invalid);_,attempts,freeze=self.run_loop(b,True)
            self.assertEqual(attempts[0]['workflow']['status'],status);self.assertEqual(freeze,[])
            self.assertNotIn('relation_researcher',[r['agent'] for r in b.requests])
            self.assertLessEqual(sum(r['agent']=='research_task_assessor' for r in b.requests),2)

    def test_archived_task_is_executed_and_failure_reaches_assessor(self):
        cat,feed,labels,protocol,registry,p=reduced_proposal();b=Backend(p);runner=DiscoveryRunner(b,None)
        reference={'task':deepcopy(TASK),'feedback':{'stop_reason':'Known binding failure'},'already_exposed_training_experience':True}
        context={'phase':'training','variable_sources':registry,'semantic_probe_qualified':True,'task_negotiation_reference':reference}
        attempts=run_loop(runner,context,cat,feed,labels,protocol,1,lambda *a:None,lambda r:None,
            derive_task_change=True,multi_input=True,semantic_gate=True,task_negotiation=True)
        self.assertEqual(attempts[0]['task_origin'],'archived_training_failure')
        self.assertNotIn('research_coordinator',[r['agent'] for r in b.requests])
        req=next(r for r in b.requests if r['agent']=='research_task_assessor')
        self.assertEqual(req['upstream']['training_history']['prior_training_failure'],reference)
        self.assertEqual(context['task_negotiation_reference'],reference)

    def test_registered_real_failure_is_bound_and_training_only(self):
        config=json.loads(Path('configs/team_task_negotiation_v1.json').read_text())
        _,_,cat,_,_,_,context,_=prepare(config);ref=context['task_negotiation_reference']
        self.assertEqual(ref['task']['market_ids'],['248881','251132','251306','252406'])
        self.assertIn('task handoff mismatch:',ref['feedback']['stop_reason'])
        bad=deepcopy(config);bad['negotiation_reference_report_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'reference changed'):negotiation.load_reference(bad,cat)

if __name__=='__main__':unittest.main()
