from copy import deepcopy
import json
from pathlib import Path
import unittest

from foretellmesh import team_semantic_consistency as semantic
from foretellmesh import team_multi_input as graph
from foretellmesh.team_research_handoff import TaskRunner,run_loop
from foretellmesh.team_research_loop import prepare
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.synthetic_sft import canonical_hash
from test_team_multi_input import setup,Backend
from test_team_lifecycle import TIMING


def verdict(context,kind='aligned'):
    return {'target_check':{'verdict':kind,'reason':'Explicit target link.'},
        'variable_checks':[{'variable_id':v['variable_id'],'verdict':'aligned','reason':'Declared measured input.'} for v in context['variables']]}

class SemanticBackend(Backend):
    def __init__(self,p,verdicts=()):super().__init__(p);self.verdicts=list(verdicts);self.reviews=0
    def generate(self,request):
        if request['agent']=='semantic_consistency_reviewer':
            self.requests.append(deepcopy(request));self.reviews+=1
            kind=self.verdicts.pop(0) if self.verdicts else 'aligned'
            if kind=='invalid':return '{}'
            return json.dumps(verdict(request['input'],kind))
        if request['agent']=='relation_researcher' and 'semantic_revision_of' in request['input']:
            self.proposal['plan']['relation']['hypothesis']='Explicitly test the same declared target using the same inputs.'
        return super().generate(request)

class Forbidden:
    def latest(self,*a):raise AssertionError('price access before gate')
    def __getitem__(self,k):raise AssertionError('label access before gate')

class SemanticTests(unittest.TestCase):
    def test_reviews_cover_exact_inputs_and_bounded_verdicts(self):
        pred={'input_ids':['x','y']};good=verdict({'variables':[{'variable_id':k} for k in pred['input_ids']]})
        self.assertEqual(semantic.validate_review(good,pred),good)
        bads=[]
        b=deepcopy(good);b['variable_checks'].pop();bads.append(b)
        b=deepcopy(good);b['variable_checks'][1]['variable_id']='x';bads.append(b)
        b=deepcopy(good);b['target_check']['verdict']='profitable';bads.append(b)
        b=deepcopy(good);b['target_check']['extra']=True;bads.append(b)
        for b in bads:
            with self.assertRaises(ValueError):semantic.validate_review(b,pred)

    def workflow(self,kinds,qualified=True):
        cat,_,_,protocol,registry,p=setup(('a',));backend=SemanticBackend(p,kinds);runner=DiscoveryRunner(backend,None)
        bound=TaskRunner(runner,{'market_ids':list(cat),'objective':'Test','change':'new_contracts'},set(),multi_input=True)
        result=graph.run_workflow(bound,{'variable_sources':registry},cat,Forbidden(),Forbidden(),protocol,
            lambda p:self.fail('unexpected freeze'),semantic_gate=True,probe_qualified=qualified)
        return result,backend

    def test_rejected_uncertain_and_invalid_reviews_block_data_and_labels(self):
        for kinds,status in [(['mismatch','mismatch'],'semantic_rejected'),(['uncertain','uncertain'],'semantic_rejected'),
                             (['invalid','invalid'],'semantic_review_failed')]:
            result,b=self.workflow(kinds)
            self.assertEqual(result['status'],status);self.assertIsNone(result['data_check'])
            self.assertNotIn('method_data_member',[r['agent'] for r in b.requests])
            self.assertLessEqual(sum(r['agent']=='relation_researcher' for r in b.requests),2)

    def test_unqualified_probe_blocks_even_when_live_review_accepts(self):
        result,_=self.workflow([],False);self.assertEqual(result['status'],'semantic_gate_unqualified')

    def test_same_fingerprint_revision_once_and_full_exact_replay(self):
        cat,feed,labels,protocol,registry,p=setup(('a',));backend=SemanticBackend(p,['mismatch','aligned'])
        runner=DiscoveryRunner(backend,None);freezes=[]
        context={'phase':'training','variable_sources':registry,'semantic_probe_qualified':True}
        def run(r):return run_loop(r,context,cat,feed,labels,protocol,1,lambda n,p:freezes.append((n,p)),lambda a:None,
            derive_task_change=True,multi_input=True,semantic_gate=True)
        original=run(runner);workflow=original[0]['workflow']
        self.assertEqual(len(workflow['semantic_reviews']),2);self.assertIsNotNone(workflow['result'])
        self.assertEqual(len(freezes),1)
        for req in backend.requests:
            self.assertNotIn('semantic_probe_qualified',req['input'])
        recorded=RecordedBackend(runner.calls);audit=AuditRunner(recorded,None,TIMING)
        self.assertEqual(run(audit),original);self.assertEqual(recorded.index,len(runner.calls))
        self.assertEqual(canonical_hash(audit.calls),canonical_hash(runner.calls))

    def test_revision_does_not_admit_prior_experiment_or_second_revision(self):
        cat,_,_,_,_,p=setup(('a',));r=p['plan']['relation'];seen=set()
        runner=TaskRunner(DiscoveryRunner(Backend(p),None),{'market_ids':list(cat)},seen,multi_input=True)
        validate=lambda v:graph.validate_relation(v,cat)
        runner.structured('relation_researcher','',{}, {},validate)
        with self.assertRaisesRegex(ValueError,'duplicate_relation'):
            runner.structured('relation_researcher','',{}, {},validate)
        ctx={'semantic_revision_of':canonical_hash(r)}
        runner.structured('relation_researcher','',ctx,{},validate)
        with self.assertRaisesRegex(ValueError,'once only'):runner.structured('relation_researcher','',ctx,{},validate)
        another=TaskRunner(runner.runner,{'market_ids':list(cat)},seen,multi_input=True)
        with self.assertRaisesRegex(ValueError,'current accepted'):another.structured('relation_researcher','',ctx,{},validate)

    def test_fixed_archived_probes_and_expected_labels_hidden(self):
        config=json.loads(Path('configs/team_semantic_consistency_v1.json').read_text())
        _,_,cat,_,_,_,context,_=prepare(config);cases=context['semantic_probe_cases']
        self.assertEqual(len(cases),3);self.assertEqual(cases[0]['relation']['predictions'][0]['target'],'252406')
        _,_,_,_,_,p=setup();b=SemanticBackend(p,['mismatch','aligned','aligned']);runner=DiscoveryRunner(b,None)
        report=semantic.run_probes(runner,cases,cat);self.assertTrue(report['qualified_for_this_bounded_run'])
        for req in b.requests:
            self.assertNotIn('expected',json.dumps(req));self.assertEqual(req['upstream'],{})
        recorded=RecordedBackend(runner.calls);audit=AuditRunner(recorded,None,TIMING)
        self.assertEqual(semantic.run_probes(audit,cases,cat),report)
        for kinds in [('aligned','aligned','aligned'),('mismatch','aligned','mismatch'),('invalid','invalid','aligned','aligned')]:
            r=semantic.run_probes(DiscoveryRunner(SemanticBackend(p,kinds),None),cases,cat)
            self.assertFalse(r['qualified_for_this_bounded_run'])
        bad=deepcopy(config);bad['semantic_probe_report_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source changed'):semantic.build_probes(bad,cat)

if __name__=='__main__':unittest.main()
