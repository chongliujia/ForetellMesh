from copy import deepcopy
import json
from pathlib import Path
import unittest

from foretellmesh import team_context_comparison as comparison
from foretellmesh.team_discovery import DiscoveryRunner,decode_agent_response
from foretellmesh.team_lifecycle_experiment import RecordedBackend,AuditRunner
from foretellmesh.synthetic_sft import canonical_hash

class PairedBackend:
    def __init__(self,bundle):self.bundle=bundle;self.requests=[]
    def generate(self,request):
        self.requests.append(deepcopy(request))
        if request['upstream']['training_history'].get('kind')!='structured_failure_history_v1':
            return self.bundle['source_initial_output']
        return json.dumps({'contract_assessments':[{'market_id':k,'disposition':'retain','reason':'Current task input.'}
            for k in request['input']['original_task']['market_ids']],
            'proposed_task':{k:request['input']['original_task'][k] for k in ('market_ids','objective')},
            'reason':'Keep the current assignment without assuming predictive advantage.'})

class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads(Path('configs/team_context_comparison_v1.json').read_text())
        _,_,cls.cat,cls.bundle=comparison.prepare(cls.config)

    def test_only_history_changes_and_full_arm_is_exact_preserved_request(self):
        full=deepcopy(self.bundle['arms']['full_history']);short=deepcopy(self.bundle['arms']['structured_history'])
        full['upstream']=short['upstream'];self.assertEqual(full,short)
        source=json.loads((Path(self.config['source_run'])/'calls.json').read_text())
        self.assertEqual(self.bundle['arms']['full_history'],source[self.bundle['source_call_index']]['request'])
        self.assertEqual(set(full['input']['original_task']['market_ids']),{'251132','251306'})

    def test_compaction_preserves_failures_and_fingerprints_without_old_prose(self):
        history=self.bundle['arms']['structured_history']['upstream']['training_history']
        self.assertTrue(history['full_history_retained_in_audit']);self.assertEqual(len(history['past_feedback_receipts']),2)
        self.assertTrue(history['rejected_experiments'])
        self.assertTrue(all(r['failed_experiment_fingerprint'] for r in history['rejected_experiments']))
        self.assertTrue(all(r['error_code']=='task_binding_mismatch' for r in history['rejected_experiments']))
        text=json.dumps(history)
        for forbidden in ('raw_output_excerpt','contract_assessments','sports event outcomes','box office'):
            self.assertNotIn(forbidden,text)
        self.assertNotIn('assessment_contract_coverage',text) # current observed failure must not enter its own input

    def test_metrics_check_assessment_ids_not_legitimate_new_proposal_ids(self):
        req=self.bundle['arms']['structured_history'];backend=PairedBackend(self.bundle)
        value=json.loads(backend.generate(req));value['proposed_task']['market_ids'].append('248881')
        metrics=comparison.output_metrics(json.dumps(value),req['input']['original_task'],self.cat)
        self.assertTrue(metrics['schema_and_routing_valid']);self.assertEqual(metrics['extra_assessed_ids'],[])
        value['contract_assessments'].append({'market_id':'248881','disposition':'retain','reason':'Old state.'})
        bad=comparison.output_metrics(json.dumps(value),req['input']['original_task'],self.cat)
        self.assertFalse(bad['schema_and_routing_valid']);self.assertEqual(bad['extra_assessed_ids'],['248881'])
        invalid=comparison.output_metrics('{}',req['input']['original_task'],self.cat)
        self.assertFalse(invalid['schema_and_routing_valid']);self.assertEqual(len(invalid['missing_assessed_ids']),2)

    def test_all_four_trials_repairs_and_exact_replay(self):
        b=PairedBackend(self.bundle);runner=DiscoveryRunner(b,None)
        records=comparison.compare(runner,self.bundle,self.cat);s=comparison.summarize(records)
        self.assertEqual([r['arm'] for r in records],comparison.ORDER)
        self.assertEqual(s['arms']['full_history']['initial_valid'],0)
        self.assertEqual(s['arms']['structured_history']['initial_valid'],2)
        self.assertEqual(len(runner.calls),6);self.assertEqual(s['exposed_training_cases'],1)
        for req in b.requests:
            self.assertEqual(req['input'],self.bundle['arms']['full_history']['input'])
            self.assertEqual(req['instruction'],self.bundle['arms']['full_history']['instruction'])
            self.assertNotIn('source_initial_output',req)
        rb=RecordedBackend(runner.calls);ar=AuditRunner(rb,None,comparison.TIMING)
        self.assertEqual(comparison.compare(ar,self.bundle,self.cat),records);self.assertEqual(ar.calls,runner.calls)
        self.assertEqual(rb.index,len(runner.calls))

    def test_fixed_design_and_source_hash_cannot_silently_change(self):
        for patch in ({'order':['structured_history']},{'max_new_tokens':2048},{'source_report_sha256':'0'*64}):
            with self.assertRaises(ValueError):comparison.prepare({**self.config,**patch})

if __name__=='__main__':unittest.main()
