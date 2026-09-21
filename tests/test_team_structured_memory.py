from copy import deepcopy
import json
import unittest

from foretellmesh import team_structured_memory as memory
from foretellmesh.team_research_handoff import run_loop, summary
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.synthetic_sft import canonical_hash
from test_team_data_catalogue import CatalogueBackend, reduced_proposal
from test_team_lifecycle import TIMING


class StructuredMemoryTests(unittest.TestCase):
    def run_workflow(self, runner, count=2):
        cat, feed, labels, protocol, registry, _ = reduced_proposal()
        frozen = []
        records = run_loop(runner, {'phase':'training', 'variable_sources':registry},
            cat, feed, labels, protocol, count, lambda n,p:frozen.append((n,p)), lambda a:None,
            derive_task_change=True, multi_input=True, semantic_gate=True, semantic_advisory=True,
            task_negotiation=True, data_discovery=True, structured_history=True)
        return records, frozen

    def test_two_attempts_keep_audit_remove_past_prose_and_replay_exactly(self):
        *_, p = reduced_proposal()
        backend = CatalogueBackend(p); runner = DiscoveryRunner(backend, None)
        records, frozen = self.run_workflow(runner)
        self.assertIsNotNone(records[0]['workflow']['result'])
        self.assertEqual(records[1]['workflow']['status'], 'duplicate_relation')
        self.assertEqual(len(frozen), 1)
        self.assertEqual(summary(records)['structured_history']['attempts_with_audited_projection'], 2)
        for record in records:
            view = record['history_view']
            self.assertEqual(view['full_history_sha256'], canonical_hash(record['history_archive']))
            self.assertEqual(view['past_calls_sha256'], canonical_hash(runner.calls[:record['call_start']]))
            self.assertEqual(len(view['past_tool_constraints']), record['attempt'] - 1)
            self.assertNotIn('hypothesis', json.dumps(view))
            self.assertNotIn('raw_output_excerpt', json.dumps(view))
            learning = record['learning_feedback_view']
            self.assertEqual(learning['feedback_sha256'], record['feedback']['feedback_sha256'])
            self.assertNotIn('assessment', learning['task_negotiation'])
            self.assertNotIn('original_task', learning['task_negotiation'])
        self.assertTrue(records[1]['history_archive']['recent_feedback'])
        self.assertEqual(len(records[1]['history_view']['attempted_experiments']), 1)
        for req in backend.requests:
            if req['agent'] == 'relation_researcher':
                self.assertEqual(req['input']['task_negotiation']['effective_task'], req['input']['active_task'])
                self.assertNotIn('assessment', req['input']['task_negotiation'])
        rb = RecordedBackend(runner.calls); ar = AuditRunner(rb, None, TIMING)
        self.assertEqual(self.run_workflow(ar), (records, frozen))
        self.assertEqual(ar.calls, runner.calls)
        self.assertEqual(rb.index, len(runner.calls))

    def test_failure_and_abstention_still_reach_learning_with_full_audit(self):
        for attr, value, expected in [('bad_query',True,'data_catalogue_failed'),
                ('invalid',True,'task_negotiation_failed'), ('decision','abstain','task_negotiation_abstained')]:
            *_, p = reduced_proposal(); backend = CatalogueBackend(p)
            setattr(backend, attr, value)
            records, frozen = self.run_workflow(DiscoveryRunner(backend, None), 1)
            record = records[0]
            self.assertEqual(record['workflow']['status'], expected)
            self.assertIsNotNone(record['reflection']); self.assertEqual(frozen, [])
            self.assertNotIn('relation_researcher', [r['agent'] for r in backend.requests])
            for failure in record['learning_feedback_view']['failure_details']:
                self.assertNotIn('raw_output_excerpt', failure)
                self.assertTrue(failure['error_code'])
            if expected != 'task_negotiation_abstained':
                self.assertTrue(record['feedback']['failure_details'][0]['raw_output_excerpt'])

    def test_missing_measurements_and_tool_results_survive_without_narrative(self):
        feedback = {'input_check':{'status':'unavailable_variables','jointly_available':0,
            'unavailable_variables':[{'variable':{'quantity':'ticket_revenue','unit':'USD',
                'definition':'Old narrative', 'market_id':'old'}}], 'failures':{'stale':3}},
            'test':{'status':'insufficient_data','valid_observations':2,'coverage':0.1,
                'eligible_for_independent_validation':False, 'observations':['private']}}
        original = deepcopy(feedback); out = memory.tool_constraints(feedback)
        self.assertEqual(out['missing_measurements'], [{'quantity':'ticket_revenue','unit':'USD','registered_source_missing':True}])
        self.assertEqual(out['input_failure_counts'], {'stale':3})
        self.assertEqual(out['test']['valid_observations'], 2)
        self.assertNotIn('private', json.dumps(out)); self.assertNotIn('Old narrative', json.dumps(out))
        self.assertEqual(feedback, original)


if __name__ == '__main__':
    unittest.main()
