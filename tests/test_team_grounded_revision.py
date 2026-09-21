from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import unittest

from foretellmesh import team_grounded_revision as grounded
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.team_research_handoff import run_loop
from foretellmesh.team_research_loop import prepare
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.schema import timestamp
from foretellmesh.mean_reversion import HistoricalBars
from test_team_data_catalogue import CatalogueBackend, reduced_proposal
from test_team_lifecycle import TIMING


class GroundedBackend(CatalogueBackend):
    def generate(self, request):
        if request['agent'] == 'research_data_catalogue' and not self.bad_query:
            value = json.loads(super().generate(request)); value['market_ids'] = ['c']
            return json.dumps(value)
        if request['agent'] == 'research_learning_member':
            self.requests.append(deepcopy(request)); feedback = request['upstream']['feedback']
            facts = feedback['tool_facts']
            row = next((r for r in facts if r['requires_response'] and r['kind'] != 'workflow_status'), facts[0])
            return json.dumps({'feedback_sha256':feedback['feedback_sha256'], 'next_focus':'Inspect actual input coverage.',
                'next_task':{'market_ids':['a','b','c'], 'objective':'Test a revised measurement hypothesis.'},
                'data_requests':[], 'diagnosis':{'fact_ids':[row['fact_id']],
                    'interpretation':'The tool screen did not qualify this method.',
                    'proposed_change':'Reconsider the inputs before a new test.'}})
        return super().generate(request)


class GroundedTests(unittest.TestCase):
    def run_workflow(self, runner, count=2):
        cat, feed, labels, protocol, registry, _ = reduced_proposal(); freezes = []
        records = run_loop(runner, {'phase':'training','variable_sources':registry}, cat, feed, labels,
            protocol, count, lambda n,p:freezes.append((n,p)), lambda a:None, derive_task_change=True,
            multi_input=True, semantic_gate=True, semantic_advisory=True, task_negotiation=True,
            data_discovery=True, structured_history=True, grounded_revision=True)
        return records, freezes

    def test_current_task_refresh_replaces_old_query_and_exact_replay(self):
        *_, p = reduced_proposal(); backend = GroundedBackend(p); runner = DiscoveryRunner(backend,None)
        records, freezes = self.run_workflow(runner)
        self.assertEqual(records[0]['data_catalogue']['query']['market_ids'], ['c'])
        self.assertEqual(records[0]['current_task_catalogue']['result']['query']['market_ids'], ['a','b'])
        self.assertIsNotNone(records[0]['workflow']['result']); self.assertEqual(len(freezes),1)
        self.assertEqual(records[1]['workflow']['status'],'duplicate_relation')
        for req in backend.requests:
            if req['agent'] == 'relation_researcher':
                self.assertEqual(req['input']['data_catalogue']['query']['market_ids'], req['input']['active_task']['market_ids'])
        for a in records:
            self.assertIsNotNone(a['reflection']); view = a['learning_feedback_view']
            self.assertFalse(view['reviewer_advice']['verified']); self.assertFalse(view['candidate_hypothesis']['verified'])
            self.assertNotIn('raw_output_excerpt',json.dumps(view))
            self.assertEqual(view['feedback_sha256'],a['feedback']['feedback_sha256'])
            prior = a['history_view']['past_tool_facts']
            self.assertEqual(len(prior),a['attempt']-1)
            self.assertNotIn(a['feedback']['feedback_sha256'],json.dumps(prior))
        rb = RecordedBackend(runner.calls); ar = AuditRunner(rb,None,TIMING)
        self.assertEqual(self.run_workflow(ar),(records,freezes)); self.assertEqual(ar.calls,runner.calls)

    def test_failed_query_assessment_and_abstention_keep_learning(self):
        for attr,value,status in [('bad_query',True,'data_catalogue_failed'),
            ('invalid',True,'task_negotiation_failed'),('decision','abstain','task_negotiation_abstained')]:
            *_, p = reduced_proposal(); b = GroundedBackend(p); setattr(b,attr,value)
            records, freezes = self.run_workflow(DiscoveryRunner(b,None),1)
            a = records[0]; self.assertEqual(a['workflow']['status'],status)
            self.assertIsNotNone(a['reflection']); self.assertIsNone(a['current_task_catalogue'])
            self.assertEqual(freezes,[])

    def test_unknown_advice_only_and_status_only_citations_rejected(self):
        cat,*_ = reduced_proposal()
        f = {'status':'no_joint_observations','feedback_sha256':'f'*64,
            'input_check':{'status':'no_joint_observations','jointly_available':0,'failures':{'x:stale_quote':3}}}
        facts = grounded.facts(f)
        good = {'feedback_sha256':f['feedback_sha256'],'next_focus':'Inspect timestamps.', 'data_requests':[],
            'next_task':{'market_ids':['a'],'objective':'Check historical availability.'},
            'diagnosis':{'fact_ids':[facts[1]['fact_id']],'interpretation':'Zero joint inputs.',
                'proposed_change':'Check the input timestamps.'}}
        self.assertEqual(grounded.validate_reflection(good,f,cat,None)['diagnosis'],good['diagnosis'])
        for ids in (['critic_opinion'],['fake'],[],[facts[0]['fact_id']],[facts[1]['fact_id']]*2):
            bad = deepcopy(good); bad['diagnosis']['fact_ids'] = ids
            with self.assertRaises(ValueError):grounded.validate_reflection(bad,f,cat,None)

    def test_refresh_has_no_future_preinit_prices_or_label_values(self):
        cat,_,_,protocol,registry,_ = reduced_proposal()
        task = {'market_ids':['a'],'objective':'Inspect.'}; start = timestamp(cat['a']['initialized_at'],'init')
        end = timestamp(protocol['as_of'],'end')
        feed = HistoricalBars({'a':[(start-timedelta(seconds=1),.1,3),(start,.2,4),(end+timedelta(seconds=1),.9,8)]})
        result = grounded.current_catalogue(task,cat,registry,feed,protocol)
        other = grounded.current_catalogue(task,cat,registry,HistoricalBars({'a':[(start,.8,4)]}),protocol)
        self.assertEqual(result,other)
        self.assertEqual(result['result']['markets'][0]['historical_trade_rows'],4)
        self.assertFalse(result['result']['outcome_labels_exposed'])
        self.assertEqual(result['task_sha256'],canonical_hash(task))

    def test_reference_is_the_preserved_unscored_failure_and_hash_locked(self):
        config = json.loads(Path('configs/team_grounded_revision_v1.json').read_text())
        *_, context, _ = prepare(config)
        reference = context['task_negotiation_reference']
        self.assertEqual(reference['feedback']['status'],'no_joint_observations')
        self.assertEqual(reference['task']['market_ids'],['253619','254223','254462','254696'])
        with self.assertRaisesRegex(ValueError,'reference changed'):
            prepare({**config,'grounded_reference_report_sha256':'0'*64})

    def test_reference_experiment_cannot_be_repeated_as_a_new_test(self):
        cat,feed,labels,protocol,registry,p = reduced_proposal()
        prior = {'status':'no_joint_observations','feedback_sha256':'a'*64,
            'relation':deepcopy(p['plan']['relation']), 'input_check':{'jointly_available':0,'status':'no_joint_observations'}}
        reference = {'task':{'market_ids':['a','b','c'],'objective':'Old task.','change':'new_contracts'},'feedback':prior}
        runner = DiscoveryRunner(GroundedBackend(p),None)
        records = run_loop(runner,{'phase':'training','variable_sources':registry,'task_negotiation_reference':reference},
            cat,feed,labels,protocol,1,lambda *a:self.fail('Repeated reference must not score'),lambda a:None,
            derive_task_change=True,multi_input=True,semantic_gate=True,semantic_advisory=True,
            task_negotiation=True,data_discovery=True,structured_history=True,grounded_revision=True)
        self.assertEqual(records[0]['workflow']['status'],'duplicate_relation')
        self.assertTrue(records[0]['history_view']['reference_experiment_fingerprint'])


if __name__ == '__main__':unittest.main()
