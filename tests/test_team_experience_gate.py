from copy import deepcopy
import json
import unittest

from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_experience_gate import facts_for, validate_facts, validate_proposal, review_case, validate_admitted
from foretellmesh.team_lifecycle import LifecycleRunner
from foretellmesh.team_trade_experience import learn_available
from test_team_lifecycle import exercise, TIMING, LearningBackend


class GateTests(unittest.TestCase):
    def setUp(self):
        self.job,self.feed,self.backend,self.runner,self.window=exercise()
        self.cases=self.window['trade_experiences']

    def test_exact_fact_gate_blocks_profit_side_exit_and_probability_target_confusion(self):
        for row in self.window['trade_reflections']:
            for key,bad in [('result_class','flat'),('side','no'),('exit_type','settle'),('forecast_target','price_increase')]:
                proposal=deepcopy(row['output']);proposal['fact_check'][key]=bad
                with self.assertRaisesRegex(ValidationError,'authoritative'):
                    validate_proposal(proposal,row['facts'])

    def test_resealed_but_inconsistent_ledger_facts_are_rejected(self):
        for key,bad in [('result_class','profit'),('total_fees','999'),('holding_seconds',-1),('event_outcome',1)]:
            facts=facts_for(next(c for c in self.cases if c['result']['result_class']=='loss'))
            facts['values'][key]=bad;facts['facts_sha256']=canonical_hash({k:v for k,v in facts.items() if k!='facts_sha256'})
            with self.assertRaises(ValidationError):validate_facts(facts)

    def test_unknown_citations_and_noncomparative_plans_do_not_pass(self):
        row=self.window['trade_reflections'][0]
        for mutate in [lambda p:p.update(evidence_refs=['invented']),
                       lambda p:p['lesson']['test_plan'].update(validation_split='same_trade'),
                       lambda p:p['lesson']['test_plan'].update(metric='always_win')]:
            proposal=deepcopy(row['output']);mutate(proposal)
            with self.assertRaises(ValidationError):validate_proposal(proposal,row['facts'])

    def test_critic_rejects_well_formatted_unsupported_lesson_and_memory_stays_empty(self):
        class Rejecting(LearningBackend):
            def generate(self,request):
                if request['agent']=='trade_reflection_critic':
                    return json.dumps({'experience_id':request['input']['facts']['experience_id'],
                        'facts_consistent':True,'evidence_supported':False,'testable_change':False,
                        'issues':['The proposed change has no measurable intervention supported by this evidence.']})
                return super().generate(request)
        runner=LifecycleRunner(Rejecting(),self.runner.catalogue,TIMING)
        learn_available(runner,self.cases,'2024-01-05T00:00:00Z',enabled=True)
        self.assertEqual(len(runner.trade_reflections),2)
        self.assertTrue(all(r['output'] is not None and not r['admitted_to_exploratory_memory'] for r in runner.trade_reflections.values()))
        self.assertNotIn('online_trade_lessons',runner.with_memory(self.job['context'],{}))
        with self.assertRaisesRegex(ValidationError,'not admitted'):
            validate_admitted(list(runner.trade_reflections.values()),'2024-01-05T00:00:00Z')

    def test_critic_failure_is_not_admission_and_no_lesson_stays_archived(self):
        class BadCritic(LearningBackend):
            def generate(self,request):
                return '{broken' if request['agent']=='trade_reflection_critic' else super().generate(request)
        runner=LifecycleRunner(BadCritic(),self.runner.catalogue,TIMING)
        row=review_case(runner,self.cases[0],'2024-01-05T00:00:00Z')
        self.assertIsNotNone(row['output']);self.assertIsNone(row['critique'])
        self.assertFalse(row['admitted_to_exploratory_memory']);self.assertIsNotNone(row['error'])
        class NoLesson(LearningBackend):
            def generate(self,request):
                result=json.loads(super().generate(request))
                if request['agent']=='trade_reflection':result['lesson']=None
                return json.dumps(result)
        row=review_case(LifecycleRunner(NoLesson(),self.runner.catalogue,TIMING),self.cases[0],'2024-01-05T00:00:00Z')
        self.assertFalse(row['admitted_to_exploratory_memory'])

    def test_forged_admission_flag_and_unchecked_legacy_record_rejected(self):
        row=deepcopy(self.window['trade_reflections'][0]);row['critique']['facts_consistent']=False
        row['critique']['issues']=['Facts contradict the ledger.']
        row['reflection_sha256']=canonical_hash({k:v for k,v in row.items() if k!='reflection_sha256'})
        with self.assertRaisesRegex(ValidationError,'not admitted'):validate_admitted([row],'2024-01-05T00:00:00Z')
        with self.assertRaisesRegex(ValidationError,'legacy'):
            validate_admitted([{'output':{'assessment':'Winning trade','lesson':None}}],'2024-01-05T00:00:00Z')

    def test_offline_freeze_preserves_actual_build_time_and_isolates_target_events(self):
        from foretellmesh.team_experience_gate import freeze_gated_memory
        from foretellmesh.team_method_memory import validate_memory
        rows=self.window['trade_reflections'];groups={c['event_group_id'] for c in self.cases}
        memory=freeze_gated_memory(rows,self.cases,groups,'2024-02-01T00:00:00Z')
        self.assertEqual(len(memory['lessons']),2)
        self.assertEqual(memory['built_at'],'2024-02-01T00:00:00Z')
        context=deepcopy(self.job['context']);context['observation_time']='2024-01-06T00:00:00Z'
        with self.assertRaisesRegex(ValidationError,'overlap'):validate_memory(memory,context)
        for m in context['markets']:m['event_group_id']='unseen-'+m['event_group_id']
        validate_memory(memory,context)
        context['observation_time']='2024-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValidationError,'later'):validate_memory(memory,context)
        with self.assertRaises(ValidationError):freeze_gated_memory(rows,self.cases,groups,'2024-01-01T00:00:00Z')
        self.assertFalse(memory['effectiveness_verified']);self.assertFalse(memory['fine_tuning_admitted'])

    def test_legacy_format_only_records_do_not_enter_current_runner_memory(self):
        self.runner.trade_reflections={'old':{'output':{'assessment':'Winning trade','lesson':{'action':'Buy more'}},'error':None}}
        self.assertNotIn('online_trade_lessons',self.runner.with_memory(self.job['context'],{}))

    def test_free_form_assessment_never_enters_decision_memory(self):
        from foretellmesh.team_trade_experience import validate_online_lessons
        context=deepcopy(self.job['context']);context['observation_time']='2024-01-05T00:00:00Z'
        views=self.runner.with_memory(context,{})['online_trade_lessons']
        self.assertTrue(all('assessment' not in json.dumps(v) for v in views))
        self.assertTrue(all(v['facts']['values']['result_class'] in ('profit','loss') for v in views))
        validate_online_lessons(views,context['observation_time'])
        views[0]['method']['proposed_change']='tampered'
        with self.assertRaisesRegex(ValidationError,'changed'):validate_online_lessons(views,context['observation_time'])


if __name__=='__main__':unittest.main()
