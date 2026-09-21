from copy import deepcopy
import json
import unittest

from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.team_research_handoff import (validate_task, validate_reflection, TaskRunner, run_loop, summary)
from foretellmesh.team_research_loop import relation_fingerprint, loop_for
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.schema import ValidationError
from foretellmesh import team_variable_contracts as typed
from test_team_research_loop import setup, LoopBackend
from test_team_lifecycle import TIMING


def task(ids=('a','b','c'), change='new_contracts'):
    return {'market_ids': list(ids), 'objective': 'Test cross-contract price dynamics at a revised forecast horizon.', 'change': change}


class Backend(LoopBackend):
    def generate(self, request):
        role = request['agent']
        if role == 'research_coordinator':
            self.requests.append(deepcopy(request)); return json.dumps(task())
        if role == 'research_learning_member':
            self.requests.append(deepcopy(request))
            return json.dumps({'feedback_sha256': request['upstream']['feedback']['feedback_sha256'],
                'next_focus': 'Revise the tested forecast horizon.', 'data_requests': [],
                'next_task': task(change='revise_current')})
        if role == 'relation_researcher' and 'repair' in request and 'duplicate_relation:' in request['repair']['error']:
            self.relation['horizon_days'] = 2
        return super().generate(request)


class HandoffTests(unittest.TestCase):
    def test_task_rejects_unknown_numeric_or_misdeclared_changes(self):
        cat, *_ = setup()
        for ids in [('outside_catalogue',), (123,), ('a','a')]:
            with self.assertRaises(ValidationError): validate_task(task(ids), cat)
        with self.assertRaises(ValidationError): validate_task(task(change='revise_current'), cat)
        with self.assertRaises(ValidationError): validate_task(task(), cat, task())
        with self.assertRaises(ValidationError): validate_task(task(('a',), 'revise_current'), cat, task())
        self.assertEqual(validate_task(task(('b',)), cat, task(('a',))), task(('b',)))

    def test_task_routing_cannot_be_ignored(self):
        cat, _, _, _, _, backend = setup(); runner = DiscoveryRunner(backend, None)
        bound = TaskRunner(runner, task(('a','b')), set())
        with self.assertRaisesRegex(ValidationError, 'ALL and ONLY'):
            bound.structured('relation_researcher', typed.RELATION, {}, {}, lambda v: typed.validate_relation(v, cat))
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0]['request']['input']['active_task'], task(('a','b')))

    def test_next_task_is_consumed_without_new_coordinator_choice(self):
        cat, feed, labels, protocol, context, _ = setup(); backend = Backend(); runner = DiscoveryRunner(backend, None)
        freezes = []
        rows = run_loop(runner, context, cat, feed, labels, protocol, 2, lambda n,p: freezes.append(n), lambda r: None)
        self.assertEqual([r['task_origin'] for r in rows], ['coordinator', 'preceding_reflection'])
        self.assertEqual(rows[1]['active_task'], rows[0]['reflection']['next_task'])
        self.assertEqual(sum(r['agent']=='research_coordinator' for r in backend.requests), 1)
        self.assertEqual(freezes, [1,2]); self.assertEqual(len(rows[1]['duplicate_repairs']), 1)
        self.assertEqual(rows[1]['workflow']['relation']['relation']['horizon_days'], 2)
        self.assertEqual(summary(rows)['tasks_received_from_reflection'], 1)
        self.assertTrue(rows[1]['feedback']['failure_details'])
        self.assertIn('raw_output_excerpt', rows[1]['feedback']['failure_details'][0])

    def test_repeated_relation_gets_one_repair_and_is_not_scored(self):
        cat, feed, labels, protocol, context, _ = setup(); backend = Backend()
        # Use the parent's relation output so it ignores duplicate repairs.
        backend.generate = lambda req: (LoopBackend.generate(backend, req) if req['agent']=='relation_researcher'
                                       else Backend.generate(backend, req))
        runner = DiscoveryRunner(backend, None); freezes = []
        rows = run_loop(runner, context, cat, feed, labels, protocol, 2, lambda n,p: freezes.append(n), lambda r: None)
        self.assertEqual(freezes,[1]); self.assertEqual(rows[1]['workflow']['status'], 'duplicate_relation')
        self.assertEqual(len(rows[1]['duplicate_repairs']),2)
        self.assertEqual(summary(rows)['scored_attempts'],1)

    def test_unknown_next_task_is_not_admitted_by_valid_reflection_text(self):
        cat, *_ = setup(); feedback = {'feedback_sha256':'a'*64}
        value = {'feedback_sha256':'a'*64,'next_focus':'Study another film.','data_requests':[],
                 'next_task':task(('Mario_not_in_catalogue',))}
        with self.assertRaises(ValidationError): validate_reflection(value, feedback, cat, task())

    def test_frozen_calls_replay_handoff_and_duplicate_repair_exactly(self):
        cat, feed, labels, protocol, context, _ = setup(); runner = DiscoveryRunner(Backend(),None); freezes={}
        def freeze(n,p): freezes[n]=(deepcopy(p),canonical_hash(runner.calls))
        first=run_loop(runner,context,cat,feed,labels,protocol,2,freeze,lambda r:None)
        backend=RecordedBackend(runner.calls); audited=AuditRunner(backend,None,TIMING)
        def check(n,p): self.assertEqual((p,canonical_hash(audited.calls)),freezes[n])
        second=run_loop(audited,context,cat,feed,labels,protocol,2,check,lambda r:None)
        self.assertEqual(first,second);self.assertEqual(backend.index,len(runner.calls))

    def test_change_metadata_is_derived_without_rewriting_model_selection(self):
        from foretellmesh.team_research_handoff import validate_derived_task
        cat, *_ = setup(); previous=task(); value=task()
        original=deepcopy(value); checked=validate_derived_task(value,cat,previous)
        self.assertEqual(value,original);self.assertEqual(checked['market_ids'],value['market_ids'])
        self.assertEqual(checked['objective'],value['objective']);self.assertEqual(checked['change'],'revise_current')
        self.assertEqual(validate_derived_task({'market_ids':['a'],'objective':'Reconsider this contract.'},cat,previous)['change'],'new_contracts')
        for ids in [[123],['unknown'],['a','a']]:
            with self.assertRaises(ValidationError):validate_derived_task({'market_ids':ids,'objective':'Test'},cat,previous)

    def test_derived_tasks_accept_valid_reflection_despite_wrong_legacy_change_tag(self):
        cat,feed,labels,protocol,context,_=setup()
        class LegacyTag(Backend):
            def generate(self,request):
                raw=super().generate(request)
                if request['agent']=='research_learning_member':
                    value=json.loads(raw);value['next_task']['change']='new_contracts';return json.dumps(value)
                return raw
        runner=DiscoveryRunner(LegacyTag(),None)
        rows=run_loop(runner,context,cat,feed,labels,protocol,2,lambda *a:None,lambda r:None,derive_task_change=True)
        self.assertEqual(rows[1]['task_origin'],'preceding_reflection')
        self.assertEqual(rows[0]['reflection']['next_task']['change'],'revise_current')
        raw=next(c['output'] for c in runner.calls if c['request']['agent']=='research_learning_member')
        self.assertEqual(json.loads(raw)['next_task']['change'],'new_contracts')
        self.assertEqual(rows[1]['active_task'],rows[0]['reflection']['next_task'])

    def test_derived_handoff_replays_normalization_and_calls(self):
        cat,feed,labels,protocol,context,_=setup();runner=DiscoveryRunner(Backend(),None)
        first=run_loop(runner,context,cat,feed,labels,protocol,2,lambda *a:None,lambda r:None,derive_task_change=True)
        backend=RecordedBackend(runner.calls);audited=AuditRunner(backend,None,TIMING)
        second=run_loop(audited,context,cat,feed,labels,protocol,2,lambda *a:None,lambda r:None,derive_task_change=True)
        self.assertEqual(first,second);self.assertEqual(backend.index,len(runner.calls))

    def test_legacy_entrypoint_is_unchanged_and_unknown_protocol_fails(self):
        from foretellmesh.team_research_loop import run_loop as legacy
        self.assertIs(loop_for({})[0],legacy)
        self.assertIs(loop_for({'handoff_protocol':'executable_task_v1'})[0],run_loop)
        with self.assertRaises(ValidationError): loop_for({'handoff_protocol':'unknown'})

if __name__ == '__main__': unittest.main()
