from copy import deepcopy
import json
import unittest
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_discovery import HistoricalCatalogue, explore_episode
from foretellmesh.team_discovery_experiment import current_account
from foretellmesh.team_memory_loop import MemoryRunner, rules_snapshot, validate_study
from foretellmesh.team_method_memory import learning_evidence, freeze_memory, validate_memory, validate_lessons
from foretellmesh.team_outcome_learning import binary_prompt
from foretellmesh.team_memory_experiment import spaced
from test_team_discovery import job, Backend, OBJECTIVE
from test_team_learning import AT, feed, policy


class ToolBackend(Backend):
    def generate(self,request):
        if request['agent']=='investigation':
            self.requests.append(deepcopy(request))
            return json.dumps({'investigations':[{'tool':'rules_snapshot','market_ids':['a','b'],'lookback_days':0,
                'expected_direction':'unspecified','claim':'These two contracts may resolve on the same condition.',
                'falsifier':'The resolution rules specify different events or dates.'}],'no_finding_reason':None})
        return super().generate(request)


def episode():
    j=job();cat=HistoricalCatalogue([j,job('b')],feed=feed());b=ToolBackend()
    account=current_account([],[],feed(),{'execution_policy':policy(),'objective':OBJECTIVE},AT)
    e=explore_episode(j['context'],j['labels'],feed(),policy(),MemoryRunner(b,cat),account,recorded_latency=1)
    return j,cat,e,b


def memory():
    j,cat,e,_=episode();evidence=learning_evidence([j],[e]);edited={'lessons':[{
        'fact_refs':['e0.feedback.forecast_brier'],'when_applicable':'Price history is sparse.',
        'action':'Read exact resolution rules before testing a dependency.',
        'failure_condition':'Rules concern different events and cannot establish a dependency.'}],'no_lesson_reason':None}
    return freeze_memory([j],[e],evidence,edited,'2030-01-01T00:00:00Z',cat)


def target_memory():
    m=memory();m['source_public_through']='2023-01-01T00:00:00Z';m['source_event_group_ids']=['disjoint-learning-group']
    m['memory_sha256']=canonical_hash({k:v for k,v in m.items() if k!='memory_sha256'});return m


class MemoryTests(unittest.TestCase):
    def test_rules_snapshot_uses_visible_rules_and_fresh_prices_no_inferred_relation(self):
        j,cat,e,b=episode();self.assertIsNone(e['decision_error'])
        tool=e['discovery']['investigations'][0]['tool_result']
        self.assertFalse(tool['relationship_verified']);self.assertEqual(len(tool['rule_records']),2)
        req=next(r for r in b.requests if r['agent']=='forecast');prompt=binary_prompt(req,'a')
        self.assertIn('rule_records',prompt);self.assertIn('relationship_verified',prompt)
        future=HistoricalCatalogue([job('a','2030-01-01T00:00:00Z')])
        with self.assertRaisesRegex(ValidationError,'future/unknown'):
            rules_snapshot({'tool':'rules_snapshot','market_ids':['a'],'lookback_days':0},future,feed(),AT,10800)
        poisoned=deepcopy(req);t=poisoned['upstream']['tools'][0];row=t['rule_records'][0]
        row['initialized_at']='2030-01-01T00:00:00Z';row['record_sha256']=canonical_hash({k:v for k,v in row.items() if k!='record_sha256'})
        t['result_sha256']=canonical_hash({k:v for k,v in t.items() if k!='result_sha256'})
        with self.assertRaisesRegex(ValidationError,'future rule'):binary_prompt(poisoned,'a')

    def test_memory_rejects_tampering_later_sources_and_event_overlap(self):
        m=target_memory();j=job();validate_memory(m,j['context'])
        m['lessons'][0]['action']='Buy every contract.'
        with self.assertRaisesRegex(ValidationError,'hash'):validate_memory(m,j['context'])
        for key,value,match in [('source_public_through','2030-01-01T00:00:00Z','later'),
            ('source_event_group_ids',[j['context']['markets'][0]['event_group_id']],'overlap'),
            ('built_at','2000-01-01T00:00:00Z','predates')]:
            m=target_memory();m[key]=value;m['memory_sha256']=canonical_hash({k:v for k,v in m.items() if k!='memory_sha256'})
            with self.assertRaisesRegex(ValidationError,match):validate_memory(m,j['context'])
        with self.assertRaisesRegex(ValidationError,'overlap'):validate_memory(target_memory(),j['context'],['disjoint-learning-group'])

    def test_memory_reaches_all_decision_roles_including_binary_readout_but_not_reflection(self):
        j,cat,base,_=episode();m=target_memory();b=ToolBackend()
        e=explore_episode(j['context'],j['labels'],feed(),policy(),MemoryRunner(b,cat,m),base['account_at_observation'],recorded_latency=1)
        self.assertIsNone(e['decision_error'])
        self.assertEqual({r['agent'] for r in b.requests},{'catalogue_search','investigation','research_critic','forecast','risk','discovery_reflection'})
        for r in b.requests:self.assertEqual('offline_team_memory' in r['upstream'],r['agent']!='discovery_reflection')
        r=next(r for r in b.requests if r['agent']=='forecast')
        self.assertIn(m['lessons'][0]['action'],binary_prompt(r,'a'))
        self.assertTrue(all('offline_team_memory' not in c['request']['upstream'] for c in base['calls']))

    def test_learning_sources_and_citations_are_checked(self):
        j,cat,e,_=episode();j['partition']='development'
        with self.assertRaisesRegex(ValidationError,'learning'):learning_evidence([j],[e])
        m=target_memory();edited={'lessons':m['lessons'],'no_lesson_reason':None};edited['lessons'][0]['fact_refs']=['invented']
        with self.assertRaisesRegex(ValidationError,'invented'):validate_lessons(edited,m['cited_facts'])
        self.assertEqual(validate_lessons({'lessons':[],'no_lesson_reason':'No reliable reusable method emerged.'},{} )['lessons'],[])

    def test_explicit_development_catalogue_cannot_mix_partitions(self):
        j=job();j['partition']='development'
        HistoricalCatalogue([j],partition='development')
        with self.assertRaisesRegex(ValidationError,'partitions'):HistoricalCatalogue([j,job('b')],partition='development')

    def test_serialization_roundtrip_and_legacy_reference_order_are_bound(self):
        from foretellmesh.team_method_memory import validate_evidence
        j,cat,e,_=episode();original=learning_evidence([j],[e])
        restored=json.loads(json.dumps(e,sort_keys=True))
        self.assertEqual(original,learning_evidence([j],[restored]))
        legacy=deepcopy(original);legacy['cases'][0]['fact_refs'].reverse()
        self.assertEqual(validate_evidence([j],[restored],legacy),legacy)
        wrong=deepcopy(legacy);wrong['facts']['e0.feedback.market_brier']=.999
        with self.assertRaisesRegex(ValidationError,'facts changed'):validate_evidence([j],[restored],wrong)
        wrong=deepcopy(legacy);wrong['cases'][0]['fact_refs'].append(wrong['cases'][0]['fact_refs'][0])
        with self.assertRaisesRegex(ValidationError,'duplicate'):validate_evidence([j],[restored],wrong)

    def test_current_protocol_excludes_used_peers_and_keeps_source_time_separate(self):
        from pathlib import Path
        from foretellmesh.team_memory_experiment import prepare
        config=json.loads(Path('configs/team_method_memory_v1.json').read_text())
        learning,evaluation,lc,ec,_,meta=prepare(config)
        self.assertEqual((len(learning),len(evaluation)),(6,6))
        excluded=set(meta['excluded_groups'])
        self.assertFalse(excluded & {r['event_group_id'] for r in ec.rows.values()})
        self.assertFalse({r['event_group_id'] for r in lc.rows.values()} & {r['event_group_id'] for r in ec.rows.values()})
        self.assertTrue(all(j['partition']=='train' for j in learning))
        self.assertTrue(all(j['partition']=='development' for j in evaluation))
        self.assertTrue(all(set(j['labels'])=={m['market_id'] for m in j['context']['markets']} for j in evaluation))
        self.assertTrue(all(j['context']['memory']==[] for j in evaluation))
        self.assertLess(max(l['resolution_time'] for j in learning for l in j['labels'].values()),
                        min(j['context']['observation_time'] for j in evaluation))
        self.assertGreater(min(l['available_at'] for j in learning for l in j['labels'].values()),
                           max(j['context']['observation_time'] for j in evaluation))

    def test_future_prices_cannot_change_snapshot(self):
        from foretellmesh.mean_reversion import HistoricalBars
        from foretellmesh.schema import timestamp
        j,cat,_,_=episode();f=feed();at=timestamp(AT,'at')
        changed=HistoricalBars({mid:[(t,.99 if t>at else p,n) for t,p,n in rows] for mid,rows in f.data.items()})
        query={'tool':'rules_snapshot','market_ids':['a','b'],'lookback_days':0}
        self.assertEqual(rules_snapshot(query,cat,f,AT,10800),rules_snapshot(query,cat,changed,AT,10800))

    def test_rules_study_has_no_price_direction_and_selection_fixed(self):
        spec={'tool':'rules_snapshot','market_ids':['a'],'lookback_days':0,'expected_direction':'positive',
              'claim':'Rules might share conditions.','falsifier':'Different dates.'}
        with self.assertRaisesRegex(ValidationError,'direction'):
            validate_study({'investigations':[spec],'no_finding_reason':None},{'a'},{'a'})
        self.assertEqual(spaced(list(range(9)),6),[0,1,3,4,6,8])
