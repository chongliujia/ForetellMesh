from copy import deepcopy
from datetime import timedelta
import json
import unittest

from foretellmesh.schema import timestamp, iso, ValidationError
from foretellmesh.team_discovery import (HistoricalCatalogue, DiscoveryRunner, explore_episode,
    validate_study, validate_reviews, validate_reflection, catalogue_query_payload, directional_test, compact_coverage)
from foretellmesh.team_portfolio import replay_portfolio, target_progress, continuous_feedback
from foretellmesh.team_learning import TeamRunner, run_episode
from foretellmesh.team_discovery_experiment import current_account
from foretellmesh.team_outcome_learning import binary_prompt
from foretellmesh.synthetic_sft import canonical_hash
from test_team_learning import context, labels, feed, policy, ScriptBackend, AT


def job(mid='a', initialized='2024-01-01T00:00:00Z'):
    c=context(); c['markets']=[m for m in c['markets'] if m['market_id'] == mid]; c['episode_id']='episode-'+mid
    return {'context':c,'partition':'train','labels':{mid:labels()[mid]},'provenance':{mid:{'initialized_at':initialized}}}


OBJECTIVE={'initial_cash':'100','target_net_return':'0.10','target_terminal_cash':'110',
    'basis':'single_continuous_settled_account','deadline':None,'force_trades':False,'automatic_fine_tuning':False}


class Backend:
    def __init__(self): self.requests=[]
    def generate(self, request):
        self.requests.append(deepcopy(request)); role=request['agent']
        if role == 'catalogue_search': value={'text':'','limit':6,'offset':0}
        elif role == 'investigation': value={'investigations':[{'market_ids':['a','b'],'lookback_days':3,
            'claim':'Markets a and b price changes may co-move.','falsifier':'Opposite movements in a later independent window.',
            'expected_direction':'positive'}],'no_finding_reason':None}
        elif role == 'research_critic':
            value={'reviews':[{'investigation_id':r['investigation_id'],'verdict':'retain_for_testing',
                'evidence_refs':[r['tool_result']['result_sha256']],'reason':'Historical changes co-move.',
                'next_check':'Repeat on independent later observations.'} for r in request['upstream']['investigations']]}
        elif role == 'forecast': value={'forecasts':[{'market_id':m['market_id'],'probability':.8,'consider_trade':True}
            for m in request['input']['markets']],'unknowns':['The relationship is unvalidated.']}
        elif role == 'risk': value={'veto_markets':[],'risks':['Sparse historical evidence.']}
        else: value={'observations':[{'fact_id':'forecast_brier','lesson':'One outcome cannot validate the relationship.',
                                     'next_check':'Check new event groups.'}],'no_lesson_reason':None}
        return json.dumps(value)


class DiscoveryTests(unittest.TestCase):
    def test_exact_search_envelope_preserves_arguments_and_rejects_ambiguous_wrappers(self):
        query={'text':'relationship','limit':6,'offset':0}
        self.assertEqual(catalogue_query_payload({'output':{'search':query}}),query)
        cat=HistoricalCatalogue([job()])
        with self.assertRaises(ValidationError):
            cat.search(catalogue_query_payload({'output':{'search':query,'extra':'ignored?'}}),AT)
        with self.assertRaises(ValidationError):
            cat.search(catalogue_query_payload({'output':{'search':{**query,'limit':1000}}}),AT)

    def test_search_filters_future_before_ranking_and_keeps_heldout_out(self):
        cat=HistoricalCatalogue([job('a'),job('b','2025-01-01T00:00:00Z')])
        r=cat.search({'text':'','offset':0,'limit':6},AT)
        self.assertEqual([m['market_id'] for m in r['records']],['a'])
        self.assertEqual(r['visible_count'],1)
        self.assertNotIn('"outcome"',json.dumps(r)); self.assertNotIn('"resolution_time"',json.dumps(r))
        bad=job(); bad['partition']='development'
        with self.assertRaisesRegex(ValidationError,'held-out'): HistoricalCatalogue([bad])

    def test_search_paginates_browsing_and_can_report_no_match(self):
        cat=HistoricalCatalogue([job('a'),job('b')])
        r=cat.search({'text':'','offset':0,'limit':1},AT); self.assertEqual(r['next_offset'],1)
        self.assertEqual(cat.search({'text':'','offset':1,'limit':1},AT)['records'][0]['market_id'],'b')
        self.assertEqual(cat.search({'text':'nonexistententity','offset':0,'limit':6},AT)['matched_count'],0)

    def test_study_needs_real_ids_and_a_falsifier(self):
        value={'investigations':[{'market_ids':['a','invented'],'lookback_days':7,'claim':'claim','falsifier':'counterexample',
                                 'expected_direction':'positive'}],'no_finding_reason':None}
        with self.assertRaisesRegex(ValidationError,'scope'):validate_study(value,{'a','b'},{'a'})
        value['investigations'][0]['market_ids']=['a','b']; value['investigations'][0]['falsifier']=''
        with self.assertRaisesRegex(ValidationError,'falsifier'):validate_study(value,{'a','b'},{'a'})
        with self.assertRaises(ValidationError):validate_study({'investigations':[],'no_finding_reason':None},{'a'},{'a'})

    def test_single_direction_measures_price_change_not_pair_correlation(self):
        spec={'market_ids':['a'],'lookback_days':7,'claim':'Market a price rose.',
              'falsifier':'Last price below first.','expected_direction':'positive'}
        validate_study({'investigations':[spec],'no_finding_reason':None},{'a'},{'a'})
        tool={'series':{'a':[{'price':.2},{'price':.4}]}}
        result=directional_test(spec,tool)
        self.assertEqual(result['kind'],'single_market_net_price_change')
        self.assertEqual(result['status'],'direction_supported_in_window')
        self.assertFalse(result['predictive_effectiveness_verified'])
        spec['expected_direction']='negative'
        self.assertEqual(directional_test(spec,tool)['status'],'direction_not_supported_in_window')
        self.assertEqual(directional_test(spec,{'series':{'a':[{'price':.4}]}})['status'],'insufficient_data')
        spec['market_ids']=['a','b']
        result=directional_test(spec,{'paired_daily_changes':0,'pearson_change_correlation':None})
        self.assertEqual(result['kind'],'paired_daily_change_correlation')
        self.assertEqual(result['status'],'insufficient_data')

    def test_coverage_is_asof_and_matches_study_samples(self):
        from foretellmesh.team_learning import run_query
        f=feed(); cat=HistoricalCatalogue([job('a'),job('b')],feed=f)
        query={'text':'','offset':0,'limit':6}
        result=cat.search(query,AT,{'a'})
        meta=result['coverage']['b']['windows_days']['7']
        pair=run_query({'tool':'pair_changes','market_ids':['a','b'],'lookback_days':7},f,AT)
        self.assertEqual(meta['daily_samples'],len(pair['series']['b']))
        self.assertEqual(meta['overlap_with_targets']['a']['paired_daily_changes'],pair['paired_daily_changes'])
        # Change every future price without changing any visible coverage or ranking.
        from foretellmesh.mean_reversion import HistoricalBars
        at=timestamp(AT,'cutoff')
        changed=HistoricalBars({mid:[(t,.99 if t>at else p,n) for t,p,n in rows] for mid,rows in f.data.items()})
        self.assertEqual(result,HistoricalCatalogue([job('a'),job('b')],feed=changed).search(query,AT,{'a'}))
        with self.assertRaisesRegex(ValidationError,'future/unknown'):
            cat.search(query,AT,{'unknown'})

    def test_coverage_prompt_tables_preserve_every_value(self):
        raw=HistoricalCatalogue([job('a'),job('b')],feed=feed()).search({'text':'','offset':0,'limit':6},AT,{'a','b'})['coverage']
        value=compact_coverage(raw); rebuilt={}
        for mid,days,n,first,last in value['windows']:
            meta=rebuilt.setdefault(mid,{'as_of':value['as_of'],'max_quote_age_seconds':value['max_quote_age_seconds'],'windows_days':{}})
            meta['windows_days'][str(days)]={'daily_samples':n,'first_source_time':first,'last_source_time':last,'overlap_with_targets':{}}
        for mid,days,target,samples,changes in value['pairs']:
            rebuilt[mid]['windows_days'][str(days)]['overlap_with_targets'][target]={
                'paired_daily_samples':samples,'paired_daily_changes':changes}
        self.assertEqual(rebuilt,raw)
        self.assertEqual(compact_coverage({}),{})

    def test_placeholder_echo_is_not_a_testable_hypothesis(self):
        value={'investigations':[{'market_ids':['503634','503348'],'lookback_days':14,
            'claim':'tentative relationship','falsifier':'observation that would count against it',
            'expected_direction':'positive'}],'no_finding_reason':None}
        with self.assertRaisesRegex(ValidationError,'placeholder'):
            validate_study(value,{'503634','503348'},{'503634'})
        row=value['investigations'][0]
        row.update(claim='503634 and 503348 have positively correlated daily price changes.',
                   falsifier='Nonpositive correlation in the selected window contradicts this direction.')
        validate_study(value,{'503634','503348'},{'503634'})
        # Structured market_ids already bind identity; do not reject substantive descriptions
        # solely because they omit a redundant literal ID in prose.
        row['claim']='The selected Egypt football and China total-medal contracts have positive daily co-movement.'
        validate_study(value,{'503634','503348'},{'503634'})

    def test_team_searches_beyond_initial_context_and_preserves_registered_test(self):
        j=job(); backend=Backend(); cat=HistoricalCatalogue([j,job('b')])
        account=current_account([],[],feed(),{'execution_policy':policy(),'objective':OBJECTIVE},AT)
        ep=explore_episode(j['context'],j['labels'],feed(),policy(),DiscoveryRunner(backend,cat),account,recorded_latency=1)
        self.assertIsNone(ep['decision_error']); self.assertIsNone(ep['reflection_error'])
        self.assertEqual(ep['discovery']['investigations'][0]['specification']['market_ids'],['a','b'])
        self.assertEqual(len(j['context']['markets']),1)
        self.assertFalse(ep['effectiveness_verified']); self.assertFalse(ep['fine_tuning_triggered'])
        request=next(r for r in backend.requests if r['agent']=='forecast')
        self.assertIn('Will event b happen?',binary_prompt(request,'a'))
        poisoned=deepcopy(request); peer=poisoned['upstream']['retrieved_markets'][0]
        peer['initialized_at']='2030-01-01T00:00:00Z'
        peer['record_sha256']=canonical_hash({k:v for k,v in peer.items() if k!='record_sha256'})
        with self.assertRaisesRegex(ValidationError,'future retrieved'):binary_prompt(poisoned,'a')
        for r in backend.requests[:-1]:self.assertNotIn('realized_outcomes',json.dumps(r))
        self.assertEqual(backend.requests[-1]['input']['phase'],'retrospective')
        review=ep['discovery']['review']; review['reviews'][0]['evidence_refs']=['invented']
        with self.assertRaisesRegex(ValidationError,'evidence'):validate_reviews(review,ep['discovery']['investigations'])

    def test_empty_reflection_needs_explicit_reason_and_facts_cannot_be_invented(self):
        with self.assertRaises(ValidationError):validate_reflection({'observations':[],'no_lesson_reason':None},{'forecast_brier':.2})
        validate_reflection({'observations':[],'no_lesson_reason':'No overlapping price samples.'},{})
        with self.assertRaisesRegex(ValidationError,'invented'):
            validate_reflection({'observations':[{'fact_id':'fake','lesson':'x','next_check':'y'}],'no_lesson_reason':None},{})

    def test_snapshot_has_no_future_settlement_and_goal_does_not_enable_training(self):
        j=job(); backend=Backend(); cat=HistoricalCatalogue([j,job('b')])
        state=current_account([],[],feed(),{'execution_policy':policy(),'objective':OBJECTIVE},AT)
        ep=explore_episode(j['context'],j['labels'],feed(),policy(),DiscoveryRunner(backend,cat),state,recorded_latency=1)
        cutoff=timestamp(AT,'at')+timedelta(minutes=11)
        snap=replay_portfolio([j],[ep],feed(),policy(),through=cutoff)['snapshot']
        self.assertEqual(len(snap['positions']),1); self.assertEqual(snap['realized_net_pnl'],'0')
        changed=deepcopy(j); changed['labels']['a']['outcome']=0
        self.assertEqual(snap,replay_portfolio([changed],[ep],feed(),policy(),through=cutoff)['snapshot'])
        pending=replay_portfolio([j],[ep],feed(),policy(),through=timestamp(AT,'at')+timedelta(minutes=1))['snapshot']
        self.assertEqual(pending['pending_orders'],1); self.assertEqual(pending['positions'],[])
        terminal=replay_portfolio([j],[ep],feed(),policy()); progress=target_progress(terminal,OBJECTIVE)
        self.assertEqual(progress['target_terminal_cash'],'110'); self.assertFalse(progress['fine_tuning_triggered'])
        terminal['final_cash']='110'; terminal['return_fraction']='0.10'
        progress=target_progress(terminal,OBJECTIVE)
        self.assertTrue(progress['terminal_target_reached']); self.assertFalse(progress['effectiveness_verified'])

    def test_repeat_observation_does_not_reset_capital_or_invent_a_second_trade(self):
        j=job(); cat=HistoricalCatalogue([j,job('b')]); cfg={'execution_policy':policy(),'objective':OBJECTIVE}
        state=current_account([],[],feed(),cfg,AT)
        ep=explore_episode(j['context'],j['labels'],feed(),policy(),DiscoveryRunner(Backend(),cat),state,
            recorded_latency=1,feedback_transform=lambda d,t,f:continuous_feedback([],[],j,d,t,f,feed(),policy()))
        later=deepcopy(j); later['context']['episode_id']='later'
        at=iso(timestamp(AT,'at')+timedelta(hours=1)); later['context']['observation_time']=at
        later['context']['markets'][0]['input']['observation_time']=at
        state=current_account([j],[ep],feed(),cfg,at)
        self.assertLess(float(state['cash']),100); self.assertEqual(len(state['positions']),1)
        ep2=explore_episode(later['context'],later['labels'],feed(),policy(),DiscoveryRunner(Backend(),cat),state,
            recorded_latency=1,feedback_transform=lambda d,t,f:continuous_feedback([j],[ep],later,d,t,f,feed(),policy()))
        self.assertEqual(ep2['feedback']['facts']['filled_trades'],0)
        self.assertEqual(ep2['feedback']['facts']['paper_net_pnl'],'0')
        portfolio=replay_portfolio([j,later],[ep,ep2],feed(),policy())
        self.assertEqual(portfolio['filled_trades'],1)
        self.assertEqual(portfolio['final_cash'],ep['feedback']['account']['final_cash'])


if __name__ == '__main__': unittest.main()
