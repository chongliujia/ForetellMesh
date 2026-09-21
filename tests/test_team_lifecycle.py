from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
import json
import unittest

from foretellmesh.paper_trading import simulate
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_discovery import HistoricalCatalogue
from foretellmesh.team_lifecycle import LifecycleRunner, run_window, replay, validate_review
from foretellmesh.team_trade_experience import learn_available, validate_online_lessons, trade_experiences
from foretellmesh.team_outcome_learning import binary_prompt
import test_paper_trading
from test_team_discovery import job
from test_team_method_memory import ToolBackend
from test_team_learning import AT, policy


TIMING={'min_review_seconds':60,'max_review_seconds':2592000,'max_signal_valid_seconds':604800,
        'max_order_ttl_seconds':86400,'failure_review_seconds':86400}


class ExitTests(unittest.TestCase):
    def setUp(self):
        self.helper=test_paper_trading.PaperTests();self.helper.setUp();self.t=self.helper.t
        self.config=self.helper.config

    def run_case(self, sell_price='.7', outcome=0, *, limit='.1', sell_delay=1, signals=None, settlement=600, through=None):
        h=self.helper
        class Feed:
            def next(f,market,t,deadline):
                delay=1 if t<h.t+timedelta(seconds=50) else sell_delay
                price='.4' if t<h.t+timedelta(seconds=50) else sell_price
                return None if price is None or t+timedelta(seconds=delay)>deadline else (D(price),t+timedelta(seconds=delay))
            def latest(f,market,t):return (D('.4'),t)
        exit=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D(limit))
        return simulate(signals or [h.signal(),exit],[h.settlement(outcome=outcome,offset=settlement)],
            Feed(),self.config,lifecycle=True,through=through)

    def test_sale_closes_position_charges_each_fee_once_and_no_redemption(self):
        r=self.run_case();buy=next(v for v in r['ledger'] if v['kind']=='fill')
        sale=next(v for v in r['ledger'] if v['kind']=='sell_fill')
        self.assertEqual(r['early_exit_count'],1);self.assertEqual(r['settlement_count'],0)
        self.assertEqual(D(r['final_cash']),D(100)-D(buy['cost'])+D(sale['proceeds']))
        self.assertEqual(D(r['net_pnl']),D(sale['proceeds'])-D(buy['cost']))
        self.assertEqual(D(r['fees_paid']),D(buy['fee'])+D(sale['fee']))
        self.assertGreater(D(r['net_pnl']),0)
        self.assertEqual(r,self.run_case(outcome=1))

    def test_no_position_sell_cannot_short_or_sell_new_position_by_old_identity(self):
        h=self.helper;s=replace(h.signal('exit',offset=100),action='sell',position_id='missing',min_exit_price=D('.1'))
        r=self.run_case(signals=[h.signal(),s])
        self.assertEqual(r['early_exit_count'],0);self.assertEqual(r['settlement_count'],1)
        self.assertEqual(r['hold_reasons']['position_not_held'],1)

    def test_no_side_exit_uses_complement_and_can_realize_loss(self):
        h=self.helper;buy=replace(h.signal(p='.1'),action='buy')
        exit=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D('.1'))
        r=self.run_case(signals=[buy,exit]);sale=next(v for v in r['ledger'] if v['kind']=='sell_fill')
        self.assertEqual(sale['side'],'no');self.assertEqual(D(sale['price']),D('.29'))
        self.assertLess(D(r['net_pnl']),0)

    def test_no_fill_or_bad_first_print_keeps_position_until_settlement(self):
        for price in (None,'.05'):
            r=self.run_case(sell_price=price)
            self.assertEqual(r['early_exit_count'],0);self.assertEqual(r['settlement_count'],1)
            self.assertEqual(len([v for v in r['ledger'] if v['kind']=='cancel']),1)

    def test_settlement_wins_race_with_sale_and_cannot_double_pay(self):
        r=self.run_case(sell_delay=490)
        # TTL prevents this quote: expand TTL to put the print exactly at settlement.
        h=self.helper;s=replace(h.signal('exit',offset=100),action='sell',position_id='s',
            min_exit_price=D('.1'),order_ttl_seconds=1000)
        r=self.run_case(sell_delay=490,signals=[h.signal(),s])
        self.assertEqual(r['early_exit_count'],0);self.assertEqual(r['settlement_count'],1)
        self.assertIn('settled_before_sell',[v.get('reason') for v in r['ledger']])

    def test_hold_cancels_pending_sale_and_does_not_force_close(self):
        h=self.helper;exit=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D('.1'))
        hold=replace(h.signal('keep',offset=120),action='hold')
        r=self.run_case(sell_delay=50,signals=[h.signal(),exit,hold])
        self.assertEqual(r['early_exit_count'],0);self.assertEqual(r['settlement_count'],1)
        self.assertIn('superseded_review',[v.get('reason') for v in r['ledger']])

    def test_pending_sale_reserves_shares_not_cash_and_hides_future_fill(self):
        r=self.run_case(sell_delay=50,through=self.t+timedelta(seconds=130))
        s=r['snapshot'];self.assertEqual(s['reserved'],'0');self.assertEqual(s['pending_orders'],1)
        self.assertEqual(s['positions'][0]['position_id'],'s')
        self.assertNotIn('price',s['open_orders'][0]);self.assertEqual(s['open_orders'][0]['action'],'sell')

    def test_expired_decisions_and_fills_never_execute(self):
        h=self.helper
        for expiry in (5,10,11):
            buy=replace(h.signal(),action='buy',expires_at=self.t+timedelta(seconds=expiry))
            r=self.run_case(signals=[buy]);self.assertEqual(r['entry_count'],0)
        exit=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D('.1'),
            expires_at=self.t+timedelta(seconds=111))
        r=self.run_case(signals=[h.signal(),exit]);self.assertEqual(r['early_exit_count'],0)

    def test_legacy_output_unchanged_and_extension_opt_in_required(self):
        h=self.helper
        r=simulate([h.signal()],[h.settlement()],h.feed(),self.config)
        self.assertNotIn('execution_protocol',r)
        with self.assertRaisesRegex(ValidationError,'opt-in'):
            simulate([replace(h.signal(),action='hold')],[h.settlement()],h.feed(),self.config)

    def test_new_position_cannot_be_sold_by_old_exit_intent(self):
        h=self.helper
        first=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D('.1'))
        second=replace(h.signal('second',offset=200),action='buy',probability=D('.99'))
        old=replace(h.signal('old',offset=250),action='sell',position_id='s',min_exit_price=D('.1'))
        r=self.run_case(signals=[h.signal(),first,second,old])
        self.assertEqual(r['entry_count'],2);self.assertEqual(r['early_exit_count'],1)
        self.assertEqual(r['settlement_count'],1);self.assertEqual(r['hold_reasons']['position_not_held'],1)

    def test_expired_review_cannot_cancel_valid_pending_sale(self):
        h=self.helper;sale=replace(h.signal('exit',offset=100),action='sell',position_id='s',min_exit_price=D('.1'))
        expired=replace(h.signal('expired',offset=120),action='hold',expires_at=self.t+timedelta(seconds=125))
        r=self.run_case(sell_delay=50,signals=[h.signal(),sale,expired])
        self.assertEqual(r['early_exit_count'],1);self.assertEqual(r['hold_reasons']['expired_decision'],1)


class LearningBackend(ToolBackend):
    def __init__(self, bad_review=False):
        super().__init__();self.bad_review=bad_review

    def generate(self,request):
        role=request['agent']
        if role=='position_review':
            self.requests.append(deepcopy(request))
            if self.bad_review:return '{broken'
            account=request['upstream']['account_state'];positions={p['market_id']:p for p in account['positions']}
            sold=D(account['realized_net_pnl'])!=0;actions=[]
            for m in request['input']['markets']:
                p=positions.get(m['market_id'])
                actions.append({'market_id':m['market_id'],'action':'sell' if p else 'hold' if sold else 'buy',
                    'position_id':p['position_id'] if p else None,'min_exit_price':.1 if p else None,
                    'reason':'The supplied updated evidence warrants testing an exit.'})
            return json.dumps({'actions':actions,'signal_valid_for_seconds':86400,'order_ttl_seconds':3600,
                               'next_review_after_seconds':86400})
        if role=='trade_reflection':
            self.requests.append(deepcopy(request));case=request['input']['facts']
            return json.dumps({'experience_id':case['experience_id'],
                'fact_check':{k:case['values'][k] for k in ('side','result_class','exit_type','forecast_target')},
                'evidence_refs':['net_pnl','total_fees'],'assessment':'Fees and the realized path are recorded; one trade is inconclusive.',
                'lesson':{'when_applicable':'Trading on sparse price histories.',
                    'proposed_change':'Check evidence availability before trusting a price relationship.',
                    'falsifier':'Independent event groups show no improvement after costs.',
                    'test_plan':{'baseline':'Unmodified team','intervention':'Check coverage before testing a price relation',
                        'metric':'after_cost_net_pnl','pass_condition':'Higher net PnL on disjoint event groups',
                        'validation_split':'disjoint_event_groups'}}})
        if role=='trade_reflection_critic':
            self.requests.append(deepcopy(request))
            return json.dumps({'experience_id':request['input']['facts']['experience_id'],
                'facts_consistent':True,'evidence_supported':True,'testable_change':True,'issues':[]})
        return super().generate(request)


def exercise(*, learning=True, bad=False):
    j=job();at=timestamp(AT,'at')
    # Synthetic paths deliberately cover one winning and one losing trade; no skill claim.
    other=job('b');j['context']['markets']+=other['context']['markets'];j['labels'].update(other['labels'])
    j['provenance'].update(other['provenance'])
    f=HistoricalBars({m:[(at-timedelta(days=1),.4,1),(at,.4,1),(at+timedelta(minutes=10),.4,1),
        (at+timedelta(days=1),exit_q,1),(at+timedelta(days=1,minutes=10),exit_q,1),
        (at+timedelta(days=2),exit_q,1)] for m,exit_q in [('a',.7),('b',.2)]})
    b=LearningBackend(bad);runner=LifecycleRunner(b,HistoricalCatalogue([j],feed=f),TIMING)
    r=run_window(j,f,policy(),runner,end=at+timedelta(days=3),max_reviews=3,recorded_latencies=[1,1,1],learning=learning)
    return j,f,b,runner,r


class LifecycleTests(unittest.TestCase):
    def test_adaptive_review_hold_and_trade_experience_coverage_win_and_loss(self):
        j,f,b,runner,r=exercise();a=r['observation_account']
        self.assertEqual(a['entry_count'],2);self.assertEqual(a['early_exit_count'],2)
        self.assertEqual(len(r['trade_experiences']),2);self.assertEqual(len(r['trade_reflections']),2)
        self.assertEqual({c['result']['result_class'] for c in r['trade_experiences']},{'profit','loss'})
        self.assertEqual(len([q for q in b.requests if q['agent']=='trade_reflection']),2)
        self.assertTrue(all(c['result']['event_outcome'] is None for c in r['trade_experiences']))
        self.assertEqual([e['context']['observation_time'] for e in r['episodes']],
            ['2024-01-02T00:00:00Z','2024-01-03T00:00:00Z','2024-01-04T00:00:00Z'])
        self.assertFalse(r['fine_tuning_triggered']);self.assertFalse(r['effectiveness_verified'])

    def test_lessons_reach_roles_and_probability_readout_only_after_close(self):
        j,f,b,runner,r=exercise();early=[q for q in b.requests if q.get('input',{}).get('observation_time',AT)<'2024-01-04T00:00:00Z']
        self.assertTrue(all('online_trade_lessons' not in q.get('upstream',{}) for q in early))
        later=[q for q in b.requests if q['agent']=='forecast'][-1]
        self.assertEqual(len(later['upstream']['online_trade_lessons']),2)
        self.assertIn('Check evidence availability',binary_prompt(later,'a'))
        for q in b.requests:
            if q['agent'] in ('trade_reflection','trade_reflection_critic'):continue
            self.assertNotIn('"outcome"',json.dumps(q));self.assertNotIn('"resolution_time"',json.dumps(q))

    def test_evaluation_records_all_experience_without_learning(self):
        _,_,b,_,r=exercise(learning=False)
        self.assertEqual(len(r['trade_experiences']),2);self.assertEqual(r['trade_reflections'],[])
        self.assertFalse(any(q['agent']=='trade_reflection' for q in b.requests))

    def test_bad_decision_repair_preserved_no_trade_or_invented_experience(self):
        _,_,b,_,r=exercise(bad=True)
        self.assertTrue(all(e['decision'] is None for e in r['episodes']))
        self.assertEqual(r['observation_account']['entry_count'],0);self.assertEqual(r['trade_experiences'],[])
        self.assertEqual(len([q for q in b.requests if q['agent']=='position_review']),6)

    def test_replay_rejects_tampered_snapshot_and_future_lessons(self):
        j,f,b,runner,r=exercise();eps=deepcopy(r['episodes']);eps[1]['account_at_observation']['cash']='999'
        from foretellmesh.paper_trading import Settlement
        labels=[Settlement(mid,timestamp(l['resolution_time'],'settle'),l['outcome']) for mid,l in j['labels'].items()]
        with self.assertRaisesRegex(ValidationError,'snapshot'):
            replay(eps,labels,f,policy(),TIMING)
        rows=deepcopy(r['trade_reflections'])
        with self.assertRaisesRegex(ValidationError,'future'):
            validate_online_lessons(rows,AT)
        rows[0]['output']['assessment']='tampered'
        with self.assertRaisesRegex(ValidationError,'changed'):
            validate_online_lessons(rows,'2025-01-01T00:00:00Z')

    def test_settlement_feedback_waits_for_verified_availability_and_open_is_not_loss(self):
        j,f,b,runner,r=exercise();ep=r['episodes'][:1]
        from foretellmesh.paper_trading import Settlement
        settlements=[Settlement(mid,timestamp(l['resolution_time'],'s'),l['outcome']) for mid,l in j['labels'].items()]
        open_account=replay(ep,settlements,f,policy(),TIMING,through=timestamp('2024-01-03T00:00:00Z','at'))
        cases=trade_experiences(ep,open_account,j['labels'],policy())
        self.assertTrue(all(c['status']=='open' and c['result'] is None for c in cases))
        cases=trade_experiences(ep,replay(ep,settlements,f,policy(),TIMING),j['labels'],policy())
        runner.trade_reflections={}
        learn_available(runner,cases,'2024-01-12T12:00:00Z',enabled=True)
        self.assertEqual(runner.trade_reflections,{})
        learn_available(runner,cases,'2024-01-13T00:00:00Z',enabled=True)
        self.assertEqual(len(runner.trade_reflections),2)

    def test_development_online_learning_rejected(self):
        j=job();j['partition']='development';f=HistoricalBars({})
        runner=LifecycleRunner(LearningBackend(),HistoricalCatalogue([j],partition='development'),TIMING)
        with self.assertRaisesRegex(ValidationError,'training partition'):
            run_window(j,f,policy(),runner,end=timestamp('2024-01-03T00:00:00Z','end'),max_reviews=1,learning=True)

    def test_failed_reflections_are_archived_and_not_injected_as_lessons(self):
        j,f,b,runner,r=exercise();runner.trade_reflections={};runner.calls=[]
        class Broken:
            def generate(self,request):return '{broken'
        runner.backend=Broken()
        learn_available(runner,r['trade_experiences'],'2024-01-05T00:00:00Z',enabled=True)
        self.assertEqual(len(runner.trade_reflections),2)
        self.assertTrue(all(v['output'] is None and v['error'] for v in runner.trade_reflections.values()))
        self.assertEqual(len(runner.calls),4)
        self.assertNotIn('online_trade_lessons',runner.with_memory(j['context'],{}))

    def test_model_call_replay_reproduces_complete_window_without_gpu(self):
        from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
        j,f,b,runner,r=exercise()
        calls=[c for e in r['episodes'] for c in e['calls']]+r['closing_reflection_calls']
        backend=RecordedBackend(calls);audit=AuditRunner(backend,runner.catalogue,TIMING)
        rebuilt=run_window(j,f,policy(),audit,end=timestamp(AT,'at')+timedelta(days=3),
            max_reviews=3,recorded_latencies=[1,1,1],learning=True)
        self.assertEqual(r,rebuilt);self.assertEqual(len(calls),backend.index)

    def test_reflection_summary_preserves_all_decisions_and_full_archive_identity(self):
        from foretellmesh.team_trade_experience import reflection_context
        _,_,_,_,r=exercise()
        for case in r['trade_experiences']:
            before=deepcopy(case);summary=reflection_context(case)
            self.assertEqual(case,before)
            self.assertEqual(summary['experience_sha256'],case['experience_sha256'])
            self.assertEqual(len(summary['decision_summaries']),len(case['decisions']))
            self.assertEqual([v['episode_sha256'] for v in summary['decision_summaries']],
                [v['episode_sha256'] for v in case['decisions']])
            self.assertTrue(summary['full_decisions_archived'])


if __name__=='__main__':unittest.main()
