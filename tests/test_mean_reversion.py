from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
import unittest

from foretellmesh.data import strict_json
from foretellmesh.mean_reversion import (MarketWindow, HistoricalBars, features,
    entry_choice, cadence_report, simulate)
from foretellmesh.paper_trading import Settlement
from foretellmesh.schema import ValidationError, iso

ROOT=Path(__file__).resolve().parents[1]
T=datetime(2025,2,1,tzinfo=timezone.utc)
H=timedelta(hours=1)
POLICY=strict_json((ROOT/'configs/mean_reversion_weekly_v1.json').read_text())['policy']
FREE={'entry_price_premium':'0','fee_fraction':'0'}
COST={'entry_price_premium':'0.01','fee_fraction':'0.01'}


def bars(prices,days=3):
    # Prices at each tick and the first subsequent execution block.
    data=[(T+h*H,D('.5'),1) for h in range(-75,0)]
    for h in range(days*24+1):
        p=D(str(prices(h)))
        data.extend([(T+h*H,p,1),(T+h*H+timedelta(seconds=60),p,1)])
    return HistoricalBars({'a':data})


def replay(feed,days=3,**kw):
    end=T+timedelta(days=days)
    return simulate([MarketWindow('a','g',T-timedelta(days=4),end)],
        [Settlement('a',end+H,kw.pop('outcome',1))],feed,T,end,POLICY,kw.pop('costs',FREE),**kw)


class MeanReversionTests(unittest.TestCase):
    def test_asof_features_exclude_current_bar_and_future_changes(self):
        feed=bars(lambda h:.4)
        first=features(feed,'a',T,POLICY)
        self.assertEqual(first['mean_yes_price'],.5)
        self.assertAlmostEqual(first['z_score'],-10)
        changed=HistoricalBars({'a':[(ts,D('.99') if ts>T else p,n) for ts,p,n in feed.data['a']]})
        self.assertEqual(features(changed,'a',T,POLICY),first)
        self.assertEqual(first['trade_count_24h'],24)
        no=HistoricalBars({})
        self.assertEqual(features(no,'a',T,POLICY)['status'],'no_quote')
        self.assertEqual(features(HistoricalBars({'a':[(T-4*H,D('.5'),1)]}),'a',T,POLICY)['status'],'stale_quote')

    def test_round_trip_costs_direction_and_regime_filter(self):
        state=features(bars(lambda h:.4),'a',T,POLICY)
        self.assertEqual(entry_choice(state,POLICY,COST)['side'],'yes')
        state['yes_price']=.6;state['z_score']=10;state['change_24h']=.1
        self.assertEqual(entry_choice(state,POLICY,COST)['side'],'no')
        state.update(yes_price=.48,z_score=-2,change_24h=-.02)
        self.assertIsNone(entry_choice(state,POLICY,COST))  # Costs consume apparent reversion.
        self.assertIsNotNone(entry_choice(state,POLICY,FREE))
        state['change_24h']=.151
        self.assertIsNone(entry_choice(state,POLICY,FREE))
        state.update(z_score=None,mean_yes_price=None)
        forced=entry_choice(state,POLICY,COST,mandatory=True)
        self.assertEqual(forced['side'],'no');self.assertIsNone(forced['target_yes_price'])
        self.assertLess(D(forced['expected_round_trip_edge']),0)

    def test_yes_and_no_targets_exit_with_fee_cash_reconciliation(self):
        for initial,side in [(.4,'yes'),(.6,'no')]:
            r=replay(bars(lambda h:initial if h==0 else .5),costs=COST,weekly=False)
            buys=[x for x in r['ledger'] if x['kind']=='buy_fill']
            sells=[x for x in r['ledger'] if x['kind']=='sell_fill']
            self.assertEqual(len(buys),1);self.assertEqual(len(sells),1)
            self.assertEqual(buys[0]['side'],side);self.assertEqual(sells[0]['exit_reason'],'mean_reached')
            self.assertGreater(D(r['net_pnl']),0)
            self.assertEqual(D(r['final_cash']),D('100')-D(buys[0]['cost'])+D(sells[0]['proceeds']))
            self.assertEqual(D(r['fees_paid']),D(buys[0]['fee'])+D(sells[0]['fee']))
            self.assertEqual(r['settlement_count'],0)
            other=replay(bars(lambda h:initial if h==0 else .5),costs=COST,weekly=False,outcome=0)
            self.assertEqual(r,other)  # Labels have no effect on completed price round trips.

    def test_stop_loss_holding_limit_and_scope_end(self):
        r=replay(bars(lambda h:.4 if h==0 else .3),weekly=False)
        sells=[x for x in r['ledger'] if x['kind']=='sell_fill']
        self.assertEqual(sells[0]['exit_reason'],'stop_loss')
        self.assertLess(D(sells[0]['net_pnl']),0)
        r=replay(bars(lambda h:.4),weekly=False)
        self.assertEqual(next(x for x in r['ledger'] if x['kind']=='sell_fill')['exit_reason'],'holding_limit')
        # At two days, a buy filled after the first tick has not reached 48 h;
        # closing at the declared end prevents a holding-period extension.
        r=replay(bars(lambda h:.4,days=2),days=2,weekly=False)
        self.assertEqual(next(x for x in r['ledger'] if x['kind']=='sell_fill')['exit_reason'],'scope_end')

    def test_first_adverse_print_cancels_without_hunting_for_better_fill(self):
        feed=bars(lambda h:.4 if h==0 else .5)
        feed=HistoricalBars({'a':[(t,D('.49') if t==T+timedelta(seconds=60) else p,n) for t,p,n in feed.data['a']]+
                             [(T+timedelta(seconds=120),D('.4'),1)]})
        r=replay(feed,weekly=False)
        self.assertEqual(r['entry_count'],0)
        self.assertEqual(next(x for x in r['ledger'] if x['kind']=='cancel')['reason'],'price_limit')
        self.assertEqual(D(r['final_cash']),100)

    def test_weekly_no_edge_control_fills_and_retries_missing_execution(self):
        feed=bars(lambda h:.5,days=16)
        # No first attempt fill at day 6: the next tick retries instead of
        # resetting cadence on the failed order.
        feed=HistoricalBars({'a':[(t,p,n) for t,p,n in feed.data['a'] if t!=T+144*H+timedelta(seconds=60)]})
        r=replay(feed,days=16,natural=False,weekly=True,costs=COST)
        self.assertTrue(r['cadence']['compliant'])
        self.assertGreaterEqual(r['entry_count'],2)
        self.assertTrue(all(x['reason']=='weekly_minimum' for x in r['decisions']))
        self.assertEqual(r['decisions'][0]['time'],iso(T+144*H))
        self.assertEqual(r['decisions'][1]['time'],iso(T+145*H))
        self.assertLess(D(r['final_cash']),100)
        self.assertTrue(any(x.get('reason')=='no_post_decision_print' for x in r['ledger']))
        self.assertEqual(r['settlement_count'],0)
        self.assertLessEqual(r['cadence']['actual_max_gap_hours'],168)

    def test_unfillable_orders_do_not_satisfy_weekly_requirement(self):
        feed=HistoricalBars({'a':[(T+h*H,D('.5'),1) for h in range(-75,10*24+1)]})
        r=replay(feed,days=10,natural=False,weekly=True)
        self.assertFalse(r['cadence']['compliant']);self.assertEqual(r['entry_count'],0)
        self.assertGreater(len(r['decisions']),0);self.assertEqual(D(r['final_cash']),100)
        self.assertEqual(r['cadence']['actual_max_gap_hours'],240)

    def test_cadence_boundaries_orders_and_settlements(self):
        week=T+timedelta(days=7)
        ignored=[{'time':iso(T+H),'kind':'buy_order'},{'time':iso(T+2*H),'kind':'settle'}]
        self.assertFalse(cadence_report(ignored,T,week,604800)['compliant'])
        fill={'time':iso(week),'kind':'buy_fill'}
        self.assertTrue(cadence_report(ignored+[fill],T,week,604800)['compliant'])
        self.assertFalse(cadence_report([fill],T,T+timedelta(days=14),604800)['compliant'])
        self.assertTrue(cadence_report([],T,week-timedelta(seconds=1),604800)['compliant'])

    def test_cash_and_group_exposure_limits_with_many_concurrent_signals(self):
        one=bars(lambda h:.4,days=10).data['a']
        end=T+timedelta(days=10)
        markets=[MarketWindow(str(i),'same_group',T-timedelta(days=4),end) for i in range(20)]
        r=simulate(markets,[Settlement(m.market_id,end+H,0) for m in markets],
            HistoricalBars({m.market_id:one for m in markets}),T,end,POLICY,COST,weekly=True)
        self.assertTrue(all(D(x['cash'])>=0 for x in r['equity_curve']))
        self.assertTrue(all(D(x['cash'])>=95 for x in r['equity_curve'][:60]))
        self.assertEqual(D(r['final_cash']),100+D(r['net_pnl']))

    def test_invalid_identity_labels_and_policy_rejected(self):
        end=T+timedelta(days=3);m=MarketWindow('a','g',T-H,end)
        with self.assertRaises(ValidationError):simulate([m,m],[Settlement('a',end+H,1)],bars(lambda h:.4),T,end,POLICY,FREE)
        with self.assertRaises(ValidationError):simulate([m],[Settlement('a',end,1)],bars(lambda h:.4),T,end,POLICY,FREE)
        bad=deepcopy(POLICY);bad['training']=True
        with self.assertRaises(ValidationError):simulate([m],[Settlement('a',end+H,1)],bars(lambda h:.4),T,end,bad,FREE)

    def test_weekly_tail_participation_is_small_feasible_and_keeps_normal_filter(self):
        v2=strict_json((ROOT/'configs/mean_reversion_weekly_v2.json').read_text())['policy']
        state=features(bars(lambda h:.995),'a',T,v2)
        self.assertIsNone(entry_choice(state,v2,COST))
        self.assertIsNone(entry_choice(state,POLICY,COST,mandatory=True))
        self.assertEqual(entry_choice(state,v2,COST,mandatory=True)['side'],'no')
        self.assertEqual(entry_choice(state,v2,FREE,mandatory=True)['side'],'yes')
        end=T+timedelta(days=10)
        r=simulate([MarketWindow('a','g',T-timedelta(days=4),end)],
            [Settlement('a',end+H,1)],bars(lambda h:.995,days=10),T,end,v2,COST,natural=False)
        self.assertTrue(r['cadence']['compliant'])
        self.assertTrue(all(D(x['cost'])<=1 for x in r['ledger'] if x['kind']=='buy_fill'))
        self.assertLess(D(r['final_cash']),100)  # Participation is not an alpha signal.


if __name__=='__main__':unittest.main()
