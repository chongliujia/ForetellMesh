from copy import deepcopy
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
import unittest

from foretellmesh.paper_trading import Signal,Settlement,simulate,validate_policy
from foretellmesh.schema import ValidationError


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.t=datetime(2025,1,1,tzinfo=timezone.utc)
        self.config={'initial_cash':'100','max_trade_usd':'2','max_event_usd':'5','max_portfolio_usd':'20',
                     'min_trade_usd':'1','min_edge':'.03','entry_price_premium':'.01','fee_fraction':'.01',
                     'max_quote_age_seconds':10800,'fill_window_seconds':300}

    def signal(self,sid='s',market='m',group='g',p='.8',q='.4',offset=0):
        t=self.t+timedelta(seconds=offset)
        return Signal(sid,market,group,t,t+timedelta(seconds=10),t,D(q),None if p is None else D(p))

    def settlement(self,market='m',outcome=1,offset=600):
        return Settlement(market,self.t+timedelta(seconds=offset),outcome)

    def feed(self,price='.4',delay=1):
        class Feed:
            def next(f,market,t,deadline):return (D(price),t+timedelta(seconds=delay)) if price is not None else None
            def latest(f,market,t):return D(price or '.4'),t
        return Feed()

    def test_cash_conservation_costs_and_yes_settlement(self):
        r=simulate([self.signal()],[self.settlement()],self.feed(),self.config)
        fill=next(v for v in r['ledger'] if v['kind']=='fill')
        shares=D(fill['shares']);cost=D(fill['cost']);fee=D(fill['fee'])
        self.assertEqual(cost,D('1.999998'))
        self.assertEqual(D(r['final_cash']),D(100)-cost+shares)
        self.assertEqual(D(r['net_pnl']),shares-cost)
        self.assertEqual(r['filled_trades'],1);self.assertEqual(r['real_orders_sent'],0)
        self.assertGreater(D(r['sampled_equity_proxy_max_drawdown_usd']),0)

    def test_no_side_and_losses_are_preserved(self):
        for y in (0,1):
            r=simulate([self.signal(p='.1')],[self.settlement(outcome=y)],self.feed(),self.config)
            fill=next(v for v in r['ledger'] if v['kind']=='fill')
            self.assertEqual(fill['side'],'no')
            self.assertEqual(D(r['final_cash']),D(100)-D(fill['cost'])+D(fill['shares'])*(1-y))
            self.assertEqual(r['winning_trades'],1-y)

    def test_market_copy_holds_even_without_costs(self):
        for premium,fee in [('0','0'),('.01','.01')]:
            c={**self.config,'entry_price_premium':premium,'fee_fraction':fee}
            r=simulate([self.signal(p='.4')],[self.settlement()],self.feed(),c)
            self.assertEqual(r['orders'],0);self.assertEqual(r['final_cash'],'100')
            self.assertEqual(r['hold_reasons'],{'no_cost_adjusted_edge':1})

    def test_missing_or_adverse_next_print_cancels_not_selects_future_winner(self):
        for price,reason in [(None,'no_post_decision_print'),('.9','price_limit')]:
            r=simulate([self.signal()],[self.settlement()],self.feed(price),self.config)
            self.assertEqual(r['orders'],1);self.assertEqual(r['filled_trades'],0)
            self.assertEqual(r['final_cash'],'100')
            self.assertEqual(next(v['reason'] for v in r['ledger'] if v['kind']=='cancel'),reason)

    def test_pending_cash_event_caps_and_repeat_contracts(self):
        signals=[self.signal(str(i),'m'+str(i),offset=0) for i in range(4)]
        signals.append(self.signal('repeat','m0',offset=1))
        settlements=[self.settlement('m'+str(i),outcome=0) for i in range(4)]
        r=simulate(signals,settlements,self.feed(delay=30),self.config)
        orders=[v for v in r['ledger'] if v['kind']=='order']
        self.assertEqual([D(v['budget']) for v in orders],[D(2),D(2),D(1)])
        self.assertEqual(r['filled_trades'],3)
        self.assertEqual(r['hold_reasons'],{'capital_or_exposure_limit':1,'already_exposed_to_contract':1})
        self.assertGreaterEqual(D(r['final_cash']),D(95))

    def test_settlement_before_fill_releases_reservation_and_blocks_late_buys(self):
        r=simulate([self.signal(),self.signal('later',offset=40)],[self.settlement(offset=20)],self.feed(delay=30),self.config)
        self.assertEqual(r['final_cash'],'100');self.assertEqual(r['filled_trades'],0)
        self.assertEqual(r['hold_reasons'],{'already_settled':1})

    def test_labels_do_not_change_pre_settlement_decisions(self):
        a=simulate([self.signal()],[self.settlement(outcome=0)],self.feed(),self.config)
        b=simulate([self.signal()],[self.settlement(outcome=1)],self.feed(),self.config)
        self.assertEqual([r for r in a['ledger'] if r['kind']!='settle'],[r for r in b['ledger'] if r['kind']!='settle'])

    def test_repeating_decimal_trade_prices_reconcile_exactly(self):
        signals=[self.signal(str(i),'m'+str(i),'g'+str(i),offset=i) for i in range(3)]
        yy=[self.settlement('m'+str(i),outcome=i%2,offset=600+i) for i in range(3)]
        r=simulate(signals,yy,self.feed(str(D(1)/D(3))),self.config)
        flows=sum((D(row['payout']) for row in r['ledger'] if row['kind']=='settle'),D(0))-sum(
            (D(row['cost']) for row in r['ledger'] if row['kind']=='fill'),D(0))
        self.assertEqual(D(r['final_cash']),D(100)+flows)
        self.assertEqual(D(r['net_pnl']),flows)

    def test_future_stale_duplicate_and_invalid_inputs(self):
        signal=self.signal()
        from dataclasses import replace
        for bad in [replace(signal,quote_time=signal.observation_time+timedelta(seconds=1)),replace(signal,probability=D('NaN'))]:
            with self.assertRaises(ValidationError):simulate([bad],[self.settlement()],self.feed(),self.config)
        with self.assertRaises(ValidationError):simulate([signal,signal],[self.settlement()],self.feed(),self.config)
        stale=replace(signal,quote_time=signal.observation_time-timedelta(hours=4))
        r=simulate([stale],[self.settlement()],self.feed(),self.config)
        self.assertEqual(r['hold_reasons'],{'stale_decision_quote':1})
        with self.assertRaises(ValidationError):simulate([signal],[],self.feed(),self.config)
        with self.assertRaises(ValidationError):simulate([signal],[self.settlement()],self.feed(delay=-1),self.config)
        with self.assertRaises(ValidationError):validate_policy({**self.config,'initial_cash':'NaN'})


if __name__=='__main__':unittest.main()
