from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import unittest

from foretellmesh.event_allocation import choice, simulate, validate_policy
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.paper_trading import Settlement
from foretellmesh.schema import ValidationError, iso

T = datetime(2024, 1, 1, tzinfo=timezone.utc)
P = dict(initial_cash='100', max_trade_usd='2', min_trade_usd='1', max_event_usd='5', max_portfolio_usd='20',
         entry_edge='.03', exit_edge='.005', uncertainty_margin='.02', stop_price_points='.05',
         min_yes_price='.05', max_yes_price='.95', max_positions=4, max_entries_per_day=2,
         quote_max_age_seconds=10800, order_ttl_seconds=3600, cooldown_seconds=86400, monitor_seconds=3600, review_seconds=259200, material_probability_change='.05')
C = dict(entry_price_premium='.01', fee_fraction='.01')


def signal(mid='1', p=.8, observed=T, ready=None, expires=None, sid='a'):
    return dict(sample_id=sid, market_id=mid, observation_time=iso(observed), available_at=iso(ready or observed+timedelta(seconds=60)),
                expires_at=iso(expires or observed+timedelta(days=7)), reference_probability=.4,
                content=None if p is None else {'probability': p})


def run(signals=None, prices=None, days=5, outcome=1, markets=1, **kwargs):
    # These tests retain the historical v2 protocol; admission has separate tests.
    kwargs.setdefault('legacy_unvalidated_research', True)
    mids = [str(i+1) for i in range(markets)]
    mm = [dict(market_id=m, event_group_id='g', initialized_at=iso(T-timedelta(days=1))) for m in mids]
    if prices is None:
        prices = {m: [(T+timedelta(minutes=5*i), D('.4'), 1) for i in range(days*288)] for m in mids}
    return simulate([signal()] if signals is None else signals, mm,
        [Settlement(m, T+timedelta(days=days), outcome) for m in mids], HistoricalBars(prices), P, C, T,
        T+timedelta(days=days+1), **kwargs)


class EventAllocationTests(unittest.TestCase):
    def test_holds_past_research_expiry_and_48_hours_until_settlement(self):
        r = run(signals=[signal(expires=T+timedelta(days=1))])
        self.assertEqual(r['metrics']['entry_count'], 1)
        self.assertEqual(r['metrics']['exit_count'], 0)
        self.assertEqual(r['metrics']['settlement_count'], 1)
        self.assertGreater(r['metrics']['mean_holding_hours'], 119)
        self.assertEqual(r['metrics']['held_to_settlement_fraction'], 1)
        self.assertTrue(any(d['reason'] == 'hold_position_without_fresh_forecast' for d in r['decisions']))

    def test_matched_short_control_exits_after_48_hours(self):
        r = run(signals=[signal(expires=T+timedelta(days=1))], max_holding_seconds=172800)
        sells = [x for x in r['ledger'] if x['kind'] == 'sell_fill']
        self.assertEqual(len(sells), 1); self.assertEqual(sells[0]['reason'], 'holding_limit')
        self.assertLess(r['metrics']['mean_holding_hours'], 50)
        self.assertLess(D(r['metrics']['final_cash']), 100)

    def test_no_advantage_and_failed_output_and_cash_control_do_not_trade(self):
        for ss in ([signal(p=.4)], [signal(p=None)], []):
            r = run(signals=ss); self.assertEqual(r['metrics']['entry_count'], 0); self.assertEqual(r['metrics']['final_cash'], '100')
        self.assertEqual(run(cash_only=True)['metrics']['entry_count'], 0)
        self.assertEqual(run(anchor_only=True)['metrics']['entry_count'], 0)

    def test_expired_or_failed_signal_cancels_pending_entry(self):
        for row in (signal(p=None, observed=T+timedelta(seconds=61), ready=T+timedelta(seconds=120), sid='failed'),):
            r = run(signals=[signal(), row]); self.assertEqual(r['metrics']['entry_count'], 0)
            self.assertTrue(any(x.get('reason') == 'expired_failed_or_superseded_signal' for x in r['ledger']))
        r = run(signals=[signal(expires=T+timedelta(seconds=100))])
        self.assertEqual(r['metrics']['entry_count'], 0)

    def test_available_time_prevents_backfill_and_labels_do_not_affect_decisions(self):
        a = run(outcome=0); b = run(outcome=1)
        self.assertEqual(a['decisions'], b['decisions'])
        self.assertEqual([x for x in a['ledger'] if x['kind'] != 'settle'], [x for x in b['ledger'] if x['kind'] != 'settle'])
        self.assertGreater(D(b['metrics']['final_cash']), 100); self.assertLess(D(a['metrics']['final_cash']), 100)
        self.assertTrue(all(x['time'] >= iso(T+timedelta(seconds=60)) for x in a['ledger'] if x['kind'] == 'buy_order'))

    def test_no_post_decision_print_never_fabricates_entry(self):
        r = run(prices={'1': [(T, D('.4'), 1)]})
        self.assertEqual(r['metrics']['entry_count'], 0); self.assertEqual(r['metrics']['final_cash'], '100')

    def test_edge_rechecked_at_first_print_not_later_better_print(self):
        prices = {'1': [(T, D('.4'), 1), (T+timedelta(minutes=5), D('.95'), 1), (T+timedelta(minutes=10), D('.4'), 1)]}
        r = run(signals=[signal(expires=T+timedelta(minutes=30))], prices=prices)
        self.assertEqual(r['metrics']['entry_count'], 0)
        self.assertTrue(any(x.get('reason') == 'edge_eroded' for x in r['ledger']))

    def test_price_risk_exit_remains_active_without_signal(self):
        prices = {'1': [(T+timedelta(minutes=5*i), D('.4') if i < 576 else D('.3'), 1) for i in range(5*288)]}
        r = run(signals=[signal(expires=T+timedelta(days=1))], prices=prices)
        self.assertEqual(r['metrics']['exit_count'], 1)
        self.assertEqual(next(x['reason'] for x in r['ledger'] if x['kind'] == 'sell_fill'), 'stop_loss')

    def test_fresh_evidence_can_exit_and_sunk_cost_is_not_recharged(self):
        r = run(signals=[signal(), signal(p=.2, observed=T+timedelta(days=2), sid='changed')])
        self.assertTrue(any(x.get('reason') == 'advantage_realized' for x in r['ledger'] if x['kind'] == 'sell_fill'))
        p = validate_policy(P)
        self.assertIsNotNone(choice(D('.45'), D('.5'), p, C, {'side': 'yes', 'cost': D(1000)}))
        self.assertIsNone(choice(D('.50'), D('.5'), p, C, {'side': 'yes', 'cost': D(0)}))
        self.assertIsNone(choice(D('.52'), D('.5'), p, C))

    def test_cash_risk_caps_and_fees_reconcile(self):
        ss = [signal(mid=str(i), sid=str(i)) for i in range(1, 7)]
        r = run(signals=ss, markets=6)
        self.assertLessEqual(sum(D(x['cost']) for x in r['ledger'] if x['kind'] == 'buy_fill'), 5)
        fee = sum((D(x['fee']) for x in r['ledger'] if 'fee' in x), D(0))
        self.assertEqual(fee, D(r['metrics']['fees_paid']))
        self.assertEqual(D(r['metrics']['net_pnl']), D(r['metrics']['final_cash'])-100)
        self.assertTrue(all(D(x['cash']) >= 0 and x['positions'] <= 4 for x in r['equity_curve']))

    def test_future_prices_cannot_change_earlier_actions(self):
        a = run(); prices = {'1': [(T+timedelta(minutes=5*i), D('.4') if i < 576 else D('.9'), 1) for i in range(5*288)]}
        b = run(prices=prices)
        cutoff = iso(T+timedelta(days=2))
        self.assertEqual([d for d in a['decisions'] if d['time'] < cutoff], [d for d in b['decisions'] if d['time'] < cutoff])

    def test_hourly_monitor_does_not_trigger_ordinary_retrading(self):
        r = run()
        ready = T+timedelta(seconds=60)
        between = [d for d in r['decisions'] if iso(ready) < d['time'] < iso(ready+timedelta(days=3))]
        self.assertTrue(between)
        self.assertTrue(all(not d['ordinary_review'] for d in between))
        self.assertEqual(r['metrics']['entry_count'], 1)

    def test_small_research_refresh_does_not_trigger_a_trade_review(self):
        r = run(signals=[signal(), signal(p=.81, observed=T+timedelta(days=1), sid='small')])
        refreshed = [d for d in r['decisions'] if d['trigger'] == 'research' and d['signal_id'] == 'small']
        self.assertTrue(refreshed)
        self.assertTrue(all(not d['ordinary_review'] and not d['material_research_update'] for d in refreshed))

    def test_configuration_bounds(self):
        for key, value in [('entry_edge', '-.01'), ('exit_edge', '.04'), ('max_trade_usd', 'NaN'), ('max_positions', 0)]:
            with self.assertRaises(ValidationError): validate_policy({**P, key: value})

if __name__ == '__main__': unittest.main()
