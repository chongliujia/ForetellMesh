from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

from foretellmesh.data import strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.mean_reversion import HistoricalBars, features
from foretellmesh.paper_trading import Settlement
from foretellmesh.schema import ValidationError
from foretellmesh.trading_rl_execution import audit, coverage_summary, evaluate_lifetime, load_spec, original_ledger, run
import test_trading_rl as training_fixtures
import test_trading_rl_diagnostic as diagnostic_fixtures
from test_trading_rl_env import AVAILABLE, CONFIG, COST, FREE, D, H, ROOT, T, env_for, fixture, finish

if AVAILABLE:
    from foretellmesh.trading_rl_execution_env import PendingAllocationEnv


def with_prints(data, future):
    raw = {mid: [(t, q, n) for t, q, n in bars if t <= data.start]
                + [(data.start + timedelta(seconds=sec), D(str(q)), 1) for sec, q in future.get(mid, [])]
           for mid, bars in data.feed.data.items()}
    feed = HistoricalBars(raw)
    return replace(data, feed=feed, states=[
        {m.market_id: features(feed, m.market_id, t, CONFIG['environment']) for m in data.windows
         if m.initialized_at <= t <= m.retire_at} for t in data.ticks])


def pending_env(data, labels, ttl=14400, costs=COST, **settings):
    config = {**CONFIG['environment'], **settings}
    return PendingAllocationEnv(data, labels, config, costs, order_ttl_seconds=ttl, record=True)


@unittest.skipUnless(AVAILABLE, 'optional environment dependencies unavailable')
class PendingOrderTests(unittest.TestCase):
    def test_original_300_second_actions_ledger_rewards_and_gym_contract(self):
        from gymnasium.utils.env_checker import check_env
        data, labels = fixture(days=10)
        new = pending_env(data, labels, ttl=300)
        check_env(new, skip_render_check=True)
        old = env_for(data, labels)
        old.reset()
        new.reset()
        self.assertEqual(finish(new, True), finish(old, True))
        self.assertEqual(new.decisions, old.decisions)
        self.assertEqual(original_ledger(new.ledger), old.ledger)
        self.assertEqual(new.curve, old.curve)

    def test_pending_buy_reserves_cash_and_blocks_duplicate_then_fills_once(self):
        data, labels = fixture(days=3)
        data = with_prints(data, {'0': [(9000, .5)]})
        env = pending_env(data, labels)
        env.reset()
        env.step(1)
        self.assertEqual(env.cash, 99)
        self.assertEqual(env._exposure(), 1)
        self.assertEqual(env._mark(env.time)[0], 100)
        self.assertEqual(len(env.pending), 1)
        self.assertTrue(env.mask[0])
        self.assertFalse(env.mask[1:].any())
        env._order(env.time, 'buy', '0', side='yes', budget=D(1))
        self.assertEqual(env.order_id, 1)
        env.step(0)
        env.step(0)
        self.assertEqual(len(env.fills), 1)
        self.assertFalse(env.pending)
        self.assertEqual(env.last_fill, T + timedelta(seconds=9000))
        self.assertEqual(env.cash + env.positions['0']['cost'], 100)

    def test_expiry_releases_reservation_without_counting_a_trade(self):
        data, labels = fixture(days=3)
        data = with_prints(data, {'0': [(18000, .5)]})
        env = pending_env(data, labels)
        env.reset()
        env.step(1)
        for _ in range(3):
            env.step(0)
        self.assertEqual(env.time, T + 4 * H)
        self.assertFalse(env.pending)
        self.assertEqual(env.cash, 100)
        self.assertEqual(env._exposure(), 0)
        self.assertEqual(env.last_fill, T)
        self.assertFalse(env.fills)
        self.assertEqual(env.cancellations['no_post_decision_print'], 1)
        env.step(0)
        self.assertFalse(env.fills)

    def test_reference_on_expiry_is_accepted_but_one_second_later_is_not(self):
        for delay, fills in ((3600, 1), (3601, 0)):
            with self.subTest(delay=delay):
                data, labels = fixture(days=3)
                data = with_prints(data, {'0': [(delay, .5)]})
                env = pending_env(data, labels, ttl=3600)
                env.reset()
                env.step(1)
                self.assertEqual(len(env.fills), fills)
                self.assertFalse(env.pending)

    def test_first_adverse_print_cancels_without_waiting_for_later_good_price(self):
        data, labels = fixture(days=3)
        data = with_prints(data, {'0': [(8100, .7), (9000, .5)]})
        env = pending_env(data, labels)
        env.reset()
        env.step(1)
        env.step(0)
        env.step(0)
        self.assertFalse(env.fills)
        self.assertFalse(env.pending)
        self.assertEqual(env.cash, 100)
        self.assertEqual(env.cancellations['price_limit'], 1)

    def test_pending_position_and_event_limits_cannot_be_overbooked(self):
        data, labels = fixture(days=3, markets=6)
        data = with_prints(data, {})
        env = pending_env(data, labels, max_entries_per_day=8)
        env.reset()
        for mid in ('0', '1', '2', '3'):
            env._order(T, 'buy', mid, side='yes', budget=D(1))
        env._refresh()
        self.assertEqual(len(env.pending), 4)
        self.assertFalse(env.mask[1:].any())
        self.assertEqual(env.lifecycle_summary()['peak_open_or_reserved_positions'], 4)
        with self.assertRaises(ValidationError):
            env._order(T, 'buy', '4', side='yes', budget=D(1))
        self.assertEqual(env.cash, 96)
        other = pending_env(data, labels, max_entries_per_day=8)
        other.reset()
        other._order(T, 'buy', '0', side='yes', budget=D(2))
        other._order(T, 'buy', '1', side='yes', budget=D(2))
        with self.assertRaises(ValidationError):
            other._order(T, 'buy', '2', side='yes', budget=D(2))
        self.assertEqual(other._exposure('g'), 4)

    def test_future_schedule_and_label_are_not_visible_in_pending_snapshot(self):
        data, labels = fixture(days=3)
        a = pending_env(with_prints(data, {'0': [(9000, .5)]}), labels)
        b = pending_env(with_prints(data, {'0': [(12600, .9)]}), [Settlement('0', labels[0].time, 0)])
        a.reset()
        b.reset()
        a.step(1)
        b.step(1)
        self.assertEqual(a.obs.tolist(), b.obs.tolist())
        self.assertEqual(a.action_masks().tolist(), b.action_masks().tolist())
        self.assertEqual(a.pending_snapshot(), b.pending_snapshot())
        self.assertEqual(a.ledger, b.ledger)
        self.assertNotIn('reference_time', a.pending_snapshot()[0])

    def test_partial_exit_replaced_by_full_risk_exit_without_double_sell(self):
        data, labels = fixture(lambda h: .4, days=3)
        data = with_prints(data, {'0': [(60, .4), (10800, .5)]})
        env = pending_env(data, labels, costs=FREE, max_holding_seconds=3600)
        env.reset()
        env.step(2)
        initial_shares = env.positions['0']['shares']
        self.assertTrue(env.mask[5])
        env.step(5)
        self.assertEqual(len(env.pending), 1)
        self.assertFalse(env.mask[5])
        self.assertFalse(env.mask[6])
        self.assertEqual(env.auto['0'], 'holding_limit')
        env.step(0)
        sells = [r for r in env.fills if r['kind'] == 'sell_fill']
        self.assertEqual(len(sells), 1)
        self.assertEqual(D(sells[0]['shares']), initial_shares)
        self.assertTrue(sells[0]['full_exit'])
        self.assertEqual(env.cancellations['replaced_by_risk_exit'], 1)
        self.assertEqual(env.risk_replacements, 1)
        self.assertFalse(env.positions)
        self.assertFalse(env.pending)

    def test_existing_full_exit_is_not_repeated_each_tick(self):
        data, labels = fixture(days=3)
        data = with_prints(data, {'0': [(60, .5), (10800, .5)]})
        env = pending_env(data, labels, max_holding_seconds=3600)
        env.reset()
        env.step(1)
        env.step(6)
        self.assertIsNone(env.auto['0'])
        env.step(0)
        self.assertEqual(sum(r['kind'] == 'sell_order' for r in env.ledger), 1)
        self.assertEqual(sum(r['kind'] == 'sell_fill' for r in env.fills), 1)

    def test_settlement_cancels_pending_exit_and_does_not_count_as_trade(self):
        data, labels = fixture(days=3)
        data = replace(data, windows=[replace(data.windows[0], retire_at=T + 25 * H)])
        data = with_prints(data, {'0': [(60, .5), (24 * 3600, .5), (27 * 3600, .5)]})
        env = pending_env(data, [Settlement('0', T + 26 * H, 1)])
        env.reset()
        env.step(1)
        while env.time < T + 24 * H:
            env.step(0)
        env.step(6)
        env.step(0)
        self.assertEqual(env.time, T + 26 * H)
        self.assertEqual(env.settlement_count, 1)
        self.assertEqual(env.cancellations['settled_before_fill'], 1)
        self.assertFalse(env.pending)
        self.assertFalse(env.positions)
        self.assertEqual(len(env.fills), 1)
        self.assertEqual(env.last_fill, T + timedelta(seconds=60))

    def test_calendar_clips_normal_exit_and_terminal_window_stays_fixed(self):
        data, labels = fixture(days=3)
        data = with_prints(data, {'0': [(60, .5), (70 * 3600, .5), (72 * 3600 + 60, .5)]})
        env = pending_env(data, labels, max_holding_seconds=10 * 86400)
        env.reset()
        env.step(1)
        while env.time < T + 71 * H:
            env.step(0)
        env.step(6)
        self.assertTrue(env.done)
        orders = [r for r in env.ledger if r['kind'] == 'sell_order']
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]['expires_at'], '2025-02-04T00:00:00Z')
        self.assertEqual(orders[1]['reason'], 'calendar_end')
        self.assertEqual(orders[1]['expires_at'], '2025-02-04T00:05:00Z')
        summary = env.summary()
        self.assertEqual(summary['cadence']['filled_transactions_in_period'], 1)
        self.assertAlmostEqual(summary['reward_sum_usd'], float(summary['net_pnl']), places=8)

    def test_partition_and_ttl_restrictions(self):
        data, labels = fixture(days=3)
        for ttl in (True, 0, 7200, 86400):
            with self.assertRaises(ValidationError):
                pending_env(data, labels, ttl=ttl)
        for split in ('validation', 'test'):
            with self.assertRaises(ValidationError):
                evaluate_lifetime(replace(data, partition=split), labels, CONFIG, COST, 300)

    def test_coverage_counts_reference_timing_without_claiming_executability(self):
        data, labels = fixture(days=3, prints=False)
        spec = {'order_ttl_seconds': [300, 3600, 14400]}
        coverage = coverage_summary(data, spec)
        self.assertEqual(coverage['decision_steps'], 72)
        self.assertEqual(coverage['quote_status_counts'], {'at_least_one_ready_quote': 72})
        self.assertEqual(coverage['retrospective_raw_reference_ticks_by_ttl'], {'300': 0, '3600': 72, '14400': 72})
        self.assertFalse(coverage['policy_input'])
        self.assertFalse(coverage['executable_liquidity_proven'])


@unittest.skipUnless(training_fixtures.HAS_RL, 'optional PPO dependencies unavailable')
class ExecutionArtifactTests(unittest.TestCase):
    def setUp(self):
        fixture_case = diagnostic_fixtures.CadenceDiagnosticArtifactTests()
        fixture_case.setUp()
        self.addCleanup(fixture_case.doCleanups)
        self.f = fixture_case
        self.spec = strict_json((ROOT / 'configs/allocation_execution_lifetime_v1.json').read_text())
        self.spec['reference_hashes'] = fixture_case.spec['reference_hashes']
        self.path = fixture_case.dataset / 'execution.json'
        self.path.write_text(json_text(self.spec))

    def test_all_lifetimes_and_costs_replay_without_training_or_heldout_reads(self):
        class FrozenPolicy:
            def predict(self, obs, *, action_masks, deterministic):
                return next(i for i, valid in enumerate(action_masks) if valid), None
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertFalse(path.name.startswith(('validation.', 'test.')), str(path))
            return original(path, *args, **kwargs)
        out = self.f.dataset / 'execution_run'
        with patch.object(Path, 'open', guard), patch('sb3_contrib.MaskablePPO.load', return_value=FrozenPolicy()), \
                patch('sb3_contrib.MaskablePPO.learn', side_effect=AssertionError('must not train')):
            report = run(self.f.dataset, self.f.store, self.f.source, self.path, out)
            self.assertFalse(report['training_performed'])
            self.assertEqual(report['original_300s_replays_matched'], 15)
            self.assertEqual(audit(out)['arms_reproduced'], 45)
            target = out / 'cost_assumption/14400/ppo_7.pending.jsonl'
            target.write_text(target.read_text() + '{}\n')
            with self.assertRaises(ValidationError):
                audit(out)

    def test_spec_does_not_allow_extra_windows_or_other_partitions(self):
        for key, value in [('partition', 'validation'), ('order_ttl_seconds', [300, 3600, 7200]),
                           ('closeout_ttl_seconds', 14400), ('training_performed', True)]:
            spec = deepcopy(self.spec)
            spec[key] = value
            self.path.write_text(json_text(spec))
            with self.assertRaises(ValidationError):
                load_spec(self.f.source, self.path)


class ExecutionCostTests(unittest.TestCase):
    def setUp(self):
        self.decompose = runpy.run_path(str(ROOT / 'scripts/summarize_execution_costs.py'))['decompose']
        self.buy = {'kind': 'buy_fill', 'shares': '4', 'price': '.51', 'fee': '.0204', 'cost': '2.0604'}

    def test_same_fill_decomposition_handles_partial_exits_and_rejects_bad_fees(self):
        ledger = [self.buy,
                  {'kind': 'sell_fill', 'shares': '2', 'price': '.59', 'fee': '.0118', 'proceeds': '1.1682'},
                  {'kind': 'sell_fill', 'shares': '2', 'price': '.54', 'fee': '.0108', 'proceeds': '1.0692'}]
        metrics = {'initial_cash': '100', 'final_cash': '100.1770', 'fees_paid': '.0430', 'net_pnl': '.1770'}
        result = self.decompose(ledger, metrics, D('.01'))
        self.assertEqual(D(result['same_fill_raw_price_pnl_usd']), D('.3'))
        self.assertEqual(D(result['price_premium_drag_usd']), D('.08'))
        with self.assertRaises(ValidationError):
            self.decompose(ledger, {**metrics, 'fees_paid': '.01'}, D('.01'))

    def test_settlement_payout_is_not_an_extra_trading_fee(self):
        metrics = {'initial_cash': '100', 'final_cash': '101.9396', 'fees_paid': '.0204', 'net_pnl': '1.9396'}
        result = self.decompose([self.buy, {'kind': 'settle', 'payout': '4'}], metrics, D('.01'))
        self.assertEqual(D(result['same_fill_raw_price_pnl_usd']), 2)
        self.assertEqual(D(result['price_premium_drag_usd']), D('.04'))
        self.assertEqual(D(result['fees_usd']), D('.0204'))


if __name__ == '__main__':
    unittest.main()
