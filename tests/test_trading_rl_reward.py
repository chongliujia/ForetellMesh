from dataclasses import replace
from datetime import timedelta
import unittest

from foretellmesh.schema import ValidationError, iso
from test_trading_rl_env import AVAILABLE, CONFIG, COST, D, H, T, fixture

if AVAILABLE:
    from foretellmesh.trading_rl_reward import CadenceRewardEnv, overdue_seconds
    from foretellmesh.trading_rl_execution_env import PendingAllocationEnv


def fill(hour, kind='buy_fill'):
    return {'time': iso(T + hour * H), 'kind': kind}


@unittest.skipUnless(AVAILABLE, 'optional environment dependencies unavailable')
class CadenceRewardTests(unittest.TestCase):
    def test_overdue_time_is_integral_not_repeated_total_lateness(self):
        self.assertEqual(overdue_seconds(T, T + 168 * H, T, []), 0)
        self.assertEqual(overdue_seconds(T, T + 170 * H, T, []), 7200)
        self.assertEqual(sum(overdue_seconds(T + h * H, T + (h + 1) * H, T, []) for h in range(170)), 7200)
        self.assertEqual(overdue_seconds(T + 169 * H, T + 170 * H, T, []), 3600)

    def test_fills_reset_at_exact_times_including_multiple_fills_and_deadline(self):
        self.assertEqual(overdue_seconds(T, T + 336 * H, T, [fill(168)]), 0)
        self.assertEqual(overdue_seconds(T, T + 340 * H, T, [fill(170), fill(338.5, 'sell_fill')]), 9000)
        self.assertEqual(overdue_seconds(T + 169 * H, T + 170 * H, T, [fill(169.5)]), 1800)
        # A fill at the left boundary resets before this interval begins accruing cost.
        self.assertEqual(overdue_seconds(T + 169 * H, T + 170 * H, T, [fill(169)]), 0)

    def test_orders_cancels_settlements_and_post_calendar_fills_do_not_reset(self):
        ignored = [fill(168, k) for k in ('buy_order', 'sell_order', 'cancel', 'settle')]
        ignored += [fill(-1), fill(171)]
        self.assertEqual(overdue_seconds(T, T + 170 * H, T, ignored), 7200)

    def test_microseconds_and_invalid_intervals(self):
        self.assertEqual(overdue_seconds(T, T + 168 * H + timedelta(microseconds=1), T, []), D('.000001'))
        for start, end, anchor, cadence in ((T, T-H, T, 604800), (T, T+H, T+H, 604800),
                                          (T, T+H, T, 0), (T, T+H, T, True),
                                          (T.replace(tzinfo=None), T, T, 604800)):
            with self.assertRaises(ValidationError):
                overdue_seconds(start, end, anchor, [], cadence)

    def test_zero_weight_preserves_economic_replay_and_learning_reward(self):
        data, labels = fixture(days=10)
        old = PendingAllocationEnv(data, labels, CONFIG['environment'], COST, order_ttl_seconds=3600, record=True)
        new = CadenceRewardEnv(data, labels, CONFIG['environment'], COST, order_ttl_seconds=3600, record=True)
        old.reset(); new.reset()
        while not old.done:
            action = old.fixed_action(weekly_only=True)
            a, b = old.step(action), new.step(action)
            self.assertEqual(a[0].tolist(), b[0].tolist())
            self.assertEqual(a[1:4], b[1:4])
        self.assertEqual(old.summary(), new.summary())
        self.assertEqual(old.ledger, new.ledger)
        self.assertEqual(old.decisions, new.decisions)
        self.assertEqual(new.reward_summary()['cadence_penalty_usd'], '0')

    def test_penalty_never_changes_cash_and_episode_reset_is_explicit(self):
        data, labels = fixture(days=10)
        env = CadenceRewardEnv(data, labels, CONFIG['environment'], COST, order_ttl_seconds=3600,
                               overdue_usd_per_hour='0.10', record=True)
        env.reset()
        total = 0
        while not env.done:
            # Explicitly refusing the cadence guard exercises constraint cost. Invalid actions cannot fill.
            _, reward, _, _, info = env.step(0)
            total += reward
        self.assertEqual(env.cash, 100)
        self.assertFalse(env.fills)
        self.assertEqual(env.reward_total, 0)
        self.assertAlmostEqual(total, -7.2)
        self.assertEqual(D(info['episode_summary']['learning_reward']['cadence_penalty_usd']), D('7.2'))
        self.assertEqual(D(env.reward_summary()['overdue_hours']), 72)
        self.assertFalse(env.summary()['policy_requirements_met'])
        env.reset()
        self.assertEqual(env.late_seconds_total, 0)
        self.assertEqual(env.penalty_total, 0)
        self.assertEqual(env.training_reward_total, 0)

    def test_exact_week_endpoint_still_fails_hard_gate_with_zero_integrated_cost(self):
        data, labels = fixture(days=7)
        env = CadenceRewardEnv(data, labels, CONFIG['environment'], COST, order_ttl_seconds=3600, overdue_usd_per_hour='0.10')
        env.reset()
        while not env.done:
            env.step(0)
        self.assertEqual(env.late_seconds_total, 0)
        self.assertFalse(env.summary()['policy_requirements_met'])

    def test_reward_scope_and_nonfinite_or_negative_weight_rejected(self):
        data, labels = fixture(days=3)
        for value in ('-1', 'NaN', 'Infinity'):
            with self.assertRaises(ValidationError):
                CadenceRewardEnv(data, labels, CONFIG['environment'], COST, overdue_usd_per_hour=value)
        for partition in ('validation', 'test'):
            with self.assertRaises(ValidationError):
                CadenceRewardEnv(replace(data, partition=partition), labels, CONFIG['environment'], COST, order_ttl_seconds=3600)


if __name__ == '__main__':
    unittest.main()
