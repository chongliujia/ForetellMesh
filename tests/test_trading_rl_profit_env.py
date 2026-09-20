from dataclasses import replace
import unittest

from foretellmesh.schema import ValidationError
from foretellmesh.mean_reversion import HistoricalBars, features
from foretellmesh.paper_trading import Settlement
from test_trading_rl_env import AVAILABLE, CONFIG, COST, D, H, fixture

if AVAILABLE:
    from foretellmesh.trading_rl_profit_env import ProfitAllocationEnv, observation_schema


def env_for(data, labels, costs=COST, **kwargs):
    return ProfitAllocationEnv(data, labels, CONFIG['environment'], costs, record=True, **kwargs)


@unittest.skipUnless(AVAILABLE, 'optional environment dependencies unavailable')
class ProfitEnvTests(unittest.TestCase):
    def test_wait_can_keep_cash_indefinitely_without_activity_penalty_or_orders(self):
        data, labels = fixture(days=16)
        env = env_for(data, labels)
        env.reset()
        while not env.done:
            self.assertTrue(env.action_masks()[0])
            self.assertFalse(env.due)
            _, reward, _, _, info = env.step(0)
            self.assertEqual(reward, 0)
            self.assertNotIn('cadence_due', info)
        summary = env.summary()
        self.assertEqual(env.cash, 100)
        self.assertFalse(env.ledger)
        self.assertTrue(summary['policy_requirements_met'])
        self.assertFalse(summary['outperformed_cash'])
        self.assertFalse(summary['periodic_activity_required'])
        self.assertNotIn('cadence', summary)
        self.assertEqual(summary['activity_penalty_usd'], '0')
        self.assertTrue(all(r['mask'][0] and not r['weekly_due'] for r in env.decisions))

    def test_explicit_fee_and_premium_inputs_and_no_weekly_side_or_size_restriction(self):
        data, labels = fixture(days=16)
        env = env_for(data, labels, {'fee_fraction': '.015', 'entry_price_premium': '.02'})
        env.reset()
        for _ in range(8 * 24):
            env.step(0)
        self.assertEqual(env.obs[3:5].tolist(), [1.5, 2.0])
        self.assertTrue(env.action_masks()[1:5].all())
        self.assertTrue(env.observation_space.contains(env.obs))
        schema = observation_schema()
        self.assertEqual(schema['dimension'], 88)
        self.assertFalse(schema['cadence_features_present'])
        self.assertNotIn('hours_since_fill', schema['global'])

    def test_fee_charged_once_per_fill_and_reward_telescopes_without_extra_deduction(self):
        cash = []
        for rate in ('0', '.01', '.02'):
            data, labels = fixture(days=3)
            env = env_for(data, labels, {'fee_fraction': rate, 'entry_price_premium': '0'})
            env.reset()
            rewards = [env.step(1)[1], env.step(6)[1]]
            while not env.done:
                rewards.append(env.step(0)[1])
            fills = [r for r in env.ledger if r['kind'] in ('buy_fill', 'sell_fill')]
            self.assertEqual(len(fills), 2)
            self.assertEqual(sum(D(r['fee']) for r in fills), env.fees)
            self.assertEqual(env.cash, D(100) - D(fills[0]['cost']) + D(fills[1]['proceeds']))
            self.assertAlmostEqual(sum(rewards), float(env.cash - 100), places=7)
            cash.append(env.cash)
        self.assertEqual(cash[0], 100)
        self.assertGreater(cash[0], cash[1])
        self.assertGreater(cash[1], cash[2])

    def test_cost_curriculum_is_visible_before_action_and_reproducible_on_seeded_reset(self):
        data, labels = fixture(days=3)
        schedule = [COST, {**COST, 'fee_fraction': '0'}, {**COST, 'fee_fraction': '.02'}]
        env = env_for(data, labels, cost_schedule=schedule)
        self.assertEqual(env.reset(seed=7)[0][3], 1)
        self.assertEqual(env.step(0)[4]['execution_costs']['fee_fraction'], '0.01')
        self.assertEqual(env.reset()[0][3], 0)
        self.assertEqual(env.reset()[0][3], 2)
        self.assertEqual(env.reset()[0][3], 1)
        self.assertEqual(env.reset(seed=7)[0][3], 1)
        with self.assertRaises(ValidationError):
            env_for(data, labels, cost_schedule=[])
        with self.assertRaises(ValidationError):
            env_for(data, labels, cost_schedule=schedule, legacy_observation=True)

    def test_legacy_observations_preserve_meaning_without_reenabling_cadence(self):
        data, labels = fixture(days=16)
        current = env_for(data, labels)
        legacy = env_for(data, labels, legacy_observation=True)
        current.reset(); legacy.reset()
        for _ in range(168):
            self.assertEqual(current.action_masks().tolist(), legacy.action_masks().tolist())
            current.step(0); legacy.step(0)
        self.assertEqual(legacy.obs[3:5].tolist(), [1, 0])
        self.assertEqual(current.obs[3:5].tolist(), [1, 1])
        self.assertEqual(current.obs[:3].tolist(), legacy.obs[:3].tolist())
        self.assertEqual(current.obs[5:].tolist(), legacy.obs[5:].tolist())
        self.assertTrue(legacy.mask[0])

    def test_risk_exits_still_work_without_forced_participation(self):
        data, labels = fixture(days=5)
        env = env_for(data, labels)
        env.reset()
        env.step(1)
        while not env.done:
            env.step(0)
        self.assertEqual(env.summary()['entry_count'], 1)
        self.assertEqual(env.summary()['exit_count'], 1)
        sells = [r for r in env.ledger if r['kind'] == 'sell_fill']
        self.assertEqual(sells[0]['exit_reason'], 'holding_limit')
        self.assertFalse(any(r.get('reason') == 'weekly' for r in env.ledger))

    def test_future_prices_and_resolution_cannot_change_fee_conditioned_decisions(self):
        data, labels = fixture(days=3)
        feed = HistoricalBars({'0': [(t, D('.9') if t > data.start + 10 * H else q, n)
                                    for t, q, n in data.feed.data['0']]})
        future = replace(data, feed=feed, states=[{'0': features(feed, '0', t, CONFIG['environment'])} for t in data.ticks])
        a = env_for(data, labels)
        b = env_for(future, [Settlement('0', labels[0].time, 0)])
        a.reset(); b.reset()
        for h in range(10):
            self.assertEqual(a.obs.tolist(), b.obs.tolist())
            self.assertEqual(a.mask.tolist(), b.mask.tolist())
            a.step(1 if h == 0 else 0); b.step(1 if h == 0 else 0)
        self.assertEqual(a.ledger, b.ledger)
        self.assertEqual(a.decisions, b.decisions)

    def test_gym_contract_and_scope_cost_guards(self):
        from gymnasium.utils.env_checker import check_env
        data, labels = fixture(days=3)
        check_env(env_for(data, labels), skip_render_check=True)
        for costs in ({'fee_fraction': '0'}, {**COST, 'fee_fraction': 'NaN'},
                      {**COST, 'fee_fraction': '-.01'}, {**COST, 'fee_fraction': '.051'},
                      {**COST, 'entry_price_premium': '.06'}):
            with self.assertRaises(ValidationError):
                env_for(data, labels, costs)
        for partition in ('validation', 'test'):
            with self.assertRaises(ValidationError):
                env_for(replace(data, partition=partition), labels)
        env = env_for(data, labels)
        env.reset()
        with self.assertRaises(ValidationError):
            env._order(env.time, 'buy', '0', side='yes', budget=D(1), reason='weekly')


if __name__ == '__main__':
    unittest.main()
