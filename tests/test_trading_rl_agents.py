from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foretellmesh.data import strict_json
from foretellmesh.schema import ValidationError, iso
from foretellmesh.trading_rl import runtime
from foretellmesh.trading_rl_agents import aggregates, evaluate_one, replay, train_models
from foretellmesh.trading_rl_agents_env import AGENT_FEATURES, AgentAllocationEnv, SignalIndex, signal_rows
from foretellmesh.trading_rl_profit_env import ProfitAllocationEnv
from foretellmesh.evaluation import json_text
from test_trading_rl_env import CONFIG, COST, T, fixture


def response(t=T, mid='0', p=.7, seconds=30, status='completed'):
    return {'sample_id': f'{mid}:{iso(t)}', 'market_id': mid, 'observation_time': iso(t),
        'seconds': seconds, 'feature_seconds': 0,
        'result': {'status': status, 'prediction': {'probability': p, 'confidence': 'medium',
            'unknowns': ['future release'], 'observation_time': iso(t)},
            'stages': {'market_quant': {'market_view': 'no_independent_edge'},
                       'game_theory': {'market_view': 'mixed'}}}}


def index_for(rr=None):
    rr = rr or [response()]
    return SignalIndex(signal_rows(rr, 604800, reference_probabilities={r['sample_id']: .5 for r in rr}), {'0', '1', '2', '3'})


class AgentAllocationTests(unittest.TestCase):
    def test_latency_queue_ttl_and_failed_refresh(self):
        rr = [response(seconds=30), response(mid='1', seconds=45)]
        index = index_for(rr)
        self.assertIsNone(index.at('0', T))
        self.assertIsNotNone(index.at('0', T + timedelta(seconds=30)))
        self.assertIsNone(index.at('1', T + timedelta(seconds=74)))
        self.assertIsNotNone(index.at('1', T + timedelta(seconds=75)))
        self.assertIsNone(index.at('0', T + timedelta(days=7)))
        rr = [response(), response(t=T + timedelta(hours=1), status='failed')]
        index = index_for(rr)
        self.assertIsNotNone(index.at('0', T + timedelta(hours=1)))
        self.assertIsNone(index.at('0', T + timedelta(hours=1, seconds=30)))

    def test_invalid_identity_probability_and_time_rejected(self):
        for rr in ([response(), response()], [response(p=float('nan'))], [response(p=1.01)],
                   [response(seconds=-1)], [response(t=T + timedelta(hours=1)), response()]):
            with self.assertRaises(ValidationError):
                index_for(rr)
        with self.assertRaises(ValidationError):
            SignalIndex(signal_rows([response(mid='99')], 3600), {'0'})
        rec = signal_rows([response()], 3600)
        rec[0]['available_at'] = iso(T)
        with self.assertRaises(ValidationError):
            SignalIndex(rec, {'0'})
        bad = response(seconds=-1); bad['feature_seconds'] = 30
        with self.assertRaises(ValidationError):
            index_for([bad])

    def test_content_ablation_preserves_price_input_masks_and_cash_ledger(self):
        data, labels = fixture(days=3, markets=4)
        index = index_for()
        a = AgentAllocationEnv(data, labels, CONFIG['environment'], COST, signals=index, content_enabled=True, record=True)
        b = AgentAllocationEnv(data, labels, CONFIG['environment'], COST, signals=index, content_enabled=False, record=True)
        c = ProfitAllocationEnv(data, labels, CONFIG['environment'], COST, record=True)
        for env in (a, b, c):
            env.reset()
        self.assertEqual(a.observation_space.shape, (88 + 4 * len(AGENT_FEATURES),))
        np.testing.assert_array_equal(a.obs, b.obs)
        for i in range(72):
            np.testing.assert_array_equal(a.mask, b.mask); np.testing.assert_array_equal(a.mask, c.mask)
            np.testing.assert_array_equal(a.obs[:88], c.obs)
            np.testing.assert_array_equal(a.obs[:88], b.obs[:88])
            if i == 1:
                self.assertEqual(a.obs[88], 1)
                np.testing.assert_array_equal(a.obs[88:90], b.obs[88:90])
                self.assertAlmostEqual(a.obs[90], .7)
                self.assertTrue(np.all(b.obs[90:88 + len(AGENT_FEATURES)] == 0))
            for env in (a, b, c):
                env.step(0)
        self.assertEqual(a.summary()['final_cash'], '100')
        self.assertEqual(a.ledger, b.ledger); self.assertEqual(a.ledger, c.ledger)
        self.assertGreater(a.summary()['decision_ticks_with_signal'], 0)
        self.assertTrue(all(d['mask'][0] for d in a.decisions))

    def test_future_output_and_future_outcome_do_not_change_prior_observations(self):
        data, labels = fixture(days=3)
        rr = [response(), response(t=T + timedelta(hours=10), p=.1)]
        other = deepcopy(rr); other[1]['result']['prediction']['probability'] = .9
        from dataclasses import replace
        aa = []
        for results, ll in [(rr, labels), (other, [replace(s, outcome=1 - s.outcome) for s in labels])]:
            env = AgentAllocationEnv(data, ll, CONFIG['environment'], COST, signals=index_for(results), content_enabled=True)
            env.reset(); observed = []
            for _ in range(10):
                observed.append((env.obs.tolist(), env.mask.tolist())); env.step(0)
            aa.append(observed)
        self.assertEqual(*aa)

    def test_market_anchor_contains_only_the_original_price_not_agent_forecast(self):
        index = index_for(); now = T + timedelta(hours=1)
        agent, _ = index.features('0', now, .55, content_enabled=True)
        anchor, _ = index.features('0', now, .55, content_enabled=False, anchor_only=True)
        control, _ = index.features('0', now, .55, content_enabled=False)
        self.assertEqual(agent[:2], anchor[:2]); self.assertEqual(anchor[:2], control[:2])
        self.assertAlmostEqual(agent[2], .7); self.assertAlmostEqual(anchor[2], .5)
        self.assertAlmostEqual(anchor[3], -.05)
        self.assertTrue(all(x == 0 for x in anchor[4:]))
        changed = index_for([response(p=.1)])
        self.assertEqual(anchor, changed.features('0', now, .55, content_enabled=False, anchor_only=True)[0])

    def test_invalid_and_terminal_actions_do_not_mutate_signal_counters(self):
        data, labels = fixture(days=2)
        env = AgentAllocationEnv(data, labels, CONFIG['environment'], COST, signals=index_for(), content_enabled=True)
        env.reset()
        for action in (-1, env.action_space.n):
            with self.assertRaises(ValidationError):
                env.step(action)
            self.assertEqual(env.total_slots, 0)
        while not env.done:
            env.step(0)
        before = env.summary()
        with self.assertRaises(ValidationError):
            env.step(0)
        self.assertEqual(before, env.summary())

    def test_real_optimizer_paired_families_and_frozen_replay(self):
        spec = strict_json((Path(__file__).resolve().parents[1] / 'configs/allocation_multi_agent_v1.json').read_text())
        spec['seeds'] = [7]; spec['ppo'].update(total_timesteps=192, n_steps=64, batch_size=32)
        spec['scenarios'] = {'cost_assumption': spec['scenarios']['cost_assumption'],
                             'fee_zero': spec['scenarios']['fee_zero'], 'fee_high': spec['scenarios']['fee_high']}
        data, labels = fixture(days=2); index = index_for(); runtime(CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = train_models(root, data, labels, CONFIG, spec, index)
            self.assertEqual(trained['price_control_7']['initial_parameters_sha256'],
                             trained['multi_agent_7']['initial_parameters_sha256'])
            self.assertNotEqual(trained['price_control_7']['trained_parameters_sha256'],
                                trained['multi_agent_7']['trained_parameters_sha256'])
            for r in trained.values():
                self.assertGreater(r['parameter_update_l2'], 0)
                self.assertEqual(r['timesteps_by_fee_fraction'], {'0.01': 96, '0': 48, '0.02': 48})
            (root / 'training_report.json').write_text(json_text({'models': trained}))
            values = replay(root, data, labels, CONFIG, spec, index)
            self.assertEqual(values, replay(root, data, labels, CONFIG, spec, index, reproduce=True))
            self.assertEqual(len(aggregates(values, spec)), 3)
            for scenario in values.values():
                for arm, r in scenario['3600'].items():
                    m = r['metrics']
                    self.assertFalse(m['periodic_activity_required'])
                    self.assertAlmostEqual(m['reward_sum_usd'], float(m['final_cash']) - 100)
            path = root / 'cost_assumption/3600/multi_agent_7.decisions.jsonl'
            path.write_text(path.read_text() + '{}\n')
            with self.assertRaises(ValidationError):
                replay(root, data, labels, CONFIG, spec, index, reproduce=True)


if __name__ == '__main__':
    unittest.main()
