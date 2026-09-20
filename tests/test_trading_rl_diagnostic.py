from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.paper_trading import Settlement
from foretellmesh.schema import ValidationError
from foretellmesh.trading_rl import freeze_data
from foretellmesh.trading_rl_data import prepare
from foretellmesh.trading_rl_diagnostic import (
    CORE_SOURCE, audit, decision_diagnostic, diagnose, reference_config, run,
)
import test_trading_rl as training_fixtures
from test_trading_rl_env import AVAILABLE, CONFIG, D, ROOT, T, env_for, fixture, finish


@unittest.skipUnless(AVAILABLE, 'optional environment dependencies unavailable')
class CadenceDiagnosticTests(unittest.TestCase):
    def test_diagnostic_does_not_change_actions_fills_or_reward(self):
        data, labels = fixture(days=16)
        expected = env_for(data, labels)
        expected.reset()
        finish(expected, weekly_only=True)
        got = diagnose(data, labels, CONFIG, weekly_only=True)
        self.assertEqual(got['summary']['metrics'], expected.summary())
        self.assertEqual(got['decisions'], expected.decisions)
        self.assertEqual(got['ledger'], expected.ledger)
        self.assertGreater(got['summary']['due_decisions']['decision_count'], 0)

    def test_next_hour_print_is_not_a_five_minute_fill(self):
        data, labels = fixture(days=3, prints=False)
        env = env_for(data, labels)
        env.reset()
        row = decision_diagnostic(env, 1)
        self.assertEqual(row['retrospective']['category'], 'legal_actions_but_no_acceptable_reference')
        self.assertEqual(row['causal']['visible_legal_actions'], 4)
        for offer in row['retrospective']['offers']:
            self.assertEqual(offer['status'], 'no_post_decision_print')
            self.assertEqual(offer['next_recorded_print_delay_seconds'], 3600)
        env.step(1)
        self.assertEqual(env.cancellations['no_post_decision_print'], 1)
        self.assertFalse(env.fills)

    def test_first_adverse_print_not_later_favorable_one(self):
        data, labels = fixture(lambda h: .4 if h == 0 else .5, days=3)
        raw = [(t, D('.49') if t == T + timedelta(seconds=60) else q, n) for t, q, n in data.feed.data['0']]
        raw.append((T + timedelta(seconds=120), D('.4'), 1))
        data = replace(data, feed=HistoricalBars({'0': raw}))
        env = env_for(data, labels)
        env.reset()
        row = decision_diagnostic(env, 1)
        offer = next(o for o in row['retrospective']['offers'] if o['op'] == 'yes_1')
        self.assertEqual(offer['status'], 'price_limit')
        self.assertEqual(offer['reference_delay_seconds'], 60)
        self.assertEqual(row['retrospective']['category'], 'visible_alternative_reference_available')
        env.step(1)
        self.assertEqual(env.cancellations['price_limit'], 1)

    def test_four_slots_can_exclude_only_acceptable_reference(self):
        data, labels = fixture(days=3, markets=5)
        raw = {mid: [(t, q, n) for t, q, n in bars if mid == '4' or t <= T or (t.minute == 0 and t.second == 0)]
               for mid, bars in data.feed.data.items()}
        data = replace(data, feed=HistoricalBars(raw))
        env = env_for(data, labels)
        env.reset()
        row = decision_diagnostic(env, 1)
        self.assertEqual(row['causal']['slots'], ['0', '1', '2', '3'])
        self.assertEqual(row['causal']['all_legal_actions'], 20)
        self.assertEqual(row['causal']['visible_legal_actions'], 16)
        self.assertEqual(row['retrospective']['category'], 'truncated_alternative_reference_available')
        self.assertEqual({o['market_id'] for o in row['retrospective']['offers'] if o['status'] == 'reference_acceptable'}, {'4'})

    def test_future_prices_only_change_retrospective_output(self):
        data, labels = fixture(days=3)
        other = replace(data, feed=HistoricalBars({'0': [(t, D('.9') if t > T else q, n) for t, q, n in data.feed.data['0']]}))
        a, b = env_for(data, labels), env_for(other, [Settlement('0', labels[0].time, 0)])
        a.reset()
        b.reset()
        before = deepcopy((a.cash, a.positions, a.attempts, a.events, a.ledger))
        first, second = decision_diagnostic(a, 1), decision_diagnostic(b, 1)
        self.assertEqual(first['causal'], second['causal'])
        self.assertNotEqual(first['retrospective'], second['retrospective'])
        self.assertEqual(before, (a.cash, a.positions, a.attempts, a.events, a.ledger))

    def test_weekly_tail_rejection_is_not_reported_as_missing_print(self):
        data, labels = fixture(lambda h: .995, days=10)
        env = env_for(data, labels)
        env.reset()
        for _ in range(144):
            env.step(0)
        row = decision_diagnostic(env, 0)
        self.assertTrue(row['causal']['weekly_due'])
        self.assertEqual(row['retrospective']['category'], 'no_legal_action')
        gates = row['causal']['markets'][0]['buy_blockers']
        self.assertIn('all_in_unit_cost_ge_1', gates['yes_1'])
        self.assertIn('weekly_higher_price_side_only', gates['no_1'])
        self.assertEqual(row['retrospective']['offers'], [])

    def test_automatic_exit_cancellation_uses_same_bid_rule(self):
        data, labels = fixture(lambda h: .1 if h <= 0 else .001, days=3)
        env = env_for(data, labels)
        env.reset()
        env.step(1)
        row = decision_diagnostic(env, 0)
        automatic = [o for o in row['retrospective']['offers'] if o['automatic']]
        self.assertEqual(len(automatic), 1)
        self.assertEqual(automatic[0]['status'], 'nonpositive_synthetic_bid')
        self.assertEqual(row['causal']['automatic_exit_reasons'], {'0': 'stop_loss'})
        env.step(0)
        self.assertEqual(env.cancellations['nonpositive_synthetic_bid'], 1)

    def test_no_trade_exact_endpoint_still_fails_with_zero_overdue_duration(self):
        data, labels = fixture(days=7, prints=False)
        result = diagnose(data, labels, CONFIG, weekly_only=True)['summary']
        self.assertFalse(result['metrics']['cadence']['compliant'])
        self.assertEqual(len(result['violations']), 1)
        self.assertEqual(result['total_overdue_hours'], 0)
        self.assertEqual(result['violations'][0]['warning_reference_actions_arriving_by_deadline'], 0)

    def test_training_only_and_invalid_action_rejected(self):
        data, labels = fixture(days=3)
        for partition in ('validation', 'test'):
            with self.assertRaises(ValidationError):
                diagnose(replace(data, partition=partition), labels, CONFIG)
        env = env_for(data, labels)
        env.reset()
        with self.assertRaises(ValidationError):
            decision_diagnostic(env, 5)


@unittest.skipUnless(training_fixtures.HAS_RL, 'optional PPO dependencies unavailable')
class CadenceDiagnosticArtifactTests(unittest.TestCase):
    def setUp(self):
        # Reuse an admitted artificial dataset; create fake frozen checkpoint
        # files and stub only model loading. No optimizer or new training runs.
        fixture_case = training_fixtures.AllocationDataTests()
        fixture_case.setUp()
        self.addCleanup(fixture_case.doCleanups)
        self.dataset, self.store = fixture_case.root, fixture_case.store
        self.config = deepcopy(fixture_case.c)
        self.config['seeds'] = [7, 17, 27]
        self.source = self.dataset / 'source_run'
        self.source.mkdir()
        (self.source / 'config.json').write_text(json_text(self.config))
        source_dir = ROOT / 'src/foretellmesh'
        shutil.copytree(source_dir, self.source / 'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
        data = prepare(self.dataset, self.store, 'train', self.config)
        freeze_data(self.source, data)
        trained = {'seeds': {}, 'artifact_hashes': {}}
        for seed in self.config['seeds']:
            dest = self.source / f'seed_{seed}'
            dest.mkdir()
            (dest / 'trained.zip').write_bytes(b'fixture only, not a PPO model')
            trained['seeds'][str(seed)] = {'trained_sha256': sha256_file(dest / 'trained.zip')}
        for name in CORE_SOURCE:
            key = 'source_snapshot/foretellmesh/' + name
            trained['artifact_hashes'][key] = sha256_file(self.source / key)
        (self.source / 'training_report.json').write_text(json_text(trained))
        (self.source / 'report.json').write_text('{}\n')
        self.spec = strict_json((ROOT / 'configs/allocation_cadence_diagnostic_v1.json').read_text())
        self.spec['reference_hashes'] = {name: sha256_file(self.source / name) for name in self.spec['reference_hashes']}
        self.spec_path = self.dataset / 'diagnostic.json'
        self.spec_path.write_text(json_text(self.spec))

    def test_run_audit_never_reads_validation_test_or_trains(self):
        class FrozenPolicy:
            def predict(self, obs, *, action_masks, deterministic):
                return (next(i for i, ok in enumerate(action_masks) if ok), None)
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertFalse(path.name.startswith(('validation.', 'test.')), str(path))
            return original(path, *args, **kwargs)
        out = self.dataset / 'diagnostic_run'
        with patch.object(Path, 'open', guard), patch('sb3_contrib.MaskablePPO.load', return_value=FrozenPolicy()), \
                patch('sb3_contrib.MaskablePPO.learn', side_effect=AssertionError('diagnostic must not train')):
            report = run(self.dataset, self.store, self.source, self.spec_path, out)
            self.assertEqual(len(report['arms']), 5)
            self.assertFalse(report['training_performed'])
            self.assertFalse(report['validation_replayed'])
            self.assertFalse(report['final_test_opened'])
            self.assertEqual(audit(out)['arms_reproduced'], 5)
            with self.assertRaises(ValidationError):
                run(self.dataset, self.store, self.source, self.spec_path, out)
            path = out / 'ppo_7.steps.jsonl'
            path.write_text(path.read_text() + '{}\n')
            with self.assertRaises(ValidationError):
                audit(out)

    def test_scope_and_checkpoint_tampering_rejected(self):
        changed = deepcopy(self.spec)
        changed['partition'] = 'validation'
        with self.assertRaises(ValidationError):
            reference_config(self.source, changed)
        (self.source / 'seed_7/trained.zip').write_bytes(b'changed')
        with self.assertRaises(ValidationError):
            reference_config(self.source, self.spec)


if __name__ == '__main__':
    unittest.main()
