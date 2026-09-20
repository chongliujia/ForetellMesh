from copy import deepcopy
from pathlib import Path
import runpy
import shutil
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError
from foretellmesh.trading_rl import artifacts, runtime
import foretellmesh.trading_rl_profit as profit
import test_trading_rl as fixtures
import test_trading_rl_retrain as prior_fixtures


@unittest.skipUnless(fixtures.HAS_RL, 'optional masked PPO dependencies unavailable')
class ProfitExperimentTests(unittest.TestCase):
    def setUp(self):
        f = prior_fixtures.RetrainTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.reference = f.f.root / 'reference'
        snapshot = self.reference / 'source_snapshot/foretellmesh'
        snapshot.mkdir(parents=True)
        for name in profit.SOURCE_FILES[2:]:
            shutil.copyfile(Path(profit.__file__).with_name(name), snapshot / name)
        for name in ('train.catalog.jsonl', 'train.features.jsonl'):
            shutil.copyfile(f.source / name, self.reference / name)
        shutil.copyfile(f.config, self.reference / 'config.json')
        models = {}
        for seed in (7, 17, 27):
            name = f'net_equity_{seed}'
            dest = self.reference / 'models' / name
            dest.mkdir(parents=True)
            shutil.copyfile(f.source / f'seed_{seed}/trained.zip', dest / 'trained.zip')
            models[name] = {'trained_sha256': sha256_file(dest / 'trained.zip')}
        (self.reference / 'training_report.json').write_text(json_text({'models': models}))
        (self.reference / 'plan.json').write_text(json_text({'paths': {'source': str(f.source), 'execution': str(f.execution)}}))
        (self.reference / 'report.json').write_text(json_text({'status': 'completed', 'partition': 'train',
            'plan_sha256': sha256_file(self.reference / 'plan.json'), 'artifact_hashes': artifacts(self.reference)}))
        (self.reference / 'audit.json').write_text(json_text({'status': 'passed', 'report_sha256': sha256_file(self.reference / 'report.json')}))
        self.spec = strict_json((fixtures.ROOT / 'configs/allocation_profit_fee_v1.json').read_text())
        self.spec['reference_hashes'] = {name: sha256_file(self.reference / name) for name in self.spec['reference_hashes']}
        self.spec['ppo'] = {**f.c['ppo'], 'gamma': 1.0}
        self.config = f.f.root / 'profit.json'
        self.config.write_text(json_text(self.spec))

    def test_real_training_cash_control_replay_fee_decomposition_and_tamper(self):
        output = self.f.f.root / 'profit_run'
        original = Path.open
        evaluate = profit.evaluate_one

        def guard(path, *args, **kwargs):
            self.assertFalse(path.name.startswith(('validation.', 'test.')), str(path))
            return original(path, *args, **kwargs)

        def frozen_first(*args, **kwargs):
            trained = strict_json((output / 'training_report.json').read_text())
            self.assertEqual(len(trained['models']), 6)
            self.assertTrue((output / 'evaluation_plan.json').exists())
            return evaluate(*args, **kwargs)

        with patch.object(Path, 'open', guard), patch.object(profit, 'evaluate_one', frozen_first):
            report = profit.run(self.f.f.root, self.f.f.store, self.reference, self.config, output)
            self.assertFalse(report['periodic_activity_required'])
            self.assertFalse(report['final_test_opened'])
            for ttl in report['scenarios'].values():
                arms = ttl['3600']
                self.assertEqual(arms['cash_only']['metrics']['final_cash'], '100')
                self.assertEqual(arms['cash_only']['metrics']['entry_count'], 0)
                self.assertTrue(arms['cash_only']['metrics']['policy_requirements_met'])
                self.assertFalse(arms['cash_only']['metrics']['outperformed_cash'])
            result = profit.audit(output)
            self.assertEqual(result['arms_reproduced'], 66)
            (output / 'audit.json').write_text(json_text(result))
            with patch('sys.argv', ['summarize_execution_costs.py', '--run', str(output)]):
                runpy.run_path(str(fixtures.ROOT / 'scripts/summarize_execution_costs.py'), run_name='__main__')
            costs = strict_json((output / 'cost_decomposition.json').read_text())
            self.assertEqual(len(costs['scenarios']), 6)
        trained = strict_json((output / 'training_report.json').read_text())['models']
        for row in trained.values():
            self.assertEqual(row['actual_timesteps'], 64)
            self.assertGreater(row['parameter_update_l2'], 0)
        for seed in (7, 17, 27):
            self.assertEqual(trained[f'fixed_fee_{seed}']['initial_parameters_sha256'],
                             trained[f'variable_fee_{seed}']['initial_parameters_sha256'])
        with self.assertRaises(ValidationError):
            profit.run(self.f.f.root, self.f.f.store, self.reference, self.config, output)
        path = output / 'fee_interpolation/3600/variable_fee_7.ledger.jsonl'
        path.write_text(path.read_text() + '{}\n')
        with self.assertRaises(ValidationError):
            profit.audit(output)

    def test_scope_cadence_discount_and_fee_spec_changes_rejected(self):
        for name, value in [('periodic_activity_required', True), ('activity_penalty_usd', '.10'),
                            ('partition', 'validation'), ('final_test_opened', True),
                            ('ppo', {**self.spec['ppo'], 'gamma': .995}),
                            ('scenarios', {**self.spec['scenarios'], 'cost_assumption': {'fee_fraction': '.02', 'entry_price_premium': '.01'}})]:
            changed = deepcopy(self.spec)
            changed[name] = value
            self.config.write_text(json_text(changed))
            with self.assertRaises(ValidationError):
                profit.load_spec(self.reference, self.config)
        self.config.write_text(json_text(self.spec))
        (self.reference / 'models/net_equity_7/trained.zip').write_bytes(b'changed')
        with self.assertRaises(ValidationError):
            profit.load_spec(self.reference, self.config)

    def test_real_optimizer_sees_each_fee_before_acting_across_episode_resets(self):
        from test_trading_rl_env import fixture, CONFIG
        data, labels = fixture(days=2)
        spec = deepcopy(self.spec)
        spec['ppo'].update(total_timesteps=192, n_steps=64, batch_size=32)
        runtime(CONFIG)
        output = self.f.f.root / 'curriculum'
        trained = profit.train_models(output, data, labels, CONFIG, spec)
        for seed in (7, 17, 27):
            fixed, variable = trained[f'fixed_fee_{seed}'], trained[f'variable_fee_{seed}']
            self.assertEqual(fixed['timesteps_by_fee_fraction'], {'0.01': 192})
            self.assertEqual(variable['timesteps_by_fee_fraction'], {'0.01': 96, '0': 48, '0.02': 48})
            self.assertEqual(fixed['initial_parameters_sha256'], variable['initial_parameters_sha256'])
            self.assertNotEqual(fixed['trained_parameters_sha256'], variable['trained_parameters_sha256'])


if __name__ == '__main__':
    unittest.main()
