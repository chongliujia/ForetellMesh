from copy import deepcopy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError
from foretellmesh.trading_rl import artifacts, freeze_data, runtime
from foretellmesh.trading_rl_data import prepare, settlements
from foretellmesh.trading_rl_diagnostic import CORE_SOURCE
from foretellmesh.trading_rl_retrain import audit, load_spec, run
import foretellmesh.trading_rl_retrain as retrain
import test_trading_rl as fixtures


@unittest.skipUnless(fixtures.HAS_RL, 'optional masked PPO dependencies unavailable')
class RetrainTests(unittest.TestCase):
    def setUp(self):
        from sb3_contrib import MaskablePPO
        from foretellmesh.trading_rl_env import AllocationEnv
        f = fixtures.AllocationDataTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.c = deepcopy(f.c)
        self.c['seeds'] = [7, 17, 27]
        runtime(self.c)
        self.source = f.root / 'original'
        snapshot = self.source / 'source_snapshot/foretellmesh'
        snapshot.mkdir(parents=True)
        for name in CORE_SOURCE:
            shutil.copyfile(Path(retrain.__file__).with_name(name), snapshot / name)
        (self.source / 'config.json').write_text(json_text(self.c))
        (self.source / 'report.json').write_text(json_text({'fixture': True}))
        data = prepare(f.root, f.store, 'train', self.c)
        freeze_data(self.source, data)
        labels = settlements(f.root, data)
        seeds = {}
        h = self.c['ppo']
        for seed in self.c['seeds']:
            dest = self.source / f'seed_{seed}'
            dest.mkdir()
            env = AllocationEnv(data, labels, self.c['environment'], self.c['scenarios']['cost_assumption'])
            model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                policy_kwargs={'net_arch': h['net_arch']}, **{k: h[k] for k in (
                    'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef')})
            model.save(dest / 'initial.zip')
            model.save(dest / 'trained.zip')
            seeds[str(seed)] = {f'{stage}_sha256': sha256_file(dest / f'{stage}.zip') for stage in ('initial', 'trained')}
            env.close()
        (self.source / 'training_report.json').write_text(json_text({'seeds': seeds, 'artifact_hashes': artifacts(self.source)}))
        self.execution = f.root / 'execution'
        snapshot = self.execution / 'source_snapshot/foretellmesh'
        snapshot.mkdir(parents=True)
        for name in retrain.SOURCE_FILES[2:]:
            shutil.copyfile(Path(retrain.__file__).with_name(name), snapshot / name)
        execution_spec = strict_json((fixtures.ROOT / 'configs/allocation_execution_lifetime_v1.json').read_text())
        execution_spec['reference_hashes'] = {name: sha256_file(self.source / name) for name in execution_spec['reference_hashes']}
        (self.execution / 'config.json').write_text(json_text(execution_spec))
        (self.execution / 'report.json').write_text(json_text({'status': 'completed', 'partition': 'train',
                                                             'artifact_hashes': artifacts(self.execution)}))
        (self.execution / 'audit.json').write_text(json_text({'status': 'passed', 'report_sha256': sha256_file(self.execution / 'report.json')}))
        self.spec = strict_json((fixtures.ROOT / 'configs/allocation_ppo_execution_retrain_v1.json').read_text())
        self.spec['ppo'] = self.c['ppo']
        self.spec['execution_reference_hashes'] = {name: sha256_file(self.execution / name) for name in self.spec['execution_reference_hashes']}
        self.config = f.root / 'retrain.json'
        self.config.write_text(json_text(self.spec))

    def test_six_real_updates_paired_frozen_and_audited_without_heldout_reads(self):
        out = self.f.root / 'retrain'
        original = Path.open
        evaluate = retrain.evaluate_lifetime

        def guard(path, *args, **kwargs):
            self.assertFalse(path.name.startswith(('validation.', 'test.')), str(path))
            return original(path, *args, **kwargs)

        def frozen_first(*args, **kwargs):
            report = strict_json((out / 'training_report.json').read_text())
            self.assertEqual(len(report['models']), 6)
            self.assertTrue((out / 'evaluation_plan.json').exists())
            for name in report['models']:
                self.assertTrue((out / 'models' / name / 'trained.zip').exists())
            return evaluate(*args, **kwargs)

        with patch.object(Path, 'open', guard), patch.object(retrain, 'evaluate_lifetime', frozen_first):
            result = run(self.f.root, self.f.store, self.source, self.execution, self.config, out)
            self.assertFalse(result['validation_replayed'])
            self.assertFalse(result['final_test_opened'])
            trained = strict_json((out / 'training_report.json').read_text())
            for row in trained['models'].values():
                self.assertEqual(row['actual_timesteps'], 64)
                self.assertGreater(row['parameter_update_l2'], 0)
                self.assertEqual(row['initial_parameters_sha256'], row['original_initial_parameters_sha256'])
                self.assertNotEqual(row['initial_parameters_sha256'], row['trained_parameters_sha256'])
            self.assertEqual(audit(out)['arms_reproduced'], 66)
        with self.assertRaises(ValidationError):
            run(self.f.root, self.f.store, self.source, self.execution, self.config, out)
        ledger = out / 'cost_assumption/3600/net_equity_7.ledger.jsonl'
        ledger.write_text(ledger.read_text() + '{}\n')
        with self.assertRaises(ValidationError):
            audit(out)

    def test_protocol_and_original_initialization_tampering_rejected(self):
        for name, value in [('partition', 'validation'), ('final_test_opened', True),
                            ('training_order_ttl_seconds', 14400), ('rewards', {'net_equity': '0', 'net_equity_cadence': '1'})]:
            changed = deepcopy(self.spec)
            changed[name] = value
            self.config.write_text(json_text(changed))
            with self.assertRaises(ValidationError):
                load_spec(self.source, self.execution, self.config)
        self.config.write_text(json_text(self.spec))
        (self.source / 'seed_7/initial.zip').write_bytes(b'changed')
        with self.assertRaises(ValidationError):
            load_spec(self.source, self.execution, self.config)


if __name__ == '__main__':
    unittest.main()
