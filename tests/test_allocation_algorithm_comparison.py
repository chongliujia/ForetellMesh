import json
from pathlib import Path
import tempfile
import unittest
from foretellmesh.allocation_algorithm_comparison import load_spec, replay, train
from foretellmesh.schema import ValidationError

CONFIG = Path(__file__).resolve().parents[1]/'configs/allocation_algorithm_comparison_v1.json'

class ComparisonGuardTests(unittest.TestCase):
    def test_train_only_scope_and_equal_budgets(self):
        self.assertEqual(load_spec(CONFIG)['total_timesteps'], 131072)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'config.json'
            for key, value in [('partition', 'test'), ('validation_replayed', True), ('llm_calls', 1),
                               ('periodic_activity_required', True), ('total_timesteps', 32768)]:
                spec = json.loads(CONFIG.read_text()); spec[key] = value; path.write_text(json.dumps(spec))
                with self.assertRaises(ValidationError): load_spec(path)

    def test_ppo_report_counts_optimizer_steps_instead_of_epochs(self):
        import torch
        from unittest.mock import patch
        from test_trading_rl_env import fixture
        torch.set_num_threads(1)
        spec = load_spec(CONFIG); spec['families'] = ['ppo']; spec['seeds'] = [7]
        spec['total_timesteps'] = 128
        spec['ppo'].update(total_timesteps=128, n_steps=32, batch_size=16, n_epochs=2)
        data, labels = fixture(days=3)
        original = torch.optim.Adam.step; calls = []
        def counted(optimizer, *args, **kwargs):
            calls.append(1); return original(optimizer, *args, **kwargs)
        with tempfile.TemporaryDirectory() as d, patch.object(torch.optim.Adam, 'step', counted):
            report = train(Path(d), data, labels, spec)['ppo_7']
        self.assertEqual(report['gradient_updates'], len(calls))
        self.assertEqual(report['gradient_updates'], 16)
        self.assertEqual(report['optimizer_epochs'], 8)

    def test_incomplete_model_population_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'training_report.json').write_text(json.dumps({'status': 'completed', 'models': {}}))
            with self.assertRaisesRegex(ValidationError, 'all checkpoints'):
                replay(root, None, None, load_spec(CONFIG))

if __name__ == '__main__': unittest.main()
