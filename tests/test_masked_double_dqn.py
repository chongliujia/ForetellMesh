import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from foretellmesh.masked_double_dqn import MaskedDoubleDQN, double_q_targets
from foretellmesh.schema import ValidationError

H = dict(learning_rate=.001, gamma=1., buffer_size=512, batch_size=32, learning_starts=32,
         train_freq=1, target_update_interval=40, epsilon_initial=1., epsilon_final=.05,
         epsilon_decay_steps=1000, max_grad_norm=10.)

class CueEnv:
    """Known positive/negative delayed payoff; masks forbid repeat entry."""
    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.cue = self.rng.choice([-1., 1.]); self.phase = 0; self.bought = 0
        return self.obs(), {}
    def obs(self): return np.array([self.cue, self.phase / 4, self.bought], dtype=np.float32)
    def action_masks(self): return np.array([True, self.phase == 0])
    def step(self, action):
        assert self.action_masks()[action]
        if action: self.bought = 1
        self.phase += 1
        terminal = self.phase == 4
        reward = (-.02 if action else 0) + (self.cue * self.bought if terminal else 0)
        return self.obs(), reward, terminal, False, {}

class DoubleDQNTests(unittest.TestCase):
    def test_online_selects_target_evaluates_and_illegal_max_is_masked(self):
        value = double_q_targets(torch.tensor([1., 2.]), torch.tensor([False, True]),
            torch.tensor([[3., 2., 99.], [0., 0., 0.]]), torch.tensor([[4., 50., 999.], [99., 99., 99.]]),
            torch.tensor([[True, True, False], [False, False, False]]), .5)
        self.assertEqual(value.tolist(), [3., 2.])
        with self.assertRaises(ValidationError):
            double_q_targets(torch.zeros(1), torch.tensor([False]), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

    def test_exploration_masks_save_load_and_no_resume(self):
        torch.set_num_threads(1)
        model = MaskedDoubleDQN(3, 2, H, 7)
        for _ in range(100):
            self.assertEqual(model.predict(np.zeros(3), action_masks=[True, False], deterministic=False, epsilon=1)[0], 0)
        model.learn(CueEnv(), 64)
        self.assertGreater(model.gradient_updates, 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pt'
            model.action_count = np.int64(model.action_count)  # legacy Gym checkpoint metadata
            model.save(path); loaded = MaskedDoubleDQN.load(path)
            self.assertIs(type(loaded.action_count), int)
            for cue in (-1., 1.):
                obs = np.array([cue, 0, 0], dtype=np.float32)
                self.assertEqual(model.predict(obs, action_masks=[True, True]), loaded.predict(obs, action_masks=[True, True]))
            with self.assertRaises(ValidationError): loaded.learn(CueEnv(), 10)
        with self.assertRaises(ValidationError): model.predict(np.zeros(3), action_masks=[False, False])

    def test_learns_delayed_reward_and_abstention_on_synthetic_data(self):
        torch.set_num_threads(1)
        model = MaskedDoubleDQN(3, 2, H, 7); model.learn(CueEnv(), 4000)
        for cue, expected in [(-1., 0), (1., 1)]:
            self.assertEqual(model.predict(np.array([cue, 0, 0], dtype=np.float32), action_masks=[True, True])[0], expected)

if __name__ == '__main__': unittest.main()
