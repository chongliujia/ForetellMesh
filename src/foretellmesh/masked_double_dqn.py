"""Small CPU Double DQN with masks in exploration, inference and Bellman targets."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .schema import ValidationError


def double_q_targets(rewards, terminal, online_next, target_next, next_masks, gamma=1.):
    """Online network selects a legal action; target network evaluates that action."""
    if not torch.all(next_masks.any(dim=1) | terminal):
        raise ValidationError('nonterminal state has no legal action')
    selected = online_next.masked_fill(~next_masks, -torch.inf).argmax(dim=1)
    value = target_next.gather(1, selected[:, None]).squeeze(1)
    return torch.where(terminal, rewards, rewards + gamma * value)


class MaskedDoubleDQN:
    def __init__(self, observation_dim, action_count, config, seed):
        if (observation_dim < 1 or action_count < 1 or not 0 < config['learning_rate'] < 1
                or not 0 < config['gamma'] <= 1 or not 0 <= config['epsilon_final'] <= config['epsilon_initial'] <= 1
                or any(type(config[k]) is not int or config[k] < 1 for k in ('buffer_size', 'batch_size',
                    'learning_starts', 'train_freq', 'target_update_interval', 'epsilon_decay_steps'))
                or config['max_grad_norm'] <= 0):
            raise ValidationError('invalid Double DQN configuration')
        self.observation_dim = int(observation_dim); self.action_count = int(action_count)
        self.config = dict(config); self.seed = seed; self.num_timesteps = 0; self.gradient_updates = 0
        torch.manual_seed(seed); self.rng = np.random.default_rng(seed)
        self.q = nn.Sequential(nn.Linear(observation_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_count))
        self.target = deepcopy(self.q); self.target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config['learning_rate'])

    def predict(self, observation, *, action_masks, deterministic=True, epsilon=None):
        mask = np.asarray(action_masks, dtype=bool)
        if mask.shape != (self.action_count,) or not mask.any():
            raise ValidationError('invalid inference mask')
        if not deterministic and self.rng.random() < float(epsilon):
            return int(self.rng.choice(np.flatnonzero(mask))), None
        with torch.no_grad():
            q = self.q(torch.as_tensor(observation, dtype=torch.float32)[None])[0]
            return int(q.masked_fill(~torch.as_tensor(mask), -torch.inf).argmax()), None

    def learn(self, env, total_timesteps):
        if self.num_timesteps or type(total_timesteps) is not int or total_timesteps < 1:
            raise ValidationError('fresh training only; checkpoints support inference, not resumption')
        h = self.config; size = h['buffer_size']; dim = self.observation_dim
        obs_buf = np.empty((size, dim), dtype=np.float32); next_buf = np.empty_like(obs_buf)
        actions = np.empty(size, dtype=np.int64); rewards = np.empty(size, dtype=np.float32)
        done_buf = np.empty(size, dtype=bool); masks = np.empty((size, self.action_count), dtype=bool)
        obs, _ = env.reset(seed=self.seed); episodes = []; losses = []; fees = {}
        for step in range(total_timesteps):
            epsilon = h['epsilon_initial'] + (h['epsilon_final'] - h['epsilon_initial']) * min(1., step / h['epsilon_decay_steps'])
            mask = env.action_masks()
            action, _ = self.predict(obs, action_masks=mask, deterministic=False, epsilon=epsilon)
            next_obs, reward, terminal, truncated, info = env.step(action)
            if truncated:
                raise ValidationError('time-limit truncation requires an explicit terminal-observation adapter')
            if not np.isfinite(reward) or not np.isfinite(next_obs).all():
                raise ValidationError('nonfinite replay transition')
            i = step % size
            obs_buf[i] = obs; next_buf[i] = next_obs; actions[i] = action; rewards[i] = reward; done_buf[i] = terminal
            masks[i] = env.action_masks()
            fee = info.get('execution_costs', {}).get('fee_fraction', 'synthetic')
            fees[fee] = fees.get(fee, 0) + 1
            self.num_timesteps += 1
            if step + 1 >= h['learning_starts'] and (step + 1) % h['train_freq'] == 0:
                ids = self.rng.integers(0, min(step + 1, size), size=h['batch_size'])
                ob = torch.from_numpy(obs_buf[ids]); nx = torch.from_numpy(next_buf[ids])
                with torch.no_grad():
                    targets = double_q_targets(torch.from_numpy(rewards[ids]), torch.from_numpy(done_buf[ids]),
                        self.q(nx), self.target(nx), torch.from_numpy(masks[ids]), h['gamma'])
                estimates = self.q(ob).gather(1, torch.from_numpy(actions[ids])[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(estimates, targets)
                self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), h['max_grad_norm'])
                self.optimizer.step(); self.gradient_updates += 1; losses.append(float(loss.detach()))
            if (step + 1) % h['target_update_interval'] == 0:
                self.target.load_state_dict(self.q.state_dict())
            if terminal:
                episodes.append({'timesteps': step + 1, **info.get('episode_summary', {})})
                obs, _ = env.reset()
            else:
                obs = next_obs
            if (step + 1) % 32768 == 0:
                print(f'Double DQN seed {self.seed}: {step + 1}/{total_timesteps}', flush=True)
        return {'episodes': episodes, 'timesteps_by_fee_fraction': fees,
                'gradient_updates': self.gradient_updates, 'mean_last_100_losses': float(np.mean(losses[-100:])) if losses else None,
                'replay_buffer_bytes': sum(x.nbytes for x in (obs_buf, next_buf, actions, rewards, done_buf, masks))}

    def save(self, path):
        torch.save({'observation_dim': self.observation_dim, 'action_count': self.action_count,
                    'config': self.config, 'seed': self.seed, 'num_timesteps': self.num_timesteps,
                    'gradient_updates': self.gradient_updates, 'q': self.q.state_dict(), 'target': self.target.state_dict()}, Path(path))

    @classmethod
    def load(cls, path):
        # Gym Discrete.n in older checkpoints is a numpy.int64. Permit only its
        # scalar/dtype constructors, retaining weights-only loading throughout.
        with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Int64DType]):
            saved = torch.load(Path(path), map_location='cpu', weights_only=True)
        model = cls(saved['observation_dim'], saved['action_count'], saved['config'], saved['seed'])
        model.q.load_state_dict(saved['q']); model.target.load_state_dict(saved['target'])
        model.num_timesteps = saved['num_timesteps']; model.gradient_updates = saved['gradient_updates']; model.q.eval()
        return model
