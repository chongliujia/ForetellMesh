"""Training-only, fixed-budget masked PPO/Double-DQN comparison; no model promotion."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import time
import statistics

from .data import strict_json, sha256_file
from .evaluation import json_text
from .schema import ValidationError
from .sft_data import jsonl
from .market_development import rows
from .trading_rl import runtime, freeze_data, artifacts
from .trading_rl_data import prepare, settlements
from .trading_rl_profit import make_env, evaluate_one
from .masked_double_dqn import MaskedDoubleDQN


def load_spec(path):
    s = strict_json(path.read_text())
    expected = {'partition': 'train', 'families': ['ppo', 'double_dqn'], 'seeds': [7, 17, 27],
        'periodic_activity_required': False, 'activity_penalty_usd': '0', 'validation_replayed': False,
        'final_test_opened': False, 'foundation_model_updated': False, 'llm_calls': 0, 'real_orders_sent': 0,
        'default_promotion': False, 'checkpoint_selection': 'last_fixed_step_all_seeds_no_selection',
        'signal_source': 'causal_price_features_only', 'objective': 'maximize_after_cost_terminal_equity'}
    if any(s.get(k) != v for k, v in expected.items()) or s['ppo']['gamma'] != 1 or s['double_dqn']['gamma'] != 1:
        raise ValidationError('unsupported comparison scope')
    if s['total_timesteps'] % s['ppo']['n_steps'] or s['ppo']['total_timesteps'] != s['total_timesteps']:
        raise ValidationError('budgets differ')
    return s


def train(root, data, labels, s):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    out = {}; schedule = [s['scenarios'][x] for x in s['training_cost_schedule']]
    for family in s['families']:
        for seed in s['seeds']:
            name = f'{family}_{seed}'; dest = root/'models'/name; dest.mkdir(parents=True)
            env = make_env(data, labels, s, s, schedule[0], cost_schedule=schedule)
            started = time.perf_counter()
            if family == 'ppo':
                h = s['ppo']
                model = MaskablePPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                    policy_kwargs={'net_arch': h['net_arch']}, **{k: h[k] for k in (
                        'n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef')})
                model.set_logger(configure(str(dest), ['csv']))
                params = model.policy; ext = '.zip'
            else:
                model = MaskedDoubleDQN(env.observation_space.shape[0], env.action_space.n, s['double_dqn'], seed)
                params = model.q; ext = '.pt'
            initial = {k: v.detach().clone() for k, v in params.state_dict().items()}
            model.save(dest/('initial'+ext))
            if family == 'ppo':
                episodes = []; fee_steps = Counter()
                class Capture(BaseCallback):
                    def _on_step(self):
                        for info in self.locals['infos']:
                            fee_steps[info['execution_costs']['fee_fraction']] += 1
                            if 'episode_summary' in info: episodes.append({'timesteps': self.num_timesteps, **info['episode_summary']})
                        if self.num_timesteps % 32768 == 0: print(f'{name}: {self.num_timesteps}/{s["total_timesteps"]}', flush=True)
                        return True
                model.learn(total_timesteps=s['total_timesteps'], callback=Capture())
                details = {'episodes': episodes, 'timesteps_by_fee_fraction': dict(fee_steps), 'optimizer_epochs': model._n_updates,
                           'gradient_updates': model._n_updates * (h['n_steps'] // h['batch_size'])}
                model.logger.close()
            else:
                details = model.learn(env, s['total_timesteps'])
            delta = sum(float(torch.sum((v.detach()-initial[k])**2)) for k, v in params.state_dict().items())**.5
            if not delta > 0 or model.num_timesteps != s['total_timesteps']: raise ValidationError('training update/budget failure')
            model.save(dest/('trained'+ext)); env.close()
            out[name] = {'family': family, 'seed': seed, 'seconds': time.perf_counter()-started,
                'actual_timesteps': model.num_timesteps, 'parameter_update_l2': delta,
                'parameters': sum(p.numel() for p in params.parameters()),
                'initial_path': str((dest/('initial'+ext)).relative_to(root)),
                'trained_path': str((dest/('trained'+ext)).relative_to(root)), **details}
            for stage in ('initial', 'trained'): out[name][stage+'_sha256'] = sha256_file(root/out[name][stage+'_path'])
            print(f'{name} frozen in {out[name]["seconds"]:.1f}s', flush=True)
    return out


def replay(root, data, labels, s, reproduce=False):
    from sb3_contrib import MaskablePPO
    tr = strict_json((root/'training_report.json').read_text())
    expected = {f'{f}_{seed}' for f in s['families'] for seed in s['seeds']}
    if tr['status'] != 'completed' or set(tr['models']) != expected or tr['plan_sha256'] != sha256_file(root/'plan.json'):
        raise ValidationError('all checkpoints must be frozen first')
    models = {}
    for name, r in tr['models'].items():
        for stage in ('initial', 'trained'):
            if sha256_file(root/r[stage+'_path']) != r[stage+'_sha256']: raise ValidationError('checkpoint changed')
        models[name] = (MaskedDoubleDQN.load(root/r['trained_path']) if r['family'] == 'double_dqn'
                        else MaskablePPO.load(root/r['trained_path'], device='cpu'))
        if models[name].num_timesteps != s['total_timesteps']: raise ValidationError('wrong checkpoint budget')
    result = {}
    for scenario, costs in s['scenarios'].items():
        result[scenario] = {}; dest = root/'replays'/scenario
        if not reproduce: dest.mkdir(parents=True)
        for name in ['cash_only', 'mean_reversion', *sorted(models)]:
            value = evaluate_one(data, labels, s, s, costs, name, models.get(name))
            result[scenario][name] = {k: value[k] for k in ('metrics', 'lifecycle')}
            for field in ('ledger', 'decisions', 'equity_curve', 'pending'):
                p = dest/f'{name}.{field}.jsonl'
                if reproduce:
                    if rows(p) != value[field]: raise ValidationError('replay differs: '+str(p))
                else: p.write_text(jsonl(value[field]))
        print(f'{"Reproduced" if reproduce else "Evaluated"} {scenario}', flush=True)
    return result


def run(spec_path, output, reproduce=False):
    s = load_spec(spec_path); resources = runtime(s)
    dataset = Path(s['dataset_path']); store = Path(s['trade_store_path'])
    data = prepare(dataset, store, 'train', s); labels = settlements(dataset, data)
    if reproduce:
        report = strict_json((output/'report.json').read_text())
        for name, digest in report['artifact_hashes'].items():
            if sha256_file(output/name) != digest: raise ValidationError('artifact changed: '+name)
        if s != strict_json((output/'config.json').read_text()): raise ValidationError('config changed')
        if replay(output, data, labels, s, True) != report['results']: raise ValidationError('summary differs')
        (output/'audit.json').write_text(json_text({'status': 'passed', 'report_sha256': sha256_file(output/'report.json'),
            'full_training_replay_reproduced': True, 'validation_replayed': False, 'final_test_opened': False}))
        return
    if output.exists(): raise ValidationError('output already exists')
    output.mkdir(parents=True); shutil.copyfile(spec_path, output/'config.json')
    shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    plan = {'scope': 'training_diagnostics_only', 'config_sha256': sha256_file(spec_path), 'resources': resources,
            'data': freeze_data(output, data), 'artifact_hashes': artifacts(output)}
    (output/'plan.json').write_text(json_text(plan))
    print(json_text(plan['data']), flush=True)
    models = train(output, data, labels, s)
    (output/'training_report.json').write_text(json_text({'status': 'completed', 'plan_sha256': sha256_file(output/'plan.json'), 'models': models}))
    result = replay(output, data, labels, s)
    summary = {scenario: {f: {'mean_final_cash': statistics.fmean(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds']),
        'min_final_cash': min(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds']),
        'max_final_cash': max(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds'])}
        for f in s['families']} for scenario, values in result.items()}
    (output/'report.json').write_text(json_text({'status': 'completed', 'scope': 'training_diagnostics_only',
        'validation_replayed': False, 'final_test_opened': False, 'default_promotion': False,
        'results': result, 'summary': summary, 'artifact_hashes': artifacts(output)}))
    print(json_text(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--reproduce', action='store_true')
    a = p.parse_args(); run(a.config, a.output, a.reproduce)
