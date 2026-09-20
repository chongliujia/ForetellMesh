"""Train real PPO on flat, costly prices; synthetic signals are explicit fixtures."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.trading_rl import runtime, artifacts
from foretellmesh.trading_rl_agents import train_models, load_models, evaluate_one
from foretellmesh.trading_rl_agents_env import SignalIndex, signal_rows
from foretellmesh.schema import iso
from test_trading_rl_env import CONFIG, T, fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root = args.output
    if root.exists():
        raise ValueError('learning-check output already exists')
    c = strict_json(args.config.read_text())
    spec_path = Path(c['training_config']); spec = deepcopy(strict_json(spec_path.read_text()))
    costs = {k: c[k] for k in ('fee_fraction', 'entry_price_premium')}
    # Keep the fee and flat path fixed for this unambiguous always-cash test.
    spec['training_cost_schedule'] = ['cost_assumption']; spec['scenarios'] = {'cost_assumption': costs}
    spec['seeds'] = c['seeds']
    spec['families'] = c['families']
    data, labels = fixture(lambda h: c['constant_price'], days=c['fixture_days'], markets=c['fixture_markets'])
    results = []
    for i in range(c['fixture_markets']):
        results.append({'sample_id': f'synthetic-neutral-{i}', 'market_id': str(i), 'observation_time': iso(T),
            'seconds': 0, 'feature_seconds': 0, 'result': {'status': 'completed',
            'prediction': {'probability': .5, 'confidence': 'low', 'unknowns': [], 'observation_time': iso(T)},
            'stages': {'market_quant': {'market_view': 'no_independent_edge'},
                       'game_theory': {'market_view': 'no_independent_edge'}}}})
    index = SignalIndex(signal_rows(results, c['signal_ttl_seconds']), {m.market_id for m in data.windows})
    root.mkdir(parents=True); shutil.copyfile(args.config, root / 'config.json')
    shutil.copytree(Path(__file__).resolve().parents[1] / 'src/foretellmesh', root / 'source_snapshot/foretellmesh',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'plan.json').write_text(json_text({'config': c, 'ppo_spec': spec, 'environment': CONFIG['environment'],
        'resources': runtime(CONFIG), 'source_script_sha256': sha256_file(Path(__file__)),
        'fixture_source_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'tests/test_trading_rl_env.py'),
        'training_config_sha256': sha256_file(spec_path), 'artifact_hashes': artifacts(root)}))
    trained = train_models(root, data, labels, CONFIG, spec, index)
    (root / 'training_report.json').write_text(json_text({'models': trained}))
    models = load_models(root, spec); evaluated = {}
    for arm, model in models.items():
        result = evaluate_one(data, labels, CONFIG, spec, index, costs, arm, model)
        again = evaluate_one(data, labels, CONFIG, spec, index, costs, arm, model)
        if result != again:
            raise ValueError('frozen toy replay differs')
        m = result['metrics']; evaluated[arm] = {'metrics': m,
            'passed': m['final_cash'] == c['required_final_cash'] and m['entry_count'] == c['required_entries']}
    report = {'status': 'completed', 'kind': c['kind'], 'neutral_signals_are_synthetic': True,
        'llm_calls': 0, 'real_market_data_used': False, 'real_orders_sent': 0,
        'all_passed': all(r['passed'] for r in evaluated.values()), 'results': evaluated,
        'plan_sha256': sha256_file(root / 'plan.json'), 'training_report_sha256': sha256_file(root / 'training_report.json'),
        'artifact_hashes': artifacts(root)}
    (root / 'report.json').write_text(json_text(report))
    print(json_text({'all_passed': report['all_passed'], 'cash': {k: v['metrics']['final_cash'] for k, v in evaluated.items()}}))


if __name__ == '__main__':
    main()
