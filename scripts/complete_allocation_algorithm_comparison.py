"""Finish evaluation of all frozen checkpoints after a metadata-only loader fix."""
import argparse
from pathlib import Path
import shutil
import statistics
from foretellmesh.allocation_algorithm_comparison import load_spec, replay, run
from foretellmesh.trading_rl import artifacts, runtime
from foretellmesh.trading_rl_data import prepare, settlements
from foretellmesh.data import strict_json, sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError
import foretellmesh.masked_double_dqn as ddqn


def finish(root):
    if (root/'report.json').exists() or (root/'replays').exists(): raise ValidationError('evaluation already started')
    s = load_spec(root/'config.json'); runtime(s)
    tr = strict_json((root/'training_report.json').read_text())
    if tr['status'] != 'completed' or tr['plan_sha256'] != sha256_file(root/'plan.json'): raise ValidationError('training incomplete')
    source = Path(ddqn.__file__).parent; prior = root/'source_snapshot/foretellmesh'
    changed = [p.name for p in source.glob('*.py') if sha256_file(p) != sha256_file(prior/p.name)]
    if changed != ['masked_double_dqn.py']: raise ValidationError('unexpected post-freeze changes: '+str(changed))
    # Original training code and checkpoint bytes remain untouched.
    shutil.copytree(source, root/'evaluation_source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(Path(__file__), root/'completion_script.py')
    (root/'evaluation_fix.json').write_text(json_text({'kind': 'numpy_int64_checkpoint_metadata_loader_compatibility',
        'training_report_sha256': sha256_file(root/'training_report.json'), 'changed_modules': changed,
        'original_module_sha256': sha256_file(prior/'masked_double_dqn.py'),
        'evaluation_module_sha256': sha256_file(source/'masked_double_dqn.py'),
        'weights_modified': False, 'additional_training': False,
        'weights_only_loading': True, 'allowlist': ['numpy._core.multiarray.scalar', 'numpy.dtype', 'numpy.dtypes.Int64DType']}))
    data = prepare(Path(s['dataset_path']), Path(s['trade_store_path']), 'train', s)
    labels = settlements(Path(s['dataset_path']), data)
    result = replay(root, data, labels, s)
    summary = {scenario: {f: {'mean_final_cash': statistics.fmean(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds']),
        'min_final_cash': min(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds']),
        'max_final_cash': max(float(values[f'{f}_{seed}']['metrics']['final_cash']) for seed in s['seeds'])}
        for f in s['families']} for scenario, values in result.items()}
    (root/'report.json').write_text(json_text({'status': 'completed', 'scope': 'training_diagnostics_only',
        'validation_replayed': False, 'final_test_opened': False, 'default_promotion': False,
        'results': result, 'summary': summary, 'artifact_hashes': artifacts(root)}))
    print(json_text(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path)
    p.add_argument('--reproduce', action='store_true'); a = p.parse_args()
    if a.reproduce: run(a.root/'config.json', a.root, True)
    else: finish(a.root)
