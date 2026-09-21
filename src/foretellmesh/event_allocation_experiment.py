"""Frozen-signal, no-training comparison of event holding and a 48-hour cap."""
import argparse
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .event_allocation import simulate, validate_policy
from .market_development import rows
from .mean_reversion import HistoricalBars
from .schema import ValidationError, timestamp, iso
from .sft_data import jsonl
from .trading_agent_signals import verify
from .trading_rl import artifacts
from .trading_rl_data import prepare, settlements
from .trading_rl_agents_env import signal_rows
from .trading_rl_profit import evaluate_one

ARMS = {'cash_only': {'cash_only': True}, 'agent_event': {}, 'agent_48h': {'max_holding_seconds': 172800},
        'anchor_event': {'anchor_only': True}, 'anchor_48h': {'anchor_only': True, 'max_holding_seconds': 172800}}


def load_spec(path):
    s = strict_json(path.read_text())
    fixed = {'kind': 'event_allocation_fixed_signal_baseline_v2', 'partition': 'train', 'training': False,
        'new_llm_calls': 0, 'foundation_model_updated': False, 'validation_replayed': False,
        'final_test_opened': False, 'real_orders_sent': 0, 'default_promotion': False,
        'replay_end': '2025-01-01T00:00:00Z', 'arms': ARMS,
        'legacy_control': 'mean_reversion', 'settlement_fee_fraction': '0'}
    if any(s.get(k) != v for k, v in fixed.items()): raise ValidationError('unsupported event baseline scope')
    validate_policy(s['policy'])
    for k, ref in s['sources'].items():
        if sha256_file(Path(ref['path'])) != ref['sha256']: raise ValidationError('source changed: '+k)
    return s


def prepare_inputs(s):
    dataset = Path(s['sources']['dataset']['path']).parent
    store = Path(s['sources']['store']['path']).parent
    reference = Path(s['sources']['comparison']['path']).parent
    signals_root = Path(s['sources']['signals']['path']).parent
    comparison = strict_json((reference/'report.json').read_text())
    audit = strict_json((reference/'audit.json').read_text())
    if audit['status'] != 'passed' or audit['report_sha256'] != sha256_file(reference/'report.json'):
        raise ValidationError('comparison not audited')
    c = strict_json((reference/'config.json').read_text())
    if sha256_file(reference/'config.json') != comparison['artifact_hashes']['config.json'] or c['scenarios'] != s['scenarios']:
        raise ValidationError('common costs/configuration changed')
    data = prepare(dataset, store, 'train', c)
    labels = settlements(dataset, data)
    end = timestamp(s['replay_end'], 'administrative cutoff')
    if any(x.time >= end for x in labels): raise ValidationError('training cohort crosses fixed cutoff')
    sc, sp, results = verify(signals_root, rebuild_inputs=True)
    # verify() just rebuilt the original inputs and replayed every raw Agent
    # response. Do not require a separate optional audit marker in the old run.
    raw_inputs = rows(signals_root/'inputs.jsonl')
    refs = {r['sample_id']: r['input']['market']['probability'] for r in raw_inputs}
    records = signal_rows(results, sc['signal_ttl_seconds'], reference_probabilities=refs)
    admitted = {x.market_id: x for x in data.windows}
    for row in results:
        if row['market_id'] not in admitted or row['event_group_id'] != admitted[row['market_id']].event_group_id:
            raise ValidationError('signal outside admitted training cohort')
    # Mark and fill coverage extends past research boundaries, using only admitted
    # training-market IDs and a public administrative cutoff, never a label feature.
    feed = HistoricalBars.from_store(store/'prices.sqlite', admitted, data.start, end)
    markets = [{k: r[k] for k in ('market_id', 'event_group_id', 'initialized_at')} for r in data.catalog]
    return c, data, labels, records, markets, feed, comparison


def run(config_path, output, reproduce=False):
    s = load_spec(config_path)
    if output.exists() and not reproduce: raise ValidationError('output already exists')
    c, data, labels, signals, markets, feed, prior = prepare_inputs(s)
    if reproduce:
        report = strict_json((output/'report.json').read_text())
        if strict_json((output/'config.json').read_text()) != s: raise ValidationError('config differs')
        for name, digest in report['artifact_hashes'].items():
            if sha256_file(output/name) != digest: raise ValidationError('artifact changed: '+name)
        if rows(output/'signals.jsonl') != signals: raise ValidationError('signals changed')
    else:
        output.mkdir(parents=True); shutil.copyfile(config_path, output/'config.json')
        shutil.copytree(Path(__file__).parent, output/'source_snapshot/foretellmesh', ignore=shutil.ignore_patterns('__pycache__'))
        (output/'signals.jsonl').write_text(jsonl(signals)); (output/'markets.jsonl').write_text(jsonl(markets))
        covered = {r['market_id'] for r in signals if r['content'] is not None}
        meta = {'admitted_markets': len(markets), 'admitted_events': len({m['event_group_id'] for m in markets}),
            'signal_jobs': len(signals), 'valid_signal_jobs': sum(r['content'] is not None for r in signals),
            'markets_with_valid_signals': len(covered), 'events_with_valid_signals': len({m['event_group_id'] for m in markets if m['market_id'] in covered}),
            'markets_without_signals': sorted(set(feed.data)-covered), 'start': iso(data.start), 'administrative_end': s['replay_end'],
            'last_research_observation': iso(data.end), 'native_price_rows': sum(n for rr in feed.data.values() for _, _, n in rr),
            'bindings': data.bindings, 'research_cutoffs': {x.market_id: iso(x.retire_at) for x in data.windows}}
        (output/'plan.json').write_text(json_text({'spec': s, 'data': meta, 'artifact_hashes': artifacts(output)}))
    result = {}
    for scenario, costs in s['scenarios'].items():
        result[scenario] = {}; dest = output/'replays'/scenario
        if not reproduce: dest.mkdir(parents=True)
        for name, options in s['arms'].items():
            value = simulate(signals, markets, labels, feed, s['policy'], costs, data.start,
                             timestamp(s['replay_end'], 'cutoff'), legacy_unvalidated_research=True, **options)
            result[scenario][name] = value['metrics']
            for field in ('ledger', 'decisions', 'equity_curve'):
                p = dest/f'{name}.{field}.jsonl'
                if reproduce:
                    if rows(p) != value[field]: raise ValidationError('event replay differs: '+str(p))
                else: p.write_text(jsonl(value[field]))
        old = evaluate_one(data, labels, c, c, costs, 'mean_reversion')
        if old['metrics'] != prior['results'][scenario]['mean_reversion']['metrics']:
            raise ValidationError('legacy mean-reversion control changed')
        result[scenario]['legacy_mean_reversion'] = old['metrics']
        for field in ('ledger', 'decisions', 'equity_curve', 'pending'):
            p = dest/f'legacy_mean_reversion.{field}.jsonl'
            if reproduce:
                if rows(p) != old[field]: raise ValidationError('legacy replay differs')
            else: p.write_text(jsonl(old[field]))
        print(('Reproduced' if reproduce else 'Evaluated')+' '+scenario+': '+str({k: v['final_cash'] for k, v in result[scenario].items()}), flush=True)
    if reproduce:
        if result != report['results']: raise ValidationError('metrics differ')
        (output/'audit.json').write_text(json_text({'status': 'passed', 'report_sha256': sha256_file(output/'report.json'),
            'replay_count': sum(len(v) for v in result.values()), 'all_ledgers_decisions_curves_identical': True,
            'original_signal_inputs_rebuilt': True, 'original_agent_responses_replayed': True,
            'new_training': False, 'new_llm_calls': 0, 'validation_replayed': False, 'final_test_opened': False}))
    else:
        (output/'report.json').write_text(json_text({'status': 'completed', 'scope': 'training_only_frozen_signal_baseline',
            'results': result, 'plan_sha256': sha256_file(output/'plan.json'), 'artifact_hashes': artifacts(output),
            'training': False, 'new_llm_calls': 0, 'validation_replayed': False, 'final_test_opened': False,
            'default_promotion': False, 'real_orders_sent': 0,
            'limitations': ['Print-based hypothetical fills; no historical depth or live fee calibration.',
                'Uncertainty margin is fixed sensitivity protection, not measured calibration.',
                'Signal coverage is inherited; unanalysed new events remain missing.',
                'Agent/event versus Agent/48h share all rules except holding cap; legacy mean reversion uses different entries.',
                'Hourly and event marks may miss intrahour drawdowns; missing/stale marks explicitly counted.',
                'Settlement payouts use audited labels in the executor only; settlement fee assumed zero.',
                'No opportunity-cost model or forecast quality improvement claim.']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--reproduce', action='store_true')
    a = p.parse_args(); run(a.config, a.output, a.reproduce)
