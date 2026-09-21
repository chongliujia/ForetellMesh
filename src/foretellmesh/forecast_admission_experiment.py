"""Reproduce the old ledger and test explicit research-only signal quarantine."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from .data import sha256_file, strict_json
from .evaluation import json_text
from .event_allocation import simulate
from .event_allocation_experiment import load_spec, prepare_inputs
from .forecast_admission import ForecastAdmission, unqualified_generation_records
from .market_development import rows
from .schema import ValidationError, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .trading_rl import artifacts


def run(config_path, output, reproduce=False):
    c = strict_json(config_path.read_text())
    expected = {'kind': 'forecast_admission_mechanism_v1', 'training': False, 'new_llm_calls': 0,
                'validation_replayed': False, 'final_test_opened': False, 'real_orders_sent': 0,
                'default_promotion': False, 'method_qualifications': {},
                'arms': ['cash_only', 'legacy_unvalidated', 'admitted_event']}
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValidationError('unsupported qualification experiment; no grants may be issued here')
    reference = Path(c['reference'])
    prior = strict_json((reference / 'report.json').read_text())
    prior_audit = strict_json((reference / 'audit.json').read_text())
    if (sha256_file(reference / 'report.json') != c['reference_report_sha256']
            or prior_audit['status'] != 'passed' or prior_audit['report_sha256'] != c['reference_report_sha256']):
        raise ValidationError('audited historical reference changed')
    for name, digest in prior['artifact_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(reference / name) != digest:
            raise ValidationError('historical artifact changed')
    if output.exists() and not reproduce:
        raise ValidationError('output already exists')
    s = load_spec(reference / 'config.json')
    _, data, labels, signals, markets, feed, _ = prepare_inputs(s)
    source = Path(s['sources']['signals']['path']).parent
    # Source verification/reconstruction has completed inside prepare_inputs().
    method_id = canonical_hash({'model_and_protocol': strict_json((source / 'config.json').read_text()),
                                'agents_sha256': sha256_file(source / 'agent_config.json'),
                                'source_plan_sha256': sha256_file(source / 'plan.json')})
    records = unqualified_generation_records(signals, rows(source / 'inputs.jsonl'),
                                             rows(source / 'results.jsonl'), method_id)
    gate = ForecastAdmission(signals, records, c['method_qualifications'])
    inventory = dict(Counter(gate.assess(row, timestamp(row['available_at'], 'availability'))['reason'] for row in signals))
    if reproduce:
        report = strict_json((output / 'report.json').read_text())
        if strict_json((output / 'config.json').read_text()) != c:
            raise ValidationError('run config changed')
        for name, digest in report['artifact_hashes'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(output / name) != digest:
                raise ValidationError('run artifact changed')
        if rows(output / 'signals.jsonl') != signals or rows(output / 'admissions.jsonl') != records:
            raise ValidationError('signal provenance does not reproduce')
    else:
        output.mkdir(parents=True)
        shutil.copyfile(config_path, output / 'config.json')
        shutil.copytree(Path(__file__).parent, output / 'source_snapshot/foretellmesh',
                        ignore=shutil.ignore_patterns('__pycache__'))
        (output / 'signals.jsonl').write_text(jsonl(signals))
        (output / 'admissions.jsonl').write_text(jsonl(records))
        (output / 'markets.jsonl').write_text(jsonl(markets))
        (output / 'plan.json').write_text(json_text({'config': c, 'source_config': s,
            'data_bindings': data.bindings, 'method_id': method_id, 'admission_inventory': inventory,
            'artifact_hashes': artifacts(output)}))
    results = {}; checks = {}; decisions_count = 0
    for scenario, costs in s['scenarios'].items():
        dest = output / 'replays' / scenario
        if not reproduce: dest.mkdir(parents=True)
        results[scenario] = {}; checks[scenario] = {}
        for arm in c['arms']:
            opts = {'legacy_unvalidated_research': True} if arm == 'legacy_unvalidated' else {
                'forecast_admission': gate, 'cash_only': arm == 'cash_only'}
            value = simulate(signals, markets, labels, feed, s['policy'], costs, data.start,
                             timestamp(s['replay_end'], 'cutoff'), **opts)
            results[scenario][arm] = value['metrics']
            if arm == 'legacy_unvalidated':
                if value['metrics'] != prior['results'][scenario]['agent_event']:
                    raise ValidationError('legacy metrics changed')
                for field in ('ledger', 'decisions', 'equity_curve'):
                    if value[field] != rows(reference / 'replays' / scenario / f'agent_event.{field}.jsonl'):
                        raise ValidationError('legacy trajectory changed')
            else:
                if (value['ledger'] or value['metrics']['final_cash'] != '100'
                        or any(d['action'] != 'hold' or d['forecast_admission']['allowed'] for d in value['decisions'])
                        or any(point['equity_proxy'] != '100' or point['positions'] != 0 for point in value['equity_curve'])):
                    raise ValidationError('unqualified signals caused exposure or cash mismatch')
            checks[scenario][arm] = {'decision_reasons': dict(Counter(d['reason'] for d in value['decisions'])),
                'decisions': len(value['decisions']), 'ledger_rows': len(value['ledger']),
                'identical_to_historical_trajectory': arm == 'legacy_unvalidated'}
            decisions_count += len(value['decisions'])
            for field in ('ledger', 'decisions', 'equity_curve'):
                target = dest / f'{arm}.{field}.jsonl'
                if reproduce:
                    if rows(target) != value[field]: raise ValidationError('replay differs: ' + str(target))
                else:
                    target.write_text(jsonl(value[field]))
        print(('Reproduced ' if reproduce else 'Evaluated ') + scenario + ': ' +
              str({a: m['final_cash'] for a, m in results[scenario].items()}), flush=True)
    summary = {'results': results, 'checks': checks, 'admission_inventory': inventory,
               'decisions_checked': decisions_count}
    if reproduce:
        if any(report[k] != v for k, v in summary.items()):
            raise ValidationError('summary does not reproduce')
        (output / 'audit.json').write_text(json_text({'status': 'passed',
            'report_sha256': sha256_file(output / 'report.json'), 'replays': 18,
            'old_trajectories_identical': True, 'new_trajectories_identical': True,
            'original_inputs_and_raw_agent_outputs_verified': True, 'decisions_checked': decisions_count}))
    else:
        (output / 'report.json').write_text(json_text({'status': 'completed', **summary,
            'plan_sha256': sha256_file(output / 'plan.json'), 'artifact_hashes': artifacts(output),
            'training': False, 'new_llm_calls': 0, 'validation_replayed': False, 'final_test_opened': False,
            'default_promotion': False, 'qualification_grants_issued': 0,
            'interpretation': c['interpretation'],
            'limitations': ['Cash preservation follows rejection of all unqualified methods, not discovered forecasting skill.',
                'Qualification references must come from separately audited research; this run cannot approve a method.',
                'This is a retrospective mechanism test, not a deployable historical calibration or profitability result.',
                'Existing sparse historical evidence and unproven independent forecasting remain unresolved.']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reproduce', action='store_true')
    a = parser.parse_args(); run(a.config, a.output, a.reproduce)
