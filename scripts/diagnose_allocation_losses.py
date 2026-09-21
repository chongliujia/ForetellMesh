"""Attribute frozen simulator losses without retraining or opening held-out data."""
import argparse
from decimal import Decimal as D
from pathlib import Path
from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError
from summarize_execution_costs import decompose
from diagnose_allocation_optimizer import ledger_summary


def diagnose(root, output):
    if output.exists(): raise ValidationError('loss diagnostic already exists')
    report = strict_json((root/'report.json').read_text()); spec = strict_json((root/'config.json').read_text())
    audit = strict_json((root/'arithmetic_audit.json').read_text())
    if (report['scope'] != 'training_diagnostics_only' or report['final_test_opened'] or report['validation_replayed']
            or audit['status'] != 'passed' or audit['report_sha256'] != sha256_file(root/'report.json')):
        raise ValidationError('audited frozen training experiment required')
    for name in ('config.json', 'training_report.json'):
        if sha256_file(root/name) != report['artifact_hashes'][name]: raise ValidationError('artifact changed')
    consumed = {}
    def read(name):
        digest = sha256_file(root/name)
        if digest != report['artifact_hashes'][name]: raise ValidationError('artifact changed: '+name)
        consumed[name] = digest
        return rows(root/name)
    scenarios = {}; diagnostics = {}
    for scenario, arms in report['results'].items():
        costs = spec['scenarios'][scenario]; scenarios[scenario] = {}
        for arm, result in arms.items():
            ll = read(f'replays/{scenario}/{arm}.ledger.jsonl')
            scenarios[scenario][arm] = decompose(ll, result['metrics'], D(costs['entry_price_premium']))
            if scenario == 'cost_assumption':
                diagnostics[arm] = ledger_summary(ll, costs, spec['environment']['min_round_trip_edge'])
                dd = read(f'replays/{scenario}/{arm}.decisions.jsonl')
                buy_ticks = [d for d in dd if any(d['mask'][i] for i in range(1, 25) if (i-1)%6 < 4)]
                count = sum(d['action'] != 0 and (d['action']-1)%6 < 4 for d in buy_ticks)
                diagnostics[arm].update(buy_legal_ticks=len(buy_ticks), buy_actions_on_buy_legal_ticks=count,
                    buy_action_fraction_when_buy_legal=count/len(buy_ticks) if buy_ticks else None)
    means = {}
    fields = ('same_fill_raw_price_pnl_usd', 'price_premium_drag_usd', 'fees_usd', 'net_pnl_usd')
    for scenario, arms in scenarios.items():
        means[scenario] = {family: {field: str(sum((D(arms[f'{family}_{seed}'][field]) for seed in spec['seeds']), D(0))/len(spec['seeds']))
            for field in fields} for family in spec['families']}
    value = {'status': 'passed', 'source_report_sha256': sha256_file(root/'report.json'),
        'source_config_sha256': sha256_file(root/'config.json'), 'consumed_artifact_hashes': consumed,
        'script_hashes': {n: sha256_file(Path(__file__).with_name(n)) for n in ('diagnose_allocation_losses.py',
            'diagnose_allocation_optimizer.py', 'summarize_execution_costs.py')},
        'same_fill_decompositions': scenarios, 'family_means': means, 'main_cost_behavior': diagnostics,
        'semantics': 'Same recorded fill times and quantities. Arithmetic attribution, not a cost-free policy rerun or a causal exit-rule ablation.',
        'limitations': ['Prices are historical trade proxies, premium and fees are simulation assumptions.',
            'Mean-reversion targets are price proxies, not verified conditional expected returns.',
            'Exit-reason PnL is conditional on triggering that exit; it does not identify the effect of removing risk controls.',
            'Only 12 independent training event groups; no generalization claim.'],
        'new_training': False, 'validation_replayed': False, 'final_test_opened': False}
    output.mkdir(parents=True); (output/'report.json').write_text(json_text(value))
    print(json_text({'main_cost_family_means': means['cost_assumption'], 'main_cost_behavior': diagnostics}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); diagnose(a.run, a.output)
