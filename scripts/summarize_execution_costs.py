"""Decompose audited execution PnL on exactly the same fill times and quantities."""
import argparse
from decimal import Decimal as D, ROUND_DOWN
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError


def decompose(ledger, metrics, premium):
    fees = D(0)
    price_drag = D(0)
    gross = D(0)
    cash = D(metrics['initial_cash'])
    for row in ledger:
        kind = row['kind']
        if kind == 'settle':
            gross += D(row['payout'])
            cash += D(row['payout'])
        elif kind in ('buy_fill', 'sell_fill'):
            qty, price, fee = D(row['shares']), D(row['price']), D(row['fee'])
            fees += fee
            raw_price = price - premium if kind == 'buy_fill' else price + premium
            raw_notional = (qty * raw_price).quantize(D('.000001'), rounding=ROUND_DOWN)
            if kind == 'buy_fill':
                cost = D(row['cost'])
                gross -= raw_notional
                cash -= cost
                price_drag += cost - fee - raw_notional
            else:
                proceeds = D(row['proceeds'])
                gross += raw_notional
                cash += proceeds
                price_drag += raw_notional - proceeds - fee
    if (cash != D(metrics['final_cash']) or fees != D(metrics['fees_paid'])
            or gross - price_drag - fees != D(metrics['net_pnl'])):
        raise ValidationError('same-fill cost decomposition does not reconcile')
    return {'same_fill_raw_price_pnl_usd': str(gross), 'price_premium_drag_usd': str(price_drag),
            'fees_usd': str(fees), 'net_pnl_usd': metrics['net_pnl']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    output = root / 'cost_decomposition.json'
    if output.exists():
        raise ValidationError('cost decomposition already exists')
    report = strict_json((root / 'report.json').read_text())
    audit = strict_json((root / 'audit.json').read_text())
    report_hash = sha256_file(root / 'report.json')
    if audit['status'] != 'passed' or audit['report_sha256'] != report_hash:
        raise ValidationError('matching replay audit required')
    if sha256_file(root / 'plan.json') != report['plan_sha256']:
        raise ValidationError('execution plan changed')
    plan = strict_json((root / 'plan.json').read_text())
    results = {}
    for scenario, lifetimes in report['scenarios'].items():
        results[scenario] = {}
        scenarios = plan.get('execution_scenarios', plan['original_config']['scenarios'])
        premium = D(scenarios[scenario]['entry_price_premium'])
        for ttl, arms in lifetimes.items():
            results[scenario][ttl] = {}
            for arm, result in arms.items():
                name = f'{scenario}/{ttl}/{arm}.ledger.jsonl'
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValidationError('invalid ledger path')
                digest = sha256_file(root / name)
                if digest != report['artifact_hashes'][name]:
                    raise ValidationError('execution ledger changed')
                results[scenario][ttl][arm] = {**decompose(rows(root / name), result['metrics'], premium),
                                              'ledger_sha256': digest}
    output.write_text(json_text({'source_report_sha256': report_hash, 'source_script_sha256': sha256_file(Path(__file__)),
        'semantics': 'same fills and quantities, not a rerun of policy decisions without costs', 'scenarios': results}))
    print(json_text({'status': 'passed', 'decompositions': sum(len(a) for s in results.values() for a in s.values()),
                     'output': str(output), 'sha256': sha256_file(output)}))


if __name__ == '__main__':
    main()
