"""Independent arithmetic/mask checks for the fixed training-only comparison."""
import argparse
from collections import Counter
from decimal import Decimal as D, ROUND_DOWN
from pathlib import Path

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError


def check(condition, message):
    if not condition: raise ValidationError(message)


def audit(root):
    report = strict_json((root/'report.json').read_text())
    spec = strict_json((root/'config.json').read_text())
    training = strict_json((root/'training_report.json').read_text())
    reproduction = strict_json((root/'audit.json').read_text())
    check(reproduction['status'] == 'passed' and reproduction['report_sha256'] == sha256_file(root/'report.json'), 'missing full reproduction')
    for name, digest in report['artifact_hashes'].items(): check(sha256_file(root/name) == digest, 'changed artifact: '+name)
    steps = [r['timesteps_by_fee_fraction'] for r in training['models'].values()]
    check(all(s == steps[0] for s in steps), 'unequal fee curriculum exposure')
    optimizer_steps = {}
    for name, row in training['models'].items():
        if row['family'] == 'ppo':
            expected_epochs = spec['total_timesteps']//spec['ppo']['n_steps']*spec['ppo']['n_epochs']
            epochs = row.get('optimizer_epochs', row['gradient_updates'])
            check(epochs == expected_epochs, 'PPO epoch budget differs')
            steps = epochs * (spec['ppo']['n_steps']//spec['ppo']['batch_size'])
            optimizer_steps[name] = {'optimizer_steps': steps, 'optimizer_epochs': epochs,
                'legacy_gradient_updates_field_counts_epochs': 'optimizer_epochs' not in row}
        else:
            optimizer_steps[name] = {'optimizer_steps': row['gradient_updates']}
    result = {}; total_decisions = 0; total_fills = 0
    for scenario, arms in report['results'].items():
        result[scenario] = {}; fee_rate = D(spec['scenarios'][scenario]['fee_fraction'])
        for arm, value in arms.items():
            folder = root/'replays'/scenario; metrics = value['metrics']
            ledger = rows(folder/f'{arm}.ledger.jsonl'); decisions = rows(folder/f'{arm}.decisions.jsonl')
            cash = D(100); fees = D(0); kinds = Counter(); positions = {}
            for row in ledger:
                kind = row['kind']; kinds[kind] += 1
                if kind == 'settle':
                    shares, side = positions.pop(row['market'])
                    payout = shares * D(row['outcome'] if side == 'yes' else 1-row['outcome'])
                    check(payout == D(row['payout']), 'settlement payout differs'); cash += payout
                    continue
                if kind not in ('buy_fill', 'sell_fill'):
                    check(kind in ('buy_order', 'sell_order', 'cancel'), 'unexpected ledger event: '+kind)
                    continue
                notional = (D(row['shares'])*D(row['price'])).quantize(D('.000001'), rounding=ROUND_DOWN)
                expected_fee = (notional*fee_rate).quantize(D('.000001'), rounding=ROUND_DOWN)
                check(D(row['fee']) == expected_fee, 'fee arithmetic differs')
                if kind == 'buy_fill':
                    check(D(row['cost']) == notional+expected_fee, 'buy cost differs'); cash -= D(row['cost'])
                    check(row['market'] not in positions, 'duplicate position'); positions[row['market']] = (D(row['shares']), row['side'])
                else:
                    check(D(row['proceeds']) == notional-expected_fee, 'sell proceeds differ'); cash += D(row['proceeds'])
                    shares, side = positions[row['market']]; remaining = shares-D(row['shares'])
                    check(remaining >= 0 and side == row['side'], 'oversold position')
                    if remaining: positions[row['market']] = (remaining, side)
                    else: del positions[row['market']]
                fees += expected_fee; check(cash >= 0, 'negative cash'); total_fills += 1
            check(not positions, 'unsettled positions')
            check(cash == D(metrics['final_cash']) and cash-100 == D(metrics['net_pnl']) and fees == D(metrics['fees_paid']), 'ledger/summary differs')
            peak = D(100); drawdown = D(0); drawdown_fraction = D(0)
            for point in rows(folder/f'{arm}.equity_curve.jsonl'):
                equity = D(point['equity_proxy']); peak = max(peak, equity)
                drawdown = max(drawdown, peak-equity); drawdown_fraction = max(drawdown_fraction, (peak-equity)/peak)
                check(D(point['cash']) >= 0 and point['positions'] <= spec['environment']['max_positions'], 'curve risk constraint')
            check(drawdown == D(metrics['max_drawdown_usd']) and drawdown_fraction == D(metrics['max_drawdown_fraction']), 'drawdown differs')
            voluntary = waited = only_wait = 0
            for row in decisions:
                check(row['mask'][0] and row['mask'][row['action']] and row['action'] == row['executed_action'] and not row['weekly_due'], 'invalid/forced action')
                if any(row['mask'][1:]):
                    voluntary += 1; waited += row['action'] == 0
                else: only_wait += 1
            check(not metrics['periodic_activity_required'] and metrics['activity_penalty_usd'] == '0' and metrics['policy_requirements_met'], 'wrong requirements')
            if arm == 'cash_only': check(cash == 100 and not ledger, 'cash baseline changed')
            result[scenario][arm] = {'final_cash': str(cash), 'fees_paid': str(fees), 'fill_count': kinds['buy_fill']+kinds['sell_fill'],
                'decision_count': len(decisions), 'only_wait_legal': only_wait, 'nonwait_legal': voluntary,
                'voluntary_wait_count': waited, 'voluntary_wait_fraction': waited/voluntary if voluntary else None,
                'max_drawdown_fraction': metrics['max_drawdown_fraction']}
            total_decisions += len(decisions)
    out = {'status': 'passed', 'report_sha256': sha256_file(root/'report.json'), 'script_sha256': sha256_file(Path(__file__)),
        'validation_replayed': False, 'final_test_opened': False, 'total_decisions': total_decisions,
        'total_fills': total_fills, 'optimizer_accounting': optimizer_steps, 'fee_curriculum_exposure_equal': True, 'results': result}
    (root/'arithmetic_audit.json').write_text(json_text(out))
    print(json_text({k: v for k, v in out.items() if k != 'results'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); audit(p.parse_args().root)
