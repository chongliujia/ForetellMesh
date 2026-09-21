"""Independent ledger, cadence, signal-availability and holding audit."""
import argparse
from collections import Counter
from decimal import Decimal as D, ROUND_DOWN
from pathlib import Path

from foretellmesh.data import strict_json, sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.market_development import rows
from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.trading_rl_agents_env import SignalIndex
from summarize_execution_costs import decompose


def check(ok, why):
    if not ok: raise ValidationError(why)


def audit(root):
    r = strict_json((root/'report.json').read_text()); s = strict_json((root/'config.json').read_text())
    a = strict_json((root/'audit.json').read_text()); p = s['policy']
    check(a['status'] == 'passed' and a['report_sha256'] == sha256_file(root/'report.json'), 'full reproduction required')
    for name, digest in r['artifact_hashes'].items(): check(sha256_file(root/name) == digest, 'changed '+name)
    markets = {m['market_id']: m['event_group_id'] for m in rows(root/'markets.jsonl')}
    signals = SignalIndex(rows(root/'signals.jsonl'), markets)
    total_fills = total_decisions = 0; results = {}
    for scenario, values in r['results'].items():
        results[scenario] = {}; costs = s['scenarios'][scenario]; fee = D(costs['fee_fraction']); premium = D(costs['entry_price_premium'])
        for arm in s['arms']:
            m = values[arm]; prefix = root/'replays'/scenario/arm
            ledger = rows(Path(str(prefix)+'.ledger.jsonl')); decisions = rows(Path(str(prefix)+'.decisions.jsonl'))
            cash = D(100); fees = D(0); turnover = D(0); realized = D(0); positions = {}; pending = {}; orders = {}
            holds = []; gaps = []; last_entry = {}; capital_days = D(0); entries = Counter(); exits = Counter(); exit_reasons = Counter()
            def belief(row): return D(str(row['reference_probability'] if s['arms'][arm].get('anchor_only') else row['content']['probability']))
            for row in ledger:
                kind = row['kind']; mid = row['market']; t = timestamp(row['time'], 'ledger time')
                if kind.endswith('_order'):
                    check(mid not in pending and row['order_id'] not in orders, 'overlapping/duplicate order')
                    pending[mid] = row; orders[row['order_id']] = row
                    if kind == 'buy_order':
                        check(mid not in positions and D(p['min_trade_usd']) <= D(row['budget']) <= D(p['max_trade_usd']), 'invalid entry')
                        cash -= D(row['budget'])
                elif kind == 'cancel':
                    order = pending.pop(mid); check(order['order_id'] == row['order_id'], 'wrong cancellation')
                    if order['kind'] == 'buy_order': cash += D(order['budget'])
                elif kind in ('buy_fill', 'sell_fill'):
                    order = pending.pop(mid)
                    check(order['order_id'] == row['order_id'] and timestamp(order['time'], 'submission') < t <= timestamp(order['expires_at'], 'expiry'), 'fill not after decision/inside TTL')
                    shares = D(row['shares']); price = D(row['price'])
                    notional = (shares*price).quantize(D('.000001'), rounding=ROUND_DOWN)
                    charge = (notional*fee).quantize(D('.000001'), rounding=ROUND_DOWN)
                    check(charge == D(row['fee']), 'fee mismatch'); fees += charge; turnover += notional
                    if kind == 'buy_fill':
                        signal = signals.at(mid, t)
                        check(signal is not None and signal['sample_id'] == row['signal_id'] == order['signal_id'], 'stale/failed/future entry signal')
                        prob = belief(signal); side_prob = prob if row['side'] == 'yes' else 1-prob
                        check(max(D(0), side_prob-D(p['uncertainty_margin']))-price*(1+fee) >= D(p['entry_edge']), 'no entry advantage')
                        reference = D(row['reference_yes'])
                        check(price == (reference if row['side'] == 'yes' else 1-reference)+premium, 'entry price differs')
                        check(D(row['cost']) == notional+charge, 'buy arithmetic')
                        cash += D(order['budget'])-D(row['cost']); positions[mid] = row; entries[markets[mid]] += 1
                        if mid in last_entry: gaps.append((t-last_entry[mid]).total_seconds()/86400)
                        last_entry[mid] = t
                    else:
                        pos = positions.pop(mid); check(row['side'] == pos['side'] and shares == D(pos['shares']), 'oversold/mismatched position')
                        proceeds = notional-charge; check(proceeds == D(row['proceeds']), 'sell arithmetic')
                        pnl = proceeds-D(pos['cost']); check(pnl == D(row['net_pnl']), 'sell PnL')
                        cash += proceeds; realized += pnl; exits[markets[mid]] += 1; exit_reasons[row['reason']] += 1
                        hours = (t-timestamp(pos['time'], 'entry time')).total_seconds()/3600
                        check(abs(hours-row['holding_hours']) < 1e-9, 'holding hours differ'); holds.append(hours)
                        capital_days += D(pos['cost'])*D(str((t-timestamp(pos['time'], 'entry')).total_seconds()))/D(86400)
                    total_fills += 1
                elif kind == 'settle':
                    pos = positions.pop(mid); expected = D(pos['shares'])*(row['outcome'] if pos['side'] == 'yes' else 1-row['outcome'])
                    check(expected == D(row['payout']) and expected-D(pos['cost']) == D(row['net_pnl']), 'settlement arithmetic')
                    cash += expected; realized += expected-D(pos['cost'])
                    hours = (t-timestamp(pos['time'], 'entry')).total_seconds()/3600; holds.append(hours)
                    check(abs(hours-row['holding_hours']) < 1e-9, 'settlement holding time')
                    capital_days += D(pos['cost'])*D(str((t-timestamp(pos['time'], 'entry')).total_seconds()))/D(86400)
                else: raise ValidationError('unknown ledger kind')
                exposure = sum((D(pos['cost']) for pos in positions.values()), D(0))+sum((D(o['budget']) for o in pending.values() if o['kind'] == 'buy_order'), D(0))
                check(cash >= 0 and exposure <= D(p['max_portfolio_usd']) and cash+exposure == 100+realized, 'capital invariant')
                check(len(set(positions)|set(pending)) <= p['max_positions'], 'position cap')
                for group in set(markets.values()):
                    used = sum((D(v['cost']) for k,v in positions.items() if markets[k] == group), D(0))+sum((D(v['budget']) for k,v in pending.items() if v['kind'] == 'buy_order' and markets[k] == group), D(0))
                    check(used <= D(p['max_event_usd']), 'event cap')
            check(not positions and not pending and cash == D(m['final_cash']) and realized == D(m['net_pnl']) and fees == D(m['fees_paid']), 'terminal accounting')
            check(turnover == D(m['gross_turnover_usd']) and abs(capital_days-D(m['capital_days_usd'])) < D('1e-18'), 'turnover/capital days')
            check(dict(entries) == m['entries_by_event'] and dict(exits) == m['exits_by_event'], 'event counts')
            last_review = {}; last_research = {}
            for d in decisions:
                t = timestamp(d['time'], 'decision'); mid = d['market']; signal = signals.at(mid, t)
                check(d['signal_id'] == (signal['sample_id'] if signal else None), 'decision signal differs')
                material = False
                if d['trigger'] == 'research' and signal is not None:
                    b = belief(signal); material = mid not in last_research or abs(b-last_research[mid]) >= D(p['material_probability_change']); last_research[mid] = b
                check(material == d['material_research_update'], 'material-update flag differs')
                if d['ordinary_review']:
                    check(material or mid not in last_review or (t-last_review[mid]).total_seconds() >= p['review_seconds'], 'ordinary trading review too frequent')
                    last_review[mid] = t
                if d['action'] == 'buy' or d['reason'] == 'advantage_realized': check(d['ordinary_review'], 'ordinary trade triggered by hourly monitoring')
                total_decisions += 1
            curve = rows(Path(str(prefix)+'.equity_curve.jsonl')); peak = D(100); dd = D(0); ddf = D(0)
            for point in curve:
                equity = D(point['equity_proxy']); peak = max(peak, equity); dd = max(dd, peak-equity); ddf = max(ddf, (peak-equity)/peak)
                check(D(point['cash']) >= 0, 'negative curve cash')
            check(dd == D(m['hourly_and_event_sampled_max_drawdown_usd']) and ddf == D(m['hourly_and_event_sampled_max_drawdown_fraction']), 'drawdown mismatch')
            results[scenario][arm] = {'cost_decomposition': decompose(ledger, m, premium), 'holding_hours': holds,
                'within_market_reentry_gap_days': gaps, 'exit_fill_reasons': dict(exit_reasons)}
    output = {'status':'passed', 'report_sha256':sha256_file(root/'report.json'), 'script_sha256':sha256_file(Path(__file__)),
        'cases':sum(len(x) for x in results.values()), 'audited_decisions':total_decisions,'audited_buy_sell_fills':total_fills,
        'results':results,'validation_replayed':False,'final_test_opened':False}
    (root/'mechanism_audit.json').write_text(json_text(output))
    print(json_text({k:v for k,v in output.items() if k!='results'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);audit(parser.parse_args().run)
