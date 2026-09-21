"""Low-turnover event allocation with a separate, outcome-blind decision rule.

Trade prints are hypothetical execution references. Settlement belongs only to
this executor; the decision rule receives neither labels nor settlement times.
"""
from collections import Counter
from datetime import timedelta
from decimal import Decimal as D, ROUND_DOWN
import heapq

from .paper_trading import decimal
from .schema import ValidationError, binary_outcome, iso, timestamp
from .trading_rl_agents_env import SignalIndex

MICRO = D('.000001')


def validate_policy(c):
    numeric = ('initial_cash', 'max_trade_usd', 'min_trade_usd', 'max_event_usd', 'max_portfolio_usd',
               'entry_edge', 'exit_edge', 'uncertainty_margin', 'stop_price_points', 'min_yes_price', 'max_yes_price', 'material_probability_change')
    integer = ('max_positions', 'max_entries_per_day', 'quote_max_age_seconds', 'order_ttl_seconds',
               'cooldown_seconds', 'monitor_seconds', 'review_seconds')
    if set(c) != set(numeric + integer): raise ValidationError('event policy fields differ')
    p = {k: decimal(v) if k in numeric else v for k, v in c.items()}
    if (any(type(p[k]) is not int or p[k] < 1 for k in integer)
            or not 0 < p['min_trade_usd'] <= p['max_trade_usd'] <= p['max_event_usd'] <= p['max_portfolio_usd'] <= p['initial_cash']
            or not 0 <= p['exit_edge'] < p['entry_edge'] < 1 or not 0 <= p['uncertainty_margin'] < 1
            or not 0 < p['material_probability_change'] < 1 or p['review_seconds'] < p['monitor_seconds']
            or not 0 < p['stop_price_points'] < 1 or not 0 < p['min_yes_price'] < p['max_yes_price'] < 1):
        raise ValidationError('invalid event policy')
    return p


def choice(probability, q, p, costs, position=None):
    """Only current belief, price, sunk position state, risk settings and costs."""
    fee, premium = decimal(costs['fee_fraction']), decimal(costs['entry_price_premium'])
    if not 0 < q < 1 or not 0 <= probability <= 1: raise ValidationError('invalid belief/quote')
    u = p['uncertainty_margin']
    if position is not None:
        belief = probability if position['side'] == 'yes' else 1-probability
        price = q if position['side'] == 'yes' else 1-q
        net_bid = max(D(0), price-premium)*(1-fee)
        # Compare future holding value with sale proceeds; do not charge entry fees again.
        optimistic_hold = min(D(1), belief+u)
        return {'kind': 'sell', 'reason': 'advantage_realized', 'residual_edge': optimistic_hold-net_bid} if (
            net_bid > 0 and optimistic_hold-net_bid <= p['exit_edge']) else None
    if not p['min_yes_price'] <= q <= p['max_yes_price']: return None
    values = []
    for side, belief, reference in [('yes', probability, q), ('no', 1-probability, 1-q)]:
        conservative = max(D(0), belief-u); unit = (reference+premium)*(1+fee)
        if reference+premium < 1 and conservative-unit >= p['entry_edge']:
            values.append({'kind': 'buy', 'side': side, 'belief': probability, 'conservative_edge': conservative-unit})
    return max(values, key=lambda r: r['conservative_edge']) if values else None


def simulate(signals, markets, settlements, feed, config, costs, start, end, *, max_holding_seconds=None,
             anchor_only=False, cash_only=False, forecast_admission=None, legacy_unvalidated_research=False):
    """New calls require forecast admission; legacy bypass is for frozen controls only."""
    if type(legacy_unvalidated_research) is not bool or legacy_unvalidated_research and forecast_admission is not None:
        raise ValidationError('ambiguous forecast admission policy')
    p = validate_policy(config)
    if (start.tzinfo is None or end.tzinfo is None or start >= end
            or max_holding_seconds is not None and (type(max_holding_seconds) is not int or max_holding_seconds < 1)):
        raise ValidationError('invalid replay horizon')
    if set(costs) != {'fee_fraction', 'entry_price_premium'} or any(not 0 <= decimal(x) <= D('.05') for x in costs.values()):
        raise ValidationError('invalid explicit costs')
    fee = decimal(costs['fee_fraction']); premium = decimal(costs['entry_price_premium'])
    groups = {m['market_id']: m['event_group_id'] for m in markets}
    initial = {m['market_id']: timestamp(m['initialized_at'], 'initialization') for m in markets}
    if len(groups) != len(markets): raise ValidationError('duplicate market')
    labels = {}
    for s in settlements:
        binary_outcome(s.outcome)
        if s.market_id in labels or s.market_id not in groups or not start < s.time < end:
            raise ValidationError('unsettled/outside-cohort label')
        labels[s.market_id] = s
    if set(labels) != set(groups): raise ValidationError('incomplete settlement coverage')
    index = SignalIndex(signals, groups)
    for row in signals:
        if timestamp(row['observation_time'], 'observation') < initial[row['market_id']]:
            raise ValidationError('signal before market initialization')
        if anchor_only and row['reference_probability'] is None: raise ValidationError('missing market anchor')
    events = []; serial = 0
    def push(t, priority, kind, value):
        nonlocal serial
        serial += 1; heapq.heappush(events, (t, priority, serial, kind, value))
    # These settlement facts never enter choice(), candidate ranking or order sizing.
    for s in settlements: push(s.time, 0, 'settlement', s)
    for row in signals:
        ready = timestamp(row['available_at'], 'availability')
        if start <= ready < end: push(ready, 2, 'research', row['market_id'])
    now = start
    while now < end:
        push(now, 3, 'monitor', None); now += timedelta(seconds=p['monitor_seconds'])
    push(end, 3, 'monitor', None)
    cash = p['initial_cash']; positions = {}; pending = {}; resolved = set(); cooldown = {}; attempts = Counter()
    fees = D(0); turnover = D(0); realized = D(0); peak = cash; dd = D(0); ddf = D(0); occupied = D(0)
    ledger = []; curve = []; decisions = []; reasons = Counter(); holdings = []; order_id = 0; previous = start
    next_review = {}; last_research_belief = {}
    def record(t, kind, **values):
        ledger.append({'time': iso(t), 'kind': kind, **{k: str(v) if isinstance(v, D) else v for k, v in values.items()}})
    def exposure(group=None):
        return sum((v['cost'] for m, v in positions.items() if group is None or groups[m] == group), D(0)) + sum(
            (v['budget'] for m, v in pending.items() if v['kind'] == 'buy' and (group is None or groups[m] == group)), D(0))
    def mark(t):
        nonlocal peak, dd, ddf
        value = cash + sum((o['budget'] for o in pending.values() if o['kind'] == 'buy'), D(0)); stale = 0
        for mid, pos in positions.items():
            quote = feed.latest(mid, t)
            if quote is None: q = pos['reference_yes']; stale += 1
            else:
                q, qt = quote
                if qt > t or not 0 < q < 1: raise ValidationError('future/invalid mark')
                stale += (t-qt).total_seconds() > p['quote_max_age_seconds']
            price = q if pos['side'] == 'yes' else 1-q
            value += pos['shares']*max(D(0), price-premium)*(1-fee)
        peak = max(peak, value); dd = max(dd, peak-value); ddf = max(ddf, (peak-value)/peak)
        curve.append({'time': iso(t), 'cash': str(cash), 'equity_proxy': str(value), 'positions': len(positions),
                      'reserved': str(sum((o['budget'] for o in pending.values() if o['kind'] == 'buy'), D(0))), 'stale_marks': int(stale)})
    def valid_quote(mid, t):
        quote = feed.latest(mid, t)
        if quote is None: return None
        q, qt = quote
        if qt > t or not 0 < q < 1: raise ValidationError('future/invalid decision quote')
        return q if (t-qt).total_seconds() <= p['quote_max_age_seconds'] else None
    def get_belief(mid, t):
        row = index.at(mid, t)
        if row is None: return None, None
        return decimal(row['reference_probability'] if anchor_only else row['content']['probability']), row
    def admission(row, t):
        if legacy_unvalidated_research:
            return {'allowed': True, 'reason': 'legacy_unvalidated_research', 'qualification_id': None}
        if anchor_only:
            return {'allowed': False, 'reason': 'market_reference_is_not_forecast', 'qualification_id': None}
        if forecast_admission is None:
            return {'allowed': False, 'reason': 'missing_forecast_admission', 'qualification_id': None}
        return forecast_admission.assess(row, t)
    def submit(t, mid, action, row=None, budget=D(0)):
        nonlocal order_id, cash
        order_id += 1; oid = str(order_id)
        expiry = t+timedelta(seconds=p['order_ttl_seconds'])
        if action['kind'] == 'buy':
            expiry = min(expiry, timestamp(row['expires_at'], 'signal expiry')); cash -= budget
            attempts[t.date()] += 1
        order = {**action, 'order_id': oid, 'budget': budget, 'submitted_at': t, 'expires_at': expiry,
                 'signal_id': row['sample_id'] if row else None}
        pending[mid] = order
        record(t, action['kind']+'_order', market=mid, order_id=oid, reason=action.get('reason', 'cost_adjusted_advantage'),
               side=action.get('side'), budget=budget, signal_id=order['signal_id'], expires_at=iso(expiry))
        quote = feed.next(mid, t, expiry)
        if quote is not None and (not t < quote[1] <= expiry or not 0 < quote[0] < 1): raise ValidationError('invalid future fill reference')
        push(expiry if quote is None else quote[1], 1, 'fill', (mid, oid, quote))
    while events:
        t, _, _, kind, item = heapq.heappop(events)
        if t > end: raise ValidationError('pending order beyond replay horizon')
        occupied += sum((v['cost'] for v in positions.values()), D(0))*D(str((t-previous).total_seconds()))/D(86400)
        previous = t
        if kind == 'settlement':
            mid = item.market_id; resolved.add(mid)
            if mid in pending:
                order = pending.pop(mid)
                if order['kind'] == 'buy': cash += order['budget']
                record(t, 'cancel', market=mid, order_id=order['order_id'], reason='settled_before_fill')
            if mid in positions:
                pos = positions.pop(mid); payout = pos['shares']*(item.outcome if pos['side'] == 'yes' else 1-item.outcome)
                pnl = payout-pos['cost']; cash += payout; realized += pnl
                hours = (t-pos['filled_at']).total_seconds()/3600; holdings.append(hours)
                record(t, 'settle', market=mid, payout=payout, net_pnl=pnl, outcome=item.outcome, holding_hours=hours)
        elif kind == 'fill':
            mid, oid, quote = item
            if mid not in pending or pending[mid]['order_id'] != oid: continue
            order = pending.pop(mid)
            if order['kind'] == 'buy': cash += order['budget']
            why = 'no_post_decision_print' if quote is None else None
            if quote is not None:
                q, qt = quote
                if order['kind'] == 'buy':
                    belief, row = get_belief(mid, t)
                    offered = None if belief is None else choice(belief, q, p, costs)
                    if row is None or row['sample_id'] != order['signal_id']: why = 'expired_failed_or_superseded_signal'
                    elif not admission(row, t)['allowed']: why = 'forecast_admission_revoked_before_fill'
                    elif offered is None or offered['side'] != order['side']: why = 'edge_eroded'
                    else:
                        price = (q if order['side'] == 'yes' else 1-q)+premium
                        shares = (order['budget']/(price*(1+fee))).quantize(MICRO, rounding=ROUND_DOWN)
                        notional = (shares*price).quantize(MICRO, rounding=ROUND_DOWN)
                        charge = (notional*fee).quantize(MICRO, rounding=ROUND_DOWN); cost = notional+charge
                        cash -= cost; fees += charge; turnover += notional
                        positions[mid] = {'side': order['side'], 'shares': shares, 'cost': cost, 'filled_at': t, 'reference_yes': q}
                        record(t, 'buy_fill', market=mid, order_id=oid, side=order['side'], shares=shares, price=price,
                               reference_yes=q, cost=cost, fee=charge, signal_id=order['signal_id'])
                else:
                    pos = positions[mid]; price = (q if pos['side'] == 'yes' else 1-q)-premium
                    if price <= 0: why = 'nonpositive_synthetic_bid'
                    elif order['reason'] == 'advantage_realized':
                        belief, row = get_belief(mid, t)
                        if belief is None or not admission(row, t)['allowed'] or choice(belief, q, p, costs, pos) is None: why = 'exit_advantage_changed'
                    if why is None:
                        notional = (pos['shares']*price).quantize(MICRO, rounding=ROUND_DOWN)
                        charge = (notional*fee).quantize(MICRO, rounding=ROUND_DOWN); proceeds = notional-charge
                        pnl = proceeds-pos['cost']; cash += proceeds; fees += charge; turnover += notional; realized += pnl
                        hours = (t-pos['filled_at']).total_seconds()/3600; holdings.append(hours)
                        record(t, 'sell_fill', market=mid, order_id=oid, side=pos['side'], shares=pos['shares'], price=price,
                               proceeds=proceeds, fee=charge, net_pnl=pnl, reason=order['reason'], holding_hours=hours)
                        del positions[mid]; cooldown[mid] = t+timedelta(seconds=p['cooldown_seconds'])
            if why:
                reasons[why] += 1; record(t, 'cancel', market=mid, order_id=oid, reason=why)
        else:
            watched = sorted(groups) if kind == 'monitor' else [item]
            entries = []
            for mid in watched:
                if mid in resolved or initial[mid] > t or mid in pending: continue
                q = valid_quote(mid, t)
                if q is None: continue
                belief, row = get_belief(mid, t); pos = positions.get(mid)
                admitted = admission(row, t)
                action = None; reason = None
                material_update = False
                if kind == 'research' and belief is not None:
                    prior = last_research_belief.get(mid)
                    material_update = prior is None or abs(belief-prior) >= p['material_probability_change']
                    last_research_belief[mid] = belief
                routine_review = t >= next_review.get(mid, start)
                ordinary_decision = routine_review or material_update
                if ordinary_decision:
                    next_review[mid] = t+timedelta(seconds=p['review_seconds'])
                if cash_only: reason = 'cash_control'
                elif pos is not None:
                    side_price = q if pos['side'] == 'yes' else 1-q
                    entered = pos['reference_yes'] if pos['side'] == 'yes' else 1-pos['reference_yes']
                    if side_price <= entered-p['stop_price_points']: action = {'kind': 'sell', 'reason': 'stop_loss'}
                    elif max_holding_seconds is not None and (t-pos['filled_at']).total_seconds() >= max_holding_seconds:
                        action = {'kind': 'sell', 'reason': 'holding_limit'}
                    elif belief is not None and admitted['allowed'] and ordinary_decision: action = choice(belief, q, p, costs, pos)
                    reason = 'hold_position' if belief is not None else 'hold_position_without_fresh_forecast'
                elif not ordinary_decision: reason = 'between_event_reviews'
                elif belief is None: reason = 'missing_or_expired_forecast'
                elif not admitted['allowed']: reason = admitted['reason']
                elif t < cooldown.get(mid, start): reason = 'cooldown'
                elif t+timedelta(seconds=p['order_ttl_seconds']) > end: reason = 'administrative_entry_cutoff'
                else: action = choice(belief, q, p, costs); reason = 'no_cost_adjusted_advantage'
                if action and action['kind'] == 'sell': submit(t, mid, action, row)
                elif action: entries.append((action, mid, row))
                decisions.append({'time': iso(t), 'market': mid, 'trigger': kind, 'quote': str(q),
                    'signal_id': row['sample_id'] if row else None, 'ordinary_review': ordinary_decision,
                    'material_research_update': material_update, 'action': action['kind'] if action else 'hold',
                    'reason': action.get('reason', 'entry_candidate') if action else reason})
                if not legacy_unvalidated_research:
                    decisions[-1]['forecast_admission'] = admitted
            # Compare all currently offered entries before reserving scarce capital.
            for action, mid, row in sorted(entries, key=lambda v: (-v[0]['conservative_edge'], v[1])):
                budget = min(p['max_trade_usd'], cash, p['max_event_usd']-exposure(groups[mid]), p['max_portfolio_usd']-exposure())
                if (budget < p['min_trade_usd'] or len(set(positions)|set(pending)) >= p['max_positions']
                        or attempts[t.date()] >= p['max_entries_per_day']):
                    reasons['capital_position_or_entry_limit'] += 1; continue
                submit(t, mid, action, row, budget)
        if (cash < 0 or exposure() > p['max_portfolio_usd'] or len(set(positions)|set(pending)) > p['max_positions']
                or any(exposure(g) > p['max_event_usd'] for g in set(groups.values()))
                or cash+exposure() != p['initial_cash']+realized): raise ValidationError('cash/exposure invariant')
        mark(t)
    if positions or pending or cash != p['initial_cash']+realized: raise ValidationError('unresolved terminal positions')
    buys = [r for r in ledger if r['kind'] == 'buy_fill']; sells = [r for r in ledger if r['kind'] == 'sell_fill']
    settled = [r for r in ledger if r['kind'] == 'settle']
    return {'metrics': {'initial_cash': str(p['initial_cash']), 'final_cash': str(cash), 'net_pnl': str(realized),
        'fees_paid': str(fees), 'gross_turnover_usd': str(turnover), 'turnover_initial_capital_multiple': str(turnover/p['initial_cash']),
        'entry_count': len(buys), 'exit_count': len(sells), 'settlement_count': len(settled),
        'held_to_settlement_fraction': len(settled)/len(buys) if buys else None,
        'mean_holding_hours': sum(holdings)/len(holdings) if holdings else None,
        'max_holding_hours': max(holdings) if holdings else None, 'capital_days_usd': str(occupied),
        'hourly_and_event_sampled_max_drawdown_usd': str(dd), 'hourly_and_event_sampled_max_drawdown_fraction': str(ddf),
        'equity_points_with_stale_marks': sum(r['stale_marks'] > 0 for r in curve),
        'orders': sum(r['kind'].endswith('_order') for r in ledger), 'rejections': dict(reasons),
        'entries_by_event': dict(Counter(groups[r['market']] for r in buys)),
        'exits_by_event': dict(Counter(groups[r['market']] for r in sells)),
        'periodic_activity_required': False, 'training': False, 'simulation_only': True, 'real_orders_sent': 0},
        'ledger': ledger, 'decisions': decisions, 'equity_curve': curve}
