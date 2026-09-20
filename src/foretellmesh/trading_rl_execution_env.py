"""Pending-order lifecycle for offline first-print execution sensitivity tests.

This is a historical print proxy, not an exchange limit-order simulator. The
first print either fills under the original price rules or cancels the order.
"""
from datetime import timedelta
from decimal import Decimal as D
import heapq

from .schema import ValidationError, iso
from .trading_rl_env import AllocationEnv


FORCED_EXITS = {'stop_loss', 'holding_limit', 'scope_end', 'calendar_end'}


class PendingAllocationEnv(AllocationEnv):
    def __init__(self, data, settlements, config, costs, *, order_ttl_seconds: int, record: bool = False):
        if type(order_ttl_seconds) is not int or order_ttl_seconds not in (300, 3600, 14400):
            raise ValidationError('unsupported predeclared order lifetime')
        if config['fill_window_seconds'] != 300:
            raise ValidationError('terminal closeout window must remain 300 seconds')
        self.order_ttl_seconds = order_ttl_seconds
        super().__init__(data, settlements, config, costs, record=record)

    def reset(self, *, seed=None, options=None):
        self.peak_reserved_cash = D(0)
        self.peak_open_or_reserved_positions = 0
        self.risk_replacements = 0
        return super().reset(seed=seed, options=options)

    def _pending_for(self, mid):
        return next(((oid, o) for oid, o in self.pending.items() if o['market'] == mid), None)

    def _reserved_markets(self):
        return {o['market'] for o in self.pending.values() if o['kind'] == 'buy'}

    def _can_buy(self, mid, state, side, budget):
        if self._pending_for(mid) or len(set(self.positions) | self._reserved_markets()) >= self.p['max_positions']:
            return False
        return super()._can_buy(mid, state, side, budget)

    def _auto_reason(self, mid, pos):
        pending = self._pending_for(mid)
        # An existing full exit is already working. A partial exit may be
        # canceled and replaced by a newly required full risk exit.
        if pending and pending[1]['kind'] == 'sell' and pending[1]['fraction'] == 1:
            return None
        return super()._auto_reason(mid, pos)

    def _refresh(self):
        super()._refresh()
        for j, mid in enumerate(self.slots):
            if self._pending_for(mid):
                self.mask[1 + j * 6:1 + (j + 1) * 6] = False
        self.mask[0] = not (self.due and self.mask[1:].any() and not any(self.auto.values()))

    def _check(self):
        super()._check()
        pending_markets = [o['market'] for o in self.pending.values()]
        reserved = self._reserved_markets()
        count = len(set(self.positions) | reserved)
        if (len(set(pending_markets)) != len(pending_markets) or reserved & set(self.positions)
                or count > self.p['max_positions']
                or any(o['kind'] == 'sell' and o['market'] not in self.positions for o in self.pending.values())):
            raise ValidationError('pending order or reserved position invariant failed')
        self.peak_reserved_cash = max(self.peak_reserved_cash, sum(
            (o['budget'] for o in self.pending.values() if o['kind'] == 'buy'), D(0)))
        self.peak_open_or_reserved_positions = max(self.peak_open_or_reserved_positions, count)

    def _cancel(self, t, oid, reason):
        order = self.pending.pop(oid, None)
        if order is None:
            return
        if order['kind'] == 'buy':
            self.cash += order['budget']
        self._log(t, 'cancel', order_id=oid, reason=reason)
        self.events = [event for event in self.events if not (event[3] == 'fill' and event[4][0] == oid)]
        heapq.heapify(self.events)
        self._check()

    def _order(self, t, kind, mid, *, side=None, budget=None, fraction=D(1), reason='policy'):
        pending = self._pending_for(mid)
        if pending:
            oid, old = pending
            if kind == 'sell' and reason in FORCED_EXITS and old['kind'] == 'sell' and old['fraction'] < 1:
                self._cancel(t, oid, 'replaced_by_risk_exit')
                self.risk_replacements += 1
            else:
                return
        state = self.data.states[self.i].get(mid, {})
        if kind == 'buy':
            if side not in ('yes', 'no') or budget not in (D(1), D(2)) or not self._can_buy(mid, state, side, budget):
                raise ValidationError('new buy violates available cash, scope or reserved-position limits')
        elif kind != 'sell' or mid not in self.positions or not D(0) < fraction <= 1:
            raise ValidationError('invalid pending exit')
        quote = self._quote(mid, t)
        if quote is None:
            raise ValidationError('order without causal reference')
        ttl = self.p['fill_window_seconds'] if reason == 'calendar_end' else self.order_ttl_seconds
        expires = t + timedelta(seconds=ttl)
        if reason != 'calendar_end':
            expires = min(expires, self.data.end)
            if kind == 'buy':
                expires = min(expires, self.windows[mid].retire_at)
        if expires <= t:
            raise ValidationError('empty pending-order lifetime')
        self.order_id += 1
        oid = str(self.order_id)
        order = {'kind': kind, 'market': mid, 'side': side, 'budget': budget, 'fraction': fraction,
                 'reason': reason, 'reference_yes': quote[0],
                 'target': None if state.get('mean_yes_price') is None else D(str(state['mean_yes_price'])),
                 'submitted_at': iso(t), 'expires_at': iso(expires)}
        if kind == 'buy':
            self.cash -= budget
            self.attempts[t.date()] += 1
        self.pending[oid] = order
        self._log(t, kind + '_order', order_id=oid, **{k: v for k, v in order.items() if k != 'kind'})
        reference = self.data.feed.next(mid, t, expires)
        if reference and not t < reference[1] <= expires:
            raise ValidationError('invalid future execution time')
        self._push(reference[1] if reference else expires, 1, 'fill', (oid, reference))
        self._check()

    def pending_snapshot(self):
        """Causal order metadata only; omit scheduled future reference events."""
        return [{'order_id': oid, 'kind': o['kind'], 'market': o['market'], 'side': o['side'],
                 'budget': None if o['budget'] is None else str(o['budget']), 'fraction': str(o['fraction']),
                 'submitted_at': o['submitted_at'], 'expires_at': o['expires_at'],
                 'reference_yes': str(o['reference_yes']), 'reason': o['reason']}
                for oid, o in sorted(self.pending.items(), key=lambda x: int(x[0]))]

    def lifecycle_summary(self):
        return {'order_ttl_seconds': self.order_ttl_seconds, 'closeout_ttl_seconds': self.p['fill_window_seconds'],
                'peak_reserved_cash_usd': str(self.peak_reserved_cash),
                'peak_open_or_reserved_positions': self.peak_open_or_reserved_positions,
                'partial_exit_replaced_by_risk_exit': self.risk_replacements,
                'pending_orders_at_end': len(self.pending)}
