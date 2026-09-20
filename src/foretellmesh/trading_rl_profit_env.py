"""Fee-conditioned net-profit allocation, with no minimum trading frequency."""
from datetime import timedelta
from decimal import Decimal as D

from .paper_trading import decimal
from .schema import ValidationError
from .trading_rl_env import GLOBAL_FEATURES, SLOT_FEATURES
from .trading_rl_execution_env import PendingAllocationEnv

PROFIT_GLOBAL_FEATURES = (*GLOBAL_FEATURES[:3], 'fee_percentage_points', 'price_premium_cents', *GLOBAL_FEATURES[5:])
OBSERVATION_VERSION = 'profit_fee_and_premium_no_cadence_v1'


def checked_costs(costs):
    if set(costs) != {'fee_fraction', 'entry_price_premium'}:
        raise ValidationError('explicit fee and price premium required')
    values = {k: decimal(v) for k, v in costs.items()}
    if any(not 0 <= v <= D('.05') for v in values.values()):
        raise ValidationError('cost outside supported observation range [0, .05]')
    return {k: str(v) for k, v in values.items()}


class ProfitAllocationEnv(PendingAllocationEnv):
    """Costs are known before acting and may vary between training episodes.

    The 88-value shape is preserved, but slots 3/4 now contain fee/premium instead
    of the fill clock/cadence flag. Legacy observation mode is evaluation-only
    compatibility for frozen policies, never silent checkpoint reinterpretation.
    """

    def __init__(self, data, settlements, config, costs, *, cost_schedule=None,
                 legacy_observation=False, minimum_entry_buffer_seconds=86400,
                 order_ttl_seconds=3600, record=False):
        if data.partition != 'train':
            raise ValidationError('profit experiment is training-only')
        if type(legacy_observation) is not bool or (legacy_observation and cost_schedule is not None):
            raise ValidationError('legacy observations are for fixed-cost replay only')
        if type(minimum_entry_buffer_seconds) is not int or minimum_entry_buffer_seconds < 0:
            raise ValidationError('invalid entry buffer')
        costs = checked_costs(costs)
        self.cost_schedule = [checked_costs(c) for c in cost_schedule] if cost_schedule is not None else [costs]
        if not self.cost_schedule or self.cost_schedule[0] != costs:
            raise ValidationError('cost schedule must start with configured costs')
        self.legacy_observation = legacy_observation
        self.minimum_entry_buffer_seconds = minimum_entry_buffer_seconds
        self.episode_index = 0
        super().__init__(data, settlements, config, costs, order_ttl_seconds=order_ttl_seconds, record=record)

    @property
    def due(self):
        return False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.episode_index = 0
        self.costs = dict(self.cost_schedule[self.episode_index % len(self.cost_schedule)])
        self.fee = decimal(self.costs['fee_fraction'])
        self.premium = decimal(self.costs['entry_price_premium'])
        self.episode_index += 1
        obs, info = super().reset(seed=seed, options=options)
        return obs, {**info, 'execution_costs': dict(self.costs)}

    def _can_buy(self, mid, state, side, budget):
        if (self._pending_for(mid) or mid in self.positions or mid in self.resolved
                or len(set(self.positions) | self._reserved_markets()) >= self.p['max_positions']):
            return False
        if state.get('status') != 'ready' or self.time < self.cooldowns.get(mid, self.data.start):
            return False
        if self.time + timedelta(seconds=self.minimum_entry_buffer_seconds) > min(self.windows[mid].retire_at, self.data.end):
            return False
        if self.attempts[self.time.date()] >= self.p['max_entries_per_day']:
            return False
        q = decimal(state['yes_price'])
        if not D(self.p['min_yes_price']) <= q <= D(self.p['max_yes_price']):
            return False
        if (self._side_price(q, side) + self.premium) * (1 + self.fee) >= 1:
            return False
        return min(self.cash, D(self.p['max_event_usd']) - self._exposure(self.windows[mid].event_group_id),
                   D(self.p['max_portfolio_usd']) - self._exposure()) >= budget

    def _refresh(self):
        super()._refresh()
        if not self.mask[0]:
            raise ValidationError('profit policy must always allow waiting')
        if not self.legacy_observation:
            self.obs[3] = float(self.fee * 100)
            self.obs[4] = float(self.premium * 100)

    def _order(self, *args, reason='policy', **kwargs):
        if reason.startswith('weekly'):
            raise ValidationError('periodic participation is disabled')
        return super()._order(*args, reason=reason, **kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        info.pop('cadence_due', None)
        info['execution_costs'] = dict(self.costs)
        return obs, reward, done, truncated, info

    def summary(self):
        result = super().summary()
        if result.pop('weekly_entries') or result.pop('weekly_blocked_ticks'):
            raise ValidationError('unexpected periodic participation')
        result['max_fill_gap_hours'] = result.pop('cadence')['actual_max_gap_hours']
        result['policy_requirements_met'] = self.invalid_actions == 0
        result.update(periodic_activity_required=False, activity_penalty_usd='0', execution_costs=dict(self.costs),
                      objective='maximize_after_cost_terminal_equity', outperformed_cash=self.cash > D(self.p['initial_cash']),
                      observation_version='legacy_cadence_fields_inactive_v1' if self.legacy_observation else OBSERVATION_VERSION)
        return result


def observation_schema():
    return {'version': OBSERVATION_VERSION, 'global': list(PROFIT_GLOBAL_FEATURES), 'slot': list(SLOT_FEATURES),
            'slots': 4, 'dimension': len(PROFIT_GLOBAL_FEATURES) + 4 * len(SLOT_FEATURES),
            'fee_scaling': 'fee_fraction * 100', 'premium_scaling': 'entry_price_premium * 100',
            'cadence_features_present': False}
