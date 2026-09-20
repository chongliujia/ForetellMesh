"""Optional overdue-time cost for allocation learning, separate from account PnL."""
from datetime import timedelta
from decimal import Decimal

from .paper_trading import decimal
from .schema import ValidationError, timestamp
from .trading_rl_execution_env import PendingAllocationEnv


def seconds(delta: timedelta) -> Decimal:
    return Decimal(delta.days * 86400 + delta.seconds) + Decimal(delta.microseconds) / 1000000


def overdue_seconds(start, end, last_fill, fills, cadence_seconds=604800) -> Decimal:
    """Integrate time beyond the last actual fill + deadline over [start, end].

    Split transitions and full episodes have identical cost. Orders, cancellations,
    settlement and post-calendar fills cannot reset the clock inside this interval.
    """
    if (any(t.tzinfo is None or t.utcoffset() is None for t in (start, end, last_fill))
            or end < start or last_fill > start or type(cadence_seconds) is not int or cadence_seconds <= 0):
        raise ValidationError('invalid cadence reward interval')
    times = sorted({timestamp(r['time'], 'fill time') for r in fills if r['kind'] in ('buy_fill', 'sell_fill')})
    elapsed = Decimal(0)
    cursor, anchor = start, last_fill
    for fill in [t for t in times if start <= t <= end]:
        elapsed += max(Decimal(0), seconds(fill - max(cursor, anchor + timedelta(seconds=cadence_seconds))))
        cursor = anchor = fill
    return elapsed + max(Decimal(0), seconds(end - max(cursor, anchor + timedelta(seconds=cadence_seconds))))


class CadenceRewardEnv(PendingAllocationEnv):
    """Keep account/reward_total unchanged; expose the learning reward separately."""

    def __init__(self, *args, overdue_usd_per_hour='0', **kwargs):
        self.overdue_usd_per_hour = decimal(overdue_usd_per_hour)
        if self.overdue_usd_per_hour < 0:
            raise ValidationError('negative cadence cost')
        super().__init__(*args, **kwargs)
        if self.data.partition != 'train':
            raise ValidationError('allocation reward experiment is training-only')

    def reset(self, **kwargs):
        self.late_seconds_total = Decimal(0)
        self.penalty_total = Decimal(0)
        self.training_reward_total = 0.
        return super().reset(**kwargs)

    def step(self, action):
        start, last, fill_count = self.time, self.last_fill, len(self.fills)
        obs, pnl_reward, done, truncated, info = super().step(action)
        late = overdue_seconds(start, self.time, last, self.fills[fill_count:], self.p['cadence_seconds'])
        penalty = self.overdue_usd_per_hour * late / 3600
        reward = pnl_reward - float(penalty)
        self.late_seconds_total += late
        self.penalty_total += penalty
        self.training_reward_total += reward
        info.update(pnl_reward_usd=pnl_reward, overdue_seconds=str(late),
                    cadence_penalty_usd=str(penalty), training_reward_usd=reward)
        if done:
            info['episode_summary'] = {**info['episode_summary'], 'learning_reward': self.reward_summary()}
        return obs, reward, done, truncated, info

    def reward_summary(self):
        if not self.done:
            raise ValidationError('reward episode incomplete')
        expected = self.reward_total - float(self.penalty_total)
        whole = overdue_seconds(self.data.start, self.data.end, self.data.start, self.fills, self.p['cadence_seconds'])
        if abs(expected - self.training_reward_total) > 1e-7 or whole != self.late_seconds_total:
            raise ValidationError('cadence reward does not reconcile')
        return {'pnl_reward_sum_usd': self.reward_total, 'overdue_hours': str(self.late_seconds_total / 3600),
                'cadence_penalty_usd': str(self.penalty_total), 'training_reward_sum_usd': self.training_reward_total,
                'overdue_usd_per_hour': str(self.overdue_usd_per_hour)}
