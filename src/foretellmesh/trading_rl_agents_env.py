"""Allocation observations augmented by immutable, time-bounded Agent outputs."""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
import math

import gymnasium as gym
import numpy as np

from .schema import ValidationError, iso, probability, timestamp
from .synthetic_sft import canonical_hash
from .trading_rl_profit_env import ProfitAllocationEnv

OBSERVATION_VERSION = 'profit_fee_with_agent_signals_v1'
VIEWS = ('supports_yes', 'supports_no', 'mixed', 'no_independent_edge')
CONFIDENCES = ('low', 'medium', 'high')
AGENT_FEATURES = ('available', 'age_fraction', 'forecast_probability', 'forecast_minus_current_price',
    *('confidence_' + k for k in CONFIDENCES), *('quant_' + k for k in VIEWS),
    *('game_' + k for k in VIEWS), 'unknown_count_scaled')


def signal_rows(results, ttl_seconds, *, reference_probabilities=None):
    """Account for sequential shared-model latency, including same-cutoff queues."""
    if type(ttl_seconds) is not int or ttl_seconds <= 0:
        raise ValidationError('invalid signal TTL')
    if reference_probabilities is not None and set(reference_probabilities) != {r['sample_id'] for r in results}:
        raise ValidationError('reference-price population mismatch')
    ready, previous, seen, records = None, None, set(), []
    for r in results:
        observation = timestamp(r['observation_time'], 'agent observation')
        if previous is not None and observation < previous:
            raise ValidationError('signal jobs must be chronological')
        previous = observation
        if r['sample_id'] in seen:
            raise ValidationError('duplicate signal identity')
        seen.add(r['sample_id'])
        durations = [r['seconds'], r['feature_seconds']]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in durations):
            raise ValidationError('invalid agent latency')
        seconds = sum(durations)
        ready = max(observation, ready or observation) + timedelta(seconds=max(1, math.ceil(seconds)))
        result = r['result']; content = None
        if result['status'] == 'completed':
            pred = result['prediction']; stages = result['stages']
            p = probability(pred['probability'], 'agent probability')
            if timestamp(pred['observation_time'], 'forecast cutoff') != observation:
                raise ValidationError('forecast time mismatch')
            if pred['confidence'] not in CONFIDENCES or not isinstance(pred['unknowns'], list):
                raise ValidationError('invalid agent confidence/unknowns')
            views = [stages[role]['market_view'] for role in ('market_quant', 'game_theory')]
            if any(v not in VIEWS for v in views):
                raise ValidationError('invalid expert view')
            content = {'probability': p, 'confidence': pred['confidence'], 'quant_view': views[0],
                       'game_view': views[1], 'unknown_count': len(pred['unknowns'])}
        elif result['status'] != 'failed':
            raise ValidationError('unknown signal status')
        records.append({'sample_id': r['sample_id'], 'market_id': r['market_id'],
            'observation_time': iso(observation), 'available_at': iso(ready),
            'expires_at': iso(observation + timedelta(seconds=ttl_seconds)), 'content': content,
            'reference_probability': None if reference_probabilities is None else
                probability(reference_probabilities[r['sample_id']], 'historical market reference'),
            'result_sha256': canonical_hash(result)})
    return records


class SignalIndex:
    def __init__(self, records, markets):
        self.records = deepcopy(records); self.by_market = defaultdict(list); self.times = {}
        seen = set()
        for r in self.records:
            mid = r['market_id']; t = timestamp(r['observation_time'], 'observation')
            ready = timestamp(r['available_at'], 'availability'); expiry = timestamp(r['expires_at'], 'expiry')
            if mid not in markets or (mid, t) in seen or not t < ready or not t < expiry:
                raise ValidationError('invalid signal market/time')
            seen.add((mid, t)); self.by_market[mid].append((ready, t, expiry, r))
        for mid, values in self.by_market.items():
            values.sort(key=lambda item: item[0])
            if any(a[1] >= b[1] for a, b in zip(values, values[1:])):
                raise ValidationError('signal observation order differs from availability')
            self.times[mid] = [v[0] for v in values]

    def at(self, mid, now):
        i = bisect_right(self.times.get(mid, []), now) - 1
        if i < 0:
            return None
        ready, observed, expiry, row = self.by_market[mid][i]
        # Failed refreshes invalidate the old recommendation; do not backfill.
        return row if now < expiry and row['content'] is not None else None

    def features(self, mid, now, price, *, content_enabled, anchor_only=False):
        row = self.at(mid, now)
        if row is None:
            return [0.] * len(AGENT_FEATURES), None
        t = timestamp(row['observation_time'], 'observation'); expiry = timestamp(row['expires_at'], 'expiry')
        metadata = [1., (now - t).total_seconds() / (expiry - t).total_seconds()]
        c = row['content']
        content = [c['probability'], c['probability'] - price,
            *[float(c['confidence'] == v) for v in CONFIDENCES],
            *[float(c['quant_view'] == v) for v in VIEWS], *[float(c['game_view'] == v) for v in VIEWS],
            min(c['unknown_count'], 10) / 10]
        if not content_enabled:
            content = [0.] * len(content)
        if anchor_only:
            if content_enabled or row['reference_probability'] is None:
                raise ValidationError('anchor control requires historical prices and disabled Agent content')
            content[:2] = [row['reference_probability'], row['reference_probability'] - price]
        return metadata + content, row['sample_id']


class AgentAllocationEnv(ProfitAllocationEnv):
    def __init__(self, *args, signals, content_enabled, anchor_only=False, **kwargs):
        if (type(content_enabled) is not bool or type(anchor_only) is not bool
                or (content_enabled and anchor_only)):
            raise ValidationError('explicit content ablation required')
        self.signals = signals; self.content_enabled = content_enabled; self.anchor_only = anchor_only
        super().__init__(*args, **kwargs)
        self.base_dimension = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(-5., 5.,
            shape=(self.base_dimension + self.k * len(AGENT_FEATURES),), dtype=np.float32)

    def reset(self, **kwargs):
        self.signal_ticks = 0; self.signal_slots = 0; self.total_slots = 0
        self.signal_entry_attempts = 0
        return super().reset(**kwargs)

    def _refresh(self):
        super()._refresh()
        extra = []; self.signal_ids = []
        for mid in self.slots:
            quote = self._quote(mid, self.time)
            price = float(quote[0]) if quote else .5
            values, sid = self.signals.features(mid, self.time, price, content_enabled=self.content_enabled,
                                               anchor_only=self.anchor_only)
            extra.extend(values); self.signal_ids.append(sid)
        extra.extend([0.] * (len(AGENT_FEATURES) * (self.k - len(self.slots))))
        self.obs = np.concatenate((self.obs, np.asarray(extra, dtype=np.float32)))
        if not self.observation_space.contains(self.obs):
            raise ValidationError('invalid augmented observation')

    def step(self, action):
        if self.done:
            raise ValidationError('step after terminal state')
        action = int(action)
        if not self.action_space.contains(action):
            raise ValidationError('action out of range')
        ids = list(self.signal_ids)
        self.signal_ticks += any(ids); self.signal_slots += sum(x is not None for x in ids)
        self.total_slots += len(ids)
        if action and self.mask[int(action)]:
            j, op = divmod(int(action) - 1, 6)
            if op < 4 and ids[j] is not None:
                self.signal_entry_attempts += 1
        result = super().step(action)
        if self.record:
            self.decisions[-1].update(agent_signal_ids=ids, agent_content_enabled=self.content_enabled,
                                     market_anchor_only=self.anchor_only)
        return result

    def summary(self):
        result = super().summary()
        result.update(observation_version=OBSERVATION_VERSION, agent_content_enabled=self.content_enabled,
            market_anchor_only=self.anchor_only,
            decision_ticks_with_signal=self.signal_ticks, candidate_slots_with_signal=self.signal_slots,
            total_candidate_slots=self.total_slots, entry_attempts_with_signal=self.signal_entry_attempts)
        return result
