"""CLOB historical prices, isolated from Data API settlement-point semantics.

The API attests sample timestamps, not trade timestamps or bid/ask liquidity.
A one-fidelity guard excludes the edge of the observation window; it is a
conservative policy, not a claim that the API documents bucket end times.
"""
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from .schema import ValidationError, iso, probability


def clob_price_url(token, observation, policy):
    return 'https://clob.polymarket.com/prices-history?'+urlencode({
        'market': token, 'startTs': int((observation-timedelta(hours=policy['price_window_hours'])).timestamp()),
        'endTs': int(observation.timestamp()), 'fidelity': 1})


def clob_history_quote(obj, observation, policy):
    quality = {'kind': 'clob_historical_price_sample', 'guard_seconds': 60, 'issues': [],
               'trust_scope': 'Native API historical sample time; no underlying trade freshness, spread or liquidity attestation.',
               'excluded_edge_or_future_points': 0, 'candidate_points': 0}
    if obj is None:
        quality['issues'].append('price_history_missing')
        return None, quality
    history = obj.get('history')
    if not isinstance(history, list) or len(history) > 1000:
        raise ValidationError('invalid or unexpectedly large CLOB price history')
    candidates, seen = [], set()
    for row in history:
        point = row.get('t')
        if type(point) is not int or point <= 0 or point in seen:
            raise ValidationError('invalid or duplicate CLOB price time')
        seen.add(point)
        value = probability(row.get('p'), 'CLOB historical price')
        sample_time = datetime.fromtimestamp(point, timezone.utc)
        if sample_time + timedelta(seconds=60) > observation:
            quality['excluded_edge_or_future_points'] += 1
            continue
        if sample_time < observation-timedelta(hours=policy['price_window_hours']):continue
        candidates.append((sample_time, value))
    quality['candidate_points'] = len(candidates)
    if not candidates:
        quality['issues'].append('no_pre_cutoff_price_sample')
        return None, quality
    sample_time, value = max(candidates)
    age = (observation-sample_time).total_seconds()
    quality.update(sample_time=iso(sample_time), age_seconds=age)
    if age > policy['max_price_age_seconds']:quality['issues'].append('stale_historical_price')
    if not 0 < value < 1:quality['issues'].append('boundary_historical_price')
    return (None if quality['issues'] else {'probability': value, 'observed_at': iso(sample_time),
                                           'available_at': iso(sample_time+timedelta(seconds=60))}), quality
