"""Bind forecast provenance to separately reviewed simulation qualifications.

This is an authorization boundary, not a statistical skill estimator. A caller
must obtain method qualifications from a separately audited evaluation; neither
model confidence, probability disagreement nor historical PnL grants access.
The current experiment issues no method qualifications.
"""
from copy import deepcopy
import re

from .schema import ValidationError, timestamp
from .synthetic_sft import canonical_hash


class ForecastAdmission:
    def __init__(self, signals, records, qualifications=None):
        indexed = {s['sample_id']: s for s in signals}
        self.records = {r['sample_id']: deepcopy(r) for r in records}
        if len(indexed) != len(signals) or len(self.records) != len(records) or set(indexed) != set(self.records):
            raise ValidationError('admission population must exactly match signals')
        self.qualifications = deepcopy(qualifications or {})
        for method, q in self.qualifications.items():
            if (set(q) != {'qualification_id', 'evaluation_sha256', 'audit_sha256', 'available_at', 'expires_at'}
                    or not method or not q['qualification_id']
                    or any(not re.fullmatch('[0-9a-f]{64}', q[k]) for k in ('evaluation_sha256', 'audit_sha256'))
                    or timestamp(q['available_at'], 'qualification availability') >= timestamp(q['expires_at'], 'qualification expiry')):
                raise ValidationError('invalid reviewed method qualification')
        for sid, r in self.records.items():
            s = indexed[sid]
            if (set(r) != {'sample_id', 'signal_sha256', 'method_id', 'basis', 'evidence'}
                    or r['signal_sha256'] != canonical_hash(s) or not r['method_id']
                    or r['basis'] not in ('market_reference', 'market_conditioned', 'evidence_only')):
                raise ValidationError('invalid or unbound signal provenance')
            seen = set()
            for e in r['evidence']:
                if (set(e) != {'evidence_id', 'published_at', 'available_at'} or not e['evidence_id']
                        or e['evidence_id'] in seen or not timestamp(e['published_at'], 'evidence publication')
                        <= timestamp(e['available_at'], 'evidence availability')
                        <= timestamp(s['observation_time'], 'forecast cutoff')):
                    raise ValidationError('invalid evidence provenance/cutoff')
                seen.add(e['evidence_id'])

    def assess(self, signal, now):
        """No quotes, resolution labels, rewards or price-difference thresholds."""
        if signal is None:
            return {'allowed': False, 'reason': 'missing_or_expired_forecast', 'qualification_id': None}
        r = self.records.get(signal['sample_id'])
        if r is None or r['signal_sha256'] != canonical_hash(signal):
            raise ValidationError('signal differs from admission binding')
        reason = None
        q = self.qualifications.get(r['method_id'])
        if not timestamp(signal['available_at'], 'signal availability') <= now < timestamp(signal['expires_at'], 'signal expiry') or signal['content'] is None:
            reason = 'missing_or_expired_forecast'
        elif r['basis'] == 'market_reference':
            reason = 'market_reference_is_not_forecast'
        elif not r['evidence']:
            reason = 'no_cited_external_evidence'
        elif q is None:
            reason = 'forecast_method_not_qualified'
        elif timestamp(q['available_at'], 'qualification availability') > timestamp(signal['observation_time'], 'forecast cutoff'):
            reason = 'qualification_not_available_at_forecast'
        elif now >= timestamp(q['expires_at'], 'qualification expiry'):
            reason = 'forecast_qualification_expired'
        return {'allowed': reason is None, 'reason': reason or 'qualified_event_forecast',
                'qualification_id': q['qualification_id'] if q else None}


def unqualified_generation_records(signals, inputs, results, method_id):
    """Use verified original requests and citations; never relabel copied p as skill."""
    ii = {r['sample_id']: r for r in inputs}; rr = {r['sample_id']: r for r in results}
    ids = {s['sample_id'] for s in signals}
    if set(ii) != ids or set(rr) != ids or len(inputs) != len(ids) or len(results) != len(ids):
        raise ValidationError('generation/provenance population mismatch')
    records = []
    for s in signals:
        payload = ii[s['sample_id']]['input']; result = rr[s['sample_id']]['result']
        references = set()
        if result['status'] == 'completed':
            prediction = result['prediction']
            references = set(prediction['key_evidence']) | set(prediction['counter_evidence'])
        # This adapter is scoped to the existing admitted official macro pool.
        # Deterministic price-feature records are not independent event evidence.
        external = [e for e in payload['evidence'] if e['evidence_id'] in references
                    and e['evidence_id'].startswith(('fed:', 'bls:'))]
        records.append({'sample_id': s['sample_id'], 'signal_sha256': canonical_hash(s), 'method_id': method_id,
                        'basis': 'market_conditioned' if payload['market'] is not None else 'evidence_only',
                        'evidence': [{k: e[k] for k in ('evidence_id', 'published_at', 'available_at')} for e in external]})
    ForecastAdmission(signals, records)
    return records
