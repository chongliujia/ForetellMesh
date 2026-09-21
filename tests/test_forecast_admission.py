"""Synthetic qualifications exercise execution only, never historical skill."""
from copy import deepcopy
from datetime import timedelta
from decimal import Decimal as D
import unittest

from foretellmesh.forecast_admission import ForecastAdmission, unqualified_generation_records
from foretellmesh.schema import ValidationError, iso
from foretellmesh.synthetic_sft import canonical_hash
from test_event_allocation import T, run, signal


def fixture(signals, *, qualified=False, basis='evidence_only', qualification_expiry=None):
    records = [{'sample_id': s['sample_id'], 'signal_sha256': canonical_hash(s), 'method_id': 'synthetic_fixture',
                'basis': basis, 'evidence': [{'evidence_id': 'synthetic:known',
                    'published_at': iso(T-timedelta(days=2)), 'available_at': iso(T-timedelta(days=1))}]}
               for s in signals]
    qualifications = {'synthetic_fixture': {'qualification_id': 'synthetic_execution_test_only',
        'evaluation_sha256': 'a'*64, 'audit_sha256': 'b'*64,
        'available_at': iso(T-timedelta(days=1)),
        'expires_at': iso(qualification_expiry or T+timedelta(days=20))}} if qualified else {}
    return records, qualifications


class ForecastAdmissionTests(unittest.TestCase):
    def test_generation_adapter_excludes_price_features_and_does_not_self_qualify(self):
        s = signal(); input_evidence = [
            {'evidence_id': key, 'published_at': iso(T-timedelta(days=2)), 'available_at': iso(T-timedelta(days=1))}
            for key in ('market_features:1', 'fed:known', 'bls:uncited')]
        inputs = [{'sample_id': 'a', 'input': {'market': {'probability': .8}, 'evidence': input_evidence}}]
        results = [{'sample_id': 'a', 'result': {'status': 'completed', 'prediction': {
            'key_evidence': ['market_features:1', 'fed:known'], 'counter_evidence': [], 'confidence': 'high'}}}]
        records = unqualified_generation_records([s], inputs, results, 'method')
        self.assertEqual(records[0]['basis'], 'market_conditioned')
        self.assertEqual([e['evidence_id'] for e in records[0]['evidence']], ['fed:known'])
        self.assertFalse(ForecastAdmission([s], records).assess(s, T+timedelta(minutes=2))['allowed'])
        inputs[0]['input']['market'] = None
        records = unqualified_generation_records([s], inputs, results, 'method')
        self.assertEqual(records[0]['basis'], 'evidence_only')
        self.assertFalse(ForecastAdmission([s], records).assess(s, T+timedelta(minutes=2))['allowed'])

    def test_default_refuses_unreviewed_forecasts(self):
        r = run(legacy_unvalidated_research=False)
        self.assertEqual(r['metrics']['final_cash'], '100')
        self.assertFalse(r['ledger'])
        self.assertTrue(any(d['reason'] == 'missing_forecast_admission' for d in r['decisions']))

    def test_old_price_anchor_cannot_become_an_independent_edge(self):
        s = signal(p=.4)
        prices = {'1': [(T+timedelta(minutes=5*i), D('.4') if i < 288 else D('.2'), 1)
                        for i in range(5*288)]}
        old = run(signals=[s], prices=prices)
        records, _ = fixture([s], basis='market_conditioned')
        fixed = run(signals=[s], prices=prices, legacy_unvalidated_research=False,
                    forecast_admission=ForecastAdmission([s], records))
        self.assertGreater(old['metrics']['entry_count'], 0)
        self.assertEqual(fixed['metrics']['entry_count'], 0)
        self.assertEqual(fixed['metrics']['final_cash'], '100')

    def test_disagreement_and_claimed_confidence_do_not_qualify(self):
        s = signal(p=.99); s['content']['confidence'] = 'high'
        records, _ = fixture([s])
        r = run(signals=[s], legacy_unvalidated_research=False, forecast_admission=ForecastAdmission([s], records))
        self.assertEqual(r['metrics']['entry_count'], 0)
        self.assertTrue(any(d['reason'] == 'forecast_method_not_qualified' for d in r['decisions']))

    def test_qualified_synthetic_signal_can_trade_even_if_equal_to_old_reference(self):
        s = signal(p=.8); s['reference_probability'] = .8
        records, qualified = fixture([s], qualified=True)
        r = run(signals=[s], legacy_unvalidated_research=False,
                forecast_admission=ForecastAdmission([s], records, qualified))
        self.assertEqual(r['metrics']['entry_count'], 1)
        self.assertEqual(r['metrics']['settlement_count'], 1)
        # This positive control verifies the gate is not an unconditional cash rule.
        self.assertGreater(D(r['metrics']['final_cash']), 100)

    def test_market_reference_and_no_external_evidence_are_not_qualified_forecasts(self):
        s = signal()
        for basis, empty in [('market_reference', False), ('evidence_only', True)]:
            records, qualified = fixture([s], qualified=True, basis=basis)
            if empty: records[0]['evidence'] = []
            self.assertFalse(ForecastAdmission([s], records, qualified).assess(s, T+timedelta(minutes=2))['allowed'])

    def test_future_qualification_not_backfilled(self):
        s = signal(); records, qualified = fixture([s], qualified=True)
        qualified['synthetic_fixture']['available_at'] = iso(T+timedelta(seconds=30))
        gate = ForecastAdmission([s], records, qualified)
        self.assertEqual(gate.assess(s, T+timedelta(minutes=2))['reason'], 'qualification_not_available_at_forecast')

    def test_expiry_cancels_pending_order_but_risk_exit_still_works(self):
        s = signal(); records, qualified = fixture([s], qualified=True, qualification_expiry=T+timedelta(minutes=2))
        r = run(signals=[s], legacy_unvalidated_research=False,
                forecast_admission=ForecastAdmission([s], records, qualified))
        self.assertEqual(r['metrics']['entry_count'], 0)
        self.assertTrue(any(x.get('reason') == 'forecast_admission_revoked_before_fill' for x in r['ledger']))
        records, qualified = fixture([s], qualified=True, qualification_expiry=T+timedelta(days=1))
        prices = {'1': [(T+timedelta(minutes=5*i), D('.4') if i < 576 else D('.3'), 1) for i in range(5*288)]}
        r = run(signals=[s], prices=prices, legacy_unvalidated_research=False,
                forecast_admission=ForecastAdmission([s], records, qualified))
        self.assertTrue(any(x.get('reason') == 'stop_loss' for x in r['ledger'] if x['kind'] == 'sell_fill'))

    def test_hash_population_and_evidence_time_tampering_rejected(self):
        s = signal(); records, _ = fixture([s])
        for changed in ([], records+records, [{**records[0], 'signal_sha256': '0'*64}]):
            with self.assertRaises(ValidationError): ForecastAdmission([s], changed)
        records[0]['evidence'][0]['available_at'] = iso(T+timedelta(seconds=1))
        with self.assertRaises(ValidationError): ForecastAdmission([s], records)
        records, _ = fixture([s]); gate = ForecastAdmission([s], records)
        mutated = deepcopy(s); mutated['content']['probability'] = .99
        with self.assertRaises(ValidationError): gate.assess(mutated, T+timedelta(minutes=2))

    def test_labels_and_future_prices_do_not_change_earlier_admission(self):
        s = signal(); records, qualified = fixture([s], qualified=True)
        opts = dict(signals=[s], legacy_unvalidated_research=False,
                    forecast_admission=ForecastAdmission([s], records, qualified))
        a, b = run(outcome=0, **opts), run(outcome=1, **opts)
        self.assertEqual(a['decisions'], b['decisions'])
        self.assertEqual([r for r in a['ledger'] if r['kind'] != 'settle'],
                         [r for r in b['ledger'] if r['kind'] != 'settle'])


if __name__ == '__main__':
    unittest.main()
