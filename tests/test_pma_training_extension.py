import unittest
from copy import deepcopy

from foretellmesh.pma_training_inventory import dispositions, parent_family
from foretellmesh.pma_legacy import decode_legacy_question
from foretellmesh.schema import ValidationError
from foretellmesh.historical_market import fed_upper_bound
from foretellmesh.pma_training_extension import expected_legacy_fomc, protect_extension
from fractions import Fraction


def question_state(*, reset=0, resolved=1, creator=1, offset=320):
    text = b'historical rule'
    words = [32, 123, 0, 0, 0, resolved, 0, reset, 0, creator, offset, len(text)]
    return '0x' + b''.join(n.to_bytes(32, 'big') for n in words).hex() + (text + bytes((-len(text)) % 32)).hex()


class LegacyQuestionTests(unittest.TestCase):
    def test_older_official_raise_to_syntax_and_dissent_separation(self):
        text = ('The Committee decided to raise the target range for the federal funds rate to 4-3/4 to 5 percent. '
                'A dissenter preferred the target range for the federal funds rate at 4-1/2 to 4-3/4 percent.')
        self.assertEqual(fed_upper_bound(text), Fraction(5))
        with self.assertRaises(ValidationError):
            fed_upper_bound(text + ' The Committee decided to maintain the target range for the federal funds rate at 4 to 5 percent.')
        self.assertEqual(expected_legacy_fomc('Will the Fed increase interest rates by 25 bps after its March meeting?', '2023-03', 25), 1)
        with self.assertRaises(ValidationError):
            expected_legacy_fomc('Will the Fed increase interest rates by 25 bps after its May meeting?', '2023-03', 25)

    def test_ten_field_abi_is_distinct_from_current_adapter(self):
        r = decode_legacy_question(question_state())
        self.assertEqual(r['request_timestamp'], 123)
        self.assertEqual(r['ancillary_data'], 'historical rule')
        self.assertTrue(r['resolved']); self.assertFalse(r['reset'])
        self.assertTrue(decode_legacy_question(question_state(reset=1))['reset'])

    def test_wrong_offsets_flags_addresses_and_padding_rejected(self):
        for value in [question_state(offset=384), question_state(reset=2),
                      question_state(creator=2**160), question_state()[:-2]+'01']:
            with self.assertRaises(ValidationError):
                decode_legacy_question(value)


class InventoryTests(unittest.TestCase):
    def test_protected_parent_siblings_never_become_early_training_candidates(self):
        markets = {str(i): {'events': [{'id': str(i//2), 'title': 'Fed Interest Rates March 2024'}],
                           'endDate': '2024-03-20T00:00:00Z', 'resolvedBy': 'adapter'} for i in range(6)}
        cov = {k: {'time_joined_trades': 100, 'benchmark_matches': []} for k in markets}
        rr = dispositions(markets, cov, {'0': 'validation', '2': 'test', '4': 'train'}, '2025-01-01T00:00:00Z')
        status = {r['market_id']: r['status'] for r in rr}
        self.assertEqual(status['1'], 'protected_parent_sibling')
        self.assertEqual(status['3'], 'protected_parent_sibling')
        self.assertEqual(status['5'], 'early_candidate_needs_admission')
        self.assertTrue(all(r['ready_for_training'] is False for r in rr))

    def test_missing_date_and_prices_and_benchmark_hits_are_not_admitted(self):
        markets = {str(i): {'events': [], 'endDate': '2024-01-01T00:00:00Z'} for i in range(3)}
        markets['0']['endDate'] = None
        cov = {k: {'time_joined_trades': 100, 'benchmark_matches': []} for k in markets}
        cov['1']['time_joined_trades'] = 0; cov['2']['benchmark_matches'] = ['benchmark:a']
        result = dispositions(markets, cov, {}, '2025-01-01T00:00:00Z')
        self.assertEqual([r['status'] for r in result], ['date_unknown_or_outside_early_scope',
            'no_reconstructed_prices', 'reserved_benchmark_identity'])

    def test_macro_keyword_is_not_a_semantic_admission(self):
        self.assertEqual(parent_family('What will Powell say about inflation?'), 'speech_or_mention')
        self.assertEqual(parent_family('Eurozone inflation'), 'inflation_needs_country_and_period_review')



class ExtensionProtectionTests(unittest.TestCase):
    def row(self):
        return {'sample_id': 'x', 'dataset_source': 'fixture', 'dataset_version': 'v1',
            'event_id': 'polymarket:1', 'event_group_id': 'group1', 'question': 'Historical event?',
            'observation_time': '2023-03-01T00:00:00Z', 'evidence': [],
            'market': {'probability': .4, 'observed_at': '2023-03-01T00:00:00Z', 'available_at': '2023-03-01T00:00:00Z'},
            'label': {'outcome': 1, 'resolution_time': '2023-03-02T00:00:00Z', 'available_at': '2026-01-01T00:00:00Z'}}

    def test_heldout_market_or_related_event_cannot_enter_training(self):
        row = self.row(); cutoff = '2025-01-01T00:00:00Z'
        protect_extension([row], [], cutoff)
        for split in ('validation', 'test'):
            for gid, eid in [('group1', 'polymarket:9'), ('group9', 'polymarket:1')]:
                with self.assertRaises(ValidationError):
                    protect_extension([row], [{'split': split, 'event_group_id': gid, 'event_id': eid}], cutoff)

    def test_cutoff_and_future_evidence_fail_closed(self):
        for changed in ('observation_time', 'resolution_time', 'evidence'):
            row = self.row()
            if changed == 'resolution_time': row['label'][changed] = '2025-01-01T00:00:00Z'
            elif changed == 'observation_time': row[changed] = '2025-01-01T00:00:00Z'
            else: row['evidence'] = [{'evidence_id': 'later', 'text': 'future', 'source': 'fixture',
                'published_at': '2023-03-02T00:00:00Z', 'available_at': '2023-03-02T00:00:00Z'}]
            with self.assertRaises(ValidationError): protect_extension([row], [], '2025-01-01T00:00:00Z')

if __name__ == '__main__':
    unittest.main()
