from copy import deepcopy
import json
from pathlib import Path
import unittest

import test_historical_market as historical
from foretellmesh.benchmark_review import apply_review, cohort_projection, native_aliases
from foretellmesh.data import sha256_file
from foretellmesh.historical_replay_gate import blockers, preflight
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = historical.HistoricalMarketTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.fixture.collect(); self.fixture.build()
        self.rows = [json.loads(s) for s in (self.fixture.root/'built/candidates.jsonl').read_text().splitlines()]
        self.index = self.fixture.index
        self.write_index([{'event_id': 'fb:else', 'question': 'An unrelated event'}])
        aliases = native_aliases(self.rows, self.fixture.capture)
        self.policy = {'schema_version': '1', 'review_id': 'fixture',
            'heldout_index_sha256': sha256_file(self.index),
            'cohort_projection_sha256': canonical_hash(cohort_projection(self.rows, aliases)),
            'review_scope': 'Bounded fixture review', 'related_entry_ids': ['fb:else'],
            'related_entries_rationale': 'Conservative reservation',
            'groups': [{'event_group_id': self.rows[0]['record']['event_group_id'],
                        'disposition': 'evaluation_only', 'rationale': 'Distinct target'}],
            'allowed_use': 'heldout_replay', 'training': False, 'prompt_tuning': False,
            'reward_tuning': False, 'checkpoint_selection': False}
        self.review = self.fixture.root/'review.json'

    def write_index(self, entries):
        self.index.write_text(json.dumps({'schema_version': '1', 'source': 'forecastbench',
            'entries': entries, 'entries_sha256': canonical_hash(entries)}))

    def apply(self, rows=None):
        self.review.write_text(json.dumps(self.policy))
        return apply_review(self.rows if rows is None else rows, self.fixture.capture, self.index, self.review)

    def test_review_preserves_inputs_and_reserves_all_rows_for_evaluation(self):
        before = deepcopy(self.rows)
        rows, report = self.apply()
        self.assertEqual(before, self.rows)
        self.assertTrue(report['model_input_payloads_unchanged'])
        for row in rows:
            self.assertNotIn('semantic_benchmark_review_required', row['blockers'])
            self.assertIn('evaluation_only_reserved', blockers(row, 'sft'))
            self.assertIn('evaluation_only_reserved', blockers(row, 'development_replay'))
            self.assertNotIn('evaluation_only_reserved', blockers(row, 'heldout_replay'))
            self.assertFalse(row['ready_for_benchmark'])
        self.assertEqual(rows, self.apply()[0])

    def test_condition_alias_quarantines_whole_group_including_other_platform(self):
        self.write_index([{'event_id': 'polymarket:'+historical.CONDITION,
                           'question': 'A different phrasing of the same contract'}])
        self.policy.update(heldout_index_sha256=sha256_file(self.index), related_entry_ids=[])
        rows, report = self.apply()
        self.assertEqual(len(report['exact_overlap_groups']), 1)
        self.assertTrue(any(r['record']['event_id'].startswith('kalshi:') for r in rows))
        self.assertTrue(all('heldout_same_event_group_overlap' in r['blockers'] for r in rows))

    def test_rehashed_index_and_changed_input_require_new_review(self):
        self.write_index([{'event_id': 'fb:else', 'question': 'Changed question'}])
        with self.assertRaisesRegex(ValidationError, 'index changed'):self.apply()
        self.policy['heldout_index_sha256'] = sha256_file(self.index)
        rows = deepcopy(self.rows); rows[0]['original_question'] += ' revised'
        with self.assertRaisesRegex(ValidationError, 'cohort inputs'):self.apply(rows)

    def test_invalid_scope_unbound_related_ids_and_missing_group_rejected(self):
        base = deepcopy(self.policy)
        for field, value in [('prompt_tuning', True), ('related_entry_ids', ['fb:unknown']),
                             ('groups', []), ('review_scope', '')]:
            self.policy = {**base, field: value}
            with self.subTest(field=field), self.assertRaises(ValidationError):self.apply()

    def test_heldout_preflight_requires_explicit_no_tuning_policy(self):
        config = ROOT/'configs/historical_heldout_admission_v1.json'
        with self.assertRaisesRegex(ValidationError, 'frozen review'):
            preflight(self.fixture.root/'built', self.fixture.capture, self.index, config)
        # Build fixture again because the strict index is now part of the source provenance.
        self.fixture.build('strict_index')
        self.apply()
        result = preflight(self.fixture.root/'strict_index', self.fixture.capture, self.index,
                           config, benchmark_review=self.review)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['eligible_sft_rows'], 0)
        self.assertIsNone(result['score_metrics'])


if __name__ == '__main__':unittest.main()
