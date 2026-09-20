from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import unittest

import test_pma_proof_pilot as pilot
from foretellmesh.data import sha256_file
from foretellmesh.pma_cohort import (cohort_inputs, review_scope, build, split_for,
                                    validate_partitions, development_baselines)
from foretellmesh.schema import ValidationError, timestamp, iso
from foretellmesh.synthetic_sft import canonical_hash

POLICY = {'train_release_before': '2025-01-01T00:00:00Z',
          'validation_release_before': '2025-08-01T00:00:00Z',
          'test_release_before': '2026-02-01T00:00:00Z',
          'unit': 'scheduled_fomc_release', 'label_before_next_split_observation': True}


def record(name, at, outcome=1):
    final = iso(timestamp(at, 'at')+timedelta(days=1))
    return {'sample_id': name, 'dataset_source': 'fixture', 'dataset_version': 'fixture',
            'event_id': name, 'event_group_id': name, 'question': 'Event '+name+'?', 'observation_time': at,
            'evidence': [], 'market': {'probability': 0.6, 'observed_at': at, 'available_at': at},
            'label': {'outcome': outcome, 'resolution_time': final, 'available_at': final}}


class PartitionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.parts = {'train': [record('early', '2024-12-01T00:00:00Z')],
                      'validation': [record('middle', '2025-06-01T00:00:00Z')],
                      'test': [record('late', '2025-09-01T00:00:00Z')]}

    def test_cutoffs_are_exclusive_and_strictly_ordered(self):
        self.assertEqual(split_for('2024-12-31T23:59:59Z', POLICY), 'train')
        self.assertEqual(split_for(POLICY['train_release_before'], POLICY), 'validation')
        self.assertEqual(split_for(POLICY['validation_release_before'], POLICY), 'test')
        with self.assertRaises(ValueError): split_for(POLICY['test_release_before'], POLICY)
        with self.assertRaises(ValidationError): split_for('2024-01-01T00:00:00Z', {**POLICY, 'train_release_before': POLICY['test_release_before']})

    def test_related_events_and_equivalent_questions_cannot_cross_splits(self):
        validate_partitions(self.parts)
        for key, value in [('event_group_id', 'early'), ('event_id', 'early'), ('question', ' EVENT   EARLY? ')]:
            parts = deepcopy(self.parts); parts['test'][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, 'crosses partitions'):
                validate_partitions(parts)

    def test_earlier_settlement_must_precede_later_observation(self):
        self.parts['train'][0]['label'].update(resolution_time='2025-08-01T00:00:00Z', available_at='2025-08-01T00:00:00Z')
        with self.assertRaisesRegex(ValidationError, 'labels settle'): validate_partitions(self.parts)

    def test_duplicate_samples_and_missing_labels_fail(self):
        self.parts['train'].append(deepcopy(self.parts['train'][0]))
        with self.assertRaisesRegex(ValidationError, 'duplicate'): validate_partitions(self.parts)
        self.parts['train'].pop(); self.parts['train'][0]['label'] = None
        with self.assertRaisesRegex(ValidationError, 'lacks label'): validate_partitions(self.parts)

    def test_development_scores_never_read_test_targets(self):
        first = development_baselines(self.parts)
        self.parts['test'] = [{'not': 'a valid record'}]
        self.assertEqual(first, development_baselines(self.parts))
        self.assertFalse(first['test_scored'])
        self.assertEqual(first['training_event_weighted_base_rate'], 1)

    def test_group_weighted_score_does_not_count_brackets_as_independent_events(self):
        self.parts['train'] = [record('one', '2024-03-01T00:00:00Z', 0)]
        other = record('two', '2024-06-01T00:00:00Z', 1)
        self.parts['train'] += [deepcopy(other) for _ in range(3)]
        report = development_baselines(self.parts)
        self.assertEqual(report['training_event_weighted_base_rate'], 0.5)
        score = report['partitions']['train']['scores']['market']
        self.assertAlmostEqual(score['event_weighted_brier'], (0.36+0.16)/2)
        self.assertNotEqual(score['event_weighted_brier'], score['brier'])


class CohortReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = pilot.PmaProofPilotTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = self.root/'cohort.json'; self.review = self.root/'review.json'
        plan = {'schema_version': '1', 'purpose': 'pma_fomc_chronological_research_dataset',
            'minimum_formal_evaluation_groups': 20, 'formal_evaluation': False, 'automatic_training': False,
            'split_policy': POLICY, 'benchmark_index': self.fixture.plan['sources']['heldout_index'],
            'batches': [{'config': self.fixture.config.name, 'config_sha256': sha256_file(self.fixture.config),
                         'capture': self.fixture.raw.name}]}
        self.fixture.dump(self.config, plan)
        config, index, _, projection, groups = cohort_inputs(self.config)
        review = {'schema_version': '1', 'review_id': 'pma_fomc_scope_review_v1',
            'cohort_config_sha256': sha256_file(self.config), 'cohort_projection_sha256': canonical_hash(projection),
            'benchmark_index_sha256': config['benchmark_index']['sha256'], 'benchmark_entries_sha256': canonical_hash(index['entries']),
            'unselected_parent_policy': 'exclude_all_from_this_release', 'rationale': 'synthetic review fixture',
            'historical_knowledge_policy': 'retrospective fixture, no forecasting claim',
            'groups': [{'event_group_id': gid, 'disposition': 'release_in_planned_split',
                        'rationale': 'synthetic distinct event', 'related_benchmark_entry_ids': []} for gid in groups]}
        self.fixture.dump(self.review, review)

    def test_raw_capture_to_versioned_partitions_rebuilds_exactly(self):
        a, b = self.root/'release_a', self.root/'release_b'
        first, second = build(self.config, self.review, a), build(self.config, self.review, b)
        self.assertEqual(first, second)
        self.assertEqual(first['partition_counts']['validation']['observations'], 2)
        self.assertEqual(first['status'], 'incomplete_research_partitions')
        self.assertFalse(first['has_nonempty_train_validation_test'])
        self.assertFalse(first['formal_evaluation_admitted'])
        self.assertEqual(first['heldout_event_groups'], 0)
        self.assertIn('fewer_than_20_distinct_heldout_events', first['formal_evaluation_blockers'])
        self.assertFalse(first['ready_for_sft'])
        inputs = [json.loads(line) for line in (a/'partitions/validation.inputs.jsonl').read_text().splitlines()]
        self.assertTrue(all(set(r['input']) == {'question', 'observation_time', 'evidence', 'market'} for r in inputs))
        self.assertEqual(first['planned_observations'], first['released_observations']+first['excluded_observations'])

    def test_modified_review_projection_or_incomplete_group_review_fails(self):
        config, index, _, projection, groups = cohort_inputs(self.config)
        changed = deepcopy(projection); changed[0]['question'] += ' changed'
        with self.assertRaisesRegex(ValidationError, 'scope/content'):
            review_scope(self.review, self.config, config, index, changed, groups)
        review = json.loads(self.review.read_text()); review['groups'] = []; self.fixture.dump(self.review, review)
        with self.assertRaisesRegex(ValidationError, 'cover cohort'):
            review_scope(self.review, self.config, config, index, projection, groups)

    def test_reservation_excludes_whole_group_but_preserves_denominator(self):
        review = json.loads(self.review.read_text())
        review['groups'][0].update(disposition='benchmark_related_reserve', related_benchmark_entry_ids=['unrelated:1'])
        self.fixture.dump(self.review, review)
        report = build(self.config, self.review, self.root/'reserved')
        self.assertEqual((report['released_observations'], report['excluded_observations']), (0, 2))
        self.assertEqual(report['exclusion_reason_counts']['benchmark_related_reserve'], 2)

    def test_repeated_batch_cannot_inflate_independent_groups(self):
        plan = json.loads(self.config.read_text()); plan['batches'] *= 2; self.fixture.dump(self.config, plan)
        with self.assertRaisesRegex(ValidationError, 'repeated'): cohort_inputs(self.config)

    def test_exact_benchmark_match_overrides_release_annotation(self):
        config, index, _, projection, groups = cohort_inputs(self.config)
        index['entries'][0].update(event_id='polymarket:'+pilot.CONDITION, question='Other title')
        review = json.loads(self.review.read_text()); review['benchmark_entries_sha256'] = canonical_hash(index['entries'])
        self.fixture.dump(self.review, review)
        _, exact = review_scope(self.review, self.config, config, index, projection, groups)
        self.assertEqual(exact[self.fixture.group['event_group_id']], ['polymarket:'+pilot.CONDITION])


if __name__ == '__main__': unittest.main()
