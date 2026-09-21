from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.evidence_forecast import ARMS, evidence_pool, score, select_samples, transform
from foretellmesh.schema import ValidationError

C = {'evidence_window_days': 120, 'evidence_per_family': 2}
T = '2024-06-01T00:00:00Z'


def evidence(key='e', published='2024-05-01T00:00:00Z', available=None):
    return {'evidence_id': key, 'published_at': published, 'available_at': available or published,
            'text': 'Dated official evidence.',
            'source': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20240501a.htm'}


def row(key='s', group='g'):
    return {'sample_id': key, 'event_group_id': group,
            'input': {'question': 'Will the event occur?', 'observation_time': T,
                      'market': {'probability': .2, 'observed_at': T, 'available_at': T},
                      'evidence': [evidence()]}}


class EvidenceForecastTests(unittest.TestCase):
    def test_future_publication_or_availability_never_visible(self):
        r = row()
        future = evidence('future', '2024-06-02T00:00:00Z')
        delayed = evidence('delayed', '2024-05-31T00:00:00Z', '2024-06-02T00:00:00Z')
        expected = transform(r, [], 'refreshed_blind', C)
        self.assertEqual(transform(r, [future, delayed], 'refreshed_blind', C), expected)
        future['text'] = 'The future result is known.'
        self.assertEqual(transform(r, [future, delayed], 'refreshed_blind', C), expected)

    def test_blind_arms_invariant_to_market_price(self):
        a = row(); b = deepcopy(a); b['input']['market']['probability'] = .98
        for arm in ('original_blind', 'refreshed_blind'):
            self.assertEqual(transform(a, [], arm, C), transform(b, [], arm, C))
            self.assertIsNone(transform(a, [], arm, C)['input']['market'])
        self.assertNotEqual(transform(a, [], 'original_market', C), transform(b, [], 'original_market', C))

    def test_recent_per_family_limit_and_preserve_original(self):
        values = [evidence('old', '2023-01-01T00:00:00Z'), evidence('a', '2024-05-10T00:00:00Z'),
                  evidence('b', '2024-05-20T00:00:00Z'), evidence('c', '2024-05-30T00:00:00Z')]
        transformed = transform(row(), values, 'refreshed_blind', C)
        self.assertEqual([e['evidence_id'] for e in transformed['input']['evidence']], ['b', 'c', 'e'])
        self.assertEqual(row()['input']['evidence'], [evidence()])

    def test_conflicts_and_unapproved_source_rejected(self):
        a = row(); b = deepcopy(a); b['input']['evidence'][0]['text'] = 'Conflicting text.'
        with self.assertRaises(ValidationError): evidence_pool([a, b])
        a['input']['evidence'][0]['source'] = 'https://www.federalreserve.gov.attacker.example/pressreleases/monetary.htm'
        with self.assertRaises(ValidationError): evidence_pool([a])

    def test_selection_stable_and_rejects_holdout(self):
        inputs = [row('a'), row('b'), row('c')]
        members = [{'sample_id': k, 'event_id': eid, 'event_group_id': 'g', 'split': 'train'}
                   for k, eid in [('a', 'z'), ('b', 'a'), ('c', 'a')]]
        expected = select_samples(inputs, members)
        self.assertEqual([r['sample_id'] for r in expected], ['b', 'c'])
        self.assertEqual(expected, select_samples(inputs[::-1], members[::-1]))
        members[0]['split'] = 'validation'
        with self.assertRaises(ValidationError): select_samples(inputs, members)

    def test_failed_outputs_remain_missing_and_common_scores_paired(self):
        selected = [row('a'), row('b', 'other')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'partitions').mkdir()
            members = [{'sample_id': r['sample_id'], 'event_id': r['sample_id'],
                        'event_group_id': r['event_group_id'], 'dataset_version': 'v1', 'split': 'train'}
                       for r in selected]
            labels = [{'sample_id': r['sample_id'], 'label': {'outcome': i,
                       'resolution_time': '2024-06-10T00:00:00Z', 'available_at': '2024-06-10T00:00:00Z'}}
                      for i, r in enumerate(selected)]
            for name, values in [('membership.jsonl', members), ('partitions/train.labels.jsonl', labels)]:
                (root / name).write_text(''.join(json.dumps(v) + '\n' for v in values))
            (root / 'report.json').write_text(json.dumps({'artifact_hashes': {
                'partitions/train.labels.jsonl': sha256_file(root / 'partitions/train.labels.jsonl')}}))
            results = [{'original_sample_id': r['sample_id'], 'arm': arm,
                        'result': {'status': 'completed', 'prediction': {'probability': .2}}}
                       for r in selected for arm in ARMS]
            results[-1]['result'] = {'status': 'failed'}
            s = score({'dataset': str(root)}, selected, results)
            self.assertEqual(s['common_observations'], 1)
            self.assertEqual(s['all_eligible']['refreshed_blind']['coverage'], .5)
            self.assertIsNone(s['probabilities']['refreshed_blind'][-1])
            self.assertEqual(s['common']['market']['brier'], s['common']['original_market']['brier'])
            with self.assertRaises(ValidationError): score({'dataset': str(root)}, selected, results[:-1])


if __name__ == '__main__':
    unittest.main()
