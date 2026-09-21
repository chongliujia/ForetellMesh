from copy import deepcopy
import json
import unittest

from foretellmesh.team_research_loop import (admit_catalogue, relation_fingerprint,
    run_loop, summarize, validate_selection, validate_reflection)
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from test_team_variable_contracts import TypedBackend
from test_team_executable_methods import fixture
from test_team_lifecycle import TIMING


class LoopBackend(TypedBackend):
    def __init__(self): super().__init__(); self.requests = []
    def generate(self, request):
        self.requests.append(deepcopy(request)); role = request['agent']
        if role == 'research_coordinator':
            return json.dumps({'market_ids': ['a', 'b', 'c'], 'reason': 'Test shared dynamics across independent events.'})
        if role == 'research_learning_member':
            return json.dumps({'feedback_sha256': request['upstream']['feedback']['feedback_sha256'],
                'next_focus': 'Inspect a different measurable relationship after this training attempt.', 'data_requests': []})
        return super().generate(request)


def setup():
    cat, feed, labels, protocol, _ = fixture(); backend = LoopBackend()
    for k, row in cat.items(): row.update(market_id=k, question='Question '+k)
    context = {'phase': 'offline_adaptive_training_research', 'variable_sources': backend.registry}
    return cat, feed, labels, protocol, context, backend


class LoopTests(unittest.TestCase):
    def test_partition_filter_precedes_rule_and_label_access(self):
        class Forbidden(dict):
            def __getitem__(self, key): raise AssertionError('held-out proof read')
        rows = [{'event_group_id': 'dev', 'proof': Forbidden()}, {'event_group_id': 'purged', 'proof': Forbidden()},
            {'event_group_id': 'train', 'market_id': 'a', 'proof': {'initialized_at': '2024-01-01T00:00:00Z',
             'historical_question': 'Allowed', 'outcome': 1, 'resolution_time': '2024-02-01T00:00:00Z',
             'available_at': '2024-02-02T00:00:00Z'}}]
        cat, labels = admit_catalogue(rows, {'dev': 'development', 'purged': 'purged_boundary', 'train': 'train'}, '2024-01-02T00:00:00Z')
        self.assertEqual(set(cat), {'a'}); self.assertNotIn('outcome', cat['a'])
        self.assertEqual(labels['a']['outcome'], 1)
        self.assertFalse(admit_catalogue(rows, {'train': 'train'}, '2023-01-01T00:00:00Z')[0])

    def test_duplicate_does_not_rescore_and_all_attempts_feed_next_selection(self):
        cat, feed, labels, protocol, context, backend = setup(); runner = DiscoveryRunner(backend, None)
        frozen = []; saved = []
        records = run_loop(runner, context, cat, feed, labels, protocol, 3,
                           lambda n, p: frozen.append((n, p)), saved.append)
        self.assertEqual(len(saved), 3); self.assertEqual(len(frozen), 1)
        self.assertEqual(summarize(records)['statuses'], {'training_screen_passed': 1, 'duplicate_relation': 2})
        self.assertEqual(summarize(records)['qualified_attempts'], [1])
        selections = [r for r in backend.requests if r['agent'] == 'research_coordinator']
        self.assertEqual([len(r['upstream']['attempts']) for r in selections], [0, 1, 2])
        self.assertEqual(records[1]['previous_attempt_sha256'], records[0]['attempt_sha256'])
        self.assertFalse(records[0]['feedback']['effectiveness_verified'])
        for r in backend.requests:
            if r['agent'] == 'relation_researcher': self.assertNotIn('variable_sources', r['input'])
            self.assertNotIn('labels', r['input'])

    def test_missing_data_learns_without_reading_prices_or_labels(self):
        cat, _, _, protocol, context, backend = setup()
        backend.relation['variables'][0].update(quantity='external_measurement', unit='count')
        backend.mapping['bindings'][0].update(source_id=None, missing_reason='No historical measurements available.')
        class Forbidden:
            def latest(self, *args): raise AssertionError('price accessed')
            def __getitem__(self, key): raise AssertionError('label accessed')
        records = run_loop(DiscoveryRunner(backend, None), context, cat, Forbidden(), Forbidden(), protocol, 2,
                           lambda *a: self.fail('missing data frozen'), lambda a: None)
        self.assertEqual(records[0]['workflow']['status'], 'unavailable_variables')
        self.assertEqual(records[1]['workflow']['status'], 'duplicate_relation')
        self.assertEqual(summarize(records)['successful_reflections'], 2)
        self.assertEqual(summarize(records)['scored_attempts'], 0)

    def test_invalid_selection_and_failed_reflection_are_retained(self):
        cat, feed, labels, protocol, context, _ = setup()
        class Invalid:
            def generate(self, request): return '{}'
        runner = DiscoveryRunner(Invalid(), None)
        records = run_loop(runner, context, cat, feed, labels, protocol, 2, lambda *a: None, lambda a: None)
        self.assertEqual(len(runner.calls), 8)
        self.assertEqual(summarize(records)['statuses'], {'selection_failed': 2})
        self.assertEqual(summarize(records)['successful_reflections'], 0)

    def test_recorded_calls_reproduce_every_attempt_and_freeze_order(self):
        cat, feed, labels, protocol, context, backend = setup(); runner = DiscoveryRunner(backend, None)
        freezes = {}
        def freeze(n, p): freezes[n] = (deepcopy(p), canonical_hash(runner.calls))
        first = run_loop(runner, context, cat, feed, labels, protocol, 3, freeze, lambda a: None)
        replay = RecordedBackend(runner.calls); audited = AuditRunner(replay, None, TIMING)
        def check(n, p): self.assertEqual((p, canonical_hash(audited.calls)), freezes[n])
        second = run_loop(audited, context, cat, feed, labels, protocol, 3, check, lambda a: None)
        self.assertEqual(first, second); self.assertEqual(replay.index, len(runner.calls))

    def test_fingerprint_ignores_wording_aliases_but_preserves_lags(self):
        *_, backend = setup(); original = deepcopy(backend.relation); changed = deepcopy(original)
        changed['hypothesis'] = 'Different wording'; changed['variables'][0]['variable_id'] = 'renamed'
        changed['variables'][0]['definition'] = 'Equivalent wording'
        changed['variables'].reverse(); changed['bindings'].reverse()
        self.assertEqual(relation_fingerprint(original), relation_fingerprint(changed))
        changed['variables'][0]['lag_days'] += 1
        self.assertNotEqual(relation_fingerprint(original), relation_fingerprint(changed))

    def test_budget_and_retrieval_scope_cannot_be_overridden(self):
        cat, feed, labels, protocol, context, backend = setup()
        for ids in [['unknown'], ['a', 'a'], [], ['a']*9]:
            with self.assertRaises(ValidationError): validate_selection({'market_ids': ids, 'reason': 'test'}, cat)
        for budget in [0, 11, True]:
            with self.assertRaises(ValidationError):
                run_loop(DiscoveryRunner(backend, None), context, cat, feed, labels, protocol, budget, lambda *a: None, lambda a: None)
        with self.assertRaises(ValidationError):
            validate_reflection({'feedback_sha256': 'changed', 'next_focus': 'test', 'data_requests': []}, {'feedback_sha256': 'original'})

if __name__ == '__main__': unittest.main()
