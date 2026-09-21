from copy import deepcopy
from datetime import timedelta
import json
import unittest

from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import ValidationError, timestamp, iso
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_executable_methods import (
    compile_expression, calculate, validate_plan, forecast_at, execute_plan, validate_review, run_workflow)


AT = timestamp('2024-01-01T00:00:00Z', 'test')


def fixture():
    catalogue = {k: {'initialized_at': iso(AT), 'event_group_id': 'g'+k} for k in ('a', 'b', 'c')}
    feed = HistoricalBars({k: [(AT+timedelta(days=d), .3+.01*d, 1) for d in range(32)] for k in catalogue})
    labels = {k: {'outcome': 1, 'resolution_time': iso(AT+timedelta(days=31)),
                  'available_at': iso(AT+timedelta(days=32))} for k in catalogue}
    protocol = {'as_of': iso(AT+timedelta(days=30)), 'feedback_cutoff': iso(AT+timedelta(days=40)),
                'max_days_per_target': 20, 'max_quote_age_seconds': 10800, 'min_event_groups': 3,
                'min_coverage': .8, 'partition': 'train', 'independent_validation': False}
    proposal = {'plan': {'hypothesis': 'Test a fixed one-point change on synthetic paths.',
                'forecast_target': 'future_yes_price', 'horizon_days': 1, 'expression': "p('target',0)+0.01",
                'bindings': [{'target': k, 'peer': None} for k in catalogue],
                'min_observations': 8, 'min_improvement': .00005}, 'no_plan_reason': None}
    return catalogue, feed, labels, protocol, proposal


class ExecutableMethodsTests(unittest.TestCase):
    def test_expression_rejects_python_execution_and_future_features(self):
        for expression in ["__import__('os').system('true')", "p('target',-1)", "p('target',31)",
                           "p('target',True)", "p('outcome',0)", 'target.outcome', '[x for x in []]',
                           "p('target',0)**10", 'nan', "p('target',0) if True else 0", '1',
                           "p('target',0)+"+'9'*330]:
            with self.subTest(expression=expression), self.assertRaises(ValidationError): compile_expression(expression)

    def test_arithmetic_and_compound_cross_contract_relation_are_executable(self):
        root, refs = compile_expression("max(0,min(1,p('target',0)+(p('peer',0)-p('peer',2))/2))")
        self.assertEqual(refs, [('peer', 0), ('peer', 2), ('target', 0)])
        self.assertAlmostEqual(calculate(root, {('target', 0): .3, ('peer', 0): .6, ('peer', 2): .4}), .4)

    def test_bindings_must_match_formula_and_admitted_catalogue(self):
        cat, _, _, _, p = fixture()
        for binding in [{'target': 'x', 'peer': None}, {'target': 'a', 'peer': 'a'}, {'target': 'a', 'peer': 'b'}]:
            q = deepcopy(p); q['plan']['bindings'] = [binding]
            with self.assertRaises(ValidationError): validate_plan(q, cat)
        p['plan']['expression'] = "p('peer',0)"
        with self.assertRaises(ValidationError): validate_plan(p, cat)

    def test_positive_and_negative_predictive_comparisons_are_tool_owned(self):
        cat, feed, labels, protocol, p = fixture()
        result = execute_plan(p, cat, feed, labels, protocol)
        self.assertEqual(result['status'], 'training_screen_passed')
        self.assertEqual(result['valid_observations'], 60)
        self.assertEqual(result['target_event_groups'], 3)
        self.assertAlmostEqual(result['mean_group_improvement'], .0001)
        self.assertFalse(result['effectiveness_verified']); self.assertFalse(result['net_profit_evaluated'])
        p['plan']['expression'] = "p('target',0)-0.01"
        negative = execute_plan(p, cat, feed, labels, protocol)
        self.assertEqual(negative['status'], 'training_screen_failed')
        self.assertLess(negative['mean_group_improvement'], 0)

    def test_many_rows_from_one_event_are_not_independent_events(self):
        cat, feed, labels, protocol, p = fixture()
        for row in cat.values(): row['event_group_id'] = 'one-event'
        r = execute_plan(p, cat, feed, labels, protocol)
        self.assertEqual(r['valid_observations'], 60)
        self.assertEqual(r['target_event_groups'], 1)
        self.assertEqual(r['status'], 'insufficient_data')

    def test_missing_and_out_of_range_predictions_are_not_filled_or_clipped(self):
        cat, feed, labels, protocol, p = fixture()
        missing = execute_plan(p, cat, HistoricalBars({}), labels, protocol)
        self.assertEqual(missing['observations_attempted'], 60)
        self.assertEqual(missing['coverage'], 0)
        for expr, reason in [("p('target',0)/0", 'division by zero'), ("p('target',0)+2", 'prediction outside [0,1]')]:
            p['plan']['expression'] = expr
            r = execute_plan(p, cat, feed, labels, protocol)
            self.assertEqual(r['failures'], {reason: 60})

    def test_future_data_changes_labels_but_not_existing_predictions(self):
        cat, feed, labels, protocol, p = fixture()
        at = AT+timedelta(days=5)
        original = forecast_at(p['plan'], p['plan']['bindings'][0], cat, feed, at, 10800)
        changed = HistoricalBars({k: [(t, price if t <= at else .99, n) for t, price, n in rows]
                                  for k, rows in feed.data.items()})
        self.assertEqual(original, forecast_at(p['plan'], p['plan']['bindings'][0], cat, changed, at, 10800))
        class BadFeed:
            def latest(self, mid, t): return .5, t+timedelta(seconds=1)
        with self.assertRaisesRegex(ValidationError, 'future'):
            forecast_at(p['plan'], p['plan']['bindings'][0], cat, BadFeed(), at, 10800)

    def test_lag_features_respect_source_time_and_initialization(self):
        cat, feed, _, _, p = fixture(); p['plan']['expression'] = "p('target',2)"
        _, evidence, error = forecast_at(p['plan'], p['plan']['bindings'][0], cat, feed, AT+timedelta(days=1), 10800)
        self.assertEqual(error, 'before_initialization')
        _, evidence, error = forecast_at(p['plan'], p['plan']['bindings'][0], cat, feed, AT+timedelta(days=5), 10800)
        self.assertIsNone(error)
        self.assertTrue(all(timestamp(r['source_time'], 'source') <= timestamp(r['sample_time'], 'sample') for r in evidence))

    def test_target_requires_new_fresh_quote_and_windows_do_not_overlap(self):
        cat, feed, labels, protocol, p = fixture(); p['plan']['horizon_days'] = 3
        r = execute_plan(p, cat, feed, labels, protocol)
        rows = [r for r in r['observations'] if r['binding']['target'] == 'a']
        for a, b in zip(rows, rows[1:]):
            self.assertLessEqual(timestamp(a['target_evidence']['sample_time'], 'end'), timestamp(b['observation_time'], 'start'))
        self.assertTrue(all(timestamp(r['target_evidence']['sample_time'], 'target') <= timestamp(protocol['as_of'], 'cutoff') for r in rows))

    def test_event_target_is_scored_once_and_feedback_is_not_backdated(self):
        cat, feed, labels, protocol, p = fixture()
        p['plan'].update(forecast_target='resolves_yes', horizon_days=None, min_observations=3, expression="1-p('target',0)")
        r = execute_plan(p, cat, feed, labels, protocol)
        self.assertEqual(r['metric'], 'brier'); self.assertEqual(r['valid_observations'], 3)
        self.assertEqual(r['status'], 'training_screen_passed')
        protocol['feedback_cutoff'] = protocol['as_of']
        r = execute_plan(p, cat, feed, labels, protocol)
        self.assertEqual(r['valid_observations'], 0)
        self.assertEqual(r['failures'], {'target:feedback_not_available': 3})

    def test_no_final_test_or_relabelled_independence(self):
        cat, feed, labels, protocol, p = fixture()
        for field, value in [('partition', 'test'), ('independent_validation', True), ('min_event_groups', 1)]:
            bad = {**protocol, field: value}
            with self.assertRaises(ValidationError): execute_plan(p, cat, feed, labels, bad)

    def test_critic_cannot_override_numeric_verdict_even_with_well_formed_output(self):
        cat, feed, labels, protocol, p = fixture(); p['plan']['expression'] = "p('target',0)"
        r = execute_plan(p, cat, feed, labels, protocol)
        v = {k: r[k] for k in ('result_sha256', 'status', 'eligible_for_independent_validation')}
        v['limitation'] = 'Training diagnostic only.'
        validate_review(v, r)
        for key, value in [('status', 'training_screen_passed'), ('eligible_for_independent_validation', True), ('result_sha256', 'fake')]:
            with self.assertRaisesRegex(ValidationError, 'contradicts'):
                validate_review({**v, key: value}, r)

    def test_workflow_freezes_before_target_access_and_never_retries_on_bad_score(self):
        cat, feed, labels, protocol, p = fixture(); frozen = []
        class Labels(dict):
            def __getitem__(self, key):
                if not frozen: raise AssertionError('targets read before registration')
                return super().__getitem__(key)
        class Backend:
            def generate(self, request):
                if request['agent'] == 'experiment_planner':
                    self.plan_calls = getattr(self, 'plan_calls', 0)+1
                    return json.dumps(p)
                r = request['upstream']['result']
                return json.dumps({**{k: r[k] for k in ('result_sha256', 'status', 'eligible_for_independent_validation')},
                                   'limitation': 'Not an independent or net-profit comparison.'})
        backend = Backend(); runner = DiscoveryRunner(backend, None)
        r = run_workflow(runner, {}, cat, feed, Labels(labels), protocol, lambda v: frozen.append(deepcopy(v)))
        self.assertEqual(frozen, [p]); self.assertEqual(backend.plan_calls, 1)
        self.assertIsNone(r['review_error']); self.assertEqual(len(runner.calls), 2)

    def test_abstention_is_valid_and_does_not_invent_a_test(self):
        cat, feed, labels, protocol, _ = fixture()
        r = execute_plan({'plan': None, 'no_plan_reason': 'Tweet-count evidence is unavailable.'}, cat, feed, labels, protocol)
        self.assertEqual(r['status'], 'no_plan'); self.assertEqual(r['observations'], [])
        self.assertFalse(r['eligible_for_independent_validation'])


if __name__ == '__main__': unittest.main()
