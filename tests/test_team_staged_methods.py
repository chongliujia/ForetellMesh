from copy import deepcopy
from datetime import timedelta
import json
import unittest

from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.team_staged_methods import (validate_relation, validate_data, check_data,
    validate_calculation, assemble, run_staged_workflow)
from test_team_executable_methods import fixture, AT
from test_team_lifecycle import TIMING


def inputs():
    relation = {'hypothesis': 'Peer one-day Yes price changes predict the next target Yes price change.',
        'forecast_target': 'future_yes_price', 'horizon_days': 1,
        'bindings': [{'target': a, 'peer': b} for a, b in [('a', 'b'), ('b', 'c'), ('c', 'a')]]}
    request = {'inputs': [{'role': r, 'quantity': 'yes_price', 'lag_days': d}
        for r, d in [('target', 0), ('peer', 0), ('peer', 1)]], 'missing_data': []}
    calculation = {'expression': "p('target',0)+p('peer',0)-p('peer',1)",
                   'min_observations': 8, 'min_improvement': .00005, 'reason': None}
    return relation, request, calculation


class Backend:
    def __init__(self): self.relation, self.request, self.calculation = inputs(); self.roles = []
    def generate(self, request):
        role = request['agent']; self.roles.append(role)
        if role == 'relation_researcher': value = {'relation': self.relation, 'reason': None}
        elif role == 'method_data_member': value = self.request
        elif role == 'method_quant_member': value = self.calculation
        else:
            result = request['upstream']['result']
            value = {k: result[k] for k in ('result_sha256', 'status', 'eligible_for_independent_validation')}
            value['limitation'] = 'Training diagnostic; not independent effectiveness or net profit.'
        return json.dumps(value)


class StagedMethodsTests(unittest.TestCase):
    def test_relation_preserves_targets_and_requires_consistent_binding_roles(self):
        cat, _, _, _, _ = fixture(); r, _, _ = inputs()
        validate_relation({'relation': r, 'reason': None}, cat)
        for binding in [{'target': 'a', 'peer': 'a'}, {'target': 'unknown', 'peer': 'b'}, {'target': 'a', 'peer': None}]:
            bad = deepcopy(r); bad['bindings'][0] = binding
            with self.assertRaises(ValidationError): validate_relation({'relation': bad, 'reason': None}, cat)

    def test_external_required_data_stops_before_quote_access(self):
        cat, _, _, protocol, _ = fixture(); relation, request, _ = inputs()
        request['inputs'][1]['quantity'] = 'opening_weekend_gross_usd'
        class NoPrices:
            def latest(self, *args): raise AssertionError('external data cannot be replaced by prices')
        check = check_data(request, relation, cat, NoPrices(), protocol)
        self.assertEqual(check['status'], 'unsupported_data')
        self.assertEqual(check['unsupported_inputs'][0]['quantity'], 'opening_weekend_gross_usd')
        self.assertEqual(check['observations'], [])

    def test_data_roles_future_lags_and_duplicates_are_rejected(self):
        relation, request, _ = inputs()
        for role, lag in [('outcome', 0), ('peer', -1), ('peer', True), ('peer', 31)]:
            bad = deepcopy(request); bad['inputs'][0].update(role=role, lag_days=lag)
            with self.assertRaises(ValidationError): validate_data(bad, relation)
        request['inputs'].append(deepcopy(request['inputs'][0]))
        with self.assertRaisesRegex(ValidationError, 'duplicate'): validate_data(request, relation)

    def test_joint_coverage_distinguishes_disjoint_histories(self):
        cat, _, _, protocol, _ = fixture(); relation, request, _ = inputs()
        relation['bindings'] = [{'target': 'a', 'peer': 'b'}]
        data = {'a': [(AT+timedelta(days=d), .4, 1) for d in range(1, 8)],
                'b': [(AT+timedelta(days=d), .5, 1) for d in range(10, 21)]}
        check = check_data(request, relation, cat, HistoricalBars(data), protocol)
        self.assertEqual(check['status'], 'no_joint_observations')
        self.assertEqual(check['jointly_available'], 0)

    def test_calculation_must_use_exact_declared_inputs_without_rewriting(self):
        cat, _, _, _, _ = fixture(); relation, request, calculation = inputs()
        validate_calculation(calculation, relation, request, cat)
        for expression in ["p('target',0)", "p('target',0)+p('peer',0)-p('peer',2)"]:
            with self.assertRaisesRegex(ValidationError, 'formula inputs differ'):
                validate_calculation({**calculation, 'expression': expression}, relation, request, cat)
        proposal = assemble(relation, calculation)
        self.assertEqual({k: proposal['plan'][k] for k in relation}, relation)
        self.assertEqual(proposal['plan']['expression'], calculation['expression'])

    def test_each_handoff_is_bounded_and_freeze_precedes_target_access(self):
        cat, feed, labels, protocol, _ = fixture(); frozen = []
        class GuardedLabels(dict):
            def __getitem__(self, key):
                if not frozen: raise AssertionError('labels read before freeze')
                return super().__getitem__(key)
        backend = Backend(); runner = DiscoveryRunner(backend, None)
        context = {'full_research_catalogue_marker': True}
        result = run_staged_workflow(runner, context, cat, feed, GuardedLabels(labels), protocol,
                                    lambda p: frozen.append(deepcopy(p)))
        self.assertEqual(len(frozen), 1); self.assertIsNone(result['proposal_error'])
        self.assertEqual(result['status'], 'training_screen_passed')
        self.assertEqual(result['result']['valid_observations'], 60)
        self.assertEqual(backend.roles, ['relation_researcher', 'method_data_member', 'method_quant_member', 'experiment_auditor'])
        for call in runner.calls[1:]:
            self.assertNotIn('full_research_catalogue_marker', call['request']['input'])
        quant = runner.calls[2]['request']
        self.assertNotIn('observations', quant['upstream']['input_check'])
        self.assertNotIn('outcome', json.dumps(quant['input']))

    def test_missing_evidence_skips_quant_scoring_and_label_access(self):
        cat, feed, _, protocol, _ = fixture(); backend = Backend()
        backend.request['missing_data'] = ['Required historical attendance evidence is unavailable.']
        runner = DiscoveryRunner(backend, None)
        class NoLabels(dict):
            def __getitem__(self, key): raise AssertionError('labels accessed after data failure')
        def no_freeze(p): raise AssertionError('unexecutable method frozen')
        result = run_staged_workflow(runner, {}, cat, feed, NoLabels(), protocol, no_freeze)
        self.assertEqual(result['status'], 'unsupported_data')
        self.assertEqual(backend.roles, ['relation_researcher', 'method_data_member'])
        self.assertIsNone(result['proposal']); self.assertIsNone(result['result'])

    def test_invalid_formula_gets_one_repair_and_is_not_silently_corrected(self):
        cat, feed, labels, protocol, _ = fixture(); backend = Backend()
        backend.calculation['expression'] = "p('target',0)"
        runner = DiscoveryRunner(backend, None)
        result = run_staged_workflow(runner, {}, cat, feed, labels, protocol,
            lambda p: self.fail('invalid plan frozen'))
        self.assertEqual(result['status'], 'calculation_failed')
        self.assertEqual(backend.roles.count('method_quant_member'), 2)
        self.assertIsNone(result['result'])
        self.assertEqual(runner.calls[-1]['output'], runner.calls[-2]['output'])

    def test_frozen_call_replay_reproduces_stages_and_score(self):
        cat, feed, labels, protocol, _ = fixture(); runner = DiscoveryRunner(Backend(), None)
        originals = []
        first = run_staged_workflow(runner, {}, cat, feed, labels, protocol, lambda p: originals.append(deepcopy(p)))
        replay = RecordedBackend(runner.calls); audited = AuditRunner(replay, None, TIMING)
        second = run_staged_workflow(audited, {}, cat, feed, labels, protocol,
            lambda p: self.assertEqual(p, originals[0]))
        self.assertEqual(first, second); self.assertEqual(replay.index, len(runner.calls))
        self.assertEqual(canonical_hash(audited.calls), canonical_hash(runner.calls))

    def test_partial_relation_abstention_never_becomes_a_calculation(self):
        cat, feed, labels, protocol, _ = fixture()
        class Abstain:
            def generate(self, request): return json.dumps({'relation': None, 'reason': 'No supported relationship in this catalogue.'})
        runner = DiscoveryRunner(Abstain(), None)
        result = run_staged_workflow(runner, {}, cat, feed, labels, protocol, lambda p: self.fail('abstention frozen'))
        self.assertEqual(result['status'], 'no_relation'); self.assertEqual(len(runner.calls), 1)
        self.assertIsNone(result['data_check']); self.assertIsNone(result['result'])


if __name__ == '__main__': unittest.main()
