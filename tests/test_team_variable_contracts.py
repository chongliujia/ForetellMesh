from copy import deepcopy
import json
import unittest

from foretellmesh import team_variable_contracts as typed
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_discovery import DiscoveryRunner
from foretellmesh.team_lifecycle_experiment import RecordedBackend, AuditRunner
from foretellmesh.team_staged_methods import run_staged_workflow
from test_team_executable_methods import fixture
from test_team_staged_methods import inputs, Backend
from test_team_lifecycle import TIMING


def contract_fixture():
    relation, request, calculation = inputs()
    relation['variables'] = [{'variable_id': 'v'+str(i), 'definition': 'Historical '+v['role']+' Yes trade price.',
        'quantity': typed.SOURCE['quantity'], 'unit': typed.SOURCE['unit'], 'role': v['role'], 'lag_days': v['lag_days']}
        for i, v in enumerate(request['inputs'])]
    registry = [{**typed.SOURCE, 'artifact_sha256': 'a'*64, 'source_report_sha256': 'b'*64}]
    mapping = {'bindings': [{'variable_id': v['variable_id'], 'source_id': typed.SOURCE['source_id'], 'missing_reason': None}
                           for v in relation['variables']]}
    return relation, registry, mapping, calculation


class TypedBackend(Backend):
    def __init__(self):
        super().__init__(); self.relation, self.registry, self.mapping, self.calculation = contract_fixture()
    def generate(self, request):
        if request['agent'] == 'method_data_member':
            self.roles.append(request['agent']); return json.dumps(self.mapping)
        return super().generate(request)


class VariableContractTests(unittest.TestCase):
    def test_research_declares_variables_without_source_registry_or_coverage(self):
        _, registry, _, _ = contract_fixture()
        context = {'variable_sources': registry, 'price_coverage': {'a': 10}, 'catalogue': ['retained'], 'candidate': 'retained'}
        original = deepcopy(context); view = typed.research_context(context)
        self.assertNotIn('variable_sources', view); self.assertNotIn('price_coverage', view)
        self.assertTrue(view['source_availability_not_provided']); self.assertEqual(view['catalogue'], ['retained'])
        self.assertEqual(context, original)

    def test_known_revenue_to_price_substitution_is_rejected(self):
        relation, registry, mapping, _ = contract_fixture()
        relation['variables'][0].update(quantity='opening_weekend_gross', unit='USD', definition='Opening weekend movie revenue.')
        with self.assertRaisesRegex(ValidationError, 'source quantity/unit mismatch'):
            typed.validate_mapping(mapping, relation, registry)
        mapping['bindings'][0].update(source_id=None, missing_reason='No point-in-time movie revenue source is registered.')
        typed.validate_mapping(mapping, relation, registry)

    def test_research_catalogue_uses_only_callable_ids_and_never_rewrites_group_ids(self):
        row = {'market_id': 'a', 'question': 'Question', 'initialized_at': '2024-01-01T00:00:00Z',
               'event_group_id': 'group:a', 'record_sha256': 'c'*64}
        view = typed.research_context({'catalogue': [row]})
        self.assertEqual(set(view['catalogue'][0]), {'market_id', 'question', 'initialized_at'})
        cat, _, _, _, _ = fixture(); relation, _, _, _ = contract_fixture()
        relation['bindings'][0]['target'] = 'group:a'
        with self.assertRaisesRegex(ValidationError, 'exact market_id'):
            typed.validate_relation({'relation': relation, 'reason': None}, cat)
        self.assertEqual(relation['bindings'][0]['target'], 'group:a')

    def test_same_unit_does_not_make_different_quantity_interchangeable(self):
        relation, registry, mapping, _ = contract_fixture()
        relation['variables'][0]['quantity'] = 'external_asset_quote'
        with self.assertRaisesRegex(ValidationError, 'quantity/unit mismatch'):
            typed.validate_mapping(mapping, relation, registry)

    def test_matching_quantity_still_requires_exact_unit(self):
        relation, registry, mapping, _ = contract_fixture(); cat, _, _, _, _ = fixture()
        relation['variables'][0]['unit'] = 'cents_per_YES_share'
        with self.assertRaisesRegex(ValidationError, 'incorrect unit'):
            typed.validate_relation({'relation': relation, 'reason': None}, cat)
        with self.assertRaisesRegex(ValidationError, 'quantity/unit mismatch'):
            typed.validate_mapping(mapping, relation, registry)

    def test_source_metadata_cannot_be_rewritten_to_match_a_variable(self):
        _, registry, _, _ = contract_fixture()
        for key, value in [('quantity', 'movie_revenue'), ('unit', 'USD'), ('reader', 'arbitrary_sql'), ('asof_supported', False)]:
            changed = deepcopy(registry); changed[0][key] = value
            with self.assertRaisesRegex(ValidationError, 'source metadata changed'): typed.validate_registry(changed)

    def test_missing_variable_stops_before_price_and_label_access(self):
        cat, _, _, protocol, _ = fixture(); relation, registry, mapping, _ = contract_fixture()
        relation['variables'][0].update(quantity='tweet_count', unit='count', definition='Number of posts.')
        mapping['bindings'][0].update(source_id=None, missing_reason='Historical post-count source is absent.')
        class NoFeed:
            def latest(self, *args): raise AssertionError('price proxy must not be read')
        result = typed.check_sources(mapping, relation, registry, cat, NoFeed(), protocol)
        self.assertEqual(result['status'], 'unavailable_variables')
        self.assertTrue(result['availability_not_attempted']); self.assertEqual(result['observations'], [])
        self.assertEqual(result['unavailable_variables'][0]['variable']['unit'], 'count')
        with self.assertRaisesRegex(ValidationError, 'unbound variable'): typed.compiler_request(mapping, relation, registry)

    def test_cannot_drop_duplicate_invent_or_redefine_bound_variables(self):
        relation, registry, mapping, _ = contract_fixture()
        bad = deepcopy(mapping); bad['bindings'].pop()
        with self.assertRaisesRegex(ValidationError, 'population'): typed.validate_mapping(bad, relation, registry)
        for update in [{'variable_id': 'invented'}, {'variable_id': 'v1'}, {'source_id': 'invented_source'}, {'unit': 'USD'}]:
            bad = deepcopy(mapping); bad['bindings'][0].update(update)
            with self.assertRaises(ValidationError): typed.validate_mapping(bad, relation, registry)

    def test_research_variable_identity_and_time_are_validated(self):
        relation, _, _, _ = contract_fixture(); cat, _, _, _, _ = fixture()
        for update in [{'variable_id': 'v1'}, {'lag_days': -1}, {'lag_days': True}, {'role': 'outcome'}]:
            bad = deepcopy(relation); bad['variables'][0].update(update)
            with self.assertRaises(ValidationError): typed.validate_relation({'relation': bad, 'reason': None}, cat)

    def test_binding_order_does_not_change_formula_inputs_or_research_declarations(self):
        relation, registry, mapping, _ = contract_fixture(); original = deepcopy(relation)
        a = typed.compiler_request(mapping, relation, registry)
        mapping['bindings'].reverse(); b = typed.compiler_request(mapping, relation, registry)
        self.assertEqual(a, b); self.assertEqual(relation, original)
        self.assertEqual(typed.project_relation(relation), {k: v for k, v in relation.items() if k != 'variables'})

    def test_string_null_compatibility_only_records_missing_and_preserves_raw_output(self):
        cat, feed, labels, protocol, _ = fixture(); backend = TypedBackend()
        backend.mapping['bindings'][0].update(source_id='null', missing_reason='No exact source available.')
        raw_mapping = deepcopy(backend.mapping)
        normalized = typed.validate_mapping(raw_mapping, backend.relation, backend.registry)
        self.assertIsNone(normalized['bindings'][0]['source_id'])
        self.assertEqual(raw_mapping['bindings'][0]['source_id'], 'null')
        runner = DiscoveryRunner(backend, None)
        result = run_staged_workflow(runner, {'variable_sources': backend.registry}, cat, feed, labels, protocol,
            lambda p: self.fail('missing source was frozen'), variable_contract=typed)
        self.assertEqual(result['status'], 'unavailable_variables'); self.assertIsNone(result['result'])
        self.assertIn('"source_id": "null"', runner.calls[-1]['output'])
        self.assertEqual(len(runner.calls), 2)
        for reason in (None, ''):
            bad = deepcopy(raw_mapping); bad['bindings'][0]['missing_reason'] = reason
            with self.assertRaises(ValidationError): typed.validate_mapping(bad, backend.relation, backend.registry)

    def test_source_and_variable_hashes_bind_availability_result(self):
        cat, feed, _, protocol, _ = fixture(); relation, registry, mapping, _ = contract_fixture()
        result = typed.check_sources(mapping, relation, registry, cat, feed, protocol)
        self.assertEqual(result['status'], 'ready_for_calculation')
        self.assertEqual(result['jointly_available'], 60)
        self.assertEqual(result['variable_contract_sha256'], canonical_hash(relation))
        self.assertEqual(result['source_registry_sha256'], canonical_hash(registry))
        self.assertEqual(result['mapping_sha256'], canonical_hash(mapping))

    def test_full_typed_workflow_freezes_before_scoring_and_replays_exactly(self):
        cat, feed, labels, protocol, _ = fixture(); backend = TypedBackend()
        runner = DiscoveryRunner(backend, None); context = {'variable_sources': backend.registry}; frozen = []
        class GuardedLabels(dict):
            def __getitem__(self, key):
                if not frozen: raise AssertionError('target read before method freeze')
                return super().__getitem__(key)
        first = run_staged_workflow(runner, context, cat, feed, GuardedLabels(labels), protocol,
            lambda p: frozen.append(deepcopy(p)), variable_contract=typed)
        self.assertEqual(first['status'], 'training_screen_passed'); self.assertEqual(len(frozen), 1)
        self.assertNotIn('variable_sources', runner.calls[0]['request']['input'])
        self.assertEqual(runner.calls[1]['request']['input']['source_registry'], backend.registry)
        quant = next(c['request'] for c in runner.calls if c['request']['agent'] == 'method_quant_member')
        self.assertEqual(quant['input']['declared_variables'], backend.relation['variables'])
        self.assertEqual(quant['input']['source_registry_sha256'], canonical_hash(backend.registry))
        replay = RecordedBackend(runner.calls); audited = AuditRunner(replay, None, TIMING)
        second = run_staged_workflow(audited, context, cat, feed, labels, protocol,
            lambda p: self.assertEqual(p, frozen[0]), variable_contract=typed)
        self.assertEqual(first, second); self.assertEqual(canonical_hash(runner.calls), canonical_hash(audited.calls))

    def test_rejected_substitution_can_only_be_repaired_by_honest_missing_binding(self):
        cat, feed, labels, protocol, _ = fixture()
        class Repair(TypedBackend):
            def __init__(self):
                super().__init__(); self.relation['variables'][0].update(quantity='box_office_gross', unit='USD', definition='Movie revenue.')
            def generate(self, request):
                if request['agent'] == 'method_data_member' and 'repair' in request:
                    self.mapping['bindings'][0].update(source_id=None, missing_reason='No revenue source available.')
                return super().generate(request)
        backend = Repair(); runner = DiscoveryRunner(backend, None)
        result = run_staged_workflow(runner, {'variable_sources': backend.registry}, cat, feed, labels, protocol,
            lambda p: self.fail('missing data frozen'), variable_contract=typed)
        self.assertEqual(result['status'], 'unavailable_variables')
        self.assertEqual(backend.roles, ['relation_researcher', 'method_data_member', 'method_data_member'])
        self.assertIn('quantity/unit mismatch', runner.calls[1]['error'])
        self.assertIsNone(runner.calls[2]['error']); self.assertIsNone(result['result'])


if __name__ == '__main__': unittest.main()
