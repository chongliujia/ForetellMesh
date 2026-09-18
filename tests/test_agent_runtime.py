from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import time
import unittest

from foretellmesh.agent_check import ScriptedBackend, check_agent_workflow
from foretellmesh.agent_runtime import AgentRunner, OUTPUT_PROTOCOLS, PROMPTS, decode_agent_response
from foretellmesh.capabilities import load_capabilities, route_plan
from foretellmesh.peft_runtime import PeftTextBackend, SharedPeftExecutor
from foretellmesh.probability_tools import execute_probability_tool
from foretellmesh.rewards import brier_reward, critic_score_delta, score_forecast_response
from foretellmesh.schema import ValidationError, parse_record
from tests.test_pipeline import sample

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/capability_agents_v1.json"
FIXTURE = ROOT / "examples/agent_workflow_fixture_v1.json"


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config, _ = load_capabilities(CONFIG)
        self.fixture = json.loads(FIXTURE.read_text())
        self.context = parse_record({**sample(), **self.fixture['input']}).forecast_input

    def run_workflow(self, responses=None, workflow='reviewed_forecast', mode='capability'):
        backend = ScriptedBackend(responses or self.fixture['responses'], self.config['capabilities'])
        result = AgentRunner(self.config, backend).run(self.context, workflow=workflow, mode=mode)
        return result, backend

    def test_roles_share_capabilities_without_stacking(self):
        plan = route_plan(self.config, 'reviewed_forecast', 'capability')
        self.assertEqual([s['adapter'] for s in plan['steps']], ['research_tool_lora'] * 2 + ['forecast_lora'] * 2)
        self.assertEqual(plan['max_model_calls'], 8)
        self.assertFalse(plan['checkpoint_quality_verified'])
        self.assertTrue(all(s['adapter'] is None for s in route_plan(self.config, 'reviewed_forecast')['steps']))

    def test_missing_adapter_never_silently_uses_base(self):
        backend = ScriptedBackend(self.fixture['responses'])
        with self.assertRaisesRegex(ValidationError, 'not loaded'):
            AgentRunner(self.config, backend).run(self.context, mode='capability')
        self.assertEqual(backend.calls, [])

    def test_output_protocol_is_explicit_and_never_changes_inputs_or_schema(self):
        legacy, original = self.run_workflow(mode='base')
        backend = ScriptedBackend(self.fixture['responses'])
        result = AgentRunner(self.config, backend, output_protocol='plain_json_v1').run(self.context, workflow='reviewed_forecast')
        self.assertEqual(result['prediction'], legacy['prediction'])
        for before, after in zip(original.calls, backend.calls):
            self.assertEqual(before['input'], after['input'])
            self.assertEqual(before['upstream'], after['upstream'])
            self.assertEqual(before['instruction'], PROMPTS[before['agent']])
            self.assertEqual(after['instruction'], before['instruction'] + OUTPUT_PROTOCOLS['plain_json_v1'])
        with self.assertRaisesRegex(ValidationError, 'unknown output protocol'):
            AgentRunner(self.config, backend, output_protocol='arbitrary')
        bad = ScriptedBackend({'forecast': ['```json\n{}\n```'] * 2})
        result = AgentRunner(self.config, bad, output_protocol='plain_json_v1').run(self.context, workflow='single_forecast')
        self.assertEqual(result['status'], 'failed')

    def test_fence_transport_accepts_one_whole_block_and_preserves_semantic_guards(self):
        valid = self.fixture['responses']['forecast'][0]
        def run(values):
            backend = ScriptedBackend({'forecast': values})
            result = AgentRunner(self.config, backend, response_transport='single_json_fence').run(self.context, workflow='single_forecast')
            return result
        wrap = lambda value: '```json\n' + json.dumps(value) + '\n```'
        result = run([wrap(valid)])
        self.assertEqual(result['prediction'], valid)
        self.assertEqual(result['trace'][0]['decoded_transport'], 'json_fence')
        for bad in ({**valid, 'probability': 1.1}, {**valid, 'key_evidence': ['invented']},
                    {**valid, 'observation_time': '2024-01-01T00:00:00Z'}, {**valid, 'outcome': 1}):
            result = run([wrap(bad)] * 2)
            self.assertEqual(result['status'], 'failed')
            self.assertIsNone(result['prediction'])
        for raw in (wrap(valid) + '\nextra text', 'explanation\n' + wrap(valid),
                    wrap(valid) + '\n' + wrap(valid), '```json\n{"a":1,"a":2}\n```',
                    '```python\n{}\n```', '```json\n{} {}\n```'):
            with self.assertRaises(ValueError):decode_agent_response(raw, 'single_json_fence')

    def test_completed_workflow_schema_context_and_calls(self):
        result, backend = self.run_workflow()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['model_calls'], 4)
        self.assertEqual(result['prediction']['probability'], .5)
        self.assertEqual([x['agent'] for x in backend.calls], ['research', 'risk', 'forecast', 'critic'])
        for request in backend.calls:
            self.assertEqual(set(request['input']), {'question', 'observation_time', 'evidence', 'market'})
            self.assertNotIn('trace', request['upstream'])

    def test_outcomes_are_not_accepted_and_cannot_change_requests(self):
        record = parse_record({**sample(), **self.fixture['input']})
        backend = ScriptedBackend(self.fixture['responses'])
        with self.assertRaisesRegex(ValidationError, 'ForecastInput only'):
            AgentRunner(self.config, backend).run(record)
        first, b1 = self.run_workflow(mode='base')
        changed = parse_record({**sample(label=None), **self.fixture['input']})
        b2 = ScriptedBackend(self.fixture['responses'])
        AgentRunner(self.config, b2).run(changed.forecast_input, workflow='reviewed_forecast')
        self.assertEqual(b1.calls, b2.calls)

    def test_research_cannot_hide_risk_evidence_from_forecaster(self):
        raw = {**sample(), **self.fixture['input']}
        raw['evidence'] = raw['evidence'] + [{**raw['evidence'][0], 'evidence_id': 'risk_evidence'}]
        self.context = parse_record(raw).forecast_input
        responses = deepcopy(self.fixture['responses'])
        responses['risk'][0]['risks'][0]['evidence_ids'] = ['risk_evidence']
        result, backend = self.run_workflow(responses)
        self.assertEqual(result['status'], 'completed')
        for role in ('risk', 'forecast', 'critic'):
            request = next(r for r in backend.calls if r['agent'] == role)
            self.assertIn('risk_evidence', {e['evidence_id'] for e in request['input']['evidence']})

    def test_invalid_schema_is_repaired_once_then_fails_closed(self):
        responses = deepcopy(self.fixture['responses'])
        valid = responses['forecast'][0]
        responses['forecast'] = [{**valid, 'probability': 1.2}, valid]
        result, backend = self.run_workflow(responses, 'single_forecast')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['model_calls'], 2)
        self.assertIn('repair', backend.calls[1])
        responses['forecast'] = [{**valid, 'outcome': 1}, {**valid, 'probability': True}]
        result, backend = self.run_workflow(responses, 'single_forecast')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['prediction'])
        self.assertEqual(len(backend.calls), 2)

    def test_unknown_evidence_and_wrong_time_never_reach_downstream(self):
        for changes in ({'evidence_ids': ['invented']}, {'observation_time': '2025-01-01T00:00:00Z'}):
            responses = deepcopy(self.fixture['responses'])
            bad = {**responses['research'][0], **changes}
            responses['research'] = [bad, bad]
            result, backend = self.run_workflow(responses)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual({r['agent'] for r in backend.calls}, {'research'})

    def test_critic_requires_explicit_evidence_for_probability_revision(self):
        responses = deepcopy(self.fixture['responses'])
        responses['critic'][0].update(accept=False, revised_probability=.6, evidence_ids=[])
        responses['critic'] *= 2
        result, _ = self.run_workflow(responses)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['prediction'])
        self.assertEqual(result['stages']['forecast']['probability'], .5)
        self.assertIn('requires evidence', result['trace'][-1]['validation_error'])
        responses = deepcopy(self.fixture['responses'])
        responses['critic'][0].update(accept=False, revised_probability=.6, rationale='Synthetic test revision only.')
        result, _ = self.run_workflow(responses)
        self.assertEqual(result['prediction']['probability'], .6)
        self.assertEqual(result['stages']['forecast']['probability'], .5)

    def test_call_budget_input_budget_and_backend_failures(self):
        self.config['limits']['max_model_calls'] = 3
        with self.assertRaisesRegex(ValidationError, 'call budget'):
            self.run_workflow()
        self.config['limits']['max_model_calls'] = 8
        self.config['limits']['max_input_chars'] = 10
        result, backend = self.run_workflow()
        self.assertEqual(result['error'], 'input_budget_exceeded')
        self.assertEqual(backend.calls, [])
        self.config['limits']['max_input_chars'] = 24000
        backend = ScriptedBackend({})
        result = AgentRunner(self.config, backend).run(self.context)
        self.assertEqual(result['error'], 'backend_error')
        self.assertIsNone(result['prediction'])

    def test_probability_tool_path_executes_deterministic_math(self):
        responses = {'quant': [{'name': 'bayes_binary', 'arguments': {'prior': .2, 'sensitivity': .8, 'false_positive_rate': .1}}]}
        result, _ = self.run_workflow(responses, 'calculate')
        self.assertEqual(result['tool_result']['exact_fraction'], '2/3')
        self.assertIsNone(result['prediction'])
        for call in ({'name': 'shell', 'arguments': {}},
                     {'name': 'bayes_binary', 'arguments': {'prior': False, 'sensitivity': .5, 'false_positive_rate': .5}},
                     {'name': 'weighted_probability', 'arguments': {'probabilities': [.1], 'weights': [.5]}}):
            with self.assertRaises(ValidationError):
                execute_probability_tool(call)

    def test_scripted_check_report_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'check'
            report = check_agent_workflow(CONFIG, FIXTURE, out)
            self.assertIsNone(report['forecasting_metrics'])
            self.assertEqual(report['result']['model_calls'], 4)
            with self.assertRaisesRegex(ValidationError, 'already exists'):
                check_agent_workflow(CONFIG, FIXTURE, out)

    def test_config_rejects_composition_base_change_and_fake_promotion(self):
        for key, value in [('agents', {**self.config['agents'], 'forecast': ['forecast_lora', 'research_tool_lora']}),
                           ('base_model', 'different-model'),
                           ('capabilities', {**self.config['capabilities'], 'forecast_lora': {**self.config['capabilities']['forecast_lora'], 'status': 'validated'}})]:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'config.json'
                path.write_text(json.dumps({**self.config, key: value}))
                with self.assertRaises(ValidationError):load_capabilities(path)


class RewardTests(unittest.TestCase):
    def test_brier_extremes_and_calibrated_expected_reward(self):
        self.assertEqual(brier_reward(1, 1), 0)
        self.assertEqual(brier_reward(1, 0), -1)
        self.assertEqual(brier_reward(.5, 0), -.25)
        q = .3
        expected = lambda p: q * brier_reward(p, 1) + (1-q) * brier_reward(p, 0)
        self.assertGreater(expected(q), expected(.5))
        self.assertGreater(expected(q), expected(0))
        self.assertAlmostEqual(critic_score_delta(.9, .6, 0), .45)

    def test_schema_failure_has_floor_and_length_has_no_reward_bonus(self):
        fixture = json.loads(FIXTURE.read_text())
        context = parse_record({**sample(), **fixture['input']}).forecast_input
        prediction = fixture['responses']['forecast'][0]
        self.assertEqual(score_forecast_response(prediction, context, 1)['reward'], -.25)
        self.assertEqual(score_forecast_response({**prediction, 'unknowns': ['long explanation ' * 100]}, context, 1)['reward'], -.25)
        invalid = score_forecast_response({**prediction, 'probability': True}, context, 1)
        self.assertFalse(invalid['valid'])
        self.assertEqual(invalid['reward'], -1)
        with self.assertRaises(ValidationError):brier_reward(.5, True)


class FakePeft:
    def __init__(self):
        self.peft_config = {'a': {}, 'b': {}}
        self.active_adapter, self.disabled, self.trainable = 'a', False, True
    def get_layer_status(self):return []
    def eval(self):return self
    def requires_grad_(self, value):self.trainable = value;return self
    def set_adapter(self, value, inference_mode=False):self.active_adapter = value;self.trainable = not inference_mode
    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:yield
        finally:self.disabled = False


class AdapterIsolationTests(unittest.TestCase):
    def test_restore_on_exception_base_disabled_and_missing_adapter(self):
        model = FakePeft();runtime = SharedPeftExecutor(model)
        self.assertTrue(runtime.execute(None, lambda m: m.disabled))
        self.assertFalse(model.disabled)
        with self.assertRaises(RuntimeError):
            runtime.execute('b', lambda m: (_ for _ in ()).throw(RuntimeError('failure')))
        self.assertEqual(model.active_adapter, 'a')
        self.assertFalse(model.trainable)
        with self.assertRaises(ValidationError):runtime.execute('missing', lambda m: None)
        with self.assertRaises(ValidationError):runtime.execute(['a', 'b'], lambda m: None)

    def test_concurrent_requests_cannot_change_active_adapter_mid_operation(self):
        model = FakePeft();runtime = SharedPeftExecutor(model)
        def request(name):
            def operation(m):
                before = m.active_adapter
                time.sleep(.005)
                return before, m.active_adapter, m.trainable
            return runtime.execute(name, operation)
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(request, ['a', 'b'] * 3))
        self.assertEqual(values, [('a', 'a', False), ('b', 'b', False)] * 3)
        self.assertEqual(model.active_adapter, 'a')


class BackendUsageTests(unittest.TestCase):
    def test_usage_counts_generated_tokens_and_does_not_leak_across_errors(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional torch dependency is not installed')
        class Tokenizer:
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False):return [1, 2, 3]
            def decode(self, tokens, skip_special_tokens=True):return '{}'
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
            def generate(self, input_ids, **kwargs):
                return torch.cat([input_ids, torch.tensor([[4, 5]])], dim=1)
        backend = PeftTextBackend(SharedPeftExecutor(Model()), Tokenizer(), max_context_tokens=8, max_new_tokens=2)
        request = {'instruction': 'JSON only', 'input': {}, 'upstream': {}, 'adapter': None}
        self.assertEqual(backend.generate(request), '{}')
        self.assertEqual(backend.last_usage['input_tokens'], 3)
        self.assertEqual(backend.last_usage['output_tokens'], 2)
        self.assertTrue(backend.last_usage['output_reached_token_limit'])
        backend.max_context_tokens = 4
        with self.assertRaisesRegex(ValidationError, 'context budget'):backend.generate(request)
        self.assertIsNone(backend.last_usage)


if __name__ == '__main__':unittest.main()
