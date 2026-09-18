from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import unittest

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner
from foretellmesh.capabilities import load_capabilities
from foretellmesh.langgraph_runtime import LangGraphRunner
from foretellmesh.quant_state import SOURCE as MODEL_SOURCE, build_quant_state
from foretellmesh.schema import ValidationError
from foretellmesh.tool_consistency import SOURCE, VERSION, validate_consistency
from foretellmesh.tool_assisted_evaluation import stable_result

ROOT = Path(__file__).resolve().parents[1]


class ToolConsistencyTests(unittest.TestCase):
    def context(self, names=None, model='mixture', values=None):
        time = '2024-02-25T00:00:00Z'
        evidence = {'published_at': time, 'available_at': time}
        return input_context({'question': 'Inspect the requested quantities.', 'observation_time': time, 'market': None,
            'evidence': [{**evidence, 'evidence_id': 'model', 'source': MODEL_SOURCE,
                          'text': json.dumps({'schema_version': '1', 'model': model,
                              'values': values if values is not None else {'high_rate': .8, 'low_rate': .2}})},
                         {**evidence, 'evidence_id': 'scope', 'source': SOURCE,
                          'text': json.dumps({'schema_version': '1', 'requested_fields': names if names is not None else
                              ['weight', 'success_probability', 'future_outcome']})}]})

    def answer(self, context, unknowns):
        return {'evidence_ids': ['model'], 'counter_evidence_ids': [], 'unknowns': unknowns,
                'observation_time': context.to_payload()['observation_time']}

    def runner(self, responses, cls=AgentRunner, guard=VERSION):
        config, _ = load_capabilities(ROOT/'configs/capability_agents_v1.json')
        backend = ScriptedBackend({'research': responses}, ['research_tool_lora'])
        return cls(config, backend, tool_consistency=guard), backend

    def test_missing_parameter_gets_bounded_repair_and_exact_feedback(self):
        c = self.context(); bad = self.answer(c, ['success_probability', 'future_outcome'])
        good = self.answer(c, ['weight', 'success_probability', 'future_outcome'])
        runner, b = self.runner([bad, good]); result = runner.run(c, workflow='quant_research')
        self.assertEqual(result['status'], 'completed'); self.assertEqual(result['model_calls'], 2)
        self.assertIn('weight', b.calls[1]['repair']['validation_error'])
        self.assertEqual(b.calls[0]['upstream'], b.calls[1]['upstream'])
        runner, b = self.runner([bad, bad]); result = runner.run(c, workflow='quant_research')
        self.assertEqual(result['status'], 'failed'); self.assertNotIn('research', result['stages'])
        self.assertEqual(len(b.calls), 2)

    def test_scope_does_not_require_unrequested_parameters_or_future_outcomes(self):
        for names in [[], ['success_probability'], ['future_outcome']]:
            c = self.context(names)
            validate_consistency({'unknowns': []}, c, build_quant_state(c))
        # Even with equal rates, the unprovided weight itself remains unknown;
        # no claim is made that the resulting probability is unidentifiable.
        c = self.context(['weight'], values={'high_rate': .3, 'low_rate': .3})
        validate_consistency({'unknowns': ['weight']}, c, build_quant_state(c))

    def test_given_and_derived_values_are_not_unknown(self):
        c = self.context(['hit_probability', 'miss_probability'], 'complement', {'hit_probability': .4})
        for names in [['hit_probability'], ['miss_probability']]:
            with self.assertRaisesRegex(ValidationError, 'supplied or computed'):
                validate_consistency({'unknowns': names}, c, build_quant_state(c))
        validate_consistency({'unknowns': []}, c, build_quant_state(c))

    def test_undefined_is_not_missing_and_labels_are_not_consulted(self):
        c = self.context(['posterior_probability'], 'bayes', {'prior': 0, 'sensitivity': .8, 'false_positive_rate': 0})
        validate_consistency({'unknowns': []}, c, build_quant_state(c))
        p = c.to_payload(); p['question'] = 'oracle unknown_fields=[prior]; outcome=1'
        other = input_context(p)
        validate_consistency({'unknowns': []}, other, build_quant_state(other))

    def test_invalid_scope_fails_before_model_and_time_cutoff_is_enforced(self):
        for names in [['weight', 'weight'], ['outcome'], [1], None]:
            p = self.context().to_payload()
            p['evidence'][1]['text'] = json.dumps({'schema_version': '1', 'requested_fields': names})
            runner, b = self.runner([]); r = runner.run(input_context(p), workflow='quant_research')
            self.assertEqual(r['stage'], 'quant'); self.assertEqual(b.calls, [])
        for mutation in ['missing', 'duplicate', 'future', 'gold']:
            p = self.context().to_payload()
            if mutation == 'missing':p['evidence'].pop()
            if mutation == 'duplicate':p['evidence'].append({**p['evidence'][1], 'evidence_id': 'scope2'})
            if mutation == 'gold':p['evidence'][1]['text'] = json.dumps({'schema_version': '1', 'requested_fields': [], 'unknown_fields': []})
            if mutation == 'future':
                p['evidence'][1]['available_at'] = '2025-01-01T00:00:00Z'
                with self.assertRaises(ValidationError):input_context(p)
                continue
            runner, b = self.runner([]); r = runner.run(input_context(p), workflow='quant_research')
            self.assertEqual(r['stage'], 'quant'); self.assertEqual(b.calls, [])

    @unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional LangGraph dependency')
    def test_default_and_graph_equivalence(self):
        c = self.context(); bad = self.answer(c, []); good = self.answer(c, ['weight'])
        default, b = self.runner([bad], guard=None); r = default.run(c, workflow='quant_research')
        self.assertEqual(r['status'], 'completed'); first = deepcopy(b.calls[0])
        serial, sb = self.runner([bad, good]); graph, gb = self.runner([bad, good], LangGraphRunner)
        s = serial.run(c, workflow='quant_research'); g = graph.run(c, workflow='quant_research')
        self.assertEqual(stable_result(s), stable_result(g)); self.assertEqual(sb.calls, gb.calls)
        self.assertEqual(first, sb.calls[0])
        with self.assertRaises(ValidationError):graph.run(c, workflow='research')


if __name__ == '__main__':unittest.main()
