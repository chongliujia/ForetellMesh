from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import unittest

from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.capabilities import load_capabilities
from foretellmesh.schema import ValidationError
from foretellmesh.tool_assisted_diagnostic import load_config, numeric_configs, rows_for_configs
from foretellmesh.tool_consistency_diagnostic import scope_names
from foretellmesh.tool_consistency_evaluation import make_jobs, run_job, summarize, load_config as eval_config
from foretellmesh.tool_consistency import SOURCE

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional LangGraph dependency')
class ConsistencyDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(ROOT/'configs/tool_assisted_diagnostic_v1.json')
        cls.inputs, cls.judges = rows_for_configs(numeric_configs(config, {'signatures': []}))
        for row in cls.inputs:
            payload = row['variants']['tool']; evidence = deepcopy(payload['evidence'][0])
            evidence.update(evidence_id='requested_scope', source=SOURCE,
                            text=json.dumps({'schema_version': '1', 'requested_fields': scope_names(payload['question'])}))
            payload['evidence'].append(evidence); row['variants'] = {'tool': payload}
        cls.c = eval_config(ROOT/'configs/tool_consistency_evaluation_v1.json')
        cls.agents, _ = load_capabilities(ROOT/'configs/capability_agents_v1.json')
        cls.jobs = make_jobs(cls.inputs, cls.c, cls.agents)

    def test_scope_contains_only_explicit_question_fields_in_both_languages(self):
        for row, judge in zip(self.inputs, self.judges):
            self.assertEqual(scope_names(row['variants']['tool']['question']), judge['allowed_fields'])
            self.assertNotIn('meanings', scope_names(row['variants']['tool']['question']))
        for question in ['No definitions', 'Field meanings: a: text; a: duplicated', 'Field meanings: badly formed']:
            with self.assertRaises(ValidationError):scope_names(question)

    def test_jobs_are_matched_and_invalid_scope_fails_preflight(self):
        indexed = {(j['sample_id'], j['arm']): j for j in self.jobs}
        for row in self.inputs:
            a, b = (indexed[row['sample_id'], arm] for arm in ['control', 'guarded'])
            self.assertEqual(a['request'], b['request'])
            self.assertNotIn('unknown_fields', json.dumps(a['request']))
        bad = deepcopy(self.inputs); bad[0]['variants']['tool']['evidence'][-1]['text'] = '{}'
        with self.assertRaises(ValidationError):make_jobs(bad, self.c, self.agents)

    def test_first_and_repaired_scores_are_separate_and_no_result_dropping(self):
        labels = {j['sample_id']: j for j in self.judges}; records = []
        for job in self.jobs:
            j = labels[job['sample_id']]
            good = {'unknowns': j['unknown_fields'], 'observation_time': job['input']['observation_time']}
            if job['role'] == 'research':good.update(evidence_ids=[], counter_evidence_ids=[])
            else:good['risks'] = []
            bad = deepcopy(good)
            if j['condition'] == 'mixture_missing_weight':bad['unknowns'].remove('weight')
            backend = ScriptedBackend({job['role']: [bad, good]}, ['research_tool_lora'])
            result = run_job(job, self.agents, backend, self.c)
            calls = [{'request': request, 'raw_output': json.dumps(value), 'usage': {
                'input_tokens': 10, 'output_tokens': 10, 'seconds': 1., 'output_reached_token_limit': False}}
                for request, value in zip(backend.calls, [bad, good])]
            records.append({**job, 'result': result, 'calls': calls, 'seconds': len(calls), 'memory': {'peak_allocated_bytes': 100}})
        m = summarize(self.inputs, self.judges, records, self.c, self.agents)
        self.assertEqual(m['cells']['control']['first']['correct'], 56)
        self.assertEqual(m['cells']['guarded']['first']['correct'], 56)
        self.assertEqual(m['cells']['guarded']['final']['correct'], 64)
        self.assertEqual(m['matched_contrasts']['improved'], 8)
        self.assertEqual(m['cells']['guarded']['resources']['repair_calls'], 8)
        for bad in [records[:-1], records + [records[0]]]:
            with self.assertRaises(ValidationError):summarize(self.inputs, self.judges, bad, self.c, self.agents)


if __name__ == '__main__':unittest.main()
