from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.agent_baseline import tool_call_matches
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner
from foretellmesh.capabilities import load_capabilities, route_plan
from foretellmesh.capability_training import encode_capability_row, epoch_batches, load_training_config
from foretellmesh.capability_evaluation import judge_role, summarize_capability
from foretellmesh.data import sha256_file
from foretellmesh.peft_runtime import render_agent_prompt
from foretellmesh.probability_tools import execute_probability_tool
from foretellmesh.research_tool_data import build_research_tool_data, read_research_tool_data, render_task, validate_rows
from foretellmesh.schema import ValidationError, parse_record
from foretellmesh.synthetic_sft import generate_synthetic_sft
from tests.test_pipeline import sample

ROOT = Path(__file__).resolve().parents[1]


class ResearchToolDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = json.loads((ROOT / 'configs/synthetic_sft_generator_v1.json').read_text())
        generator['groups_per_family'] = {'train': 2, 'validation': 3, 'test': 3}
        g = self.root / 'generator.json'; g.write_text(json.dumps(generator))
        self.raw, self.bundle = self.root / 'raw', self.root / 'bundle'
        generate_synthetic_sft(g, self.raw)
        c = json.loads((ROOT / 'configs/research_tool_dataset_v1.json').read_text())
        c['evidence_groups'] = {p: 3 for p in ('train', 'validation', 'test')}
        self.config = self.root / 'config.json'; self.config.write_text(json.dumps(c))
        self.splits = ROOT / 'configs/synthetic_sft_splits_v1.json'

    def build(self):
        build_research_tool_data(self.raw, self.splits, self.config, self.bundle)
        return read_research_tool_data(self.bundle)

    def test_replay_determinism_group_integrity_and_input_isolation(self):
        manifest, parts, _ = self.build()
        self.assertEqual(manifest['validation_evaluation_examples'], 24)
        owners = {}
        for p, rows in parts.items():
            for row in rows:
                self.assertEqual(owners.setdefault(row['event_group_id'], p), p)
                payload = row['request']['input']
                self.assertEqual(set(payload), {'question', 'observation_time', 'evidence', 'market'})
                self.assertNotIn('target', row['request'])
                self.assertNotIn('oracle', row['request'])
                self.assertEqual(row, render_task(row['case'], p))
                if row['task'].startswith('quant'):
                    self.assertEqual([e['evidence_id'] for e in payload['evidence']], ['setup'])
                    self.assertNotIn('expected_result', render_agent_prompt(row['request']))
                if row['task']=='quant_repair':
                    self.assertEqual(set(row['request']['repair']), {'validation_error', 'instruction'})
        second = build_research_tool_data(self.raw, self.splits, self.config, self.root / 'second')
        self.assertEqual(manifest['artifact_hashes'], second['artifact_hashes'])
        with self.assertRaisesRegex(ValidationError, 'already exists'):
            build_research_tool_data(self.raw, self.splits, self.config, self.bundle)

    def test_event_variant_cross_split_is_rejected(self):
        _, parts, config = self.build()
        row = deepcopy(next(r for r in parts['train'] if r['case']['kind']=='evidence'))
        row['case']['observation_time'] = config['observation_times']['validation']
        parts['validation'].append(render_task(row['case'], 'validation'))
        with self.assertRaisesRegex(ValidationError, 'cross-partition'):validate_rows(parts, config)

    def test_semantically_changed_target_is_rejected_even_after_rehash(self):
        self.build()
        path = self.bundle / 'train.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        row = next(r for r in rows if r['task']=='research')
        row['target']['evidence_ids'] = [row['oracle']['forbidden_evidence_ids'][0]]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest_path = self.bundle / 'manifest.json'
        m = json.loads(manifest_path.read_text()); m['artifact_hashes']['train.jsonl'] = sha256_file(path)
        manifest_path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValidationError, 'deterministic case replay'):read_research_tool_data(self.bundle)

    def test_unverified_commentary_and_other_event_never_become_gold_evidence(self):
        _, parts, _ = self.build()
        for row in parts['train']:
            if row['case']['kind'] != 'evidence':continue
            target = row['target']
            refs = target.get('evidence_ids', []) + target.get('counter_evidence_ids', [])
            for risk in target.get('risks', []):refs.extend(risk['evidence_ids'])
            self.assertFalse(set(refs) & set(row['oracle']['forbidden_evidence_ids']))
            if row['task']=='risk' and not row['oracle']['risk_present']:self.assertEqual(target['risks'], [])

    def test_encoding_masks_prompt_and_matches_serving_prompt(self):
        _, parts, _ = self.build()
        class Tokenizer:
            eos_token_id = 999999
            def encode(self, text, add_special_tokens=False):return list(text.encode())
        tokenizer = Tokenizer(); row = parts['train'][0]
        prompt = render_agent_prompt(row['request'])
        encoded = encode_capability_row(tokenizer, row, 10000)
        n = len(prompt.encode())
        self.assertEqual(encoded['input_ids'][:n], list(prompt.encode()))
        self.assertEqual(encoded['labels'][:n], [-100]*n)
        self.assertEqual(encoded['labels'][n:], encoded['input_ids'][n:])
        self.assertEqual(encoded['labels'][-1], tokenizer.eos_token_id)
        with self.assertRaisesRegex(ValidationError, 'truncation is forbidden'):
            encode_capability_row(tokenizer, row, 10)

    def test_evidence_judge_checks_meaningful_targets_not_just_existing_ids(self):
        _,parts,_=self.build()
        row=next(r for r in parts['validation'] if r['task']=='research')
        output=deepcopy(row['target']);output['evidence_ids']=[row['oracle']['forbidden_evidence_ids'][0]]
        result=judge_role(row,output)
        self.assertTrue(result['schema_valid'])
        self.assertFalse(result['task_correct'])
        self.assertTrue(result['includes_forbidden_evidence'])
        row=next(r for r in parts['validation'] if r['task']=='risk' and r['oracle']['risk_present'])
        output=deepcopy(row['target']);output['risks'][0]['scenario']='An invented technical failure.'
        self.assertFalse(judge_role(row,output)['task_correct'])
        self.assertTrue(judge_role(row,output)['schema_valid'])
        self.assertTrue(judge_role(row,row['target'])['task_correct'])

    def test_paired_metrics_count_failed_cases_and_reject_incomplete_runs(self):
        _,parts,_=self.build()
        role=next(r for r in parts['validation'] if r['task']=='quant')
        cohort={'judge.jsonl':[{'sample_id':'system1','outcome':1,'oracle_probability':.7}]}
        results=[]
        for arm in ('base','candidate'):
            common={'arm':arm,'seconds':1,'calls':[],'memory':{'peak_allocated_bytes':1}}
            results.append({**common,'kind':'role','sample_id':role['sample_id'],
                            'output':role['target'] if arm=='base' else None,'decoded_transport':'raw_json' if arm=='base' else None})
            results.append({**common,'kind':'system','sample_id':'system1',
                            'result':{'prediction':{'probability':.7} if arm=='base' else None,
                                      'status':'completed' if arm=='base' else 'failed','trace':[{'attempt':0}]}})
        report=summarize_capability([role],cohort,results)
        self.assertEqual(report['role']['candidate']['tasks']['quant']['task_accuracy'],0)
        self.assertEqual(report['system']['candidate']['scores']['missing_count'],1)
        self.assertIsNone(report['system']['candidate']['scores']['brier'])
        self.assertEqual(report['system_common_coverage']['count'],0)
        with self.assertRaisesRegex(ValidationError,'incomplete'):summarize_capability([role],cohort,results[:-1])
        with self.assertRaisesRegex(ValidationError,'duplicate'):summarize_capability([role],cohort,results+[results[0]])


class CapabilityTrainingContractTests(unittest.TestCase):
    def test_accumulation_covers_tail_and_shuffle_is_deterministic(self):
        groups = epoch_batches(19, 16, 42, 0)
        self.assertEqual([len(g) for g in groups], [16,3])
        self.assertEqual(sorted(i for g in groups for i in g), list(range(19)))
        self.assertEqual(groups, epoch_batches(19,16,42,0))
        self.assertNotEqual(groups, epoch_batches(19,16,42,1))

    def test_semantic_tool_judge_rejects_swapped_roles_even_when_result_matches(self):
        expected = {'name':'weighted_probability','arguments':{'probabilities':[.1,.9],'weights':[.2,.8]}}
        wrong = {'name':'weighted_probability','arguments':{'probabilities':[.2,.8],'weights':[.1,.9]}}
        self.assertEqual(execute_probability_tool(expected)['result'],execute_probability_tool(wrong)['result'])
        self.assertFalse(tool_call_matches(wrong, expected))

    def test_training_config_rejects_unpinned_base_and_raw_rl(self):
        config = load_training_config(ROOT / 'configs/qwen3_8b_research_tool_sft_v1.json')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            for changes in ({'model_revision':'main'}, {'optimizer':'grpo'}, {'epochs':0}, {'quantization':'4bit'}):
                p.write_text(json.dumps({**config, **changes}))
                with self.assertRaises(ValidationError):load_training_config(p)

    def test_explicit_scope_routes_frozen_roles_to_base_without_faking_adapters(self):
        config,_ = load_capabilities(ROOT / 'configs/capability_agents_v1.json')
        plan = route_plan(config, 'reviewed_forecast', 'capability', capability_scope={'research_tool_lora'})
        self.assertEqual([s['adapter'] for s in plan['steps']], ['research_tool_lora','research_tool_lora',None,None])
        self.assertEqual(plan['requires_loaded_adapters'], ['research_tool_lora'])
        fixture = json.loads((ROOT / 'examples/agent_workflow_fixture_v1.json').read_text())
        context = parse_record({**sample(), **fixture['input']}).forecast_input
        backend = ScriptedBackend(fixture['responses'], {'research_tool_lora'})
        result = AgentRunner(config,backend).run(context,workflow='reviewed_forecast',mode='capability',capability_scope={'research_tool_lora'})
        self.assertEqual(result['status'],'completed')
        self.assertEqual([r['adapter'] for r in backend.calls], ['research_tool_lora','research_tool_lora',None,None])
        with self.assertRaisesRegex(ValidationError,'not loaded'):
            AgentRunner(config,ScriptedBackend(fixture['responses'])).run(context,mode='capability',capability_scope={'research_tool_lora'})
        for mode,scope in [('base',{'research_tool_lora'}),('capability',{'missing'}),('capability',set())]:
            with self.assertRaises(ValidationError):route_plan(config,'research_forecast',mode,capability_scope=scope)


if __name__ == '__main__':unittest.main()
