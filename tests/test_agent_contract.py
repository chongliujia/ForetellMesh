from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner, agent_instruction
from foretellmesh.capabilities import load_capabilities
from foretellmesh.capability_evaluation import PREVIOUS_ADAPTER, evaluation_agents, load_evaluation_config, summarize_capability
from foretellmesh.capabilities import route_plan
from foretellmesh.data import sha256_file
from foretellmesh.peft_runtime import render_agent_prompt
from foretellmesh.schema import ValidationError
from foretellmesh.uncertainty_diagnostic import (build_uncertainty_cohort, judge_uncertainty,
    load_uncertainty_config, probe_rows, read_uncertainty_cohort, render_probe)

ROOT = Path(__file__).resolve().parents[1]


def probe_answer(row, labels):
    base = {'unknowns':labels['unknown_fields'], 'observation_time':row['input']['observation_time']}
    return {**base, 'risks':[]} if row['role']=='risk' else {
        **base,'evidence_ids':[row['input']['evidence'][0]['evidence_id']],'counter_evidence_ids':[]}


class AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.agents,_=load_capabilities(ROOT/'configs/capability_agents_v1.json')
        self.fixture=json.loads((ROOT/'examples/agent_workflow_fixture_v1.json').read_text())
        self.config=load_uncertainty_config(ROOT/'configs/uncertainty_diagnostic_v1.json')

    def test_multiline_event_retry_copies_only_visible_metadata(self):
        payload=deepcopy(self.fixture['input'])
        payload['question']='Will project "甲" succeed?\n{"weight": 63, "high": 87, "low": 19}'
        context=input_context(payload)
        good={**self.fixture['responses']['forecast'][0],'event':payload['question']}
        bad={**good,'event':payload['question'].split('\n')[0]}
        backend=ScriptedBackend({'forecast':[bad,good]})
        result=AgentRunner(self.agents,backend,output_protocol='grounded_json_v2').run(context,workflow='single_forecast')
        self.assertEqual(result['prediction'],good)
        self.assertEqual(result['trace'][0]['status'],'invalid_output')
        feedback=backend.calls[1]['repair']
        self.assertEqual(feedback['required_input_copies'],{'event':payload['question'],'observation_time':payload['observation_time']})
        body=json.loads(render_agent_prompt(backend.calls[1]).split('Input JSON:\n')[1].split('\nOutput JSON:')[0])
        self.assertEqual(body['repair']['required_input_copies']['event'],payload['question'])
        self.assertEqual(set(feedback),{'validation_error','instruction','required_input_copies'})
        self.assertEqual(backend.calls[1]['input'],payload)

    def test_new_prompt_never_repairs_the_model_answer_or_weakens_schema(self):
        context=input_context(self.fixture['input']);good=self.fixture['responses']['forecast'][0]
        for changes in ({'event':'wrong'}, {'probability':1.1}, {'outcome':1}, {'key_evidence':['invented']},
                        {'observation_time':'2026-01-01T00:00:00Z'}):
            backend=ScriptedBackend({'forecast':[{**good,**changes}]*2})
            result=AgentRunner(self.agents,backend,output_protocol='grounded_json_v2',response_transport='single_json_fence').run(context,workflow='single_forecast')
            self.assertIsNone(result['prediction']);self.assertEqual(result['status'],'failed')
            self.assertEqual(len(backend.calls),2)

    def test_legacy_repairs_and_input_cutoff_are_preserved(self):
        context=input_context(self.fixture['input']);good=self.fixture['responses']['forecast'][0]
        backend=ScriptedBackend({'forecast':[{**good,'event':'wrong'},good]})
        AgentRunner(self.agents,backend,output_protocol='plain_json_v1').run(context,workflow='single_forecast')
        self.assertEqual(set(backend.calls[1]['repair']),{'validation_error','instruction'})
        payload=deepcopy(self.fixture['input'])
        payload['evidence'][0]['available_at']='2099-01-01T00:00:00Z'
        with self.assertRaises(ValidationError):input_context(payload)
        self.assertEqual(agent_instruction('quant','plain_json_v1'),agent_instruction('quant','grounded_json_v2'))

    def test_probe_pairs_distinguish_derived_missing_and_sample_uncertainty(self):
        expected={'mixture_known':{'future_outcome'},
                  'mixture_missing_weight':{'weight','success_probability','future_outcome'},
                  'bayes_known':{'future_outcome'},
                  'bayes_missing_prior':{'prior','posterior_probability','future_outcome'},
                  'complement_parameters':set(),'complement_future':{'future_outcome'},
                  'sampling_empirical':{'population_probability','future_outcome'},
                  'sampling_population':{'future_outcome'}}
        rows,labels=probe_rows(self.config)
        self.assertEqual(len(rows),32)
        self.assertEqual(len({r['event_group_id'] for r in labels}),4)
        for row,label in zip(rows,labels):
            self.assertEqual(set(label['unknown_fields']),expected[label['condition']])
            self.assertEqual(set(label['known_fields'])|set(label['unknown_fields']),set(label['allowed_fields']))
            self.assertEqual(set(row),{'sample_id','role','input'})
            input_context(row['input'])
            self.assertTrue(judge_uncertainty(row,label,probe_answer(row,label))['correct'])

    def test_probe_scores_do_not_ignore_extra_empty_or_free_text_unknowns(self):
        row,label=render_probe(self.config,'bayes_missing_prior','en','risk')
        good=probe_answer(row,label)
        for bad_unknowns in ([], ['The future is unknown.'], label['allowed_fields'], ['future_outcome']):
            self.assertFalse(judge_uncertainty(row,label,{**good,'unknowns':bad_unknowns})['correct'])
        judged=judge_uncertainty(row,label,{**good,'unknowns':label['allowed_fields']})
        self.assertEqual(set(judged['known_as_unknown']),{'sensitivity','false_positive_rate'})
        with self.assertRaises(ValidationError):
            judge_uncertainty(row,label,{**good,'unknowns':['prior','prior']})
        self.assertFalse(judge_uncertainty(row,label,None)['correct'])

    def test_probe_archive_rejects_rehashed_target_or_input_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for filename in ('judge.jsonl','inputs.jsonl'):
                path=root/filename.replace('.','_')
                build_uncertainty_cohort(ROOT/'configs/uncertainty_diagnostic_v1.json',path)
                read_uncertainty_cohort(path)
                rows=[json.loads(s) for s in (path/filename).read_text().splitlines()]
                if filename=='judge.jsonl':rows[0]['unknown_fields']=[]
                else:rows[0]['input']['evidence'][0]['available_at']='2099-01-01T00:00:00Z'
                (path/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                manifest=json.loads((path/'manifest.json').read_text())
                manifest['artifact_hashes'][filename]=sha256_file(path/filename)
                (path/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValidationError,'replay mismatch'):read_uncertainty_cohort(path)

    def test_protocol_configs_are_explicit_and_only_support_frozen_comparisons(self):
        self.assertEqual(load_evaluation_config(ROOT/'configs/research_tool_evaluation_v1.json')['output_protocol'],'plain_json_v1')
        c=load_evaluation_config(ROOT/'configs/research_tool_evaluation_v2.json')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'c.json'
            for changes in ({'uncertainty_protocols':['grounded_json_v2']},{'output_protocol':'arbitrary'},
                            {'partition':'test'},{'default_promotion':True}):
                path.write_text(json.dumps({**c,**changes}))
                with self.assertRaises(ValidationError):load_evaluation_config(path)
        c=load_evaluation_config(ROOT/'configs/research_tool_evaluation_v3.json')
        self.assertEqual(c['arms'],['base','previous','candidate'])

    def test_previous_checkpoint_alias_preserves_role_scope_and_original_config(self):
        before=deepcopy(self.agents)
        previous=evaluation_agents(self.agents,PREVIOUS_ADAPTER)
        plan=route_plan(previous,'reviewed_forecast','capability',capability_scope={PREVIOUS_ADAPTER})
        self.assertEqual([s['adapter'] for s in plan['steps']],[PREVIOUS_ADAPTER,PREVIOUS_ADAPTER,None,None])
        self.assertEqual(self.agents,before)
        self.assertNotIn('research_tool_lora',previous['capabilities'])
        with self.assertRaises(ValidationError):evaluation_agents(self.agents,'unknown')

    def test_uncertainty_summary_counts_failures_and_requires_every_protocol_arm(self):
        row,label=render_probe(self.config,'mixture_known','en','risk')
        call={'name':'weighted_probability','arguments':{'probabilities':[.2,.8],'weights':[.5,.5]}}
        role={'sample_id':'tool','task':'quant','request':{'agent':'quant','input':row['input']},
              'target':call,'oracle':{'expected_result':.5}}
        cohort={'judge.jsonl':[{'sample_id':'system','outcome':1,'oracle_probability':.5}]}
        common={'seconds':1,'calls':[],'memory':{'peak_allocated_bytes':1}}
        results=[]
        for arm in ('base','candidate'):
            results.extend([
                {**common,'arm':arm,'kind':'role','sample_id':'tool','output':call,'decoded_transport':'raw_json'},
                {**common,'arm':arm,'kind':'system','sample_id':'system',
                 'result':{'status':'completed','prediction':{'probability':.5},'trace':[{'attempt':0}]}}])
            for protocol in ('plain_json_v1','grounded_json_v2'):
                results.append({**common,'arm':arm+':'+protocol,'kind':'uncertainty','sample_id':row['sample_id'],
                                'output':None if arm=='candidate' else probe_answer(row,label)})
        report=summarize_capability([role],cohort,results,([row],[label]))
        for protocol in ('plain_json_v1','grounded_json_v2'):
            self.assertEqual(report['uncertainty'][protocol]['candidate']['examples'],1)
            self.assertEqual(report['uncertainty'][protocol]['candidate']['correct'],0)
            self.assertEqual(report['uncertainty'][protocol]['base']['correct'],1)
        with self.assertRaisesRegex(ValidationError,'incomplete'):
            summarize_capability([role],cohort,results[:-1],([row],[label]))
        triad=[]
        for value in results:
            value=deepcopy(value)
            if value['kind']=='uncertainty':
                if value['arm'].endswith('grounded_json_v2'):continue
                value['arm']=value['arm'].replace('plain_json_v1','grounded_json_v3')
            triad.append(value)
            if value['arm'].split(':')[0]=='candidate':
                triad.append({**deepcopy(value),'arm':value['arm'].replace('candidate','previous')})
        report=summarize_capability([role],cohort,triad,([row],[label]),arms=('base','previous','candidate'),uncertainty_protocols=('grounded_json_v3',))
        self.assertEqual(report['uncertainty']['grounded_json_v3']['previous']['correct'],0)
        self.assertEqual(report['system_common_coverage']['count'],1)


if __name__=='__main__':unittest.main()
