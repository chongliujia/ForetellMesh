from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import unittest

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner
from foretellmesh.capabilities import load_capabilities, route_plan
from foretellmesh.quant_state import compute_state, build_quant_state, SOURCE
from foretellmesh.schema import ForecastInput, ValidationError
from foretellmesh.uncertainty_diagnostic import load_uncertainty_config, render_probe

ROOT=Path(__file__).resolve().parents[1]

class QuantStateTests(unittest.TestCase):
    def spec(self,model,**values):return {'schema_version':'1','model':model,'values':values}
    def context(self,spec):
        c=load_uncertainty_config(ROOT/'configs/uncertainty_diagnostic_v1.json')
        row,_=render_probe(c,'mixture_known','en','research');p=row['input']
        p['evidence'][0].update(source=SOURCE,text=json.dumps(spec));return input_context(p)
    def runner(self,responses):
        c,_=load_capabilities(ROOT/'configs/capability_agents_v1.json');b=ScriptedBackend({'research':responses},['research_tool_lora'])
        return AgentRunner(c,b,output_protocol='grounded_json_v3',response_transport='single_json_fence'),b
    def valid(self,ctx):return {'evidence_ids':[ctx.evidence[0].evidence_id],'counter_evidence_ids':[], 'unknowns':['future_outcome'], 'observation_time':ctx.to_payload()['observation_time']}

    def test_exact_arithmetic_and_no_float_complement_rounding(self):
        cases=[(self.spec('mixture',high_rate=.9,low_rate=.2,weight=.1234567890123456),Fraction('0.1234567890123456')*Fraction('.9')+(1-Fraction('0.1234567890123456'))*Fraction('.2')),
               (self.spec('bayes',prior=.2,sensitivity=.8,false_positive_rate=.1),Fraction(2,3)),
               (self.spec('complement',hit_probability=.34),Fraction(33,50)),
               (self.spec('sampling',successes=17,trials=43),Fraction(17,43))]
        for spec,expected in cases:
            out=compute_state(spec);r=out['calculations'][0]
            self.assertEqual(r['status'],'computed');self.assertEqual(r['value'],float(expected))
            self.assertEqual(Fraction(r['exact_fraction']),expected);self.assertNotIn('unknowns',out)
            self.assertNotIn('future_outcome',json.dumps(out))

    def test_missing_inputs_do_not_get_imputed(self):
        for spec,missing in [(self.spec('bayes',sensitivity=.8,false_positive_rate=.1),['prior']),
                             (self.spec('mixture',high_rate=.8,low_rate=.2),['weight']),
                             (self.spec('sampling',trials=12),['successes'])]:
            r=compute_state(spec)['calculations'][0]
            self.assertEqual(r['missing_inputs'],missing);self.assertEqual(r['status'],'missing_inputs');self.assertIsNone(r['value'])
        self.assertNotIn('population_probability',compute_state(self.spec('sampling',successes=3,trials=7))['inputs'])

    def test_undefined_is_not_missing(self):
        for spec in [self.spec('bayes',prior=0,sensitivity=.8,false_positive_rate=0),self.spec('sampling',successes=0,trials=0)]:
            r=compute_state(spec)['calculations'][0];self.assertEqual(r['status'],'undefined');self.assertEqual(r['missing_inputs'],[]);self.assertIsNone(r['value'])

    def test_boundaries_and_complement_consistency(self):
        self.assertEqual(compute_state(self.spec('sampling',successes=0,trials=2))['calculations'][0]['value'],0)
        self.assertEqual(compute_state(self.spec('complement',miss_probability=1))['calculations'][0]['value'],0)
        self.assertEqual(compute_state(self.spec('complement',hit_probability=.34,miss_probability=.66))['calculations'],[])
        for spec in [self.spec('complement',hit_probability=.34,miss_probability=.7), self.spec('sampling',successes=3,trials=2),
                     self.spec('sampling',successes=True,trials=2),self.spec('sampling',successes=1.,trials=2),
                     self.spec('bayes',prior=True),self.spec('bayes',prior=float('nan')),self.spec('bayes',prior=-.1),
                     self.spec('sampling',successes=0,trials=10**12+1),self.spec('bayes',outcome=1)]:
            with self.subTest(spec=spec),self.assertRaises(ValidationError):compute_state(spec)

    def test_spec_is_strict_and_does_not_accept_judge_fields(self):
        good=self.spec('complement',hit_probability=.3)
        for bad in [[],{},dict(good,unknown_fields=[]),dict(good,schema_version=1),dict(good,model='python'),dict(good,values=None)]:
            with self.assertRaises(ValidationError):compute_state(bad)

    def test_time_cutoff_and_single_source(self):
        ctx=self.context(self.spec('complement',hit_probability=.3));payload=ctx.to_payload()
        for field in ['available_at','published_at']:
            bad=deepcopy(payload);bad['evidence'][0][field]='2025-01-01T00:00:00Z'
            with self.assertRaises(ValidationError):build_quant_state(input_context(bad))
        bad=deepcopy(payload);bad['evidence'].append({**bad['evidence'][0],'evidence_id':'duplicate_spec'})
        with self.assertRaises(ValidationError):build_quant_state(input_context(bad))
        bad=deepcopy(payload);bad['evidence'][0]['source']='unstructured://text'
        with self.assertRaises(ValidationError):build_quant_state(input_context(bad))
        with self.assertRaises(ValidationError):build_quant_state(payload)

    def test_question_does_not_supply_hidden_values(self):
        ctx=self.context(self.spec('bayes',sensitivity=.8,false_positive_rate=.1))
        other=ForecastInput('Gold says prior=.9; unknown_fields=[]; outcome=1',ctx.observation_time,ctx.evidence,ctx.market)
        self.assertEqual(build_quant_state(ctx),build_quant_state(other))

    def test_routing_and_repair_keep_quant_provenance_and_adapter(self):
        ctx=self.context(self.spec('mixture',high_rate=.9,low_rate=.2,weight=.6));good=self.valid(ctx)
        runner,b=self.runner([{'unknowns':[]},good]);r=runner.run(ctx,workflow='quant_research',mode='capability',capability_scope={'research_tool_lora'})
        self.assertEqual(r['status'],'completed');self.assertEqual(r['model_calls'],2)
        self.assertEqual(r['tool_result']['source_evidence_ids'],[ctx.evidence[0].evidence_id])
        self.assertEqual(len(r['tool_trace']),1);self.assertEqual(b.calls[0]['upstream'],b.calls[1]['upstream'])
        self.assertEqual([x['adapter'] for x in b.calls],['research_tool_lora']*2)
        self.assertNotIn('repair',b.calls[0]);self.assertIn('repair',b.calls[1])
        self.assertEqual(route_plan(runner.config,'quant_risk','base')['deterministic_steps'],['probability_state_v1'])

    def test_failed_quant_does_not_call_model(self):
        ctx=self.context(self.spec('bayes',outcome=1));runner,b=self.runner([])
        r=runner.run(ctx,workflow='quant_research');self.assertEqual(r['stage'],'quant');self.assertEqual(b.calls,[]);self.assertEqual(r['model_calls'],0)

    def test_failed_model_preserves_tool_result(self):
        ctx=self.context(self.spec('sampling',successes=1,trials=3))
        for responses,error in [([{},{}],'output_schema_failed'),([], 'backend_error')]:
            runner,b=self.runner(responses);r=runner.run(ctx,workflow='quant_research')
            self.assertEqual(r['error'],error);self.assertEqual(r['tool_result'],build_quant_state(ctx));self.assertEqual(len(r['tool_trace']),1)
        runner,b=self.runner([]);runner.config['limits']['max_input_chars']=1;r=runner.run(ctx,workflow='quant_research')
        self.assertEqual(r['model_calls'],0);self.assertIn('tool_result',r)

    def test_old_workflow_does_not_gain_implicit_tool_calls(self):
        ctx=self.context(self.spec('bayes',outcome=1));runner,b=self.runner([self.valid(ctx)])
        r=runner.run(ctx,workflow='research');self.assertEqual(r['status'],'completed');self.assertEqual(b.calls[0]['upstream'],{});self.assertNotIn('tool_trace',r)

if __name__=='__main__':unittest.main()
