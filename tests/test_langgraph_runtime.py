from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.agent_baseline_data import input_context
from foretellmesh.agent_check import ScriptedBackend, check_agent_workflow
from foretellmesh.agent_runtime import AgentRunner
from foretellmesh.capabilities import load_capabilities, WORKFLOWS
from foretellmesh.langgraph_runtime import LangGraphRunner
from foretellmesh.quant_state import SOURCE
from foretellmesh.schema import ForecastInput, ValidationError
from foretellmesh.tool_assisted_evaluation import stable_result

ROOT=Path(__file__).resolve().parents[1]
AVAILABLE=importlib.util.find_spec('langgraph') is not None

@unittest.skipUnless(AVAILABLE,'optional foretellmesh[graph] is not installed')
class LangGraphRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config,_=load_capabilities(ROOT/'configs/capability_agents_v1.json')
        self.fixture=json.loads((ROOT/'examples/agent_workflow_fixture_v1.json').read_text())
        self.context=input_context(self.fixture['input'])
        self.responses=self.fixture['responses']
        self.responses['quant']=[{'name':'weighted_probability','arguments':{'probabilities':[.5],'weights':[1.]}}]

    def runner(self,cls=LangGraphRunner,responses=None,config=None):
        b=ScriptedBackend(self.responses if responses is None else responses,self.config['capabilities'])
        return cls(config or self.config,b,output_protocol='grounded_json_v3',response_transport='single_json_fence'),b

    def tool_context(self):
        p=self.context.to_payload();e=deepcopy(p['evidence'][0]);e.update(evidence_id='model',source=SOURCE,
            text=json.dumps({'schema_version':'1','model':'sampling','values':{'successes':2,'trials':5}}));p['evidence'].append(e)
        return input_context(p)

    def test_all_routes_match_requests_outputs_and_adapter_choices(self):
        for workflow in WORKFLOWS:
            for mode,scope in [('base',None),('capability',None),('capability',{'research_tool_lora'})]:
                ctx=self.tool_context() if workflow.startswith('quant_') else self.context
                a,ab=self.runner(AgentRunner);b,bb=self.runner();kwargs=dict(workflow=workflow,mode=mode,capability_scope=scope)
                with self.subTest(workflow=workflow,mode=mode,scope=scope):
                    x=a.run(ctx,**kwargs);y=b.run(ctx,**kwargs)
                    self.assertEqual(stable_result(x),stable_result(y));self.assertEqual(ab.calls,bb.calls)
                    self.assertEqual(y['status'],'completed');self.assertEqual(y['model_calls'],len(WORKFLOWS[workflow]))
                    for req in bb.calls:
                        expected=None if mode=='base' or scope and self.config['agents'][req['agent']] not in scope else self.config['agents'][req['agent']]
                        self.assertEqual(req['adapter'],expected)

    def test_stream_runs_distinct_role_nodes_in_order(self):
        runner,b=self.runner();g=runner.compile(workflow='reviewed_forecast')
        updates=list(g.stream({'context':self.context},stream_mode='updates'))
        self.assertEqual([next(iter(u)) for u in updates],['prepare','research','risk','forecast','critic','finalize'])
        self.assertEqual([r['agent'] for r in b.calls],['research','risk','forecast','critic'])
        self.assertEqual(set(updates[-1]['finalize']),{'result'})
        self.assertNotIn('messages',json.dumps(updates,default=str))

    def test_repair_stays_within_role_and_failure_skips_downstream(self):
        responses=deepcopy(self.responses);responses['research']=[{},{}]
        runner,b=self.runner(responses=responses);g=runner.compile(workflow='reviewed_forecast')
        updates=list(g.stream({'context':self.context},stream_mode='updates'))
        self.assertEqual([next(iter(u)) for u in updates],['prepare','research','finalize'])
        r=updates[-1]['finalize']['result'];self.assertEqual(r['error'],'output_schema_failed');self.assertEqual(len(b.calls),2)
        self.assertIsNone(r['prediction']);self.assertIn('repair',b.calls[1])
        responses['research']=[{},self.responses['research'][0]]
        runner,b=self.runner(responses=responses);r=runner.run(self.context,workflow='reviewed_forecast')
        self.assertEqual(r['status'],'completed');self.assertEqual(r['model_calls'],5)
        self.assertEqual([t['attempt'] for t in r['trace']],[0,1,0,0,0])

    def test_backend_failure_is_not_retried_by_graph(self):
        runner,b=self.runner(responses={});r=runner.run(self.context,workflow='reviewed_forecast')
        self.assertEqual(r['error'],'backend_error');self.assertEqual(len(b.calls),1);self.assertIsNone(r['prediction'])

    def test_invalid_tool_source_stops_before_llm(self):
        runner,b=self.runner();g=runner.compile(workflow='quant_research')
        updates=list(g.stream({'context':self.context},stream_mode='updates'))
        self.assertEqual([next(iter(u)) for u in updates],['prepare','quant_tools','finalize'])
        r=updates[-1]['finalize']['result'];self.assertEqual(r['error'],'invalid_tool_evidence');self.assertEqual(b.calls,[])
        self.assertEqual(r['tool_trace'][0]['status'],'failed')

    def test_time_and_label_validation_happen_before_nodes_call_model(self):
        runner,b=self.runner();g=runner.compile(workflow='reviewed_forecast')
        for context in [self.context.to_payload(),{'outcome':1}]:
            with self.assertRaises(ValidationError):g.invoke({'context':context})
        from dataclasses import replace
        from datetime import timedelta
        e=replace(self.context.evidence[0],available_at=self.context.observation_time+timedelta(days=1))
        with self.assertRaises(ValidationError):g.invoke({'context':ForecastInput(self.context.question,self.context.observation_time,(e,),None)})
        self.assertEqual(b.calls,[])

    def test_missing_adapter_and_overbudget_fail_before_model(self):
        runner,b=self.runner();b.available_adapters=set()
        with self.assertRaises(ValidationError):runner.run(self.context,mode='capability')
        self.assertEqual(b.calls,[])
        config=deepcopy(self.config);config['limits']['max_model_calls']=3
        runner,b=self.runner(config=config)
        with self.assertRaises(ValidationError):runner.run(self.context,workflow='reviewed_forecast')
        self.assertEqual(b.calls,[])

    def test_context_selection_and_critic_revision_use_original_evidence(self):
        p=self.context.to_payload();p['evidence'].append({**p['evidence'][0],'evidence_id':'counter','text':'Additional synthetic counterevidence.'})
        ctx=input_context(p);responses=deepcopy(self.responses)
        responses['risk'][0]['risks']=[{'scenario':'Counter scenario','evidence_ids':['counter']}]
        responses['critic'][0].update(accept=False,revised_probability=.4,evidence_ids=['counter'],unknowns=['Unverified assumption'])
        runner,b=self.runner(responses=responses);r=runner.run(ctx,workflow='reviewed_forecast')
        self.assertEqual([e['evidence_id'] for e in b.calls[2]['input']['evidence']],['setup','counter'])
        self.assertEqual(b.calls[1]['input'],ctx.to_payload());self.assertEqual(b.calls[3]['input'],ctx.to_payload())
        self.assertEqual(set(b.calls[2]['upstream']),{'research','risk'});self.assertEqual(set(b.calls[3]['upstream']),{'forecast','risk'})
        self.assertEqual(r['prediction']['probability'],.4);self.assertEqual(r['stages']['forecast']['probability'],.5)

    def test_compiled_config_and_invocations_are_isolated(self):
        responses=deepcopy(self.responses);responses['research']*=2
        runner,b=self.runner(responses=responses);scope={'research_tool_lora'}
        g=runner.compile(workflow='research',mode='capability',capability_scope=scope)
        runner.config['limits']['max_model_calls']=0;scope.clear()
        a=g.invoke({'context':self.context})['result'];a['stages']['research']['unknowns'].append('pollution')
        result=g.invoke({'context':self.context})['result']
        self.assertEqual(result['model_calls'],1);self.assertNotIn('pollution',result['stages']['research']['unknowns'])

    def test_cli_check_records_actual_graph_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=check_agent_workflow(ROOT/'configs/capability_agents_v1.json',ROOT/'examples/agent_workflow_fixture_v1.json',Path(tmp)/'check',engine='langgraph')
            self.assertEqual(report['result']['status'],'completed');self.assertEqual(report['orchestration']['engine'],'langgraph')
            self.assertTrue(report['orchestration']['langgraph_version']);self.assertIsNone(report['forecasting_metrics'])

class OptionalGraphTests(unittest.TestCase):
    def test_missing_optional_library_has_actionable_error(self):
        config,_=load_capabilities(ROOT/'configs/capability_agents_v1.json');runner=LangGraphRunner(config,ScriptedBackend({}))
        with patch.dict('sys.modules',{'langgraph.graph':None}):
            with self.assertRaisesRegex(RuntimeError,r'foretellmesh\[graph\]'):runner.compile()

if __name__=='__main__':unittest.main()
