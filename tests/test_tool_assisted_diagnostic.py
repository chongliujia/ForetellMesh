from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.capabilities import load_capabilities
from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.schema import ValidationError
from foretellmesh.tool_assisted_diagnostic import load_config, numeric_configs, rows_for_configs, VARIANTS, specification, fingerprint, build, read
from foretellmesh.tool_assisted_evaluation import make_jobs, load_config as eval_config, run_job, summarize, stable_result
from foretellmesh.uncertainty_diagnostic import CONDITIONS

ROOT=Path(__file__).resolve().parents[1]

class ToolAssistedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=load_config(ROOT/'configs/tool_assisted_diagnostic_v1.json');cls.configs=numeric_configs(cls.c,{'signatures':[]})
        cls.inputs,cls.judges=rows_for_configs(cls.configs)
        cls.e=eval_config(ROOT/'configs/tool_assisted_evaluation_v1.json');cls.agents,_=load_capabilities(ROOT/'configs/capability_agents_v1.json')
        cls.jobs=make_jobs(cls.inputs,cls.e,cls.agents)

    def records(self):
        labels={j['sample_id']:j for j in self.judges};rows=[]
        for job in self.jobs:
            output={'unknowns':labels[job['sample_id']]['unknown_fields'],'observation_time':job['input']['observation_time']}
            if job['role']=='research':output.update(evidence_ids=[job['input']['evidence'][0]['evidence_id']],counter_evidence_ids=[])
            else:output['risks']=[]
            b=ScriptedBackend({job['role']:[output]},['research_tool_lora']);result=run_job(job,self.agents,b,self.e)
            rows.append({**job,'calls':[{'request':b.calls[0],'raw_output':json.dumps(output),'usage':{'seconds':1.,'input_tokens':10,'output_tokens':10,'output_reached_token_limit':False}}],
                         'result':result,'seconds':1.,'memory':{'peak_allocated_bytes':100}})
        return rows

    def test_matched_inputs_and_independent_group_counts(self):
        self.assertEqual(len(self.inputs),64);self.assertEqual(len(self.jobs),384);self.assertEqual(len({j['event_group_id'] for j in self.judges}),8)
        for row in self.inputs:
            a,b,t=(row['variants'][k] for k in VARIANTS)
            self.assertEqual(b,t);self.assertEqual(a['question'],b['question']);self.assertEqual(a['evidence'],b['evidence'][:-1])
        for job in self.jobs:
            self.assertEqual(job['request']['input'],job['input'])
            self.assertEqual(set(job['request']['upstream']),{'quant'} if job['variant']=='tool' else set())
            self.assertEqual(job['request']['adapter'],None if job['arm']=='base' else 'research_tool_lora')

    def test_runtime_requests_hide_targets_and_missing_values(self):
        labels={j['sample_id']:j for j in self.judges}
        for job in self.jobs:
            text=json.dumps(job['request'])
            for name in ('unknown_fields','known_fields','outcome','case','target','event_group_id'):
                self.assertNotIn('"'+name+'":',text)
            if job['variant']=='prose':continue
            spec=json.loads(job['input']['evidence'][-1]['text']);condition=labels[job['sample_id']]['condition']
            for condition_name,key in [('bayes_missing_prior','prior'),('mixture_missing_weight','weight'),('sampling_empirical','population_probability')]:
                if condition==condition_name:self.assertNotIn(key,spec['values'])
            if job['variant']=='tool':self.assertNotIn('unknowns',job['request']['upstream']['quant'])

    def test_exclusions_cover_visible_inputs_even_when_hidden_parameter_differs(self):
        spec=specification(self.configs[0],'mixture_missing_weight');excluded={'signatures':[fingerprint(spec['model'],spec['values'])]}
        revised=numeric_configs(self.c,excluded)
        self.assertNotEqual(specification(revised[0],'mixture_missing_weight'),spec)
        self.assertEqual(numeric_configs(self.c,excluded),revised)
        seen=set()
        for c in revised:
            for family in ('mixture','bayes','complement','sampling'):
                sigs={fingerprint(s['model'],s['values']) for s in [specification(c,cond) for cond in CONDITIONS if cond.startswith(family+'_')]}
                self.assertFalse(seen&sigs);self.assertFalse(set(excluded['signatures'])&sigs);seen.update(sigs)

    def test_score_pairs_and_first_response_are_not_repaired_scores(self):
        records=self.records();m=summarize(self.inputs,self.judges,records,self.e,self.agents)
        self.assertTrue(all(v['first']['correct']==64 for v in m['cells'].values()))
        r=next(r for r in records if r['arm']=='candidate' and r['variant']=='structured')
        r['calls'].insert(0,{**deepcopy(r['calls'][0]),'raw_output':'invalid'})
        m=summarize(self.inputs,self.judges,records,self.e,self.agents)
        self.assertEqual(m['cells']['candidate:structured']['first']['correct'],63)
        self.assertEqual(m['cells']['candidate:structured']['final']['correct'],64)
        self.assertEqual(m['matched_contrasts']['candidate:structured->tool']['first']['improved'],1)
        self.assertEqual(m['matched_contrasts']['candidate:prose->structured']['first']['regressed'],1)
        self.assertEqual(m['cells']['candidate:structured']['resources']['repair_calls'],1)
        for bad in [records[:-1],records+[records[0]]]:
            with self.assertRaises(ValidationError):summarize(self.inputs,self.judges,bad,self.e,self.agents)

    def test_raw_replay_preserves_actual_schema_repair(self):
        job=next(j for j in self.jobs if j['variant']=='tool');role=job['role']
        output={'unknowns':['future_outcome'],'observation_time':job['input']['observation_time'],'evidence_ids':[],'counter_evidence_ids':[]}
        a=ScriptedBackend({role:['bad',output]},['research_tool_lora']);b=ScriptedBackend({role:['bad',output]},['research_tool_lora'])
        x=run_job(job,self.agents,a,self.e);y=run_job(job,self.agents,b,self.e)
        self.assertEqual(stable_result(x),stable_result(y));self.assertEqual(a.calls,b.calls);self.assertEqual(x['model_calls'],2)
        self.assertEqual(a.calls[0],job['request']);self.assertEqual(a.calls[0]['upstream'],a.calls[1]['upstream'])

    def test_rebuild_rejects_rehashed_label_and_hidden_input_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'cohort';build(ROOT/'configs/tool_assisted_diagnostic_v1.json',output);read(output)
            manifest=json.loads((output/'manifest.json').read_text())
            for name,mutate in [('judge.jsonl',lambda r:r.update(unknown_fields=[])),
                                ('inputs.jsonl',lambda r:r['variants']['tool']['evidence'][-1].update(text='{}'))]:
                before=(output/name).read_text();rows=[json.loads(s) for s in before.splitlines()];mutate(rows[0])
                (output/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                changed=deepcopy(manifest);changed['artifact_hashes'][name]=sha256_file(output/name);(output/'manifest.json').write_text(json_text(changed))
                with self.assertRaisesRegex(ValidationError,'replay'):read(output)
                (output/name).write_text(before);(output/'manifest.json').write_text(json_text(manifest))

if __name__=='__main__':unittest.main()
