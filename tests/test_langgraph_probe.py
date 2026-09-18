from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.langgraph_probe import archived_jobs
from foretellmesh.schema import ValidationError
from foretellmesh.sft_data import jsonl

class GraphProbeSelectionTests(unittest.TestCase):
    def write(self,path,rows):
        path.mkdir();(path/'results.jsonl').write_text(jsonl(rows))
        (path/'report.json').write_text(json_text({'status':'completed','results_sha256':sha256_file(path/'results.jsonl'),
                                                'config':{'system_workflow':'reviewed_forecast'}}))

    def test_selection_never_skips_failed_first_job_and_checks_hashes(self):
        calls=[{'request':{'input':{'question':'synthetic'},'agent':'research'},'raw_output':'invalid'}]
        tool=[];system=[]
        for arm in ('base','candidate'):
            for role in ('research','risk'):
                row={'arm':arm,'role':role,'variant':'tool','sample_id':'first:'+role,'workflow':'quant_'+role,
                     'input':{'question':'synthetic'},'calls':deepcopy(calls),'result':{'status':'failed','trace':[]}}
                tool.extend([row,{**deepcopy(row),'sample_id':'later_success','result':{'status':'completed','trace':[]}}])
            system.append({'arm':arm,'kind':'system','sample_id':'system','calls':[{'request':calls[0]['request'],'output':'invalid'}],
                           'result':{'status':'failed','trace':[]}})
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'tool',Path(tmp)/'system';self.write(a,tool);self.write(b,system)
            jobs=archived_jobs(a,b);self.assertEqual(len(jobs),6)
            self.assertTrue(all(j['expected_result']['status']=='failed' for j in jobs))
            self.assertNotIn('later_success',[j['sample_id'] for j in jobs])
            (a/'results.jsonl').write_text(jsonl(tool[:-1]))
            with self.assertRaises(ValidationError):archived_jobs(a,b)

    def test_partial_source_run_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'tool';self.write(a,[])
            (a/'report.json').write_text(json_text({'status':'running'}))
            with self.assertRaises(ValidationError):archived_jobs(a,Path(tmp)/'system')

if __name__=='__main__':unittest.main()
