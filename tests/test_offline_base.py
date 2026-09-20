from copy import deepcopy
from pathlib import Path
import unittest

from foretellmesh.evaluation import json_text
from foretellmesh.offline_base import load_config, replay
from foretellmesh.schema import ValidationError
import test_market_development as fixtures

ROOT=Path(__file__).resolve().parents[1]


class OfflineBaseTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.MarketDevelopmentTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.config=load_config(ROOT/'configs/offline_base_forecasts_v1.json')

    def test_replays_multiagent_base_and_refuses_adapter_or_output_tampering(self):
        f=self.fixture
        rr=[r for r in f.fake_results() if r['arm']=='multi_base']
        replay(f.inputs,rr,f.agents,self.config)
        for field in ('adapter','output'):
            changed=deepcopy(rr)
            if field=='adapter':changed[0]['calls'][0]['request']['adapter']='research_tool_lora'
            else:changed[0]['calls'][0]['output']='{}'
            with self.assertRaises(ValidationError):replay(f.inputs,changed,f.agents,self.config)
        with self.assertRaises(ValidationError):replay(f.inputs,rr[:-1],f.agents,self.config)

    def test_refuses_training_lora_and_test_partition(self):
        for key,value in [('training',True),('load_adapters',True),('mode','capability'),('partition','test')]:
            c=deepcopy(self.config);c[key]=value;path=self.fixture.root/'config.json';path.write_text(json_text(c))
            with self.assertRaises(ValidationError):load_config(path)


if __name__=='__main__':unittest.main()
