from copy import deepcopy
from pathlib import Path
import json
import unittest
from foretellmesh.team_reference_recheck import select_fresh,prepare,reference_modules
from foretellmesh.schema import ValidationError,timestamp
from foretellmesh.team_memory_loop import MemoryRunner


class FreshReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config={'initialized_from':'2024-09-01T00:00:00Z','resolution_before':'2025-05-01T00:00:00Z'}
        self.market={'market_id':'a','event_group_id':'ga','proof':{'initialized_at':'2024-09-06T00:00:00Z',
            'resolution_time':'2024-10-01T00:00:00Z','outcome':0}}

    def test_boundary_readmission_requires_new_chronology_and_no_exposure(self):
        a=self.market;groups={'ga':'purged_boundary'}
        self.assertEqual(select_fresh([a],groups,set(),set(),self.config)[0],[a])
        for used,related in [({'ga'},set()),(set(),{'ga'})]:
            self.assertFalse(select_fresh([a],groups,used,related,self.config)[0])
        for partition in ('train','development'):
            self.assertFalse(select_fresh([a],{'ga':partition},set(),set(),self.config)[0])
        old=deepcopy(a);old['proof']['initialized_at']='2024-08-30T00:00:00Z'
        self.assertFalse(select_fresh([old],groups,set(),set(),self.config)[0])
        late=deepcopy(a);late['proof']['resolution_time']=self.config['resolution_before']
        self.assertFalse(select_fresh([late],groups,set(),set(),self.config)[0])

    def test_selection_does_not_filter_outcome_or_future_price(self):
        a=deepcopy(self.market);a['proof']['outcome']=1;a['future_price']=.999;a['future_pnl']=1000
        self.assertEqual(len(select_fresh([a],{'ga':'purged_boundary'},set(),set(),self.config)[0]),1)
        self.assertEqual(self.market['proof']['outcome'],0)

    def test_actual_supplement_keeps_old_cohort_and_admits_three_fresh_groups(self):
        cfg=json.loads(Path('configs/team_reference_recheck_v1.json').read_text())
        source=Path(cfg['cohort'])/'report.json';before=source.read_bytes()
        jobs,cat,_,meta,modules=prepare(cfg)
        self.assertEqual(len(jobs),3);self.assertEqual(meta['admitted_market_ids'],['506598','506756','509106'])
        self.assertEqual(source.read_bytes(),before)
        self.assertEqual(set(meta['source_groups_unchanged'].values()),{'purged_boundary'})
        for j in jobs:
            self.assertEqual(j['partition'],'development')
            at=timestamp(j['context']['observation_time'],'at')
            for m in j['context']['markets']:
                self.assertLessEqual(timestamp(m['input']['market']['observed_at'],'quote'),at)
                self.assertLessEqual(timestamp(j['provenance'][m['market_id']]['initialized_at'],'init'),at)
                self.assertLess(at,timestamp(j['labels'][m['market_id']]['resolution_time'],'settlement'))
        self.assertIn('source_snapshot',modules['team_discovery'].__file__)
        self.assertNotEqual(modules['team_discovery'].DiscoveryRunner,MemoryRunner)

    def test_reference_budgets_cannot_be_silently_changed(self):
        cfg=json.loads(Path('configs/team_reference_recheck_v1.json').read_text());cfg['max_new_tokens']+=1
        with self.assertRaisesRegex(ValidationError,'budget changed'):prepare(cfg)
