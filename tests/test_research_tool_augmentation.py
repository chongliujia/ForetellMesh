from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.research_tool_augmentation import build_augmented_data, replay_augmented_task, validate_augmented_rows
from foretellmesh.research_tool_data import build_research_tool_data, read_research_tool_data
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import generate_synthetic_sft

ROOT=Path(__file__).resolve().parents[1]


class AugmentedBehaviorDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        c=json.loads((ROOT/'configs/synthetic_sft_generator_v1.json').read_text())
        c['groups_per_family']={'train':2,'validation':3,'test':3}
        g=cls.root/'generator.json';g.write_text(json.dumps(c));cls.raw=cls.root/'raw'
        generate_synthetic_sft(g,cls.raw)
        c=json.loads((ROOT/'configs/research_tool_dataset_v1.json').read_text())
        c['evidence_groups']={p:3 for p in ('train','validation','test')}
        cfg=cls.root/'legacy_config.json';cfg.write_text(json.dumps(c));cls.legacy=cls.root/'legacy'
        cls.splits=ROOT/'configs/synthetic_sft_splits_v1.json'
        build_research_tool_data(cls.raw,cls.splits,cfg,cls.legacy)
        c=json.loads((ROOT/'configs/research_tool_dataset_v2.json').read_text())
        for key in ('math_groups_per_family','parameter_audit_groups_per_family'):
            c[key]={p:1 for p in ('train','validation','test')}
        cfg=cls.root/'v2_config.json';cfg.write_text(json.dumps(c));cls.bundle=cls.root/'bundle'
        build_augmented_data(cls.legacy,cls.raw,cls.splits,cfg,cls.bundle)
        cls.manifest,cls.parts,cls.config=read_research_tool_data(cls.bundle)
        cls.source_splits=json.loads((cls.bundle/'source_splits.json').read_text())
        cls.times=json.loads((cls.bundle/'observation_times.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_old_tasks_and_eval_selection_survive_with_inherited_disjoint_groups(self):
        _,old,_=read_research_tool_data(self.legacy)
        self.assertEqual((self.bundle/'evaluation_ids.json').read_bytes(),(self.legacy/'evaluation_ids.json').read_bytes())
        owners={}
        for part,rows in self.parts.items():
            lookup={r['sample_id']:r for r in rows}
            for before in old[part]:
                after=lookup[before['sample_id']]
                self.assertEqual(before['request']['input'],after['request']['input'])
                self.assertEqual(before['target'],after['target'])
            for row in rows:
                group=row['event_group_id']
                self.assertEqual(owners.setdefault(group,part),part)
                self.assertEqual(self.source_splits[group],part)

    def test_unknown_targets_distinguish_realization_from_calculation_inputs(self):
        math=[r for r in self.parts['train'] if r['case']['kind']=='math_role']
        self.assertEqual(len(math),48)
        for row in math:
            self.assertEqual(replay_augmented_task(row['case'],'train'),row)
            self.assertEqual(set(row['request']),{'agent','adapter','instruction','input','upstream'})
            self.assertNotIn('label',row['request']['input'])
            if row['case']['view']=='parameters':self.assertEqual(row['target']['unknowns'],[])
            elif row['case']['source_case']['family']=='beta_binomial':self.assertEqual(len(row['target']['unknowns']),2)
            else:self.assertEqual(len(row['target']['unknowns']),1)

    def test_changed_targets_partitions_or_missing_role_variants_are_rejected(self):
        for kind in ('target','split','variant'):
            parts=deepcopy(self.parts);source=deepcopy(self.source_splits)
            row=next(r for r in parts['train'] if r['case']['kind']=='math_role' and r['case']['view']=='forecast')
            if kind=='target':row['target']['unknowns']=['The supplied prior is unknown.']
            elif kind=='split':source[row['event_group_id']]='test'
            else:parts['train'].remove(row)
            with self.assertRaises(ValidationError):validate_augmented_rows(parts,source,self.times)

    def test_reader_rejects_rehashed_semantically_incorrect_target(self):
        import shutil
        copy=self.root/'tampered';shutil.copytree(self.bundle,copy)
        rows=[json.loads(s) for s in (copy/'train.jsonl').read_text().splitlines()]
        row=next(r for r in rows if r['case']['kind']=='math_role')
        row['target']['unknowns']=['Invented missing input']
        (copy/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=json.loads((copy/'manifest.json').read_text())
        manifest['artifact_hashes']['train.jsonl']=sha256_file(copy/'train.jsonl')
        (copy/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError,'replay mismatch'):read_research_tool_data(copy)

    def test_new_diagnostic_cannot_be_used_as_a_math_training_case(self):
        with self.assertRaises(ValidationError):
            replay_augmented_task({'kind':'uncertainty','condition':'bayes_missing_prior'},'train')


if __name__=='__main__':unittest.main()
