from collections import defaultdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.paired_diagnostic import (
    FACTORS, FORMULAS, build_paired_cohort, load_paired_config, paired_rows, read_paired_cohort, summarize_paired,
)
from foretellmesh.paired_evaluation import decode, load_config, make_jobs, verify_excluded_source
from foretellmesh.schema import ValidationError
from foretellmesh.uncertainty_diagnostic import build_uncertainty_cohort, load_uncertainty_config, probe_rows

ROOT = Path(__file__).resolve().parents[1]


class PairedDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_config = load_uncertainty_config(ROOT/'configs/uncertainty_diagnostic_v1.json')
        cls.original, cls.labels = probe_rows(cls.source_config)
        cls.inputs, cls.judges = paired_rows(cls.original, cls.labels)
        cls.config = load_config(ROOT/'configs/paired_information_evaluation_v1.json')

    def outputs(self):
        outputs = []
        for arm in ('base', 'previous', 'candidate'):
            for row, judge in zip(self.inputs, self.judges):
                result = {'unknowns': judge['unknown_fields'], 'observation_time': row['input']['observation_time']}
                if row['role'] == 'research':result.update(evidence_ids=[row['input']['evidence'][0]['evidence_id']], counter_evidence_ids=[])
                else:result['risks'] = []
                outputs.append({'arm': arm, 'sample_id': row['sample_id'], 'output': result})
        return outputs

    def test_all_factorial_cells_keep_source_answers_and_exact_original_anchor(self):
        originals = {r['sample_id']: r for r in self.original}
        labels = {r['sample_id']: r for r in self.labels}
        parents = defaultdict(list)
        self.assertEqual(len(self.inputs), 256)
        self.assertEqual(len({j['event_group_id'] for j in self.judges}), 4)
        for row, judge in zip(self.inputs, self.judges):
            parent = judge['parent_sample_id']; source = originals[parent]
            parents[parent].append(judge['factors'])
            for key in ('known_fields', 'unknown_fields', 'allowed_fields', 'event_group_id', 'role', 'language', 'condition'):
                self.assertEqual(judge[key], labels[parent][key])
            self.assertEqual(row['input']['observation_time'], source['input']['observation_time'])
            self.assertIsNone(row['input']['market'])
            evidence = deepcopy(row['input']['evidence'])
            if judge['factors']['formula'] == 'explicit':
                suffix = ' ' + FORMULAS[judge['family']][judge['language'] == 'zh']
                self.assertTrue(evidence[0]['text'].endswith(suffix))
                evidence[0]['text'] = evidence[0]['text'][:-len(suffix)]
            self.assertEqual(evidence, source['input']['evidence'])
            if tuple(judge['factors'].values()) == ('original', 'implicit', 'inline'):
                self.assertEqual(row['input'], source['input'])
        for variants in parents.values():
            self.assertEqual({tuple(v[k] for k in FACTORS) for v in variants}, set(product(*FACTORS.values())))

    def test_each_factor_changes_only_its_component(self):
        by_parent = defaultdict(dict)
        for row, judge in zip(self.inputs, self.judges):
            by_parent[judge['parent_sample_id']][tuple(judge['factors'].values())] = row['input']
        for rows in by_parent.values():
            for wording, presentation in product(FACTORS['wording'], FACTORS['presentation']):
                a, b = rows[wording, 'implicit', presentation], rows[wording, 'explicit', presentation]
                self.assertEqual(a['question'], b['question'])
                self.assertNotEqual(a['evidence'][0]['text'], b['evidence'][0]['text'])
            for formula in FACTORS['formula']:
                evidence = [r['evidence'] for (w, f, p), r in rows.items() if f == formula]
                self.assertTrue(all(e == evidence[0] for e in evidence))

    def test_missing_values_and_answer_annotations_never_enter_requests(self):
        jobs = make_jobs(self.inputs, self.config)
        self.assertEqual(len(jobs), 768)
        self.assertEqual({j['request']['adapter'] for j in jobs}, {None, 'research_tool_lora', 'research_tool_previous_lora'})
        judges = {j['sample_id']: j for j in self.judges}
        for job in jobs:
            req = job['request']; judge = judges[job['sample_id']]
            self.assertEqual(set(req), {'agent', 'adapter', 'instruction', 'input', 'upstream'})
            self.assertEqual(req['upstream'], {})
            serialized = json.dumps(req)
            for field in ('unknown_fields', 'known_fields', 'parent_sample_id', 'outcome', 'factors'):
                self.assertNotIn('"' + field + '"', serialized)
            if judge['condition'] == 'bayes_missing_prior':self.assertNotIn('0.27', serialized)
            if judge['condition'] == 'mixture_missing_weight':self.assertNotIn('0.63', serialized)

    def test_replay_refuses_rehashed_target_or_time_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; output = root/'paired'
            build_uncertainty_cohort(ROOT/'configs/uncertainty_diagnostic_v1.json', source)
            build_paired_cohort(source, ROOT/'configs/paired_information_diagnostic_v1.json', output)
            _, inputs, judges = read_paired_cohort(output)
            self.assertEqual((inputs, judges), (self.inputs, self.judges))
            for name, mutate in [('judge.jsonl', lambda r: r.update(unknown_fields=[])),
                                 ('inputs.jsonl', lambda r: r['input']['evidence'][0].update(available_at='2025-01-01T00:00:00Z'))]:
                path = output/name; before = path.read_text(); manifest = json.loads((output/'manifest.json').read_text())
                rows = [json.loads(s) for s in before.splitlines()]; mutate(rows[0])
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                changed = deepcopy(manifest); changed['artifact_hashes'][name] = sha256_file(path)
                (output/'manifest.json').write_text(json_text(changed))
                with self.assertRaisesRegex(ValidationError, 'replay'):read_paired_cohort(output)
                path.write_text(before); (output/'manifest.json').write_text(json_text(manifest))

    def test_exact_gold_and_matched_discordant_pair_counts(self):
        outputs = self.outputs()
        result = summarize_paired(self.inputs, self.judges, outputs)
        arm = result['arms']['candidate']
        self.assertEqual(arm['correct'], 256); self.assertEqual(arm['all_variants_correct'], 32)
        for factor in FACTORS:
            self.assertEqual(arm['factors'][factor]['pairs'], 128)
            self.assertEqual(arm['factors'][factor]['both_correct'], 128)
        # A single invalid original/implicit/inline answer improves on all three adjacent edges.
        outputs[512]['output'] = None
        result = summarize_paired(self.inputs, self.judges, outputs)['arms']['candidate']
        self.assertEqual(result['correct'], 255); self.assertEqual(result['all_variants_correct'], 31)
        for factor in FACTORS:
            self.assertEqual(result['factors'][factor]['improved'], 1)
            self.assertEqual(result['factors'][factor]['regressed'], 0)
        self.assertEqual(result['cells']['original:implicit:inline']['correct'], 31)

    def test_unknown_accuracy_and_role_grounding_are_separate(self):
        outputs = self.outputs(); outputs[0]['output']['evidence_ids'] = []
        result = summarize_paired(self.inputs, self.judges, outputs)['arms']['base']
        self.assertEqual(result['correct'], 256)
        self.assertEqual(result['role_contract_correct'], 255)

    def test_scoring_rejects_missing_duplicate_or_changed_pairs(self):
        outputs = self.outputs()
        for bad in (outputs[:-1], outputs + [outputs[0]]):
            with self.assertRaises(ValidationError):summarize_paired(self.inputs, self.judges, bad)
        labels = deepcopy(self.judges); labels[1]['unknown_fields'] = []
        with self.assertRaisesRegex(ValidationError, 'answers'):summarize_paired(self.inputs, labels, outputs)
        with self.assertRaisesRegex(ValidationError, 'factor grid'):
            summarize_paired(self.inputs[:-1], self.judges[:-1], outputs)

    def test_decode_rejects_additional_fields_without_repair(self):
        job = make_jobs(self.inputs, self.config)[0]
        output = self.outputs()[0]['output']
        good = decode(json.dumps(output), job['request'], self.config, 10000)
        self.assertEqual(good['output'], output)
        output['posterior_probability'] = 0.5
        bad = decode(json.dumps(output), job['request'], self.config, 10000)
        self.assertIsNone(bad['output']); self.assertIn('unknown fields', bad['error'])
        self.assertIsNone(decode(json.dumps(output), job['request'], self.config, 1)['output'])

    def test_configs_reject_training_or_adaptive_selection(self):
        c = load_paired_config(ROOT/'configs/paired_information_diagnostic_v1.json')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'config.json'; c['training_allowed'] = True; p.write_text(json.dumps(c))
            with self.assertRaises(ValidationError):load_paired_config(p)
            p.write_text(json.dumps({**self.config, 'retries': 1}))
            with self.assertRaises(ValidationError):load_config(p)

    def test_exclusion_identity_uses_source_hash_and_semantic_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root/'bundle'; cohort = root/'cohort'
            bundle.mkdir(); (cohort/'source').mkdir(parents=True)
            source = {'source_manifest_sha256': 'fixed-source'}
            (bundle/'manifest.json').write_text(json.dumps({'excluded_diagnostic_manifest_sha256': 'fixed-source'}))
            (bundle/'excluded_diagnostic_config.json').write_text(json.dumps(self.source_config, sort_keys=True, indent=2))
            (cohort/'source/config.json').write_text(json.dumps(self.source_config))
            verify_excluded_source(source, cohort, bundle)
            (cohort/'source/config.json').write_text(json.dumps({**self.source_config, 'mixture': {}}))
            with self.assertRaises(ValidationError):verify_excluded_source(source, cohort, bundle)
            with self.assertRaises(ValidationError):verify_excluded_source({'source_manifest_sha256': 'changed'}, cohort, bundle)


if __name__ == '__main__':unittest.main()
