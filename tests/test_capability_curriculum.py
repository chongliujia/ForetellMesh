from collections import Counter
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import random
import tempfile
import unittest

from foretellmesh.capability_curriculum import (
    EVALUATION_POLICY, TRAIN_POLICY, judge_information, select_information_validation,
    select_training_rows, summarize_information,
)
from foretellmesh.capability_evaluation import load_evaluation_config, summarize_capability
from foretellmesh.capability_training import load_training_config
from foretellmesh.research_tool_counterfactual import CONDITIONS, render_counterfactual, sample_parameters
from foretellmesh.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
POLICY = {'policy': TRAIN_POLICY, 'selection_seed': 20260922, 'groups_per_family': 8}


def rows_for(partition, count, seed):
    rng = random.Random(seed)
    rows = []
    for family in CONDITIONS:
        for _ in range(count):
            params = sample_parameters(rng, family)
            for condition, scope, fmt, lang, role in product(CONDITIONS[family], ('calculation', 'forecast'),
                    ('field_names', 'aliases'), ('en', 'zh'), ('research', 'risk')):
                case = {'kind': 'information_state', 'family': family, 'parameters': params, 'condition': condition,
                        'scope': scope, 'format': fmt, 'language': lang, 'role': role,
                        'observation_time': '2024-01-15T00:00:00Z' if partition == 'train' else '2024-02-15T00:00:00Z'}
                rows.append(render_counterfactual(case, partition))
    return rows


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = rows_for('train', 8, 21)
        cls.old = [{'sample_id': 'old:' + str(i), 'partition': 'train', 'case': {'kind': 'legacy'},
                    'event_group_id': 'old', 'task': 'quant'} for i in range(4)]
        cls.train += cls.old
        cls.validation = rows_for('validation', 2, 22)

    def test_fixed_training_budget_keeps_every_old_row_and_every_new_group(self):
        selected, report = select_training_rows(self.train, POLICY)
        self.assertEqual(len(selected), 260)
        self.assertEqual(report['information_examples'], 256)
        self.assertEqual(report['legacy_examples'], 4)
        for row in self.old:self.assertIn(row, selected)
        new = [r for r in selected if r['case']['kind'] == 'information_state']
        self.assertEqual(set(Counter(r['event_group_id'] for r in new).values()), {8})
        for family in CONDITIONS:
            members = [r for r in new if r['case']['family'] == family]
            for key in ('role', 'language', 'scope', 'format'):
                self.assertEqual(set(Counter(r['case'][key] for r in members).values()), {32})
            self.assertEqual(set(Counter(r['case']['condition'] for r in members).values()), {16})
        self.assertEqual(select_training_rows(self.train, POLICY), select_training_rows(list(reversed(self.train)), POLICY))

    def test_selection_does_not_inspect_targets_and_default_preserves_original_order(self):
        changed = deepcopy(self.train)
        for row in changed:row['target'] = None; row['oracle'] = None
        _, before = select_training_rows(self.train, POLICY)
        _, after = select_training_rows(changed, POLICY)
        self.assertEqual(before, after)
        selected, report = select_training_rows(self.train)
        self.assertIs(selected, self.train)
        self.assertIsNone(report)

    def test_train_selector_refuses_heldout_rows_missing_variants_and_bad_policy(self):
        for bad in (self.validation, self.train[:-1] + [self.train[0]], self.train[1:]):
            with self.assertRaises(ValidationError):select_training_rows(bad, POLICY)
        for change in ({'groups_per_family': 7}, {'selection_seed': True}, {'policy': 'weighted_by_loss'}):
            with self.assertRaises(ValidationError):select_training_rows(self.train, {**POLICY, **change})

    def test_frozen_validation_is_balanced_disjoint_and_keeps_all_disclosure_scope_pairs(self):
        selected = select_information_validation(self.validation, EVALUATION_POLICY)
        self.assertEqual(len(selected), 128)
        self.assertEqual(len({r['event_group_id'] for r in selected}), 8)
        self.assertFalse({r['event_group_id'] for r in selected} & {r['event_group_id'] for r in self.train})
        for key in ('role', 'language', 'scope', 'format'):
            self.assertEqual(set(Counter(r['case'][key] for r in selected).values()), {64})
        self.assertEqual(selected, select_information_validation(list(reversed(self.validation)), EVALUATION_POLICY))
        report = summarize_information(selected, {r['sample_id']: r['target'] for r in selected})
        self.assertEqual(report['task_correct'], 128)
        self.assertEqual(report['scope_pairs'], {'total': 64, 'all_correct': 64})
        self.assertEqual(report['disclosure_sets'], {'total': 32, 'all_correct': 32})

    def test_judge_counts_empty_free_text_extra_unknowns_and_bad_evidence_as_failures(self):
        selected = select_information_validation(self.validation, EVALUATION_POLICY)
        row = next(r for r in selected if r['request']['agent'] == 'research' and r['case']['condition'] == 'omit_prior'
                   and r['case']['scope'] == 'forecast')
        good = row['target']
        for bad in ({**good, 'unknowns': []}, {**good, 'unknowns': ['Not yet known.']},
                    {**good, 'unknowns': list(row['oracle']['output_names'].values())},
                    {**good, 'evidence_ids': [], 'counter_evidence_ids': good['evidence_ids']}, None):
            self.assertFalse(judge_information(row, bad)['task_correct'])
        reversed_answer = {**good, 'unknowns': list(reversed(good['unknowns']))}
        self.assertTrue(judge_information(row, reversed_answer)['task_correct'])
        with self.assertRaises(ValidationError):judge_information(row, {**good, 'unknowns': [good['unknowns'][0]] * 2})
        outputs = {r['sample_id']: r['target'] for r in selected}; outputs[row['sample_id']] = None
        report = summarize_information(selected, outputs)
        self.assertEqual(report['task_correct'], 127)
        self.assertEqual(report['scope_pairs']['all_correct'], 63)
        self.assertEqual(report['disclosure_sets']['all_correct'], 31)
        with self.assertRaises(ValidationError):summarize_information(selected, {})

    def test_every_information_arm_is_required_in_integrated_evaluation(self):
        info = select_information_validation(self.validation, EVALUATION_POLICY)
        call = {'name': 'weighted_probability', 'arguments': {'probabilities': [.2, .8], 'weights': [.5, .5]}}
        role = {'sample_id': 'tool', 'task': 'quant', 'request': {'agent': 'quant', 'input': info[0]['request']['input']},
                'target': call, 'oracle': {'expected_result': .5}}
        cohort = {'judge.jsonl': [{'sample_id': 'system', 'outcome': 1, 'oracle_probability': .5}]}
        common = {'seconds': 1, 'calls': [], 'memory': {'peak_allocated_bytes': 1}}
        results = []
        for arm in ('base', 'previous', 'candidate'):
            results.extend([{**common, 'arm': arm, 'kind': 'role', 'sample_id': 'tool', 'output': call, 'decoded_transport': 'raw_json'},
                {**common, 'arm': arm, 'kind': 'system', 'sample_id': 'system',
                 'result': {'status': 'completed', 'prediction': {'probability': .5}, 'trace': [{'attempt': 0}]}}])
            results.extend({**common, 'arm': arm, 'kind': 'information', 'sample_id': r['sample_id'], 'output': r['target']} for r in info)
        kwargs = {'arms': ('base', 'previous', 'candidate'), 'information_tasks': info}
        report = summarize_capability([role], cohort, results, **kwargs)
        self.assertEqual(report['information']['candidate']['task_correct'], 128)
        with self.assertRaisesRegex(ValidationError, 'incomplete'):
            summarize_capability([role], cohort, results[:-1], **kwargs)
        with self.assertRaisesRegex(ValidationError, 'duplicate'):
            summarize_capability([role], cohort, results + [results[-1]], **kwargs)

    def test_versioned_configs_preserve_older_defaults_and_reject_unfrozen_selection(self):
        old = load_training_config(ROOT/'configs/qwen3_8b_research_tool_sft_v2.json')
        self.assertNotIn('curriculum', old)
        new = load_training_config(ROOT/'configs/qwen3_8b_research_tool_sft_v3.json')
        self.assertEqual(new['curriculum'], POLICY)
        config = load_evaluation_config(ROOT/'configs/research_tool_evaluation_v4.json')
        self.assertEqual(config['information_cohort'], EVALUATION_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'; config['information_cohort'] = 'choose_best_outputs'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValidationError):load_evaluation_config(path)


if __name__ == '__main__':unittest.main()
