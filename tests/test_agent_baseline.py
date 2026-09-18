import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.agent_baseline import summarize_baseline, tool_call_matches
from foretellmesh.agent_baseline_data import input_context, load_baseline_config, prepare_agent_baseline, read_cohort
from foretellmesh.agent_check import ScriptedBackend
from foretellmesh.agent_runtime import AgentRunner
from foretellmesh.data import sha256_file
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import generate_synthetic_sft

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / 'configs/capability_agents_v1.json'
SPLITS = ROOT / 'configs/synthetic_sft_splits_v1.json'
CONFIG = ROOT / 'configs/agent_behavior_baseline_v1.json'


class AgentBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = json.loads((ROOT / 'configs/synthetic_sft_generator_v1.json').read_text())
        generator['groups_per_family'] = {'train': 1, 'validation': 2, 'test': 1}
        path = self.root / 'generator.json'
        path.write_text(json.dumps(generator))
        self.raw, self.output = self.root / 'raw', self.root / 'cohort'
        generate_synthetic_sft(path, self.raw)

    def prepare(self):
        prepare_agent_baseline(self.raw, SPLITS, AGENTS, CONFIG, self.output)
        return read_cohort(self.output)

    def simulated_results(self, cohort):
        agents = cohort['agent_config.json']
        judges = {j['sample_id']: j for j in cohort['judge.jsonl']}
        tool_judges = {j['sample_id']: j for j in cohort['tool_judge.jsonl']}
        rows = []
        for input_row in cohort['inputs.jsonl'] + cohort['tool_inputs.jsonl']:
            sid, payload = input_row['sample_id'], input_row['input']
            observed = payload['observation_time']
            if sid in judges:
                judge = judges[sid]
                forecast = {'event': payload['question'], 'probability': judge['oracle_probability'],
                            'confidence': 'high', 'base_rate': None, 'key_evidence': judge['required_evidence_ids'],
                            'counter_evidence': [], 'unknowns': [], 'observation_time': observed}
                responses = {
                    'research': [{'evidence_ids': judge['required_evidence_ids'], 'counter_evidence_ids': [],
                                  'unknowns': [], 'observation_time': observed}],
                    'risk': [{'risks': [], 'unknowns': [], 'observation_time': observed}],
                    'forecast': [forecast],
                    'critic': [{'accept': True, 'revised_probability': None, 'rationale': 'Fixture only.',
                                'evidence_ids': ['calculation'], 'unknowns': [], 'observation_time': observed}]}
                workflows = cohort['evaluation_config.json']['workflows']
            else:
                responses = {'quant': [tool_judges[sid]['expected_call']]}
                workflows = ['calculate']
            for workflow in workflows:
                backend = ScriptedBackend(responses)
                result = AgentRunner(agents, backend).run(input_context(payload), workflow=workflow)
                self.assertEqual(result['status'], 'completed')
                rows.append({'sample_id': sid, 'workflow': workflow, 'result': result, 'seconds': 1,
                             'calls': [{'usage': {'input_tokens': 10, 'output_tokens': 5,
                                        'output_reached_token_limit': False, 'seconds': .1}} for _ in backend.calls]})
        return rows

    def test_frozen_selection_bilingual_validation_and_judge_isolation(self):
        manifest, cohort = self.prepare()
        self.assertEqual(manifest['counts'], {'forecast_inputs': 12, 'independent_event_groups': 6, 'tool_inputs': 4, 'families': 6})
        judge = cohort['judge.jsonl']
        for family in {j['family'] for j in judge}:
            selected = [j for j in judge if j['family'] == family]
            self.assertEqual({j['language'] for j in selected}, {'en', 'zh'})
            self.assertEqual(len({j['event_group_id'] for j in selected}), 1)
        for row in cohort['inputs.jsonl'] + cohort['tool_inputs.jsonl']:
            self.assertEqual(set(row['input']), {'question', 'observation_time', 'evidence', 'market'})
            self.assertEqual(row['input']['observation_time'], '2024-02-15T00:00:00Z')
            self.assertNotIn('oracle_probability', row['input'])
            self.assertNotIn('label', row['input'])
        for row in cohort['tool_inputs.jsonl']:
            self.assertEqual([e['evidence_id'] for e in row['input']['evidence']], ['setup'])
        other = self.root / 'again'
        second = prepare_agent_baseline(self.raw, SPLITS, AGENTS, CONFIG, other)
        self.assertEqual(manifest['artifact_hashes'], second['artifact_hashes'])
        with self.assertRaisesRegex(ValidationError, 'already exists'):
            prepare_agent_baseline(self.raw, SPLITS, AGENTS, CONFIG, self.output)

    def test_final_test_selection_and_unknown_config_are_rejected(self):
        config = json.loads(CONFIG.read_text())
        for modified in ({**config, 'source_partition': 'test'}, {**config, 'extra': 1}, {**config, 'groups_per_family': True}):
            path = self.root / 'invalid.json'
            path.write_text(json.dumps(modified))
            with self.assertRaises(ValidationError):load_baseline_config(path)

    def test_protocol_ablation_freezes_exactly_the_same_cohort(self):
        first, _ = self.prepare()
        config, _ = load_baseline_config(ROOT / 'configs/agent_behavior_baseline_v2.json')
        self.assertEqual(config['output_protocol'], 'plain_json_v1')
        second = prepare_agent_baseline(self.raw, SPLITS, AGENTS, ROOT / 'configs/agent_behavior_baseline_v2.json', self.root / 'v2')
        for name in ('inputs.jsonl', 'judge.jsonl', 'tool_inputs.jsonl', 'tool_judge.jsonl', 'agent_config.json'):
            self.assertEqual(first['artifact_hashes'][name], second['artifact_hashes'][name])
        third = prepare_agent_baseline(self.raw, SPLITS, AGENTS, ROOT / 'configs/agent_behavior_baseline_v3.json', self.root / 'v3')
        for name in ('inputs.jsonl', 'judge.jsonl', 'tool_inputs.jsonl', 'tool_judge.jsonl', 'agent_config.json'):
            self.assertEqual(first['artifact_hashes'][name], third['artifact_hashes'][name])

    def test_source_hash_tampering_is_rejected(self):
        path = self.raw / 'records.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValidationError, 'hash/type mismatch'):self.prepare()

    def test_rehashed_simulated_label_tampering_fails_oracle_replay(self):
        path = self.raw / 'records.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        record = next(r for r in rows if r['observation_time'].startswith('2024-02'))
        record['label']['outcome'] = 1 - record['label']['outcome']
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest_path = self.raw / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['records_sha256'] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, 'deterministic oracle replay'):self.prepare()

    def test_frozen_cohort_tampering_fails_closed(self):
        self.prepare()
        path = self.output / 'judge.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValidationError, 'artifact hash mismatch'):read_cohort(self.output)

    def test_perfect_fixture_metrics_and_tool_semantics(self):
        _, cohort = self.prepare()
        rows = self.simulated_results(cohort)
        report = summarize_baseline(cohort, rows)
        self.assertEqual(report['counts']['scored_workflow_examples'], 40)
        self.assertEqual(report['common_coverage_comparison']['count'], 12)
        for arm in report['arms'].values():
            self.assertEqual(arm['oracle_probability_mae'], 0)
            self.assertEqual(arm['within_oracle_tolerance_fraction_all_examples'], 1)
        self.assertEqual(report['tools']['arguments_match_count'], 4)
        self.assertEqual(report['critic_comparison']['probability_changes'], 0)
        original = {'name': 'weighted_probability', 'arguments': {'probabilities': [.1, .9], 'weights': [.2, .8]}}
        reversed_call = {'name': 'weighted_probability', 'arguments': {'probabilities': [.9, .1], 'weights': [.8, .2]}}
        self.assertTrue(tool_call_matches(reversed_call, original))
        self.assertFalse(tool_call_matches({'name': 'shell', 'arguments': {}}, original))
        wrong = {'name': 'weighted_probability', 'arguments': {'probabilities': [.1, .9], 'weights': [.8, .2]}}
        self.assertFalse(tool_call_matches(wrong, original))

    def test_incomplete_and_duplicate_runs_never_publish_final_scores(self):
        _, cohort = self.prepare()
        rows = self.simulated_results(cohort)
        with self.assertRaisesRegex(ValidationError, 'incomplete'):summarize_baseline(cohort, rows[:-1])
        with self.assertRaisesRegex(ValidationError, 'duplicate'):summarize_baseline(cohort, rows + [rows[0]])

    def test_failed_critic_counts_missing_and_preserves_paired_comparison(self):
        _, cohort = self.prepare()
        rows = self.simulated_results(cohort)
        row = next(r for r in rows if r['workflow'] == 'reviewed_forecast')
        row['result'].update(status='failed', prediction=None, stage='critic', error='output_schema_failed')
        metrics = summarize_baseline(cohort, rows)
        arm = metrics['arms']['reviewed_forecast']
        self.assertEqual(arm['simulated_outcome_scores']['missing_count'], 1)
        self.assertEqual(arm['resources_and_format']['completion_rate'], 11/12)
        self.assertEqual(arm['within_oracle_tolerance_fraction_all_examples'], 11/12)
        self.assertEqual(metrics['common_coverage_comparison']['count'], 11)
        self.assertEqual(metrics['critic_comparison']['forecast_available_final_missing'], 1)
        self.assertEqual(metrics['critic_comparison']['paired_count'], 11)


if __name__ == '__main__':unittest.main()
