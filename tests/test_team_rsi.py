from copy import deepcopy
import json
import sqlite3
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_learning import TeamRunner, run_episode, run_query
from foretellmesh.team_outcome_learning import binary_prompt, binary_probability
from foretellmesh.team_rsi_data import assign_groups, make_jobs
from foretellmesh.team_rsi_experiment import (build_artifact, training_rows, summarize,
    validate_exploration_config, finish_exploration, audit_exploration, run)
from foretellmesh.sft_data import jsonl
from test_team_learning import context, labels, feed, policy, ScriptBackend, AT


class TeamRsiTests(unittest.TestCase):
    def test_legacy_automatic_training_rejected_before_loading_data_or_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = root/'config.json'
            config.write_text(json.dumps({'kind': 'team_rsi_cycle_v1', 'training': {'epochs': 2}}))
            with patch('foretellmesh.team_rsi_experiment.load_partition') as load:
                with self.assertRaisesRegex(ValidationError, 'Automatic RSI fine-tuning is retired'):
                    run(config, root/'model.json', root/'output')
                load.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_exploration_cannot_enable_training_with_a_boolean_or_training_config(self):
        config = {'kind': 'team_exploration_v1', 'automatic_fine_tuning': False,
                  'effectiveness_evidence_required': True, 'final_test_opened': False, 'sampling': None}
        validate_exploration_config(config)
        for field, value in [('automatic_fine_tuning', True), ('effectiveness_evidence_required', False), ('training', {})]:
            with self.assertRaises(ValidationError): validate_exploration_config({**config, field: value})

    def test_profitable_valid_experience_does_not_trigger_fitting_or_qualification(self):
        episode = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        self.assertGreater(float(episode['feedback']['account']['net_pnl']), 0)
        job = {'partition': 'train', 'context': context(), 'labels': labels()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('foretellmesh.team_rsi_experiment.fit_outcomes') as fit:
                report = finish_exploration(root, [job], [episode]); fit.assert_not_called()
            self.assertFalse(report['fine_tuning_triggered']); self.assertFalse(report['effectiveness_verified'])
            self.assertEqual(report['admitted_training_examples'], 0)
            self.assertTrue((root/'experience_candidates.jsonl').exists())
            self.assertFalse((root/'forecast_adapter').exists())
            (root/'train.jobs.jsonl').write_text(jsonl([job]))
            (root/'train.episodes.jsonl').write_text(jsonl([episode]))
            (root/'report.json').write_text(json.dumps(report))
            with patch('foretellmesh.team_rsi_experiment.load_partition', return_value=[job]), \
                    patch('foretellmesh.team_rsi_experiment.make_feed', return_value=feed()):
                result = audit_exploration(root, report, {'execution_policy': policy()})
            self.assertEqual(result['status'], 'passed'); self.assertFalse(result['fine_tuning_performed'])

    def test_failed_exploration_is_preserved_without_requiring_training_examples(self):
        episode = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend(bad=True)), recorded_latency=1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = finish_exploration(root, [{'partition': 'train', 'context': context(), 'labels': labels()}], [episode])
            self.assertEqual(report['generated_candidates'], 0)
            self.assertFalse(report['fine_tuning_triggered'])
            self.assertIsNone(json.loads((root/'learning_artifact.json').read_text()))

    def request(self):
        return {'agent': 'forecast', 'input': context(), 'upstream': {
            'research': {'hypotheses': ['Investigate shared drivers.']},
            'tools': [run_query({'tool': 'pair_changes', 'market_ids': ['a', 'b'], 'lookback_days': 3}, feed(), AT)]},
            'adapter': None}

    def test_readout_prompt_contains_only_past_inputs_and_verified_tool_summary(self):
        req = self.request(); data = binary_prompt(req, 'a')
        self.assertIn('Investigate shared drivers.', data)
        self.assertNotIn('resolution_time', data); self.assertNotIn('feedback_facts', data)
        req['upstream']['tools'][0]['series']['a'][0]['price'] = .99
        with self.assertRaisesRegex(ValidationError, 'tool result changed'): binary_prompt(req, 'a')
        req = self.request(); result = req['upstream']['tools'][0]
        result['series']['a'][0]['source_time'] = '2030-01-01T00:00:00Z'
        result['result_sha256'] = canonical_hash({k: v for k, v in result.items() if k != 'result_sha256'})
        with self.assertRaisesRegex(ValidationError, 'future summarized quote'): binary_prompt(req, 'a')

    def test_conditional_probability_is_stable_and_moves_with_correct_answer(self):
        self.assertEqual(binary_probability([0, 0]), .5)
        self.assertGreater(binary_probability([0, 1]), .5)
        self.assertEqual(binary_probability([-10000, 10000]), 1)
        with self.assertRaises(ValidationError): binary_probability([0, float('nan')])

    def test_group_split_purges_full_lifecycle_and_keeps_siblings_together(self):
        config = {'train_before': '2024-09-01T00:00:00Z', 'development_from': '2024-09-15T00:00:00Z',
                  'development_before': '2025-02-01T00:00:00Z'}
        def row(group, start, end):
            return {'event_group_id': group, 'proof': {'initialized_at': start, 'resolution_time': end}}
        rows = [row('a', '2024-01-01T00:00:00Z', '2024-08-01T00:00:00Z'),
                row('b', '2024-10-01T00:00:00Z', '2024-11-01T00:00:00Z'),
                row('a', '2024-08-01T00:00:00Z', '2024-10-01T00:00:00Z')]
        self.assertEqual(assign_groups(rows, config), {'a': 'purged_boundary', 'b': 'development'})

    def test_event_trigger_does_not_backfill_quote_or_repeat_same_print(self):
        db = sqlite3.connect(':memory:')
        db.executescript('CREATE TABLE trades(tx TEXT,log_index INTEGER,block_number INTEGER,market_id TEXT,numerator TEXT,denominator TEXT);'
                        'CREATE TABLE blocks(block_number INTEGER,unix_time INTEGER);')
        at = int(timestamp(AT, 'at').timestamp())
        db.execute('INSERT INTO blocks VALUES(1,?)', (at,))
        db.execute("INSERT INTO trades VALUES('tx',0,1,'a','4','10')")
        markets = [{'market_id': 'a', 'event_group_id': 'a', 'proof': {
            'initialized_at': '2024-01-01T00:00:00Z', 'resolution_time': '2024-01-03T00:00:00Z',
            'available_at': '2026-01-01T00:00:00Z', 'outcome': 1, 'historical_question': 'Will event a happen?',
            'ancillary_sha256': 'a', 'proof_sha256': 'b', 'manifest_sha256': 'c'}}]
        jobs, _ = make_jobs(markets, {'a': 'train'}, db, {'observation_age_hours': [6,24,168],
            'observation_trigger': 'first_trade_after_age', 'max_quote_age_seconds': 10800, 'max_markets_per_episode': 3})
        self.assertEqual(len(jobs['train']), 1)
        self.assertEqual(jobs['train'][0]['context']['observation_time'], AT)
        self.assertEqual(jobs['train'][0]['labels']['a']['available_at'], '2026-01-01T00:00:00Z')
        self.assertNotIn('outcome', json.dumps(jobs['train'][0]['context']))

    def test_offline_reflection_is_source_bound_and_not_backdated_online_memory(self):
        episode = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        job = {'partition': 'train', 'context': context(), 'labels': labels()}
        artifact = build_artifact([job], [episode], built_at='2026-09-20T00:00:00Z')
        self.assertEqual(artifact['built_at'], '2026-09-20T00:00:00Z')
        self.assertEqual(artifact['source_public_through'], '2024-01-12T00:00:00Z')
        runner = TeamRunner(ScriptBackend(), learning_artifacts=[artifact])
        with self.assertRaisesRegex(ValidationError, 'leaks target'): runner.call('research', context(), {}, {'a', 'b'})
        c = context(); c['observation_time'] = '2024-02-01T00:00:00Z'
        for m in c['markets']:
            m['event_group_id'] = 'new-'+m['market_id']; m['input']['observation_time'] = c['observation_time']
        runner.call('research', c, {}, {'a', 'b'})
        self.assertEqual(runner.calls[-1]['request']['upstream']['offline_learning_artifacts'], [artifact])
        altered = deepcopy(artifact); altered['lessons'] = ['invented']
        with self.assertRaisesRegex(ValidationError, 'unbound'):
            TeamRunner(ScriptBackend(), learning_artifacts=[altered]).call('research', c, {}, {'a', 'b'})

    def test_outcome_supervision_admits_all_valid_calls_including_vetoed_targets(self):
        episode = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        request = episode['calls'][1]['request']; episode['readouts'] = []
        for mid in ('a', 'b'):
            prompt = binary_prompt(request, mid)
            episode['readouts'].append({'market_id': mid, 'request_sha256': canonical_hash(request),
                'prompt': prompt, 'prompt_sha256': canonical_hash(prompt), 'logits_no_yes': [0,0], 'probability': .5})
        job = {'partition': 'train', 'context': context(), 'labels': labels(), 'provenance': {'a': {}, 'b': {}}}
        rows = training_rows([job], [episode]); self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['outcome'], 1); self.assertNotIn('probability', rows[1])
        job['partition'] = 'development'
        with self.assertRaisesRegex(ValidationError, 'training source'): training_rows([job], [episode])
        self.assertEqual(summarize([episode])['forecast']['coverage'], 1)

    def test_empty_reflection_stays_empty_instead_of_inventing_training_lessons(self):
        episode = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        episode['reflection'] = {'fact_ids': [], 'lessons': [], 'next_experiments': []}
        artifact = build_artifact([{'partition': 'train', 'context': context(), 'labels': labels()}], [episode])
        self.assertEqual(artifact['lessons'], [])


if __name__ == '__main__': unittest.main()
