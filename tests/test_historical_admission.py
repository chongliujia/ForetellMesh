from copy import deepcopy
import json
from pathlib import Path
import unittest

import test_historical_market as historical
from foretellmesh.data import sha256_file
from foretellmesh.historical_replay_gate import blockers, preflight, verify_staging
from foretellmesh.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]


class HistoricalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = historical.HistoricalMarketTests(methodName='runTest')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.fixture.collect(); self.fixture.build(); self.staging = self.fixture.root/'built'
        self.rows = [json.loads(s) for s in (self.staging/'candidates.jsonl').read_text().splitlines()]

    def test_teacher_missing_blocks_sft_but_not_independent_scoring(self):
        row = next(r for r in self.rows if r['question_proof'])
        row['blockers'] = ['historical_teacher_target_missing']
        self.assertEqual(blockers(row, 'development_replay'), [])
        self.assertEqual(blockers(row, 'sft'), ['historical_teacher_target_missing'])
        row['blockers'].append('new_unreviewed_source_problem')
        self.assertIn('new_unreviewed_source_problem', blockers(row, 'development_replay'))

    def test_ready_flags_do_not_override_labels_prices_or_timestamps(self):
        row = deepcopy(self.rows[0]); row['blockers'] = []; row['ready_for_benchmark'] = True
        row['record']['label'] = None; row['record']['market'] = None
        self.assertIn('exact_settlement_proof_missing', blockers(row, 'development_replay'))
        self.assertIn('historical_price_unusable', blockers(row, 'development_replay'))
        row['record']['evidence'][0]['available_at'] = '2027-01-01T00:00:00Z'
        with self.assertRaises(ValidationError):blockers(row, 'development_replay')

    def test_rehashed_manual_promotion_cannot_pass_raw_source_reconstruction(self):
        verify_staging(self.staging, self.fixture.capture, self.fixture.index)
        self.rows[0].update(blockers=[], ready_for_benchmark=True)
        path = self.staging/'candidates.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        report = json.loads((self.staging/'report.json').read_text())
        report['artifact_hashes']['candidates.jsonl'] = sha256_file(path)
        (self.staging/'report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValidationError, 'replay from raw'):
            verify_staging(self.staging, self.fixture.capture, self.fixture.index)

    def test_preflight_reports_zero_coverage_without_spurious_scores(self):
        result = preflight(self.staging, self.fixture.capture, self.fixture.index, ROOT/'configs/historical_replay_admission_v1.json')
        self.assertEqual(result['model_calls'], 0); self.assertIsNone(result['score_metrics'])
        self.assertEqual(result['candidate_event_groups'], 1)
        self.assertEqual(result['eligible_replay_rows'], 0)
        self.assertNotIn('historical_teacher_target_missing', result['replay_blocker_counts'])
        self.assertIn('historical_market_rules_version_missing', result['replay_blocker_counts'])


if __name__ == '__main__':unittest.main()
