from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
import unittest

from foretellmesh.schema import ValidationError, iso
from foretellmesh.trading_agent_signals import historical_input, schedule
from test_trading_rl_env import CONFIG, T, fixture


class HistoricalAgentInputTests(unittest.TestCase):
    def test_future_evidence_is_excluded_and_future_mutation_has_no_effect(self):
        job = {'sample_id': 'j', 'observation_time': iso(T)}
        old = {'evidence_id': 'old', 'source': 'https://example.org/old', 'text': 'Already known.',
               'published_at': iso(T - timedelta(days=1)), 'available_at': iso(T - timedelta(hours=1))}
        future = {**old, 'evidence_id': 'future', 'available_at': iso(T + timedelta(seconds=1))}
        data = [{'input': {'question': 'Will the event happen?', 'evidence': [old, future]}}]
        result = historical_input(job, data, (Decimal('.5'), T))
        self.assertEqual(result['input']['evidence'], [old])
        data[0]['input']['evidence'][1]['text'] = 'Future outcome revealed.'
        self.assertEqual(result, historical_input(job, data, (Decimal('.5'), T)))
        with self.assertRaises(ValidationError):
            historical_input(job, data, (Decimal('.5'), T + timedelta(seconds=1)))

    def test_conflicting_evidence_and_questions_rejected(self):
        e = {'evidence_id': 'e', 'source': 'https://example.org/e', 'text': 'Known.',
             'published_at': iso(T), 'available_at': iso(T)}
        data = [{'input': {'question': 'Event?', 'evidence': [e]}},
                {'input': {'question': 'Event?', 'evidence': [{**e, 'text': 'Changed.'}]}}]
        with self.assertRaises(ValidationError):
            historical_input({'sample_id': 'j', 'observation_time': iso(T)}, data, (Decimal('.5'), T))
        data[1]['input']['question'] = 'Different event?'
        with self.assertRaises(ValidationError):
            historical_input({'sample_id': 'j', 'observation_time': iso(T)}, data, (Decimal('.5'), T))

    def test_schedule_is_causal_and_bounded_by_group_refresh(self):
        data, _ = fixture(days=17, markets=4)
        spec = {'refresh_seconds': 604800, 'markets_per_group': 2}
        original = schedule(data, CONFIG, spec)
        self.assertEqual(len(original), 6)
        self.assertEqual({r['market_id'] for r in original}, {'0', '1'})
        other = deepcopy(data)
        for state in other.states[24:]:
            state['3']['trade_count_24h'] += 1000
        changed = schedule(other, CONFIG, spec)
        self.assertEqual(original[:2], changed[:2])
        self.assertNotEqual(original[2:], changed[2:])


if __name__ == '__main__':
    unittest.main()
