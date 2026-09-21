from copy import deepcopy
from datetime import timedelta
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_learning import (MemoryBank, TeamRunner, experience_candidates, feedback_for,
    run_episode, run_query, validate_context, validate_memory, validate_output)
from foretellmesh.team_market_scope import select_candidates


AT = '2024-01-02T00:00:00Z'


def context():
    return {'episode_id': 'episode-a', 'observation_time': AT,
            'account': {'initial_cash': '100', 'simulation_only': True}, 'memory': [],
            'markets': [{'market_id': mid, 'event_group_id': 'event-'+mid, 'input': {
                'question': 'Will event '+mid+' happen?', 'observation_time': AT, 'evidence': [],
                'market': {'probability': .4, 'observed_at': AT, 'available_at': AT}}} for mid in ('a', 'b')]}


def policy():
    return {'initial_cash': '100', 'max_trade_usd': '2', 'max_event_usd': '5',
            'max_portfolio_usd': '20', 'min_trade_usd': '1', 'min_edge': '.03',
            'entry_price_premium': '.01', 'fee_fraction': '.01', 'max_quote_age_seconds': 10800,
            'fill_window_seconds': 3600}


def labels():
    return {m: {'outcome': 1, 'resolution_time': '2024-01-12T00:00:00Z',
                'available_at': '2024-01-13T00:00:00Z'} for m in ('a', 'b')}


def feed():
    at = timestamp(AT, 'at')
    return HistoricalBars({m: [(at-timedelta(days=3-i), .2+i*.05, 1) for i in range(4)] +
                           [(at+timedelta(minutes=10), .4, 1)] for m in ('a', 'b')})


class ScriptBackend:
    def __init__(self, bad=False, trade=True):
        self.requests = []; self.bad = bad; self.trade = trade

    def generate(self, request):
        self.requests.append(deepcopy(request))
        if self.bad:
            return '{broken'
        return json.dumps({
            'research': {'queries': [{'tool': 'pair_changes', 'market_ids': ['a', 'b'], 'lookback_days': 3}],
                         'hypotheses': ['Test whether changes co-move; not a causal claim.']},
            'forecast': {'forecasts': [{'market_id': m, 'probability': .8, 'consider_trade': self.trade}
                                      for m in ('a', 'b')], 'unknowns': ['No independent event evidence.']},
            'risk': {'veto_markets': ['b'], 'risks': ['Shared error risk.']},
            'reflection': {'fact_ids': ['paper_net_pnl', 'forecast_brier'],
                           'lessons': ['This outcome alone cannot establish calibration.'],
                           'next_experiments': ['Check a new event.']}}[request['agent']])


@unittest.skipUnless(importlib.util.find_spec('langgraph'), 'optional LangGraph dependency')
class TeamLearningTests(unittest.TestCase):
    def test_live_graph_tools_feedback_and_no_label_in_decision(self):
        backend = ScriptBackend()
        ep = run_episode(context(), labels(), feed(), policy(), TeamRunner(backend), recorded_latency=1)
        self.assertIsNone(ep['decision_error']); self.assertEqual(ep['feedback']['account']['filled_trades'], 1)
        self.assertEqual(len(ep['calls']), 4)
        self.assertAlmostEqual(ep['feedback']['scores']['brier'], .04)
        for request in backend.requests[:3]:
            payload = json.dumps(request)
            for hidden in ('resolution_time', 'outcome', 'feedback_facts', '2024-01-13'):
                self.assertNotIn('"'+hidden+'"', payload)
        self.assertEqual(backend.requests[-1]['input']['phase'], 'retrospective')
        self.assertEqual(ep['feedback']['account']['real_orders_sent'], 0)
        self.assertEqual(ep['feedback']['account']['final_cash'],
                         str(100 + __import__('decimal').Decimal(ep['feedback']['account']['net_pnl'])))

    def test_memory_only_after_feedback_and_not_same_group(self):
        ep = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        bank = MemoryBank([ep['memory']])
        self.assertEqual(bank.at(AT), [])
        self.assertEqual(len(bank.at('2024-01-14T00:00:00Z')), 1)
        self.assertEqual(bank.at('2024-01-14T00:00:00Z', excluded_groups=['event-a']), [])
        self.assertEqual(bank.at('2024-01-14T00:00:00Z', limit=0), [])
        c = context(); c['memory'] = [ep['memory']]
        with self.assertRaisesRegex(ValidationError, 'future memory'):
            validate_context(c)
        altered = deepcopy(ep['memory']); altered['reflection']['lessons'] = ['fabricated']
        with self.assertRaisesRegex(ValidationError, 'hash'):
            validate_memory(altered)

    def test_cash_is_valid_and_failed_team_does_not_get_default_forecasts(self):
        cash = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend(trade=False)), recorded_latency=1)
        self.assertEqual(cash['feedback']['account']['final_cash'], '100')
        failed = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend(bad=True)), recorded_latency=1)
        self.assertEqual(len(failed['calls']), 2)
        self.assertIsNone(failed['decision']); self.assertIsNone(failed['memory'])
        self.assertEqual(failed['feedback']['account']['final_cash'], '100')
        self.assertEqual(failed['feedback']['scores']['coverage'], 0)

    def test_causal_tools_staleness_and_future_poison(self):
        q = {'tool': 'pair_changes', 'market_ids': ['a', 'b'], 'lookback_days': 3}
        f = feed(); before = run_query(q, f, AT)
        f.data['a'].append((timestamp(AT, 't')+timedelta(days=1), .99, 1))
        f.times['a'].append(timestamp(AT, 't')+timedelta(days=1))
        self.assertEqual(run_query(q, f, AT), before)
        self.assertEqual(before['paired_daily_changes'], 3)
        old = run_query(q, f, '2024-02-02T00:00:00Z')
        self.assertEqual(old['paired_daily_changes'], 0)
        class Future:
            def latest(self, mid, at): return .5, at+timedelta(seconds=1)
        with self.assertRaisesRegex(ValidationError, 'future'):
            run_query(q, Future(), AT)

    def test_invalid_tools_forecasts_and_refs_fail_closed(self):
        invalid = [({'queries': [{'tool': 'shell', 'market_ids': ['a'], 'lookback_days': 1}], 'hypotheses': []}, 'research'),
                   ({'queries': [{'tool': 'history', 'market_ids': ['test'], 'lookback_days': 1}], 'hypotheses': []}, 'research'),
                   ({'queries': [{'tool': 'history', 'market_ids': ['a'], 'lookback_days': True}], 'hypotheses': []}, 'research'),
                   ({'forecasts': [{'market_id': 'a', 'probability': float('nan'), 'consider_trade': True}], 'unknowns': []}, 'forecast'),
                   ({'veto_markets': ['test'], 'risks': []}, 'risk'),
                   ({'fact_ids': ['invented'], 'lessons': [], 'next_experiments': []}, 'reflection')]
        for value, role in invalid:
            with self.subTest(role=role), self.assertRaises(ValidationError):
                validate_output(role, value, {'a'})

    def test_post_settlement_latency_cannot_fabricate_fill(self):
        ep = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=20*86400)
        self.assertEqual(ep['feedback']['account']['filled_trades'], 0)

    def test_generated_profitable_trace_is_not_automatic_sft_admission(self):
        ep = run_episode(context(), labels(), feed(), policy(), TeamRunner(ScriptBackend()), recorded_latency=1)
        candidates = experience_candidates([ep])
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(r['ready_for_training'] is False for r in candidates))
        self.assertEqual(candidates[-1]['task_phase'], 'retrospective')
        self.assertTrue(all(r['task_phase'] == 'ex_ante' for r in candidates[:-1]))

    def test_unknown_context_fields_cannot_smuggle_labels(self):
        c = context(); c['markets'][0]['input']['label'] = {'outcome': 1}
        with self.assertRaises(ValidationError): validate_context(c)
        c = context(); c['labels'] = labels()
        with self.assertRaises(ValidationError): validate_context(c)


class TopicNeutralScopeTests(unittest.TestCase):
    def test_actual_reviewed_adapter_loader_return_types(self):
        from foretellmesh.team_market_proof import runtime
        from foretellmesh.pma_proof_replay import reviewed_runtime as neg
        from foretellmesh.pma_legacy import reviewed_runtime as legacy
        # Existing immutable local source reviews; no network or market labels.
        for path, loader in [('configs/pma_proof_pilot_v1.json', neg),
                             ('configs/pma_legacy_fomc_extension_v1.json', legacy)]:
            if not Path(path).exists(): self.skipTest('local review configuration unavailable')
            try: result = runtime(Path(path), loader)
            except FileNotFoundError: self.skipTest('optional local raw source reviews unavailable')
            self.assertIsInstance(result, str)
            self.assertTrue(result.startswith('0x')); self.assertGreater(len(result), 1000)

    def rows(self):
        return [{'market_id': str(i), 'benchmark_matches': [], 'tokens': {'Yes': str(i)+'0', 'No': str(i)+'1'},
                 'created_at': AT, 'question': text, 'snapshot_closed': closed} for i, text, closed in
                [(1, 'Will it snow?', False), (2, 'Will a team win?', True), (3, 'Will the Fed cut?', True)]]

    def config(self):
        return {'kind': 'team_market_discovery_v1', 'seed': 1, 'limit': 3,
                'created_from': '2023-01-01T00:00:00Z', 'created_before': '2025-01-01T00:00:00Z'}

    def test_not_selected_by_topic_closed_status_or_input_order(self):
        rr = self.rows(); a, _ = select_candidates(rr, [], self.config())
        b, _ = select_candidates(list(reversed(rr)), [], self.config())
        self.assertEqual(a, b); self.assertEqual({r['market_id'] for r in a}, {'1', '2', '3'})
        self.assertTrue(all(not r['ready_for_training'] for r in a))
        changed = deepcopy(rr)
        for r in changed: r['question'] = 'irrelevant'; r['snapshot_closed'] = not r['snapshot_closed']
        c, _ = select_candidates(changed, [], self.config())
        self.assertEqual([r['market_id'] for r in a], [r['market_id'] for r in c])

    def test_reservations_and_duplicates(self):
        rr = self.rows(); rr[0]['benchmark_matches'] = ['reserved']
        selected, counts = select_candidates(rr, [{'event_id': 'polymarket:2', 'split': 'test'}], self.config())
        self.assertEqual([r['market_id'] for r in selected], ['3'])
        self.assertEqual(counts['exclusions']['reserved_identity'], 2)
        with self.assertRaisesRegex(ValidationError, 'duplicate'):
            select_candidates(self.rows()+self.rows(), [], self.config())


if __name__ == '__main__': unittest.main()
