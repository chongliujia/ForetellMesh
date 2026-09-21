"""Diagnostic denominators and trade attribution must preserve ledger semantics."""
import importlib.util
from pathlib import Path
import unittest

from foretellmesh.schema import ValidationError

spec = importlib.util.spec_from_file_location('optimizer_diagnostic',
    Path(__file__).resolve().parents[1] / 'scripts/diagnose_allocation_optimizer.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def decision(action, legal):
    return {'executed_action': action, 'mask': [i in legal for i in range(25)]}


class OptimizerDiagnosticTests(unittest.TestCase):
    def test_voluntary_wait_excludes_ticks_without_a_choice(self):
        result = diagnostic.decision_summary([decision(0, [0]), decision(0, [0, 1, 2]), decision(1, [0, 1])])
        self.assertEqual(result['wait_ticks'], 2)
        self.assertEqual(result['only_wait_legal_ticks'], 1)
        self.assertEqual(result['choice_ticks'], 2)
        self.assertEqual(result['voluntary_wait_fraction'], .5)
        self.assertAlmostEqual(result['uniform_legal_action_wait_probability_on_choice_ticks'], (1/3 + 1/2)/2)

    def test_illegal_actions_and_wait_guard_are_rejected(self):
        for row in [decision(1, [0]), decision(1, [1]), decision(-1, [0])]:
            with self.assertRaises(ValidationError):
                diagnostic.decision_summary([row])
        self.assertIsNone(diagnostic.decision_summary([decision(0, [0])])['voluntary_wait_fraction'])

    def test_no_side_costs_and_missing_targets_are_separate(self):
        ledger = [
            {'kind': 'buy_order', 'reference_yes': '.7', 'target': '.5', 'side': 'no'},
            {'kind': 'buy_order', 'reference_yes': '.7', 'target': '.5', 'side': 'yes'},
            {'kind': 'buy_order', 'reference_yes': '.5', 'target': None, 'side': 'yes'}]
        result = diagnostic.ledger_summary(ledger, {'fee_fraction': '.01', 'entry_price_premium': '.01'}, .01)
        self.assertEqual(result['buy_orders'], 3)
        self.assertEqual(result['buy_orders_without_historical_mean'], 1)
        self.assertEqual(result['buy_orders_positive_mean_reversion_edge'], 1)
        self.assertEqual(result['buy_orders_meeting_rule_edge_threshold'], 1)

    def test_partial_exit_pnl_and_completed_holding_time(self):
        ledger = [
            {'kind': 'buy_fill', 'market': 'a', 'time': '2024-01-01T00:00:00Z'},
            {'kind': 'sell_fill', 'market': 'a', 'time': '2024-01-01T01:00:00Z',
                'exit_reason': 'policy', 'net_pnl': '-.01', 'full_exit': False},
            {'kind': 'sell_fill', 'market': 'a', 'time': '2024-01-03T00:00:00Z',
                'exit_reason': 'holding_limit', 'net_pnl': '.02', 'full_exit': True}]
        result = diagnostic.ledger_summary(ledger, {'fee_fraction': '0', 'entry_price_premium': '0'}, .01)
        self.assertEqual(result['mean_completed_position_hours'], 48)
        self.assertEqual(result['net_pnl_by_exit_reason'], {'policy': '-0.01', 'holding_limit': '0.02'})
        self.assertEqual(result['open_positions_at_ledger_end'], 0)


if __name__ == '__main__':
    unittest.main()
