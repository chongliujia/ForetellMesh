from datetime import timedelta
import unittest

from foretellmesh.macro_prices import clob_history_quote
from foretellmesh.schema import ValidationError, timestamp


class ClobHistoricalPriceTests(unittest.TestCase):
    def setUp(self):
        self.t = timestamp('2026-07-13T12:30:00Z', 'time'); self.sec = int(self.t.timestamp())
        self.policy = {'price_window_hours': 6, 'max_price_age_seconds': 10800}

    def quote(self, rows):
        return clob_history_quote({'history': rows}, self.t, self.policy)

    def test_edge_guard_future_and_outside_window_do_not_influence_price(self):
        quote, quality = self.quote([{'t': self.sec-61, 'p': .42}, {'t': self.sec-59, 'p': .99},
                                    {'t': self.sec+1, 'p': 1}, {'t': self.sec-22000, 'p': .01}])
        self.assertEqual(quote['probability'], .42)
        self.assertEqual(timestamp(quote['available_at'], 'time'), self.t-timedelta(seconds=1))
        self.assertEqual(quality['excluded_edge_or_future_points'], 2)

    def test_empty_stale_and_boundary_latest_fail_without_older_fallback(self):
        for rows in ([], [{'t': self.sec-10801, 'p': .4}],
                     [{'t': self.sec-120, 'p': .4}, {'t': self.sec-61, 'p': 1}]):
            with self.subTest(rows=rows):self.assertIsNone(self.quote(rows)[0])

    def test_malformed_nonfinite_and_duplicate_samples_rejected(self):
        for rows in ([{'t': True, 'p': .5}], [{'t': self.sec-61, 'p': float('nan')}],
                     [{'t': self.sec-61, 'p': True}], [{'t': self.sec-61, 'p': 1.1}],
                     [{'t': self.sec-61, 'p': .4}]*2):
            with self.subTest(rows=rows), self.assertRaises(ValidationError):self.quote(rows)


if __name__ == '__main__':unittest.main()
