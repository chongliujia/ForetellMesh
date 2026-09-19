from copy import deepcopy
import unittest

from foretellmesh.macro_releases import bls_headline, fed_headline, prior_release
from foretellmesh.schema import ValidationError, timestamp
from test_historical_market import official_html


def bls_fixture(family='cpi_yoy', period='2026-06', date='July 14, 2026', day='Tuesday', value='3.5'):
    from datetime import datetime
    dt = datetime.strptime(date, '%B %d, %Y'); month = datetime.strptime(period, '%Y-%m').strftime('%B %Y')
    kind = 'cpi' if family == 'cpi_yoy' else 'empsit'
    url = 'https://www.bls.gov/news.release/archives/'+kind+'_'+dt.strftime('%m%d%Y')+'.htm'
    title = ('CONSUMER PRICE INDEX' if kind == 'cpi' else 'THE EMPLOYMENT SITUATION')+' -- '+month
    headline = (f'Over the last 12 months, the all items index increased {value} percent before seasonal adjustment.'
                if kind == 'cpi' else f'Total nonfarm employment rose, and the unemployment rate changed little at {value} percent.')
    text = (url+'\nL175: Transmission of material in this release is embargoed until\n'
            f'L176: 8:30 a.m. (ET) {day}, {date}\nL177: \nL181: {title}\nL182: \n'
            f'L183: {headline}\nL184: \nL185: Later tables discuss prior-month revisions and a future release schedule.')
    return {'url': url, 'family': family, 'period': period, 'expected_time': dt.strftime('%Y-%m-%d')+'T12:30:00Z',
            'capture_method': 'web_tool_extracted_text', 'started_at': '2026-09-19T00:00:00Z',
            'completed_at': '2026-09-19T00:00:01Z', 'response': text}


class MacroReleaseTests(unittest.TestCase):
    def test_cpi_and_unemployment_headlines_use_exact_vintage_and_units(self):
        r = bls_headline(bls_fixture()); self.assertEqual(r['value'], '3.5')
        self.assertEqual(r['unit'], 'all_items_cpi_u_yoy_nsa_percent')
        d = bls_fixture('unemployment_u3', value='4.2')
        d['response'] += '\nL190: Later revision: unemployment rate was 9.9 percent.'
        r = bls_headline(d); self.assertEqual(r['value'], '4.2')
        self.assertNotIn('9.9', r['text']); self.assertNotIn('Later tables', r['text'])

    def test_timezone_is_calendar_aware_not_a_fixed_utc_offset(self):
        d = bls_fixture(period='2026-01', date='February 13, 2026', day='Friday')
        d['expected_time'] = '2026-02-13T13:30:00Z'
        self.assertEqual(timestamp(bls_headline(d)['published_at'], 'time'), timestamp(d['expected_time'], 'time'))
        d['expected_time'] = '2026-02-13T12:30:00Z'
        with self.assertRaisesRegex(ValidationError, 'planned time'):bls_headline(d)

    def test_wrong_period_correction_date_clock_or_url_fail_closed(self):
        base = bls_fixture()
        variants = [{**base, 'period': '2026-07'}, {**base, 'completed_at': '2025-01-01T00:00:00Z'},
                    {**base, 'url': base['url'].replace('07142026', '07152026')},
                    {**base, 'response': base['response'].replace('L181:', 'L180: CORRECTION\nL181:')},
                    {**base, 'response': base['response'].replace('Tuesday', 'Friday')}]
        for d in variants:
            with self.subTest(variant=d), self.assertRaises(ValidationError):bls_headline(d)

    def test_wrong_cpi_measurement_is_not_silently_used(self):
        d = bls_fixture(); d['response'] = d['response'].replace('before seasonal adjustment', 'on a seasonally adjusted basis')
        with self.assertRaisesRegex(ValidationError, 'annual CPI'):bls_headline(d)

    def test_release_at_or_before_cutoff_only(self):
        early = bls_headline(bls_fixture(period='2026-05', date='June 10, 2026', day='Wednesday', value='4.2'))
        later = bls_headline(bls_fixture())
        self.assertEqual(prior_release([later, early], 'cpi_yoy', '2026-07-13T12:30:00Z'), early)
        self.assertEqual(prior_release([later, early], 'cpi_yoy', '2026-07-14T12:30:00Z'), later)
        self.assertIsNone(prior_release([later], 'cpi_yoy', '2026-07-13T12:30:00Z'))

    def test_fed_unicode_fraction_preserves_exact_target_value(self):
        raw = official_html('June 17, 2026', 'The Committee maintained the target range for the federal funds rate at 3‑1/2 to 3‑3/4 percent.')
        r = fed_headline(raw, 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20260617a.htm', '2026-06-17T18:00:00Z')
        self.assertEqual(r['value'], '3.75'); self.assertIn('3‑3/4', r['text'])


if __name__ == '__main__':unittest.main()
