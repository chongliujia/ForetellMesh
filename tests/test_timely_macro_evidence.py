from copy import deepcopy
from datetime import datetime, timezone
import unittest

from foretellmesh.schema import ValidationError, timestamp
from foretellmesh.timely_macro_evidence import parse_capture, select_facts


def fixture(kind='empsit', headline=None):
    day = 'June 2, 2023' if kind == 'empsit' else 'June 13, 2023'
    weekday = 'Friday' if kind == 'empsit' else 'Tuesday'
    digits = '06022023' if kind == 'empsit' else '06132023'
    title = 'THE EMPLOYMENT SITUATION -- MAY 2023' if kind == 'empsit' else 'CONSUMER PRICE INDEX - MAY 2023'
    headline = headline or ('Total nonfarm payroll employment rose by 339,000 in May, and the unemployment rate rose 0.3 percentage point to 3.7 percent. This news release presents statistics from two monthly surveys.' if kind == 'empsit' else
        'Over the last 12 months, the all items index increased 4.0 percent before seasonal adjustment. The all items less food and energy index rose 5.3 percent over the last 12 months. Table A. Revised historical table values: 9.9 999,000')
    url = f'https://www.bls.gov/news.release/archives/{kind}_{digits}.htm'
    header = f'Employment Situation ({url})\nSource: open({{"ref_id":"{url}"}})\n'
    response = header+f'Transmission of material\n8:30 a.m. (ET) {weekday}, {day}\n'+title+'\n'+headline
    for marker in ('corrected', 'correction', 'reissued'):
        response += '-'*80+f'\nEmployment Situation ({url})\nSource: find({{"ref_id":"{url}","pattern":"{marker}"}})\nNo matching text found for "{marker}"'
    spec = {'key': kind+'_'+digits, 'url': url, 'family': kind,
            'expected_publication': '2023-06-02T12:30:00Z' if kind == 'empsit' else '2023-06-13T12:30:00Z'}
    capture = {**spec, 'response': response, 'status': 'fulfilled', 'capture_method': 'web_tool_extracted_text',
               'started_at': '2026-09-20T01:00:00Z', 'completed_at': '2026-09-20T01:00:01Z'}
    return capture, spec


class TimelyEvidenceTests(unittest.TestCase):
    def test_rate_level_is_not_percentage_point_change(self):
        result = parse_capture(*fixture())
        self.assertEqual(result['values']['unemployment_u3_sa'], '3.7')
        self.assertEqual(result['values']['nonfarm_payroll_change_sa'], '339000')

    def test_payroll_unchanged_wording_keeps_numeric_change(self):
        text = 'Total nonfarm payroll employment was essentially unchanged in May (+12,000), and the unemployment rate was unchanged at 4.1 percent. This news release presents statistics from two monthly surveys.'
        r = parse_capture(*fixture(headline=text))
        self.assertEqual(int(r['values']['nonfarm_payroll_change_sa']), 12000)

    def test_cpi_table_mutations_do_not_change_narrative_facts(self):
        a, s = fixture('cpi'); b = deepcopy(a)
        b['response'] = b['response'].replace('9.9 999,000', '0.1 future outcome 1')
        self.assertEqual(parse_capture(a, s), parse_capture(b, s))
        self.assertEqual(parse_capture(a, s)['values']['cpi_core_yoy_nsa'], '5.3')

    def test_unknown_correction_date_or_source_mismatch_rejected(self):
        a, s = fixture()
        for changed in (a['response'].replace('No matching text found for "corrected"', 'A corrected release.'),
                        a['response'].replace('Transmission of material', 'Transmission of material CORRECTED release'),
                        a['response'].replace('June 2, 2023', 'June 3, 2023'),
                        a['response'].replace('MAY 2023', 'APRIL 2023')):
            with self.assertRaises(ValidationError): parse_capture({**a, 'response': changed}, s)

    def test_same_day_reissue_is_unavailable_until_next_midnight(self):
        a, s = fixture('cpi'); old = '06132023'; new = '07112024'
        a['response'] = a['response'].replace(old, new).replace('Tuesday, June 13, 2023', 'Thursday, July 11, 2024').replace('MAY 2023', 'JUNE 2024')
        note = 'This news release was reissued on July 11, 2024. These data have been removed from tables 2, 6, and 7.'
        a['response'] = a['response'].replace('Over the last 12 months', note+' Over the last 12 months', 1).replace('No matching text found for "reissued"', note)
        for r in (a, s):
            r['key'] = 'cpi_'+new; r['url'] = r['url'].replace(old, new)
            r['expected_publication'] = '2024-07-11T12:30:00Z'
        r = parse_capture(a, s)
        self.assertEqual(r['evidence']['available_at'], '2024-07-12T04:00:00Z')
        self.assertEqual(select_facts([r], timestamp('2024-07-11T18:00:00Z', 'cutoff')), [])
        self.assertEqual(len(select_facts([r], timestamp('2024-07-12T04:00:00Z', 'cutoff'))), 1)

    def test_future_release_and_delayed_availability_cannot_change_selection(self):
        old = parse_capture(*fixture()); future = deepcopy(old)
        future['evidence'].update(published_at='2023-07-01T12:30:00Z', available_at='2023-07-01T12:30:00Z')
        cutoff = datetime(2023, 6, 5, tzinfo=timezone.utc)
        self.assertEqual(select_facts([old, future], cutoff), [old])
        future['values'] = {'outcome': 1}
        self.assertEqual(select_facts([old, future], cutoff), [old])
        future['evidence']['published_at'] = '2023-06-03T12:30:00Z'
        self.assertEqual(select_facts([old, future], cutoff), [old])


if __name__ == '__main__': unittest.main()
