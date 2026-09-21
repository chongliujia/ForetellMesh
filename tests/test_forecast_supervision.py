import json
from pathlib import Path
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.forecast_supervision import build, parse_pdf_release, parse_release, require_trainable
from foretellmesh.schema import ValidationError


def fixture():
    rows = '<tr><th>Quarterly data:</th><th>Previous</th><th>New</th></tr>'
    for q, old, new in [('2023:Q1', '47.2', '40.4'), ('2023:Q2', '49.4', '42.4'),
                        ('2023:Q3', '46.1', '44.9'), ('2023:Q4', '43.5', '40.6'),
                        ('2024:Q1', 'N.A.', '31.8')]:
        rows += f'<tr><td>{q}</td><td>{old}</td><td>{new}</td></tr>'
    return ('<h1>First Quarter 2023 Survey of Professional Forecasters</h1>'
            '<p itemprop="datePublished">10 Feb ’23</p>'
            '<h3>Risk of a Negative Quarter (%)<br />Survey Means</h3><table>'+rows+'</table>')


class ForecastSupervisionTests(unittest.TestCase):
    def test_new_column_not_previous_or_outcome(self):
        r = parse_release(fixture(), 'spf-q1-2023')
        self.assertEqual([x['probability'] for x in r['forecasts']], [.404, .424, .449, .406, .318])
        self.assertEqual(r['publication_precision'], 'day')
        self.assertEqual(r['conservative_public_availability_bound'], '2023-02-11T00:00:00-05:00')

    def test_explicit_date_variants_are_equivalent(self):
        self.assertEqual(parse_release(fixture(), 'spf-q1-2023'),
                         parse_release(fixture().replace('10 Feb ’23', 'February 10, 2023'), 'spf-q1-2023'))

    def test_wrong_target_year_cannot_be_silently_repaired(self):
        with self.assertRaisesRegex(ValidationError, 'horizon'):
            parse_release(fixture().replace('2023:Q3', '2022:Q3'), 'spf-q1-2023')

    def test_title_publication_and_source_quarter_bind(self):
        for html in (fixture().replace('First Quarter 2023', 'First Quarter 2024'),
                     fixture().replace('10 Feb ’23', '10 May ’23'), '<title>Error - 404</title>'):
            with self.assertRaises(ValidationError):
                parse_release(html, 'spf-q1-2023')

    def test_missing_out_of_bounds_and_nonfinite_probabilities_rejected(self):
        for value in ('N.A.', '101', '-1', 'NaN', 'Infinity'):
            with self.assertRaises(ValidationError):
                parse_release(fixture().replace('40.4', value), 'spf-q1-2023')

    def test_swapped_headers_or_duplicate_tables_rejected(self):
        for html in (fixture().replace('<th>Previous</th><th>New</th>', '<th>New</th><th>Previous</th>'),
                     fixture()+fixture()[fixture().index('<h3>'):]):
            with self.assertRaises(ValidationError):
                parse_release(html, 'spf-q1-2023')

    def test_navigation_changes_cannot_change_teacher(self):
        a = fixture(); b = a+'<div>Updated: 20 Sep 2026. Outcome 1.</div><script>revised=0</script>'
        self.assertEqual(parse_release(a, 'spf-q1-2023'), parse_release(b, 'spf-q1-2023'))

    def test_candidate_inventory_never_qualifies_as_training_bundle(self):
        for flag in (False, True):
            with self.assertRaises(ValidationError):
                require_trainable({'kind': 'real_forecast_target_inventory', 'ready_for_sft': flag})

    def test_heading_without_survey_means_matches_old_release(self):
        self.assertEqual(parse_release(fixture(), 'spf-q1-2023'),
                         parse_release(fixture().replace('<br />Survey Means', ''), 'spf-q1-2023'))

    def test_pdf_dates_columns_and_horizons_are_validated(self):
        pdf = ('Release Date: February 10, 2023\nFIRST QUARTER 2023\f'
               'Risk of a Negative Quarter (%)\nSurvey Means\n'
               'Quarterly data: Previous New\n2023:Q1 47.2 40.4\n2023:Q2 49.4 42.4\n'
               '2023:Q3 46.1 44.9\n2023:Q4 43.5 40.6\n2024:Q1 N.A. 31.8\n')
        parsed = parse_pdf_release(pdf, 'spf-q1-2023')
        self.assertEqual(parsed.pop('pdf_page'), 2)
        self.assertEqual(parsed, parse_release(fixture(), 'spf-q1-2023'))
        for changed in (pdf.replace('FIRST QUARTER 2023', 'SECOND QUARTER 2023'),
                        pdf.replace('2023:Q3', '2022:Q3'), pdf.replace('Previous New', 'New Previous'),
                        pdf.replace('31.8', 'NaN')):
            with self.assertRaises(ValidationError):
                parse_pdf_release(changed, 'spf-q1-2023')

    def test_source_bound_inventory_rebuild_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root/'raw'; raw.mkdir()
            config = root/'config.json'
            config.write_text(json.dumps({'collection_only': True, 'automatic_training_admission': False,
                'years': list(range(2019, 2025)), 'quarters': [1, 2, 3, 4],
                'horizons': [0, 1, 2, 3, 4], 'extract_column': 'New', 'source': 'philadelphia_fed_spf',
                'limitations': ['unit fixture']}))
            keys = [f'spf-q{q}-{y}' for y in range(2019, 2025) for q in range(1, 5)]
            captures = []
            for key in keys+['spf-faqs', 'spf-overview']:
                url = 'https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/'+key
                row = {'key': key, 'url': url, 'started_at': '2026-09-20T00:00:00Z',
                       'completed_at': '2026-09-20T00:00:01Z', 'status': 'failed', 'error': 'fixture missing'}
                if key == 'spf-q1-2023':
                    artifact = raw/(key+'.html'); artifact.write_text(fixture())
                    row.update(status='captured', final_url=url, artifact=artifact.name,
                               bytes=artifact.stat().st_size, sha256=sha256_file(artifact))
                (raw/(key+'.capture.json')).write_text(json.dumps(row)); captures.append(row)
            (raw/'manifest.json').write_text(json.dumps({'captures': captures, 'config_sha256': sha256_file(config)}))
            out = root/'out'; report = build(config, raw, out)
            self.assertEqual(report['teacher_target_count'], 5)
            self.assertEqual(report['ready_for_sft_count'], 0)
            self.assertEqual(report, build(config, raw, out))
            targets = [json.loads(line) for line in (out/'targets.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['student_input'] is None and r['outcome'] is None for r in targets))
            (raw/'spf-q1-2023.html').write_text(fixture().replace('40.4', '90.4'))
            with self.assertRaisesRegex(ValidationError, 'source bytes changed'):
                build(config, raw, root/'changed')


if __name__ == '__main__':
    unittest.main()
