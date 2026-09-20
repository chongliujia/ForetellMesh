from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import test_pma_proof_pilot as pilot
from test_macro_releases import bls_fixture
from foretellmesh.data import sha256_file
from foretellmesh.pma_bls import expected_bls_outcome, reviewed_headline, load_reviewed_bls
from foretellmesh.pma_cohort import preserve_parent
from foretellmesh.pma_proof_pilot import capture, inputs
from foretellmesh.pma_proof_replay import build, observation
from foretellmesh.schema import ValidationError, timestamp, iso
from foretellmesh.synthetic_sft import canonical_hash


def reviewed_fixture(family='cpi_yoy', period='2025-04', value='2.3'):
    e = bls_fixture(family, period, 'May 13, 2025', 'Tuesday', value)
    e['correction_search'] = e['url']+'\n'+e['url']+'\n'+'\n'.join(
        f'No matching text found for "{term}"' for term in ('correct', 'reissu'))
    d = {'family': family, 'period': period, 'disposition': 'no_correction_marker_found'}
    markers = {'url': e['url'], 'response': '\n'.join(f'No matching text found for "{term}"'
                for term in ('corrected', 'correction', 'reissued'))}
    return e, d, markers


class BlsReviewTests(unittest.TestCase):
    def test_only_current_period_headline_is_extracted(self):
        e, d, m = reviewed_fixture()
        e['response'] += '\nL300: Later tables show a revised prior-month figure of 9.9 percent.'
        r = reviewed_headline(e, d, m)
        self.assertEqual(r['value'], '2.3')
        self.assertNotIn('9.9', r['text'])
        self.assertNotIn('Later tables', r['text'])

    def test_positive_missing_or_unbound_search_does_not_clear_review(self):
        e, d, m = reviewed_fixture()
        for search in ('No matching text found for "correct"', e['correction_search'].replace('reissu', 'unrelated')):
            with self.subTest(search=search), self.assertRaisesRegex(ValidationError, 'correction search'):
                reviewed_headline({**e, 'correction_search': search}, d, m)
        e, d, m = reviewed_fixture('unemployment_u3', value='4.2')
        for markers in (None, {**m, 'url': 'https://example.com'}, {**m, 'response': 'Reissued'}):
            with self.assertRaisesRegex(ValidationError, 'correction search'):
                reviewed_headline(e, d, markers)

    def test_notice_cannot_be_hidden_by_negative_search(self):
        e, d, m = reviewed_fixture()
        e['response'] += '\nL300: CORRECTED release.'
        with self.assertRaisesRegex(ValidationError, 'unreviewed correction'):
            reviewed_headline(e, d, m)

    def test_april_notice_requires_scoped_unaffected_headline_attestation(self):
        e, d, m = reviewed_fixture('unemployment_u3', value='4.2')
        note = ('(NOTE: BLS reissued this news release on June 3, 2025, to note errors. '
                'Major labor force measures were unaffected. For more information see '
                'https://www.bls.gov/bls/errata/cps-corrections-april-2025.htm.)')
        e['response'] = e['response'].replace('L183:', 'L182: '+note+'\nL182:\nL183:')
        d['disposition'] = 'april_2025_u3_unaffected_errata'
        errata = {'url': 'https://www.bls.gov/bls/errata/cps-corrections-april-2025.htm',
                  'response': ('Major labor force measures, such as the unemployment rate were unaffected. '
                               'April release will not be updated to reflect the corrected household survey estimates.')}
        with self.assertRaisesRegex(ValidationError, 'attestation'): reviewed_headline(e, d, m)
        r = reviewed_headline(e, d, m, errata)
        self.assertEqual(r['value'], '4.2')
        self.assertNotIn('June', r['text']); self.assertNotIn('reissued', r['headline_excerpt'])
        with self.assertRaisesRegex(ValidationError, 'attestation'):
            reviewed_headline({**e, 'period': '2025-03'}, {**d, 'period': '2025-03'}, m, errata)
        with self.assertRaisesRegex(ValidationError, 'attestation'):
            reviewed_headline(e, d, m, {**errata, 'response': errata['response'].replace('unaffected', 'affected')})


class BlsPredicateTests(unittest.TestCase):
    def setUp(self):
        self.group = {'family': 'cpi_yoy', 'period': '2025-04', 'release': {'published_at': '2025-05-13T12:30:00Z'}}
        self.result = {'family': 'cpi_yoy', 'period': '2025-04', 'published_at': '2025-05-13T12:30:00Z',
                       'unit': 'all_items_cpi_u_yoy_nsa_percent', 'value': '2.3'}
        self.market = {'question': 'Will annual inflation increase by 2.3% in April?',
            'description': 'Bureau of Labor Statistics: 12 month period ending April 2025 before seasonal adjustment.'}

    def test_cpi_inclusive_brackets_use_decimal_arithmetic(self):
        for words, expected in [('2.3%', 1), ('2.4%', 0), ('2.3% or less', 1), ('2.3% or greater', 1), ('2.4% or more', 0)]:
            self.market['question'] = 'Will US annual inflation be '+words+' in April?'
            self.assertEqual(expected_bls_outcome(self.market, self.group, self.result), expected)

    def test_wrong_period_measurement_or_ambiguous_title_is_rejected(self):
        variants = [{**self.market, 'question': self.market['question'].replace('April', 'March')},
                    {**self.market, 'question': self.market['question'].replace('2.3%', '≤2.3% or less')},
                    {**self.market, 'description': self.market['description'].replace('before seasonal', 'after seasonal')},
                    {**self.market, 'description': self.market['description'].replace('April 2025', 'April 2024')}]
        for m in variants:
            with self.subTest(m=m), self.assertRaises(ValidationError): expected_bls_outcome(m, self.group, self.result)
        for change in ({'period': '2025-03'}, {'unit': 'monthly'}, {'value': '2.35'}):
            with self.assertRaises(ValidationError): expected_bls_outcome(self.market, self.group, {**self.result, **change})

    def test_specific_description_cannot_contradict_title(self):
        self.market['description'] += ' The index increased by 2.4 percent over the period.'
        with self.assertRaisesRegex(ValidationError, 'disagree'): expected_bls_outcome(self.market, self.group, self.result)

    def test_legacy_repeated_percent_unit_and_greater_mean_same_bracket(self):
        self.market['description'] += ' The index increased by 2.3% percent over the period.'
        self.assertEqual(expected_bls_outcome(self.market, self.group, self.result), 1)
        self.market['question'] = 'Will annual inflation increase by 2.3% or more in April?'
        self.market['description'] = self.market['description'].replace('percent over', 'percent or greater over')
        self.assertEqual(expected_bls_outcome(self.market, self.group, self.result), 1)

    def test_legacy_u3_comparators_and_ambiguous_between(self):
        group = {'family': 'unemployment_u3', 'period': '2025-03', 'release': self.group['release']}
        result = {**self.result, 'family': 'unemployment_u3', 'period': '2025-03', 'unit': 'us_u3_sa_percent', 'value': '4.2'}
        m = {'description': 'Bureau of Labor Statistics seasonally adjusted U-3 for March 2025.'}
        for title in ['Will the March unemployment rate be 4.2% or higher?',
                      'Will the unemployment rate for March be exactly 4.2%?',
                      'Will US unemployment be 4.2% or lower in March 2025?',
                      'Will the March 2025 unemployment rate be ≥4.2%?']:
            self.assertEqual(expected_bls_outcome({**m, 'question': title}, group, result), 1)
        for title in ['Will the March 2025 unemployment rate be between 4.2%?',
                      'Will the March 2025 unemployment rate be between ≥4.2%?']:
            with self.assertRaisesRegex(ValidationError, 'ambiguous'):
                expected_bls_outcome({**m, 'question': title}, group, result)


class BlsCaptureReplayTests(unittest.TestCase):
    def setUp(self):
        f = pilot.PmaProofPilotTests(methodName='runTest'); f.setUp(); self.addCleanup(f.doCleanups); self.f = f
        root = f.root; sources = f.plan['sources']
        f.market.update(question='Will annual inflation increase by 2.3% in April?',
            description='Bureau of Labor Statistics: 12 month period ending April 2025 before seasonal adjustment.')
        f.ancillary = 'q: title: '+f.market['question']+', description: '+f.market['description']+f.ancillary.split(pilot.RULES)[1]
        bls = root/'bls'; bls.mkdir(); refs = []
        for period, date, weekday, value in [('2025-03', 'April 10, 2025', 'Thursday', '2.4'), ('2025-04', 'May 13, 2025', 'Tuesday', '2.3')]:
            e, _, _ = reviewed_fixture(period=period, value=value)
            dated = bls_fixture('cpi_yoy', period, date, weekday, value)
            dated['correction_search'] = e['correction_search'].replace(e['url'], dated['url'])
            p = bls/(period+'.json'); f.dump(p, dated)
            refs.append({'file': p.name, 'sha256': sha256_file(p), 'family': 'cpi_yoy', 'period': period, 'url': dated['url']})
        f.dump(bls/'errata.json', {})
        f.dump(bls/'manifest.json', {'schema_version': '1', 'kind': 'reviewed_bls_headline_captures', 'refs': refs,
            'refs_sha256': canonical_hash(refs), 'errata': {'file': 'errata.json', 'sha256': sha256_file(bls/'errata.json')}})
        f.dump(root/'bls_policy.json', {'schema_version': '1', 'review_id': 'pma_bls_headline_v1',
            'archive_manifest_sha256': sha256_file(bls/'manifest.json'), 'historical_tables_allowed': False,
            'whole_release_model_evidence_allowed': False, 'allowed_fields': ['headline_cpi_yoy_nsa', 'headline_unemployment_u3_sa'],
            'trust_model': 'dated_official_headline_and_indexed_correction_search_not_first_fetch_snapshot',
            'releases': [{'family': r['family'], 'period': r['period'], 'capture_sha256': r['sha256'],
                          'disposition': 'no_correction_marker_found', 'rationale': 'synthetic fixture'} for r in refs]})
        sources.update({'bls_archive': {'path': 'bls/manifest.json'}, 'bls_policy': {'path': 'bls_policy.json'}})
        f.group.update(family='cpi_yoy', period='2025-04', event_group_id='macro:us:cpi_yoy:2025-04',
            release={'url': refs[1]['url'], 'period': '2025-04', 'published_at': '2025-05-13T12:30:00Z'},
            prior_release={'url': refs[0]['url'], 'period': '2025-03', 'published_at': '2025-04-10T12:30:00Z'})
        f.state['last_update_timestamp'] = str(int(timestamp(f.group['release']['published_at'], 'release').timestamp())+7200)
        f.dump(root/'native/raw.bin', [f.market]); manifest = json.loads((root/'native/manifest.json').read_text())
        manifest['requests'][0]['sha256'] = sha256_file(root/'native/raw.bin')
        manifest['requests_sha256'] = canonical_hash(manifest['requests']); f.dump(root/'native/manifest.json', manifest)
        candidate = json.loads((root/'catalog/macro_candidates.jsonl').read_text()); candidate['question'] = f.market['question']
        f.dump(root/'catalog/macro_candidates.jsonl', candidate)
        f.dump(root/'catalog/report.json', {'artifact_hashes': {'macro_candidates.jsonl': sha256_file(root/'catalog/macro_candidates.jsonl')}})
        with sqlite3.connect(root/'prices/prices.sqlite') as db:
            for n, days in enumerate((7, 1)):
                epoch = int((timestamp(f.group['release']['published_at'], 'release')-timedelta(days=days, hours=1)).timestamp())
                db.execute('UPDATE blocks SET unix_time=? WHERE block_number=?', (epoch, n))
        f.dump(root/'prices/report.json', {'catalog_report_sha256': sha256_file(root/'catalog/report.json'),
            'artifact_hashes': {'prices.sqlite': sha256_file(root/'prices/prices.sqlite')}})
        for source in sources.values(): source['sha256'] = sha256_file(root/source['path'])
        f.plan['purpose'] = 'pma_bls_historical_proof_pilot'; f.plan['groups'] = [f.group]; f.dump(f.config, f.plan)
        self.capture = root/'bls_capture'
        with patch('foretellmesh.pma_proof_pilot.fetched', side_effect=f.fetch), patch('foretellmesh.polygon_capture.urlopen', side_effect=f.rpc), patch('builtins.print'):
            capture(f.config, self.capture, 'https://polygon.drpc.org')

    def test_raw_proofs_to_bls_records_are_deterministic_and_label_isolated(self):
        f = self.f
        a, b = f.root/'bls_a', f.root/'bls_b'
        first, second = build(f.config, self.capture, a), build(f.config, self.capture, b)
        self.assertEqual(first, second)
        self.assertEqual(first['proof_complete_observations'], 2)
        for row in map(json.loads, (a/'review_inputs.jsonl').read_text().splitlines()):
            text = json.dumps(row['input'])
            self.assertNotIn('resolution_time', text); self.assertNotIn('2.3%', row['input']['evidence'][0]['text'])
            self.assertIn('2.4%', row['input']['evidence'][0]['text'])
            self.assertTrue(row['input']['evidence'][0]['evidence_id'].startswith('bls:'))

    def test_changed_bls_capture_cannot_reuse_review(self):
        f = self.f; p = f.root/'bls/2025-03.json'; d = json.loads(p.read_text()); d['response'] += 'tamper'; f.dump(p, d)
        with self.assertRaisesRegex(ValidationError, 'artifact changed'):
            load_reviewed_bls(f.root/'bls/manifest.json', f.root/'bls_policy.json')

    def test_wrong_release_family_url_is_rejected(self):
        f = self.f; f.plan['groups'][0]['release']['url'] = f.group['release']['url'].replace('cpi_', 'empsit_'); f.dump(f.config, f.plan)
        with self.assertRaisesRegex(ValidationError, 'release URL'): inputs(f.config)

    def test_additive_parent_cannot_change_or_move_frozen_samples(self):
        f = self.f; out = f.root/'replayed'; build(f.config, self.capture, out)
        rows = [json.loads(line)['record'] for line in (out/'candidates.jsonl').read_text().splitlines()]
        parent = f.root/'parent'; (parent/'partitions').mkdir(parents=True); hashes = {}
        for split in ('train', 'validation', 'test'):
            p = parent/'partitions'/f'{split}.records.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in (rows if split == 'validation' else [])))
            hashes[str(p.relative_to(parent))] = sha256_file(p)
        f.dump(parent/'report.json', {'artifact_hashes': hashes}); digest = sha256_file(parent/'report.json')
        parts = {'train': [], 'validation': deepcopy(rows), 'test': []}
        self.assertEqual(preserve_parent(parts, parent, digest), 2)
        parts['validation'][0]['dataset_version'] = 'a-new-version'
        self.assertEqual(preserve_parent(parts, parent, digest), 2)
        parts['validation'][0]['market']['probability'] = 0.1
        with self.assertRaisesRegex(ValidationError, 'changed'): preserve_parent(parts, parent, digest)
        parts = {'train': rows, 'validation': [], 'test': []}
        with self.assertRaisesRegex(ValidationError, 'reassigned'): preserve_parent(parts, parent, digest)


if __name__ == '__main__': unittest.main()
