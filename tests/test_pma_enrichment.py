from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from foretellmesh.data import sha256_bytes, sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.pma_enrichment import build, capture, selection, token_pairs, urls_for
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash
from test_pma_data import fixture_archive


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'optional pyarrow dependency')
class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.native=self.root/'native';fixture_archive(self.native);self.markets=self.native/'polymarket/markets'
        self.archive=self.root/'capture'

    def fetch(self,url):
        rows=[]
        for source in selection(self.markets)['rows']:
            rows.append({'id':source['id'],'conditionId':source['condition_id'],'question':source['question'],
                'outcomes':source['outcomes'],'clobTokenIds':source['clob_token_ids'],'events':[{'id':'same-parent'}],
                'description':'Current description only, including a later revision','resolutionSource':'https://example.org/fixture'})
        raw=json.dumps(rows).encode()
        return {'url':url,'status':200,'sha256':sha256_bytes(raw),'started_at':'2026-09-19T00:00:00Z','completed_at':'2026-09-19T00:00:01Z'},raw

    def collected(self):
        with patch('foretellmesh.pma_enrichment.fetched',side_effect=self.fetch):capture(self.markets,self.archive)

    def test_bounded_closed_batch_and_minimum_parent_grouping(self):
        selected=selection(self.markets);urls=urls_for(selected['rows']);self.assertEqual(len(urls),1)
        self.assertIn('closed=true',urls[0]);self.assertIn('id=1&id=2',urls[0])
        self.collected();report=build(self.archive,self.root/'audit')
        self.assertEqual(report['identity_and_token_matches'],2);self.assertEqual(report['distinct_native_parent_events'],1)
        rows=[json.loads(s) for s in (self.root/'audit/identities.jsonl').read_text().splitlines()]
        self.assertTrue(all(not row['ready_for_training'] for row in rows))
        self.assertEqual(report,build(self.archive,self.root/'repeat'))

    def test_changed_body_and_incomplete_request_coverage_rejected(self):
        self.collected();path=self.archive/'raw/000.bin';original=path.read_bytes();path.write_bytes(b'[]')
        with self.assertRaises(ValidationError):build(self.archive,self.root/'bad')
        path.write_bytes(original);p=self.archive/'manifest.json';manifest=json.loads(p.read_text())
        manifest['requests']=[];manifest['requests_sha256']=canonical_hash([]);p.write_text(json_text(manifest))
        with self.assertRaises(ValidationError):build(self.archive,self.root/'missing')

    def test_identity_mismatch_and_missing_market_stay_in_report(self):
        def fetched(url):
            ref,raw=self.fetch(url);rows=json.loads(raw);rows=rows[:1];rows[0]['clobTokenIds']='["99","98"]'
            raw=json.dumps(rows).encode();ref['sha256']=sha256_bytes(raw);return ref,raw
        with patch('foretellmesh.pma_enrichment.fetched',side_effect=fetched):capture(self.markets,self.archive)
        report=build(self.archive,self.root/'audit');self.assertEqual(report['selected_markets'],2)
        self.assertEqual(report['missing_markets'],1);self.assertEqual(report['identity_and_token_matches'],0)

    def test_other_binary_labels_can_match_identity_without_becoming_yes_no(self):
        self.assertEqual(token_pairs('["Hold","Cut"]','["10","11"]'),{'Hold':'10','Cut':'11'})
        for bad in (None,'[]',[{},'11'],['10','10']):
            with self.subTest(bad=bad),self.assertRaises(ValidationError):token_pairs(['Hold','Cut'],bad)


if __name__=='__main__':unittest.main()
