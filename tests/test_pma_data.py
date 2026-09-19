from copy import deepcopy
from datetime import datetime, timezone
import json
import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

from foretellmesh.data import sha256_file
from foretellmesh.evaluation import json_text
from foretellmesh.macro_discovery import poly_inventory
from foretellmesh.pma_catalog import build as catalog_build, market_record
from foretellmesh.pma_trades import build as trade_build, insert_block, insert_trade, prepare_db, quote_at, trade_price
from foretellmesh.schema import ValidationError
from foretellmesh.synthetic_sft import canonical_hash


def trade(token='10', amount=400000, tx='a', block=1, index=0):
    return {'block_number':block,'transaction_hash':tx*64,'log_index':index,'maker_asset_id':'0','taker_asset_id':token,
            'maker_amount':amount,'taker_amount':1000000,'fee':0,'_contract':'CTF Exchange'}


def market(mid='1', question='Will annual inflation exceed 3%?', closed=True):
    return {'id':mid,'condition_id':'0x'+str(mid)*64,'question':question,'outcomes':'["Yes","No"]',
            'outcome_prices':'["1","0"]','clob_token_ids':json.dumps([str(int(mid)*10),str(int(mid)*10+1)]),
            'closed':closed,'created_at':datetime(2024,1,1,tzinfo=timezone.utc),
            'end_date':datetime(2024,2,1,tzinfo=timezone.utc),'_fetched_at':datetime(2026,2,3)}


def fixture_archive(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root.mkdir();entries=[]
    content={'markets/m.parquet':[market(),market('2','Will the Fed cut interest rates?'),market('3','Will a team win?')],
             'trades/t.parquet':[trade(),trade(amount=600000,tx='b'),trade(amount=990000,tx='c',block=2),trade(token='30',tx='d')],
             'trades/duplicate.parquet':[trade()],
             'blocks/b.parquet':[{'block_number':1,'timestamp':'2024-01-02T00:00:00Z'},
                                  {'block_number':2,'timestamp':'2024-01-04T00:00:00Z'}]}
    for relative,rows in content.items():
        path=root/'polymarket'/relative;path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist(rows),path)
        entries.append({'name':'data/polymarket/'+relative,'file':'polymarket/'+relative,'sha256':sha256_file(path),
                        'bytes':path.stat().st_size,'selected':True})
    (root/'members.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in entries))
    manifest={'schema_version':'1','kind':'pma_selective_extraction','archive_sha256':'a'*64,
              'member_ledger_sha256':sha256_file(root/'members.jsonl'),'selected':entries}
    (root/'manifest.json').write_text(json_text(manifest))


class NativePriceTests(unittest.TestCase):
    def setUp(self):
        self.tokens={'10':('m','Yes'),'11':('m','No')}
        self.db=sqlite3.connect(':memory:');self.addCleanup(self.db.close);prepare_db(self.db)

    def test_cash_direction_and_no_outcome_convert_to_same_yes_probability(self):
        a=trade_price(trade(),self.tokens)
        b=trade(token='11',amount=600000);b['maker_asset_id'],b['taker_asset_id']='11','0'
        b['maker_amount'],b['taker_amount']=1000000,600000;b=trade_price(b,self.tokens)
        self.assertEqual((a['yes_price_numerator'],a['yes_price_denominator']),('2','5'))
        self.assertEqual((b['yes_price_numerator'],b['yes_price_denominator']),('2','5'))

    def test_huge_uint256_ids_stay_exact_and_unsupported_fills_fail(self):
        token=str(2**255+123);p=trade_price(trade(token=token),{token:('m','Yes')});self.assertEqual(p['token_id'],token)
        for change in [{'maker_asset_id':'11'},{'maker_amount':True},{'taker_amount':0},{'maker_amount':1000000},
                       {'_contract':'unknown'},{'_contract':'NegRisk CTF Exchange v2'},{'maker_asset_id':'0.0'}]:
            row=trade();row.update(change)
            with self.subTest(change=change),self.assertRaises(ValidationError):trade_price(row,self.tokens)

    def test_native_negrisk_exchange_name_and_collateral_units(self):
        # Values and exchange spelling sampled from the pinned native archive.
        row=trade(token='11',amount=32125680);row.update(taker_amount=33360000,_contract='NegRisk CTF Exchange')
        parsed=trade_price(row,self.tokens)
        self.assertEqual((parsed['yes_price_numerator'],parsed['yes_price_denominator']),('37','1000'))
        self.assertEqual(parsed['exchange'],'NegRisk CTF Exchange')

    def test_dedup_conflicts_and_future_prices_cannot_change_historical_quote(self):
        parsed=trade_price(trade(),self.tokens);self.assertTrue(insert_trade(self.db,parsed,'a',0))
        self.assertFalse(insert_trade(self.db,parsed,'b',0))
        with self.assertRaisesRegex(ValidationError,'conflicting'):insert_trade(self.db,trade_price(trade(amount=900000),self.tokens),'b',0)
        insert_block(self.db,1,'2024-01-02T00:00:00Z','block',0)
        before,_=quote_at(self.db,'m','2024-01-02T00:01:00Z',120)
        insert_trade(self.db,trade_price(trade(amount=990000,tx='b',block=2),self.tokens),'a',1)
        insert_block(self.db,2,'2024-01-03T00:00:00Z','block',1)
        after,_=quote_at(self.db,'m','2024-01-02T00:01:00Z',120);self.assertEqual(before,after)
        self.assertEqual(before['probability'],.4)
        self.assertIsNone(quote_at(self.db,'m','2024-01-02T00:03:00Z',120)[0])

    def test_missing_naive_and_conflicting_block_times_fail_closed(self):
        insert_trade(self.db,trade_price(trade(),self.tokens),'trade',0)
        self.assertIsNone(quote_at(self.db,'m','2024-01-02T00:00:00Z',120)[0])
        with self.assertRaises(ValidationError):insert_block(self.db,1,'2024-01-01T00:00:00','b',0)
        insert_block(self.db,1,'2024-01-01T00:00:00Z','b',0)
        with self.assertRaises(ValidationError):insert_block(self.db,1,'2024-01-01T00:00:01Z','b',1)

    def test_progress_reader_cannot_block_writer_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'prices.sqlite'
            writer=sqlite3.connect(path,timeout=.01);reader=sqlite3.connect(path,timeout=.01)
            try:
                prepare_db(writer);insert_trade(writer,trade_price(trade(),self.tokens),'t',0);writer.commit()
                reader.execute('BEGIN')
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM trades').fetchone()[0],1)
                insert_trade(writer,trade_price(trade(tx='b'),self.tokens),'t',1);writer.commit()
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM trades').fetchone()[0],1)
                reader.commit()
                self.assertEqual(writer.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0],0)
                self.assertEqual(reader.execute('SELECT COUNT(*) FROM trades').fetchone()[0],2)
            finally:reader.close();writer.close()


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'optional pyarrow dependency')
class ArchiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.native=self.root/'native';fixture_archive(self.native)
        self.index=self.root/'heldout.json';entries=[{'event_id':'polymarket:0x'+'2'*64,'question':'heldout equivalent'}]
        self.index.write_text(json_text({'schema_version':'1','source':'forecastbench','entries':entries,'entries_sha256':canonical_hash(entries)}))
        # Exercise the existing inventory producer's real output schema, rather
        # than hand-copying a shape that can diverge from the application.
        native=poly_inventory({'family':'fomc','period':'2024-01','calendar_reference_time':'2024-01-31T19:00:00Z'},
            {'id':'event-1','markets':[{'id':'1','conditionId':'0x'+'1'*64,
                'question':'Will the Fed cut in January 2024?','closed':True}]},None,{}, {'entries':[]})
        self.reserved=self.root/'reserved.json';self.reserved.write_text(json_text({'groups':[{'event_group_id':'reserved-macro',
            'native_events':[native]}]}))

    def test_full_native_catalog_trade_time_join_and_result_isolation(self):
        catalog=self.root/'catalog';r=catalog_build(self.native,self.index,self.reserved,catalog)
        self.assertEqual(r['counts']['markets_rows'],3);self.assertEqual(r['counts']['closed_macro_keyword_candidates'],2)
        self.assertEqual(r['counts']['exact_reserved_markets'],2)
        rows=[json.loads(line) for line in (catalog/'macro_candidates.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertNotIn('outcome_prices',row);self.assertNotIn('label',row);self.assertFalse(row['ready_for_training'])
            self.assertIn('reserved_benchmark_identity_or_exact_question',row['blockers'])
        store=self.root/'store';report=trade_build(self.native,catalog,store)
        self.assertEqual(report['counts']['native_trade_rows_scanned'],5)
        self.assertEqual(report['counts']['unique_valid_selected_trades'],3)
        self.assertEqual(report['counts']['identical_duplicate_trades'],1)
        self.assertEqual(report['counts']['trades_with_block_time'],3)
        self.assertFalse((store/'prices.sqlite-wal').exists())
        with sqlite3.connect(store/'prices.sqlite') as db:
            quote,quality=quote_at(db,'1','2024-01-02T00:01:00Z',120)
        self.assertEqual(quote['probability'],.5);self.assertEqual(quality['trades_in_block'],2)
        catalog2=self.root/'catalog2';self.assertEqual(r,catalog_build(self.native,self.index,self.reserved,catalog2))
        self.assertEqual(report,trade_build(self.native,catalog2,self.root/'store2'))

    def test_shard_tampering_is_rejected_before_catalog_admission(self):
        (self.native/'polymarket/markets/m.parquet').write_bytes(b'tampered')
        with self.assertRaises(ValidationError):catalog_build(self.native,self.index,self.reserved,self.root/'bad')

    def test_terminal_prices_do_not_enter_market_projection(self):
        a=market();b=deepcopy(a);b['outcome_prices']='["0","1"]'
        ref={'file':'fixture.parquet','sha256':'a'*64}
        self.assertEqual(market_record(a,ref,0,{},{}),market_record(b,ref,0,{},{}))


if __name__=='__main__':unittest.main()
