"""Reconstruct selected PMA trade prices and join native Polygon block times.

Read Parquet in batches; keep selected trades in SQLite. No snapshot close
price, end date or fetch time substitutes for a historical transaction time.
This layer produces a historical price store, not supervised forecast targets.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
import sqlite3

from .data import sha256_file, strict_json
from .evaluation import json_text
from .pma_catalog import extraction_entries, write_row
from .schema import ValidationError, iso, timestamp
from .synthetic_sft import canonical_hash


SQLITE_CACHE_MIB = 1024


def integer(value, context):
    if type(value) is int and value >= 0:return value
    if isinstance(value,str) and re.fullmatch(r'0|[1-9][0-9]*',value):return int(value)
    raise ValidationError('invalid nonnegative integer '+context)


def trade_price(row, token_map):
    maker, taker = str(integer(row['maker_asset_id'],'maker asset')), str(integer(row['taker_asset_id'],'taker asset'))
    if (maker == '0') == (taker == '0'):raise ValidationError('unsupported token-token or collateral-collateral fill')
    token = taker if maker == '0' else maker
    if token not in token_map:raise ValidationError('unmapped outcome token')
    # The archive uses the full name; upstream SCHEMAS.md abbreviates it.
    if row['_contract'] not in ('CTF Exchange','NegRisk','NegRisk CTF Exchange'):
        raise ValidationError('unsupported exchange contract')
    collateral = integer(row['maker_amount'] if maker == '0' else row['taker_amount'],'collateral amount')
    shares = integer(row['taker_amount'] if maker == '0' else row['maker_amount'],'share amount')
    fee = integer(row['fee'],'fee')
    if not shares:raise ValidationError('zero outcome token amount')
    p = Fraction(collateral,shares)
    if not 0 < p < 1:raise ValidationError('boundary or nonunit trade price')
    market, outcome = token_map[token]
    if outcome == 'No':p = 1-p
    if outcome not in ('Yes','No'):raise ValidationError('nonbinary token outcome')
    if not isinstance(row['transaction_hash'],str):raise ValidationError('invalid trade transaction hash')
    tx = row['transaction_hash'].removeprefix('0x')
    if not re.fullmatch(r'[0-9a-fA-F]{64}',tx):raise ValidationError('invalid trade transaction hash')
    return {'transaction_hash':'0x'+tx.lower(), 'log_index':integer(row['log_index'],'log index'),
            'block_number':integer(row['block_number'],'block number'), 'market_id':market, 'token_id':token,
            'token_outcome':outcome,'yes_price_numerator':str(p.numerator),'yes_price_denominator':str(p.denominator),
            'collateral_amount':str(collateral),'share_amount':str(shares),'fee':str(fee),'exchange':row['_contract']}


def prepare_db(connection):
    # Progress readers must not prevent an ingest transaction from committing.
    connection.execute('PRAGMA journal_mode=WAL')
    # Bulk loading otherwise rewrites the growing random-key indexes at almost
    # every shard commit. Publish only after the final verified checkpoint.
    connection.execute('PRAGMA wal_autocheckpoint=0')
    connection.execute('PRAGMA busy_timeout=60000')
    connection.execute(f'PRAGMA cache_size=-{SQLITE_CACHE_MIB*1024}')
    connection.executescript('''
        CREATE TABLE trades (tx TEXT, log_index INTEGER, block_number INTEGER, market_id TEXT,
            numerator TEXT, denominator TEXT, payload_hash TEXT, source_file TEXT, source_row INTEGER,
            PRIMARY KEY(tx,log_index));
        CREATE TABLE blocks (block_number INTEGER PRIMARY KEY, unix_time INTEGER, source_file TEXT, source_row INTEGER);
        CREATE INDEX trade_market_block ON trades(market_id,block_number);
    ''')


def insert_trade(connection, parsed, file, position):
    digest=canonical_hash(parsed)
    inserted=connection.execute('INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?)',
        (parsed['transaction_hash'],parsed['log_index'],parsed['block_number'],parsed['market_id'],
         parsed['yes_price_numerator'],parsed['yes_price_denominator'],digest,file,position))
    if inserted.rowcount:return True
    prior=connection.execute('SELECT payload_hash FROM trades WHERE tx=? AND log_index=?',
                             (parsed['transaction_hash'],parsed['log_index'])).fetchone()
    if prior[0]!=digest:raise ValidationError('conflicting duplicate native trade log')
    return False


def insert_block(connection, number, value, file, position):
    number=integer(number,'block number'); dt=timestamp(value,'native block timestamp')
    if dt.microsecond:raise ValidationError('noninteger block second')
    seconds=int(dt.timestamp());old=connection.execute('SELECT unix_time FROM blocks WHERE block_number=?',(number,)).fetchone()
    if old:
        if old[0]!=seconds:raise ValidationError('conflicting duplicate native block time')
        return
    connection.execute('INSERT INTO blocks VALUES (?,?,?,?)',(number,seconds,file,position))


def quote_at(connection, market_id, observation, max_age_seconds):
    """Use the latest known pre-cutoff block; average prices within that block.

This is an explicitly defined block mean, not an executable bid/ask quote or
claim about within-block transaction ordering. Missing block times fail closed.
"""
    cutoff=int(timestamp(observation,'observation').timestamp())
    if type(max_age_seconds) is not int or max_age_seconds < 0:raise ValidationError('invalid quote freshness')
    missing=connection.execute('SELECT COUNT(*) FROM trades t LEFT JOIN blocks b USING(block_number) WHERE t.market_id=? AND b.block_number IS NULL',
                               (market_id,)).fetchone()[0]
    if missing:return None,{'reason':'selected_market_has_unmapped_block_times','missing_trades':missing}
    latest=connection.execute('SELECT t.block_number,b.unix_time FROM trades t JOIN blocks b USING(block_number) '
        'WHERE t.market_id=? AND b.unix_time<=? ORDER BY b.unix_time DESC,t.block_number DESC LIMIT 1',(market_id,cutoff)).fetchone()
    if latest is None:return None,{'reason':'no_pre_cutoff_trade'}
    block,seconds=latest
    if cutoff-seconds>max_age_seconds:return None,{'reason':'stale_trade','age_seconds':cutoff-seconds}
    rows=connection.execute('SELECT numerator,denominator FROM trades WHERE market_id=? AND block_number=?',(market_id,block)).fetchall()
    price=sum((Fraction(int(n),int(d)) for n,d in rows),Fraction())/len(rows)
    return {'probability':float(price),'block_number':block,'unix_time':seconds}, {'kind':'last_pre_cutoff_block_mean_trade_price',
        'trades_in_block':len(rows),'age_seconds':cutoff-seconds,'is_executable_quote':False}


def build(extraction, catalog, output):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    if output.exists():raise ValidationError('PMA trade store exists')
    manifest,entries=extraction_entries(extraction); report=strict_json((catalog/'report.json').read_text())
    if (report['extraction_manifest_sha256']!=sha256_file(extraction/'manifest.json')
            or report['artifact_hashes']['macro_candidates.jsonl']!=sha256_file(catalog/'macro_candidates.jsonl')
            or report['artifact_hashes']['shards.jsonl']!=sha256_file(catalog/'shards.jsonl')):
        raise ValidationError('PMA catalog binding changed')
    markets=[strict_json(line) for line in (catalog/'macro_candidates.jsonl').read_text().splitlines()]
    if not 1<=len(markets)<=2000:raise ValidationError('unbounded macro trade selection')
    token_map={}
    for market in markets:
        for outcome,token in (market['tokens'] or {}).items():
            owner=market['market_id'],outcome
            if token in token_map and token_map[token]!=owner:raise ValidationError('outcome token maps to multiple markets')
            token_map[token]=owner
    token_set=pa.array(sorted(token_map),type=pa.string());counts=Counter();issues=Counter();block_issues=Counter()
    output.mkdir(parents=True);connection=sqlite3.connect(output/'prices.sqlite',timeout=60);prepare_db(connection)
    columns=['block_number','transaction_hash','log_index','maker_asset_id','taker_asset_id','maker_amount','taker_amount','fee','_contract']
    infos={row['file']:row for row in map(strict_json,(catalog/'shards.jsonl').read_text().splitlines())}
    try:
        with (output/'trade_issues.jsonl').open('w') as rejected:
            # The hash-bound ledger preserves deterministic extraction order.
            # Follow that disk layout instead of seeking across 40k shards in
            # filename order; duplicate conflicts remain order-independent.
            for entry in entries:
                if not entry['file'].startswith('polymarket/trades/'):continue
                path=extraction/entry['file']
                if sha256_file(path)!=entry['sha256']:raise ValidationError('trade shard changed')
                parquet=pq.ParquetFile(path);offset=0
                for batch in parquet.iter_batches(batch_size=65536,columns=columns):
                    counts['native_trade_rows_scanned']+=batch.num_rows
                    mask=pc.or_(pc.is_in(batch['maker_asset_id'],value_set=token_set),pc.is_in(batch['taker_asset_id'],value_set=token_set))
                    positions=pc.indices_nonzero(mask).to_pylist();filtered=batch.filter(mask).to_pylist()
                    for position,row in zip(positions,filtered):
                        counts['selected_raw_trade_rows']+=1
                        try:parsed=trade_price(row,token_map)
                        except ValidationError as exc:
                            issues[str(exc)]+=1;write_row(rejected,{'file':entry['file'],'row':offset+position,'reason':str(exc)});continue
                        if insert_trade(connection,parsed,entry['file'],offset+position):counts['unique_valid_selected_trades']+=1
                        else:counts['identical_duplicate_trades']+=1
                    offset+=batch.num_rows
                counts['trade_shards_scanned']+=1
                if counts['trade_shards_scanned']%100==0:connection.commit()
                if counts['trade_shards_scanned']%1000==0:print(f'Scanned {counts["trade_shards_scanned"]} trade shards; selected {counts["unique_valid_selected_trades"]} unique trades',flush=True)
        connection.commit()
        needed=[row[0] for row in connection.execute('SELECT DISTINCT block_number FROM trades ORDER BY block_number')]
        with (output/'block_issues.jsonl').open('w') as rejected:
            for entry in entries:
                if not entry['file'].startswith('polymarket/blocks/'):continue
                bounds=infos[entry['file']]['column_bounds'].get('block_number')
                local=needed if not bounds else needed[bisect_left(needed,bounds['min']):bisect_right(needed,bounds['max'])]
                if not local:continue
                path=extraction/entry['file']
                if sha256_file(path)!=entry['sha256']:raise ValidationError('block shard changed')
                values=pa.array(local,type=pa.int64());offset=0
                for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['block_number','timestamp']):
                    mask=pc.is_in(batch['block_number'],value_set=values);positions=pc.indices_nonzero(mask).to_pylist()
                    for position,row in zip(positions,batch.filter(mask).to_pylist()):
                        try:insert_block(connection,row['block_number'],row['timestamp'],entry['file'],offset+position)
                        except ValidationError as exc:
                            if 'conflicting duplicate' in str(exc):raise
                            block_issues[str(exc)]+=1
                            write_row(rejected,{'file':entry['file'],'row':offset+position,'block_number':row['block_number'],'reason':str(exc)})
                    offset+=batch.num_rows
                connection.commit();counts['block_shards_read']+=1
        counts['required_distinct_blocks']=len(needed)
        counts['matched_block_times']=connection.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]
        counts['trades_with_block_time']=connection.execute('SELECT COUNT(*) FROM trades JOIN blocks USING(block_number)').fetchone()[0]
        with (output/'market_coverage.jsonl').open('w') as stream:
            for market in markets:
                mid=market['market_id'];native=connection.execute('SELECT COUNT(*) FROM trades WHERE market_id=?',(mid,)).fetchone()[0]
                joined,first,last=connection.execute('SELECT COUNT(*),MIN(unix_time),MAX(unix_time) FROM trades JOIN blocks USING(block_number) WHERE market_id=?',(mid,)).fetchone()
                counts['markets_with_trades']+=native>0;counts['markets_with_time_joined_trades']+=joined>0
                write_row(stream,{'market_id':mid,'unique_trades':native,'time_joined_trades':joined,'first_trade_unix':first,'last_trade_unix':last,
                    'benchmark_matches':market['benchmark_matches'],'ready_for_training':False,'ready_for_scoring':False})
        # The published artifact is the main database, never an unbound WAL.
        if connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]:
            raise ValidationError('active readers prevented final database checkpoint')
    finally:connection.close()
    result={'schema_version':'1','kind':'pma_historical_trade_store','dataset_version':report['dataset_version'],
        'catalog_report_sha256':sha256_file(catalog/'report.json'),'selection_sha256':sha256_file(catalog/'macro_candidates.jsonl'),
        'selected_markets':len(markets),'sqlite_cache_budget_mib':SQLITE_CACHE_MIB,'sqlite_checkpoint_policy':'final_explicit_truncate',
        'counts':dict(sorted(counts.items())),'rejected_trade_reasons':dict(sorted(issues.items())),
        'rejected_block_reasons':dict(sorted(block_issues.items())),
        'artifact_hashes':{name:sha256_file(output/name) for name in ['prices.sqlite','trade_issues.jsonl','block_issues.jsonl','market_coverage.jsonl']},
        'model_calls':0,'admitted_training_rows':0,'admitted_evaluation_rows':0,
        'limitations':['Raw logs and block times rely on the archived dataset; chain completeness is not independently certified.',
            'Prices represent fills, not bid/ask midpoint, depth or guaranteed executability.',
            'Every candidate stays in coverage, including missing or unsupported trade histories.',
            'No labels or future snapshot fields are stored as price features; no training targets are emitted.']}
    (output/'report.json').write_text(json_text(result));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('extraction','catalog','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();print(json_text(build(args.extraction,args.catalog,args.output)))


if __name__=='__main__':main()
