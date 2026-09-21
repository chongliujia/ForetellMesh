"""Offline audit of the topic-neutral data and first team interface artifacts."""
import argparse
import json
from pathlib import Path
import sqlite3

from foretellmesh.data import sha256_file, strict_json
from foretellmesh.evaluation import json_text
from foretellmesh.pma_trades import load_selection, quote_at
from foretellmesh.schema import iso
from foretellmesh.synthetic_sft import canonical_hash
from foretellmesh.team_learning import require
from foretellmesh.team_market_proof import audit as audit_proofs
from datetime import datetime, timezone


def audit(catalog, selection, store, native, proofs, output):
    require(not output.exists(), 'foundation audit output exists')
    cat = strict_json((catalog/'report.json').read_text())
    markets, source = load_selection(catalog, cat, selection)
    report = strict_json((store/'report.json').read_text())
    require(report['selection_sha256'] == sha256_file(source)
            and report['discovery_selection_report_sha256'] == sha256_file(selection/'report.json')
            and report['admitted_training_rows'] == 0 and report['admitted_evaluation_rows'] == 0,
            'discovery price store is not bound or claims admission')
    for name, digest in report['artifact_hashes'].items():
        path = (store/name).resolve()
        require(path.is_relative_to(store.resolve()) and sha256_file(path) == digest, 'price artifact changed')
    coverage = [strict_json(s) for s in (store/'market_coverage.jsonl').read_text().splitlines()]
    require({r['market_id'] for r in coverage} == {m['market_id'] for m in markets}, 'coverage population differs')
    checked = 0
    with sqlite3.connect((store/'prices.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'SQLite integrity failure')
        require(db.execute('SELECT COUNT(*) FROM trades').fetchone()[0] == report['counts']['unique_valid_selected_trades'],
                'price row count differs')
        require(db.execute('SELECT COUNT(*) FROM trades JOIN blocks USING(block_number)').fetchone()[0]
                == report['counts']['trades_with_block_time'], 'price timestamp join count differs')
        for row in coverage:
            if not row['time_joined_trades']: continue
            for seconds in (row['first_trade_unix'], row['last_trade_unix']):
                value, _ = quote_at(db, row['market_id'], iso(datetime.fromtimestamp(seconds, timezone.utc)), 10800)
                require(value is not None and value['unix_time'] <= seconds and 0 < value['probability'] < 1,
                        'invalid actual historical quote')
                checked += 1
    manifest = strict_json((native/'manifest.json').read_text())
    require(manifest['requests_sha256'] == canonical_hash(manifest['requests']), 'native manifest changed')
    for ref in manifest['requests']:
        path = (native/ref['file']).resolve()
        require(path.is_relative_to(native.resolve()) and sha256_file(path) == ref['sha256'], 'native response changed')
    proof_audit = audit_proofs(selection, native, proofs)
    result = {'status': 'passed', 'kind': 'team_learning_foundation_audit_v1',
              'selected_markets': len(markets), 'actual_asof_quotes_checked': checked,
              'native_responses_verified': len(manifest['requests']), 'proof_audit': proof_audit,
              'training_admission': False, 'training_performed': False,
              'source_report_hashes': {name: sha256_file(root/'report.json') for name, root in
                  [('catalog', catalog), ('selection', selection), ('store', store), ('native', native), ('proofs', proofs)]}}
    output.mkdir(parents=True); (output/'audit.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'selection', 'store', 'native', 'proofs', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json_text(audit(a.catalog, a.selection, a.store, a.native, a.proofs, a.output)))
