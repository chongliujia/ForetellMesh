"""Build an additive, training-only overlay; never read validation/test examples."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import re
import sqlite3

from .benchmark_review import read_index, exact_overlaps
from .data import sha256_bytes, sha256_file, strict_json, question_key
from .evaluation import json_text
from .historical_market import fed_upper_bound
from .historical_sources import fed_statement
from .pma_enrichment import token_pairs
from .pma_legacy import inputs, urls, reviewed_runtime, audit_contract
from .pma_proof_pilot import require, read_http, read_jsonl
from .pma_trades import quote_at
from .pma_proof_replay import utc_seconds
from .schema import iso, timestamp, parse_record
from .synthetic_sft import canonical_hash


def expected_legacy_fomc(question, period, change):
    month = datetime.strptime(period, '%Y-%m').strftime('%B')
    match = re.fullmatch(r'Will the Fed (decrease|increase|raise) interest rates by (0|25|50) bps after its '
                         + re.escape(month) + r' meeting\?', question)
    require(match is not None and change % 25 == 0, 'unreviewed legacy FOMC bracket')
    target = int(match[2]) * (-1 if match[1] == 'decrease' else 1)
    return int(change == target)


def protect_extension(rows, parent_membership, cutoff):
    protected = {r['event_group_id'] for r in parent_membership if r['split'] != 'train'}
    ids = {r['event_id'] for r in parent_membership if r['split'] != 'train'}
    for r in rows:
        parsed = parse_record(r)
        require(r['event_group_id'] not in protected and r['event_id'] not in ids, 'extension overlaps heldout')
        require(parsed.label is not None and timestamp(r['observation_time'], 'observation') < timestamp(cutoff, 'cutoff')
                and timestamp(r['label']['resolution_time'], 'settlement') < timestamp(cutoff, 'cutoff'),
                'extension crosses training cutoff')


def write_rows(path, rr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json_text(r).replace('\n', '')+'\n' for r in rr))


def build(config_path, capture, review_path, output):
    require(not output.exists(), 'training overlay already exists')
    c, paths, selected = inputs(config_path)
    cr = strict_json((capture/'report.json').read_text()); review = strict_json(review_path.read_text())
    require(cr['config_sha256'] == sha256_file(config_path), 'capture config changed')
    for key, name in [('index_sha256', 'index.json'), ('http_manifest_sha256', 'http/manifest.json'), ('selection_sha256', 'selection.json')]:
        require(sha256_file(capture/name) == cr[key], 'capture changed: ' + name)
    require(strict_json((capture/'selection.json').read_text()) == {'config_sha256': sha256_file(config_path),
        'market_ids': [r['market_id'] for r in selected]}, 'capture selection mismatch')
    require(review['kind'] == 'pma_legacy_fomc_training_scope_review_v1' and review['config_sha256'] == sha256_file(config_path)
            and review['selection_sha256'] == c['selection_sha256'] and review['benchmark_sha256'] == sha256_file(paths['benchmark'])
            and review['allowed_partition'] == 'train' and review['preserve_parent_partitions'] is True,
            'unbound scope review')
    reviewed = {g['event_group_id']: g for g in review['groups']}
    require(set(reviewed) == {g['event_group_id'] for g in c['groups']} and len(reviewed) == len(review['groups'])
            and all(g['disposition'] == 'training_candidate' and g['rationale'] for g in reviewed.values()), 'scope review incomplete')
    parent = paths['parent'].parent; pr = strict_json(paths['parent'].read_text())
    def parent_rows(name):
        require(sha256_file(parent/name) == pr['artifact_hashes'][name], 'parent artifact changed')
        return read_jsonl(parent/name)
    membership = parent_rows('membership.jsonl')
    old_inputs = parent_rows('partitions/train.inputs.jsonl'); old_labels = parent_rows('partitions/train.labels.jsonl')
    old_records = parent_rows('partitions/train.records.jsonl')
    old_ids = {r['event_id'].removeprefix('polymarket:') for r in membership if r['split'] == 'train'}
    require(not old_ids & {r['market_id'] for r in selected}, 'extension repeats old market')
    index = read_index(paths['benchmark']); captured = strict_json((capture/'index.json').read_text())
    require([r['market_id'] for r in captured] == [r['market_id'] for r in selected], 'incomplete capture population')
    manifest, http = read_http(capture/'http')
    require([r['url'] for r in manifest['requests']] == urls(c, selected), 'HTTP population differs')
    def get(url, json=True):
        ref, raw = http[url]; require(ref['status'] == 200, 'HTTP source unavailable: ' + url)
        return strict_json(raw.decode()) if json else raw
    runtime = reviewed_runtime(paths); prepared = {}; group_errors = {}
    for g in c['groups']:
        try:
            event = get('https://gamma-api.polymarket.com/events/' + g['native_event_id'])
            members = {r['market_id'] for r in selected if r['native_event_id'] == g['native_event_id']}
            require(event['id'] == g['native_event_id'] and {m['id'] for m in event['markets']} == members
                    and len(event['markets']) == len(members), 'native sibling coverage differs')
            ss = get('https://data-api.polymarket.com/v2/resolutions?event_id=' + g['native_event_id'])['data']
            states = {s['condition_id']: s for s in ss}; require(len(states) == len(ss), 'duplicate resolution state')
            docs = [fed_statement(get(ref['url'], json=False), ref['url'], ref['published_at']) for ref in (g['prior_release'], g['release'])]
            prepared[g['event_group_id']] = (g, event, states, *docs)
        except (ValueError, KeyError, TypeError) as e:
            group_errors[g['event_group_id']] = str(e)
    tr = strict_json(paths['prices'].read_text()); dbpath = paths['prices'].parent/'prices.sqlite'
    require(sha256_file(dbpath) == tr['artifact_hashes']['prices.sqlite'], 'price store changed')
    wal = Path(str(dbpath)+'-wal'); require(not wal.exists() or wal.stat().st_size == 0, 'uncheckpointed trade database')
    version = 'sha256:' + canonical_hash({'config': sha256_file(config_path), 'review': sha256_file(review_path),
        'capture': sha256_file(capture/'report.json'), 'parent': sha256_file(paths['parent']),
        'code': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')}})
    records = []; contracts = []; exclusions = []
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        for row, evidence in zip(selected, captured):
            mid = row['market_id']; gid = row['event_group_id']; m = row['market']; proof = None; errors = []
            try:
                require(gid in prepared, group_errors.get(gid, 'missing group'))
                g, event, states, prior, official = prepared[gid]
                require(evidence['status'] == 'captured', 'incomplete capture: ' + evidence.get('error', 'unknown'))
                root = capture/'rpc'/mid
                require(sha256_file(root/'proof.json') == evidence['proof_sha256']
                        and sha256_file(root/'manifest.json') == evidence['manifest_sha256'], 'contract capture changed')
                current = next(x for x in event['markets'] if x['id'] == mid)
                require(all(current.get(k) == m.get(k) for k in ('question', 'description', 'conditionId', 'questionID'))
                        and current['resolvedBy'].lower() == m['resolvedBy'].lower()
                        and token_pairs(current['outcomes'], current['clobTokenIds']) == row['candidate']['tokens'], 'native identity changed')
                clob = get('https://clob.polymarket.com/clob-markets/' + m['conditionId'])
                require(clob['c'] == m['conditionId'] and [(x['o'], x['t']) for x in clob['t']] ==
                        [(o, row['candidate']['tokens'][o]) for o in ('Yes', 'No')], 'CLOB mapping conflict')
                require(not exact_overlaps(['polymarket:'+mid, 'polymarket:'+m['conditionId']], [m['question']], index), 'benchmark overlap')
                proof = audit_contract(root, row, states[m['conditionId']], runtime)
                change = (fed_upper_bound(official['text']) - fed_upper_bound(prior['text'])) * 100
                require(expected_legacy_fomc(m['question'], g['period'], change) == proof['outcome'], 'official payout conflict')
            except (ValueError, KeyError, TypeError, StopIteration) as e:
                proof = None; errors.append(str(e))
            contracts.append({'market_id': mid, 'event_group_id': gid, 'proof': proof, 'proof_errors': errors})
            g = next(g for g in c['groups'] if g['event_group_id'] == gid)
            for days in (7, 1):
                at = timestamp(g['release']['published_at'], 'release') - timedelta(days=days)
                why = list(errors); record = None
                if proof is not None:
                    try:
                        prior, official = prepared[gid][-2:]
                        require(timestamp(proof['initialized_at'], 'init') <= at < timestamp(official['published_at'], 'release')
                                <= timestamp(proof['resolution_time'], 'resolution'), 'observation chronology conflict')
                        quote, quality = quote_at(db, mid, iso(at), 10800)
                        require(quote is not None, 'historical_quote_unavailable:' + str(quality))
                        record = {'sample_id': 'pma:'+mid+':'+iso(at), 'dataset_source': 'foretellmesh_pma_macro_training_extension',
                            'dataset_version': version, 'event_id': 'polymarket:'+mid, 'event_group_id': gid,
                            'question': proof['historical_question'], 'observation_time': iso(at),
                            'evidence': [{'evidence_id': 'fed:'+sha256_bytes(prior['text'].encode())[:16], 'text': prior['text'],
                                'source': prior['source'], 'published_at': prior['published_at'], 'available_at': prior['published_at']}],
                            'market': {'probability': quote['probability'], 'observed_at': utc_seconds(quote['unix_time']),
                                       'available_at': utc_seconds(quote['unix_time'])},
                            'label': {'outcome': proof['outcome'], 'resolution_time': proof['resolution_time'], 'available_at': proof['available_at']}}
                        protect_extension([record], membership, c['train_before'])
                    except (ValueError, KeyError, TypeError) as e:
                        record = None; why.append(str(e))
                if record is None:
                    exclusions.append({'market_id': mid, 'event_group_id': gid, 'observation_time': iso(at), 'reasons': why})
                else:
                    records.append(record)
    protect_extension(records, membership, c['train_before'])
    require(records, 'no new admissible training records')
    all_records = [{**r, 'dataset_version': version} for r in old_records] + records
    require(len({r['sample_id'] for r in all_records}) == len(all_records), 'duplicate training sample')
    owners = {}
    for r in all_records:
        q = question_key(r['question']); require(owners.get(q, r['event_id']) == r['event_id'], 'duplicate equivalent training question')
        owners[q] = r['event_id']; parse_record(r)
    output.mkdir(parents=True)
    write_rows(output/'partitions/train.records.jsonl', all_records)
    write_rows(output/'partitions/train.inputs.jsonl', old_inputs + [{'sample_id': r['sample_id'], 'input': parse_record(r).forecast_input.to_payload()} for r in records])
    write_rows(output/'partitions/train.labels.jsonl', old_labels + [{'sample_id': r['sample_id'], 'label': r['label']} for r in records])
    write_rows(output/'membership.jsonl', [{**r, 'dataset_version': version} for r in membership if r['split'] == 'train'] + [
        {k: r[k] for k in ('sample_id', 'event_id', 'event_group_id', 'dataset_version')} | {'split': 'train'} for r in records])
    for name in pr['artifact_hashes']:
        if name.startswith('audit/') and name.endswith('/contracts.jsonl'):
            values = [r for r in parent_rows(name) if r['market_id'] in old_ids]
            if values:
                write_rows(output/name, values)
    write_rows(output/'audit/legacy/contracts.jsonl', contracts); write_rows(output/'exclusions.jsonl', exclusions)
    report = {'kind': 'pma_additive_training_only_overlay', 'status': 'released', 'dataset_version': version,
        'config_sha256': sha256_file(config_path), 'capture_report_sha256': sha256_file(capture/'report.json'),
        'review_sha256': sha256_file(review_path), 'parent_report_sha256': sha256_file(paths['parent']),
        'parent_training_rows_preserved': len(old_records), 'new_training_rows': len(records),
        'new_event_groups': sorted({r['event_group_id'] for r in records}), 'candidate_contracts': len(selected),
        'proof_complete_contracts': sum(r['proof'] is not None for r in contracts),
        'exclusion_reasons': dict(Counter(s for r in exclusions for s in r['reasons'])),
        'partition_counts': {'train': {'observations': len(all_records), 'contracts': len({r['event_id'] for r in all_records}),
                                     'event_groups': len({r['event_group_id'] for r in all_records})}},
        'artifact_hashes': {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'heldout_inputs_opened': False, 'heldout_labels_opened': False, 'new_training_started': False,
        'limitations': ['Training-only additive overlay; original validation/test remain at parent and are not copied or opened.',
            'Retrospective source/provider attestations; no pretraining-contamination or executable-depth guarantee.',
            'New events are FOMC only; CPI/employment diversity and contemporaneous news remain limited.',
            'Selection is all native brackets of eight predeclared events; failures retained, no payout-based selection.']}
    (output/'report.json').write_text(json_text(report)); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('config', 'capture', 'review', 'output'): p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(); r = build(a.config, a.capture, a.review, a.output)
    print(json_text({k: r[k] for k in ('status', 'new_training_rows', 'new_event_groups', 'partition_counts', 'exclusion_reasons')}))
