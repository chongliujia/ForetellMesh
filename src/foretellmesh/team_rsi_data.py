"""Freeze a reviewed, non-macro, chronological team replay cohort.

Inputs use receipt-verified immutable rules and past prints. Outcome labels and
their capture times remain on the grading side. No existing holdout is opened.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3

from .benchmark_review import read_index, exact_overlaps
from .data import sha256_file, sha256_bytes, strict_json
from .evaluation import json_text
from .pma_trades import quote_at
from .schema import iso, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash
from .team_learning import require, validate_context
from .team_market_proof import audit as audit_proofs, prepare as proof_inputs


def assign_groups(markets, config):
    """Purge lifecycle-straddling groups, not just individual observations."""
    grouped = defaultdict(list)
    for market in markets: grouped[market['event_group_id']].append(market)
    train_end = timestamp(config['train_before'], 'train end')
    dev_start = timestamp(config['development_from'], 'development start')
    dev_end = timestamp(config['development_before'], 'development end')
    require(train_end < dev_start < dev_end, 'invalid chronological boundaries')
    assigned = {}
    for group, rows in grouped.items():
        start = min(timestamp(r['proof']['initialized_at'], 'init') for r in rows)
        end = max(timestamp(r['proof']['resolution_time'], 'resolution') for r in rows)
        assigned[group] = 'train' if end < train_end else 'development' if dev_start <= start and end < dev_end else 'purged_boundary'
    return assigned


def make_jobs(markets, groups, db, config):
    jobs = {'train': [], 'development': []}; exclusions = []; seen = set()
    for anchor in sorted(markets, key=lambda m: m['market_id']):
        split = groups[anchor['event_group_id']]
        if split not in jobs: continue
        init = timestamp(anchor['proof']['initialized_at'], 'init')
        for hours in config['observation_age_hours']:
            at = init+timedelta(hours=hours)
            if config.get('observation_trigger', 'fixed_clock') == 'first_trade_after_age':
                first = db.execute('SELECT MIN(b.unix_time) FROM trades t JOIN blocks b USING(block_number) '
                                   'WHERE t.market_id=? AND b.unix_time>=?',
                                   (anchor['market_id'], int(at.timestamp()))).fetchone()[0]
                if first is None: continue
                at = datetime.fromtimestamp(first, timezone.utc)
            eid = 'team-rsi:'+anchor['market_id']+':'+iso(at)
            if eid in seen: continue
            if at >= timestamp(anchor['proof']['resolution_time'], 'settlement'):
                exclusions.append({'episode_id': eid, 'reason': 'already_settled_at_scheduled_observation'}); continue
            available = []
            for candidate in markets:
                if groups[candidate['event_group_id']] != split: continue
                proof = candidate['proof']; mid = candidate['market_id']
                if not timestamp(proof['initialized_at'], 'init') <= at < timestamp(proof['resolution_time'], 'settlement'): continue
                q, _ = quote_at(db, mid, iso(at), config['max_quote_age_seconds'])
                if q is None: continue
                available.append((candidate, q))
            available.sort(key=lambda x: (x[0]['market_id'] != anchor['market_id'], canonical_hash(x[0]['market_id'])))
            if not available or available[0][0]['market_id'] != anchor['market_id']:
                exclusions.append({'episode_id': eid, 'reason': 'anchor_quote_missing_or_stale'}); continue
            chosen = available[:config['max_markets_per_episode']]
            context = {'episode_id': eid, 'observation_time': iso(at), 'markets': [],
                       'account': {'initial_cash': '100', 'simulation_only': True}, 'memory': []}
            labels = {}; provenance = {}
            for market, quote in chosen:
                mid = market['market_id']; proof = market['proof']
                qt = iso(datetime.fromtimestamp(quote['unix_time'], timezone.utc))
                context['markets'].append({'market_id': mid, 'event_group_id': market['event_group_id'],
                    'input': {'question': proof['historical_question'], 'observation_time': iso(at), 'evidence': [],
                        'market': {'probability': quote['probability'], 'observed_at': qt, 'available_at': qt}}})
                labels[mid] = {k: proof[k] for k in ('outcome', 'resolution_time', 'available_at')}
                provenance[mid] = {k: proof[k] for k in ('initialized_at', 'ancillary_sha256', 'proof_sha256', 'manifest_sha256')}
            validate_context(context); require(eid not in seen, 'duplicate episode'); seen.add(eid)
            jobs[split].append({'context': context, 'labels': labels, 'provenance': provenance,
                               'anchor_group_id': anchor['event_group_id'], 'partition': split})
    for split in jobs:
        jobs[split].sort(key=lambda j: (j['context']['observation_time'], j['context']['episode_id']))
    return jobs, exclusions


def build(review_path, selection, native, proofs, store, index_path, output):
    require(not output.exists(), 'RSI cohort exists')
    config = strict_json(review_path.read_text())
    require(config['kind'] == 'team_rsi_cohort_review_v1'
            and config['selection_report_sha256'] == sha256_file(selection/'report.json')
            and config['benchmark_index_sha256'] == sha256_file(index_path)
            and config['observation_age_hours'] == [6, 24, 168]
            and config['max_markets_per_episode'] == 3 and config['max_quote_age_seconds'] == 10800,
            'unbound RSI cohort review')
    index = read_index(index_path); audit_proofs(selection, native, proofs)
    proof_report = strict_json((proofs/'report.json').read_text())
    require(config.get('observation_trigger', 'fixed_clock') in ('fixed_clock', 'first_trade_after_age'), 'invalid trigger')
    source, _, _ = proof_inputs(selection, native, proof_report['selected'])
    reviews = {r['market_id']: r for r in config['entries']}
    require(len(reviews) == len(config['entries']) and set(reviews) == {r['market_id'] for r in source}, 'incomplete scope review')
    verified = {r['market_id']: r for r in map(strict_json, (proofs/'proofs.jsonl').read_text().splitlines())}
    prices = strict_json((store/'report.json').read_text())
    require(prices['discovery_selection_report_sha256'] == sha256_file(selection/'report.json')
            and prices['artifact_hashes']['prices.sqlite'] == sha256_file(store/'prices.sqlite'), 'unbound discovery prices')
    markets = []; rejected = []; parent_groups = {}
    for candidate in source:
        mid = candidate['market_id']; review = reviews[mid]; proof = verified[mid]
        require(review['question_sha256'] == sha256_bytes(candidate['question'].encode()) and review['rationale'], 'review question changed')
        reason = None
        if review['disposition'] == 'exclude_reserved_family': reason = 'reserved_semantic_family'
        elif review['disposition'] == 'exclude_pilot_scope': reason = 'outside_nonpolitical_pilot'
        else:
            require(review['disposition'] == 'candidate_replay', 'unknown review disposition')
            if proof['status'] != 'chain_proof_verified': reason = 'historical_proof_incomplete'
            elif exact_overlaps(['polymarket:'+mid, 'polymarket:'+candidate['condition_id']], [candidate['question']], index):
                reason = 'reserved_exact_identity'
        if reason:
            rejected.append({'market_id': mid, 'reason': reason}); continue
        parents = proof['native_event_ids']; require(parents, 'missing parent event')
        for parent in parents:
            require(parent_groups.setdefault(parent, review['event_group_id']) == review['event_group_id'], 'native siblings split across groups')
        markets.append({'market_id': mid, 'event_group_id': review['event_group_id'],
                        'native_event_ids': parents, 'proof': deepcopy(proof['proof'])})
    groups = assign_groups(markets, config)
    with sqlite3.connect((store/'prices.sqlite').resolve().as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        jobs, excluded_jobs = make_jobs(markets, groups, db, config)
    require(jobs['train'] and jobs['development'], 'both isolated training and development episodes required')
    output.mkdir(parents=True)
    for split, rows in jobs.items(): (output/(split+'.jobs.jsonl')).write_text(jsonl(rows))
    (output/'markets.jsonl').write_text(jsonl(markets)); (output/'exclusions.jsonl').write_text(jsonl(rejected+excluded_jobs))
    (output/'review.json').write_text(json_text(config))
    result = {'kind': 'team_rsi_replay_cohort_v1', 'status': 'frozen_replay_cohort_not_teacher_sft',
        'source_hashes': {'review': sha256_file(review_path), 'proof_report': sha256_file(proofs/'report.json'),
                         'price_report': sha256_file(store/'report.json'), 'benchmark_index': sha256_file(index_path)},
        'sources': {'review': str(review_path), 'selection': str(selection), 'native': str(native),
                    'proofs': str(proofs), 'store': str(store), 'index': str(index_path)},
        'groups': groups, 'counts': {split: {'episodes': len(rows),
            'markets': len({m['market_id'] for j in rows for m in j['context']['markets']}),
            'event_groups': len({m['event_group_id'] for j in rows for m in j['context']['markets']})} for split, rows in jobs.items()},
        'rejected_market_reasons': dict(Counter(r['reason'] for r in rejected)),
        'excluded_episode_reasons': dict(Counter(r['reason'] for r in excluded_jobs)),
        'artifact_hashes': {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        'existing_holdouts_opened': False, 'final_test_opened': False,
        'limitations': ['Small prefix of a fixed discovery sample, not representative of the full archive.',
            'No external contemporaneous news; immutable contract rules and historical trade prices only.',
            'Question/label verification relies on reviewed adapter behavior and one public RPC provider.',
            'Historical replay cannot rule out foundation-model pretraining contamination.',
            'Outcome supervision is a separate grading target, not a 0/1 probability teacher.']}
    (output/'report.json').write_text(json_text(result)); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('review','selection','native','proofs','store','index','output'): p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(); print(json_text(build(a.review,a.selection,a.native,a.proofs,a.store,a.index,a.output)))
