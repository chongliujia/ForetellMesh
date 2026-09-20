"""Offline rule/settlement audit and time-bounded PMA pilot observations."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
import re
import sqlite3

from .benchmark_review import exact_overlaps, read_index
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .historical_market import fed_upper_bound
from .historical_sources import fed_statement, normalize
from .pma_enrichment import token_pairs
from .pma_bls import load_reviewed_bls, expected_bls_outcome
from .pma_proof_pilot import inputs, require, read_http, http_urls
from .pma_trades import quote_at
from .polygon_proof_audit import (INITIALIZED, GET_QUESTION, GET_UPDATES, EMPTY_UPDATES,
                                  abi_bytes, word, dynamic_text, decode_question)
from .polygon_rules_audit import extract_source
from .polygon_settlement_audit import (ProofArchive, CTF, CTF_RESOLVED, DENOMINATOR, NUMERATOR,
    UMA_RESOLVED, NEG_RISK, REPORTED, receipt_log, settlement_event, uint_result, binary_payout)
from .schema import ValidationError, timestamp, iso, parse_record
from .synthetic_sft import canonical_hash

REVIEW_ADAPTER = '0x2f5e3684cb1f318ec51b00edba38d79ac2c0aa9d'
# Historical deployment: Polymarket/neg-risk-ctf-adapter addresses.json (137).
# Also embedded as immutable ctf in the reviewed adapter's verified runtime.
REVIEW_OPERATOR = '0x71523d0f655b41e805cec45b17163f528b59b820'


def utc_seconds(seconds):
    return iso(datetime.fromtimestamp(seconds, timezone.utc))


def reviewed_runtime(paths):
    policy = strict_json(paths['rules_policy'].read_text())
    require(policy.get('schema_version') == '1' and policy.get('review_id') == 'pma_uma_2f5e_rules_v1'
            and policy.get('adapter') == REVIEW_ADAPTER
            and policy.get('trust_model') == 'polygonscan_source_verification_plus_single_public_rpc'
            and policy.get('reviewed_properties') == ['immutable_initialized_ancillary', 'append_only_creator_updates',
                                                     'no_proxy_delegatecall_or_selfdestruct'], 'unsupported rule review')
    manifest = strict_json(paths['rules_source'].read_text())
    refs = [r for r in manifest['requests'] if r['url'] == policy['source_url']]
    require(len(refs) == 1 and refs[0]['status'] == 200 and refs[0]['request'] is None, 'verified source page missing')
    ref = refs[0]; root = paths['rules_source'].parent; path = (root/ref['file']).resolve()
    require(path.is_relative_to(root) and sha256_file(path) == ref['sha256'] == policy['source_page_sha256'], 'reviewed source page changed')
    sources, runtime = extract_source(path.read_text())
    require({k: sha256_bytes(v['content'].encode()) for k, v in sources.items()} == policy['source_sha256']
            and sha256_bytes(runtime) == policy['runtime_sha256'], 'reviewed source/runtime differs')
    return policy, '0x'+runtime.hex()


def audit_contract(root, row, state, runtime):
    """Bind initialization, reviewed code, empty updates, UMA report and final CTF."""
    archive = ProofArchive(root); proof = strict_json((root/'proof.json').read_text())
    market = row['market']; adapter = market['resolvedBy'].lower(); qid = market['negRiskRequestID']
    condition = market['conditionId']; native_qid = market['questionID']
    require(adapter == REVIEW_ADAPTER and market.get('negRisk') is True
            and strict_json(market['outcomes']) == ['Yes', 'No']
            and proof['market_id'] == market['id'] and proof['question_id'] == qid
            and proof['condition_id'] == condition
            and state['question_id'] == qid and state['condition_id'] == condition,
            'unsupported or conflicting market/adapter identity')
    require(state.get('status') == 'resolved' and state.get('was_disputed') is False
            and state.get('extended_review') is False and state.get('was_arbitrated', False) is False,
            'disputed or unsupported resolution lifecycle')
    head = archive.get(proof['finalized_head'], 'eth_getBlockByNumber', ['finalized', False])
    initial = archive.get(proof['initial_block'], 'eth_getBlockByNumber')
    require(proof['initial_block']['request']['params'] == [initial['number'], False], 'unbound initialization block')
    receipt = archive.get(proof['initial_receipt'], 'eth_getTransactionReceipt', [state['transaction_hash']])
    logs = [l for l in receipt['logs'] if l['address'] == adapter and l['topics'][:2] == [INITIALIZED, qid]]
    require(len(logs) == 1 and len(logs[0]['topics']) == 4, 'ambiguous initialization')
    log = logs[0]; receipt_log(log, receipt, initial)
    creator_word = log['topics'][3]; initialized = int(log['topics'][2], 16)
    require(initialized == int(initial['timestamp'], 16) and int(creator_word, 16) < 2**160
            and int(initial['number'], 16) < int(head['number'], 16)
            and initialized < int(head['timestamp'], 16)
            <= timestamp(proof['finalized_head']['completed_at'], 'capture').timestamp(), 'invalid initialization chronology')
    raw = abi_bytes(log['data'])
    require(word(raw, 0) == 128, 'unsupported initialized text ABI')
    ancillary = dynamic_text(raw, 128)
    prefix = 'q: title: '+market['question']+', description: '+market['description']
    # Captured historical templates use three explicit metadata boundaries.
    # Match the whole rule text and, when present, bind the native market ID;
    # accepting an arbitrary suffix would hide amendments to the description.
    boundaries = (', res_data: ', ' res_data: ', f' market_id: {market["id"]} res_data: ')
    require(any(normalize(ancillary).startswith(normalize(prefix)+boundary) for boundary in boundaries),
            'native rules differ from initialization')
    require(f'Updates made by the question creator via the bulletin board at {adapter}'.lower() in ancillary.lower()
            and ancillary.endswith(',initializer:'+creator_word[-40:]), 'unsupported creator-update clause')
    current = decode_question(archive.get(proof['question_state'], 'eth_call',
        [{'to': adapter, 'data': GET_QUESTION+qid[2:]}, head['number']]))
    require(current['creator'] == '0x'+creator_word[-40:] and current['ancillary_data'] == ancillary
            and current['request_timestamp'] == initialized and current['resolved'], 'question state/reset disagreement')
    updates = archive.get(proof['creator_updates'], 'eth_call',
        [{'to': adapter, 'data': GET_UPDATES+qid[2:]+creator_word[2:]}, head['number']])
    require(updates == EMPTY_UPDATES, 'nonempty creator updates require historical review')
    require(len(proof['code']) == 2, 'missing runtime coverage')
    for ref, n in zip(proof['code'], (initial['number'], head['number'])):
        require(archive.get(ref, 'eth_getCode', [adapter, n]) == runtime, 'deployed runtime differs from reviewed source')
    oracle, final = proof['oracle'], proof['ctf']
    require(all(r['question_id'] == qid and r['condition_id'] == condition
                and r['finalized_head'] == proof['finalized_head'] for r in (oracle, final)), 'settlement question or pin differs')
    ob, ol, ore = settlement_event(archive, oracle, adapter, [None, qid])
    require(len(ol['topics']) == 3 and ol['topics'][:2] == [UMA_RESOLVED, qid], 'unsupported UMA report')
    payouts = binary_payout(ol['data']); outcome = payouts[0]
    require(int(ol['topics'][2], 16) == outcome*10**18, 'UMA price/payout conflict')
    reports = [l for l in ore['logs'] if l['address'] == REVIEW_OPERATOR and l['topics'] == [REPORTED, native_qid]]
    require(len(reports) == 1, 'NegRisk report binding missing')
    receipt_log(reports[0], ore, ob)
    require(reports[0]['data'] == qid+f'{outcome:064x}', 'NegRisk native/request identity conflict')
    fb, fl, _ = settlement_event(archive, final, CTF, [CTF_RESOLVED, condition])
    require(fl['topics'] == [CTF_RESOLVED, condition, '0x'+NEG_RISK[2:].rjust(64, '0'), native_qid]
            and binary_payout(fl['data'], ctf=True) == payouts, 'CTF native identity or payout conflict')
    n = int(fb['number'], 16)
    require(uint_result(archive, final['previous_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n-1)]) == 0
            and uint_result(archive, final['resolved_denominator'], [{'to': CTF, 'data': DENOMINATOR+condition[2:]}, hex(n)]) == 1,
            'final settlement boundary not proven')
    require(len(final['numerators']) == 2 and [uint_result(archive, ref,
        [{'to': CTF, 'data': NUMERATOR+condition[2:]+f'{i:064x}'}, hex(n)]) for i, ref in enumerate(final['numerators'])] == payouts,
        'CTF state and log payouts conflict')
    require(int(initial['number'], 16) < int(ob['number'], 16) < n < int(head['number'], 16)
            and initialized < int(ob['timestamp'], 16) <= int(fb['timestamp'], 16) <= int(head['timestamp'], 16),
            'backwards settlement chronology')
    require(Fraction(state['price']) == outcome*10**18, 'API result disagrees with chain')
    # Every proof is served by one identified provider. This is not a consensus proof.
    endpoints = {ref['url'] for ref in archive.used}
    chains = {ref['url'] for ref, obj in archive.refs.values() if ref['status'] == 200
              and ref['request']['method'] == 'eth_chainId' and ref['request']['params'] == []
              and obj.get('result') == '0x89' and 'error' not in obj}
    require(len(endpoints) == 1 and endpoints <= chains, 'unbound mixed-provider proof')
    for ref in (proof['question_state'], proof['creator_updates'], *proof['code']):
        require(timestamp(ref['completed_at'], 'capture').timestamp() >= int(head['timestamp'], 16), 'state pin after capture')
    return {'market_id': market['id'], 'historical_question': prefix,
            'ancillary_sha256': sha256_bytes(ancillary.encode()), 'initialized_at': utc_seconds(initialized),
            'rule_valid_through': utc_seconds(int(head['timestamp'], 16)), 'creator_update_count': 0,
            'runtime_sha256': sha256_bytes(bytes.fromhex(runtime[2:])), 'outcome': outcome,
            'oracle_report_time': utc_seconds(int(ob['timestamp'], 16)), 'resolution_time': utc_seconds(int(fb['timestamp'], 16)),
            'settlement_block': n, 'settlement_block_hash': fb['hash'], 'settlement_transaction': fl['transactionHash'],
            'available_at': max((r['completed_at'] for r in archive.used), key=lambda t: timestamp(t, 'capture')),
            'proof_sha256': sha256_file(root/'proof.json'), 'manifest_sha256': sha256_file(root/'manifest.json')}


def expected_outcome(question, period, change):
    date = datetime.strptime(period, '%Y-%m')
    month, year = date.strftime('%B'), date.strftime('%Y')
    # The reviewed legacy brackets and the newer rounding rule agree on this
    # grid. Other changes need their actual rule text interpreted separately.
    require(change % 25 == 0, 'off-grid decision requires rounding-rule review')
    endings = [f'{month} {year}', f'{year} {month}', f'its {month} {year}', f'its {year} {month}']
    if question in {f'No change in Fed interest rates after {suffix} meeting?' for suffix in endings} | {
            f'No change in Fed raise interest rates after its {year} {month} meeting?'}:
        return int(change == 0)
    # These two explicit 2024 sibling contracts cover changes between the other
    # 25bp-grid brackets. Their year comes from the frozen parent and rules.
    legacy_other = {'2024-11': 'Will the FED change rates to another level after Nov meeting?',
                    '2024-12': 'Will the FED change rates to another level after December meeting?'}
    if question == legacy_other.get(period):
        return 0
    match = re.fullmatch(r'(?:Will the )?Fed (decreases?|increases?|raises?) interest rates by '
                         r'(25|25\+|50|50\+|75\+) bps after (?:'+ '|'.join(map(re.escape, endings))+r') meeting\?', question)
    require(match, 'question outside reviewed FOMC bracket grammar')
    signed = -change if match[1].startswith('decrease') else change
    threshold = int(match[2].rstrip('+'))
    return int(signed >= threshold if match[2].endswith('+') else signed == threshold)


def official_outcome(row, group, prior, official):
    if group.get('family') in ('cpi_yoy', 'unemployment_u3'):
        return expected_bls_outcome(row['market'], group, official)
    return expected_outcome(row['market']['question'], group['period'],
                            (fed_upper_bound(official['text'])-fed_upper_bound(prior['text']))*100)


def observation(row, group, proof, prior, official, quote, at, dataset_version):
    require(timestamp(proof['initialized_at'], 'init') <= timestamp(at, 'observation')
            < timestamp(official['published_at'], 'public result') <= timestamp(proof['oracle_report_time'], 'oracle')
            <= timestamp(proof['resolution_time'], 'resolution'), 'historical observation chronology conflict')
    require(timestamp(at, 'observation') <= timestamp(proof['rule_valid_through'], 'rule pin'), 'future observation beyond rule proof')
    expected = official_outcome(row, group, prior, official)
    require(expected == proof['outcome'], 'official statement disagrees with contract payout')
    market = None if quote is None else {'probability': quote['probability'], 'observed_at': utc_seconds(quote['unix_time']),
                                        'available_at': utc_seconds(quote['unix_time'])}
    record = {'sample_id': 'pma:'+row['market_id']+':'+at, 'dataset_source': 'pma_polymarket_historical_pilot',
              'dataset_version': dataset_version, 'event_id': 'polymarket:'+row['market_id'], 'event_group_id': group['event_group_id'],
              'question': proof['historical_question'], 'observation_time': at,
              'evidence': [{'evidence_id': ('bls:' if group.get('family') else 'fed:')+sha256_bytes(prior['text'].encode())[:16], 'text': prior['text'],
                            'source': prior['source'], 'published_at': prior['published_at'], 'available_at': prior['published_at']}],
              'market': market, 'label': {'outcome': proof['outcome'], 'resolution_time': proof['resolution_time'],
                                         'available_at': proof['available_at']}}
    parsed = parse_record(record)
    return record, parsed.forecast_input.to_payload()


def build(config_path, capture, output):
    config, paths, selected = inputs(config_path)
    require(not output.exists(), 'pilot replay output exists')
    selection = strict_json((capture/'selection.json').read_text()); report = strict_json((capture/'report.json').read_text())
    require(selection == {'config': config, 'config_sha256': sha256_file(config_path), 'market_ids': [r['market_id'] for r in selected]},
            'capture selection or policy changed')
    for key, name in [('selection_sha256', 'selection.json'), ('index_sha256', 'index.json'), ('http_manifest_sha256', 'http/manifest.json')]:
        require(sha256_file(capture/name) == report[key], 'capture binding changed')
    index = strict_json((capture/'index.json').read_text())
    require([r['market_id'] for r in index] == [r['market_id'] for r in selected], 'capture coverage incomplete')
    manifest, http = read_http(capture/'http')
    require([r['url'] for r in manifest['requests']] == http_urls(config, selected), 'capture URL coverage differs')
    def get(url, json=True):
        ref, raw = http[url]
        require(ref['status'] == 200, 'HTTP source failed: '+url)
        return ref, strict_json(raw.decode()) if json else raw
    policy, runtime = reviewed_runtime(paths)
    bls = load_reviewed_bls(paths['bls_archive'], paths['bls_policy']) if 'bls_archive' in paths else {}
    price_report = strict_json(paths['price_store'].read_text()); price_path = paths['price_store'].parent/'prices.sqlite'
    require(price_report['catalog_report_sha256'] == sha256_file(paths['catalog'])
            and sha256_file(price_path) == price_report['artifact_hashes']['prices.sqlite'], 'price store/candidate binding changed')
    # The completed checkpointed DB is immutable; ignore no pending WAL silently.
    wal = Path(str(price_path)+'-wal')
    require(not wal.exists() or wal.stat().st_size == 0, 'price DB has uncheckpointed WAL')
    benchmark = read_index(paths['heldout_index'])
    code_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    version = 'sha256:'+canonical_hash({'config': sha256_file(config_path), 'capture_report': sha256_file(capture/'report.json'),
                                      'code': code_hashes})
    output.mkdir(parents=True)
    contracts, candidates, review_inputs, outcomes, groups = [], [], [], [], []
    by_group = {g['event_group_id']: g for g in config['groups']}
    prepared = {}; selected_by_event = {}
    for row in selected:
        selected_by_event.setdefault(row['native_event_id'], set()).add(row['market_id'])
    for group in config['groups']:
        entry = {**group, 'market_ids': sorted(selected_by_event[group['native_event_id']], key=int),
                 'grouping_status': 'same_scheduled_release', 'split': None,
                 'cross_parent_and_benchmark_semantic_review': 'pending'}
        try:
            _, event = get('https://gamma-api.polymarket.com/events/'+group['native_event_id'])
            require(event['id'] == group['native_event_id'] and len({m['id'] for m in event['markets']}) == len(event['markets'])
                    and {m['id'] for m in event['markets']} == selected_by_event[group['native_event_id']], 'native parent coverage changed')
            _, resolution = get('https://data-api.polymarket.com/v2/resolutions?event_id='+group['native_event_id'])
            states = {s['condition_id']: s for s in resolution['data']}
            require(len(states) == len(resolution['data']), 'duplicate resolution conditions')
            docs = []
            for source in (group['prior_release'], group['release']):
                if bls:
                    fact = bls[(group['family'], source['period'])]
                    require(fact['source'] == source['url'] and timestamp(fact['published_at'], 'BLS release')
                            == timestamp(source['published_at'], 'planned release'), 'BLS release binding differs')
                    docs.append(fact)
                else:
                    _, raw = get(source['url'], json=False)
                    docs.append(fed_statement(raw, source['url'], source['published_at']))
            prepared[group['event_group_id']] = (event, states, *docs)
        except (ValidationError, KeyError, ValueError, TypeError) as exc:
            entry['source_error'] = str(exc)
        groups.append(entry)
    with sqlite3.connect(price_path.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
        for row, captured in zip(selected, index):
            m = row['market']; group = by_group[row['event_group_id']]; proof = None; errors = []
            try:
                require(captured['status'] == 'captured_not_audited', 'incomplete chain capture: '+captured.get('error', 'unknown'))
                root = capture/'rpc'/row['market_id']
                require(sha256_file(root/'proof.json') == captured['proof_sha256']
                        and sha256_file(root/'manifest.json') == captured['manifest_sha256'], 'contract proof binding changed')
                require(row['event_group_id'] in prepared, 'native parent or official source incomplete')
                event, states, prior, official = prepared[row['event_group_id']]
                current = next(x for x in event['markets'] if x['id'] == m['id'])
                require(all(current.get(k) == m.get(k) for k in ('conditionId', 'questionID', 'negRiskRequestID', 'question', 'description', 'negRisk'))
                        and current['resolvedBy'].lower() == m['resolvedBy'].lower()
                        and token_pairs(current['outcomes'], current['clobTokenIds']) == row['candidate']['tokens'], 'native market mapping/rules changed')
                _, clob = get('https://clob.polymarket.com/clob-markets/'+m['conditionId'])
                require(clob['c'] == m['conditionId'] and [(x['o'], x['t']) for x in clob['t']]
                        == [(o, row['candidate']['tokens'][o]) for o in strict_json(m['outcomes'])],
                        'CLOB token/outcome mapping conflict')
                proof = audit_contract(root, row, states[m['conditionId']], runtime)
                require(official_outcome(row, group, prior, official) == proof['outcome'], 'official outcome conflict')
            except (ValidationError, KeyError, ValueError, TypeError, StopIteration) as exc:
                proof = None; errors.append(str(exc))
            aliases = ['polymarket:'+m['id'], 'polymarket:'+m['conditionId']]
            matches = exact_overlaps(aliases, [m['question']], benchmark)
            contracts.append({'market_id': m['id'], 'event_group_id': row['event_group_id'], 'proof': proof,
                              'proof_errors': errors, 'benchmark_exact_matches': matches,
                              'existing_reservations': row['candidate']['benchmark_matches']})
            for days in config['observation_days_before']:
                at = iso(timestamp(group['release']['published_at'], 'release')-timedelta(days=days))
                quote, quality = quote_at(db, m['id'], at, config['max_price_age_seconds'])
                data_errors = list(errors); record = payload = None
                if proof is not None:
                    try:
                        _, _, prior, official = prepared[row['event_group_id']]
                        record, payload = observation(row, group, proof, prior, official, quote, at, version)
                    except (ValidationError, KeyError, ValueError, TypeError) as exc:
                        data_errors.append(str(exc))
                if quote is None: data_errors.append('historical_quote_unavailable:'+quality['reason'])
                blockers = ['semantic_benchmark_and_cross_parent_review_required', 'chronological_split_not_frozen',
                            'insufficient_independent_event_groups_for_formal_evaluation']
                if matches or row['candidate']['benchmark_matches']: blockers.append('reserved_benchmark_overlap')
                sid = 'pma:'+m['id']+':'+at
                candidates.append({'sample_id': sid, 'event_group_id': row['event_group_id'], 'observation_time': at,
                    'record': record, 'price_quality': quality, 'data_blockers': data_errors, 'release_blockers': blockers,
                    'proof_complete': not data_errors and record is not None,
                    'ready_for_training': False, 'ready_for_scoring': False, 'split': None})
                if record is not None:
                    outcomes.append({'sample_id': sid, 'label': record['label'], 'official_crosscheck': 'matched'})
                if not data_errors and payload is not None:
                    review_inputs.append({'sample_id': sid, 'event_group_id': row['event_group_id'], 'input': payload,
                                          'usage': 'review_only'})
    artifacts = {'contracts.jsonl': contracts, 'candidates.jsonl': candidates, 'review_inputs.jsonl': review_inputs,
                 'outcomes.jsonl': outcomes, 'event_groups.jsonl': groups}
    for name, rows in artifacts.items():
        (output/name).write_text(''.join(json_text(r).replace('\n', '')+'\n' for r in rows))
    result = {'schema_version': '1', 'kind': 'pma_historical_proof_pilot', 'dataset_version': version,
        'config_sha256': sha256_file(config_path), 'capture_report_sha256': sha256_file(capture/'report.json'), 'code_hashes': code_hashes,
        'selected_contracts': len(selected), 'event_groups': len(groups), 'planned_observations': len(candidates),
        'verified_rule_and_settlement_contracts': sum(c['proof'] is not None for c in contracts),
        'verified_observation_labels': len(outcomes), 'proof_complete_observations': len(review_inputs),
        'proof_complete_event_groups': len({c['event_group_id'] for c in candidates if c['proof_complete']}),
        'data_blocker_counts': dict(Counter(b for c in candidates for b in c['data_blockers'])),
        'release_blocker_counts': dict(Counter(b for c in candidates for b in c['release_blockers'])),
        'admitted_training_rows': 0, 'admitted_evaluation_rows': 0, 'model_calls': 0,
        'artifact_hashes': {name: sha256_file(output/name) for name in artifacts},
        'rule_review_id': policy['review_id'], 'status': 'proof_pilot_not_formal_release',
        'limitations': ['Provider/explorer attestations, not an independent chain-consensus proof.',
            'Official dated releases assert historical publication; current retrieval is not a contemporaneous local snapshot.',
            'Only one prior release statement/headline is evidence; this is not a complete historical information set.',
            'Block-mean trade prices are not executable quotes. Missing prices remain in the denominator.',
            'Same-event grouping is complete only within the selected parents; correlated parents and benchmark overlaps need review.',
            'No train/validation/test split, teacher targets or forecast-quality claims; pretrained historical knowledge remains a risk.']}
    (output/'report.json').write_text(json_text(result))
    return result
