"""Bounded macro-market inventory. Current metadata never becomes replay evidence.

This stage freezes a *search scope*, including missing slots and exclusions.
It does not freeze a scoring dataset or certify settlement, rules, or timestamps.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .benchmark_review import exact_overlaps, read_index
from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .market_dataset import now
from .schema import ValidationError, iso, timestamp
from .synthetic_sft import canonical_hash

GAMMA = 'https://gamma-api.polymarket.com/events/'
RESOLUTIONS = 'https://data-api.polymarket.com/v2/resolutions?event_id='
KALSHI = 'https://external-api.kalshi.com/trade-api/v2/'
FORBIDDEN = ('training', 'prompt_tuning', 'reward_tuning', 'checkpoint_selection')


def load_plan(path):
    return validate_plan(strict_json(path.read_text()))


def validate_plan(plan):
    if (plan.get('schema_version') != '1' or plan.get('purpose') != 'evaluation_discovery'
            or any(plan.get(k) is not False for k in FORBIDDEN)
            or not plan.get('groups') or len(plan['groups']) > 30
            or plan.get('observation_days_before') != [7, 1]):
        raise ValidationError('invalid macro discovery policy')
    seen_groups, seen_native = set(), set()
    for g in plan['groups']:
        if (g['event_group_id'] in seen_groups or g['family'] not in ('fomc', 'cpi_yoy', 'unemployment_u3')
                or not re.fullmatch(r'\d{4}-\d{2}', g['period']) or not g['scope_rationale']):
            raise ValidationError('invalid or duplicate discovery group')
        seen_groups.add(g['event_group_id'])
        if timestamp(g['calendar_reference_time'], 'calendar time') >= timestamp(plan['scope_cutoff'], 'cutoff'):
            raise ValidationError('future calendar slot in historical scope')
        for native in g['polymarket_event_ids']:
            if not isinstance(native, str) or not native.isdigit() or ('poly', native) in seen_native:
                raise ValidationError('duplicate or invalid native event')
            seen_native.add(('poly', native))
        for native in g['kalshi_event_tickers']:
            # These mappings were verified in their native series; KXUE is international.
            series = {'fomc': 'KXFED', 'cpi_yoy': 'KXCPIYOY'}.get(g['family'])
            expected = (series + '-' + datetime.strptime(g['period'], '%Y-%m').strftime('%y%b').upper()) if series else None
            if native != expected or ('kalshi', native) in seen_native:
                raise ValidationError('unverified Kalshi period or country mapping')
            seen_native.add(('kalshi', native))
    if len(request_urls(plan)) > 100:raise ValidationError('discovery request budget exceeded')
    return plan


def request_urls(plan):
    urls = []
    for g in plan['groups']:
        for event in g['polymarket_event_ids']:
            urls.extend([GAMMA+event, RESOLUTIONS+event])
        for event in g['kalshi_event_tickers']:
            urls.extend([KALSHI+'events/'+event+'?with_nested_markets=true',
                         KALSHI+'historical/markets?event_ticker='+event+'&limit=1000'])
    return urls


def fetch(url):
    """Only called with URLs derived from the validated, bounded plan."""
    try:
        with urlopen(Request(url, headers={'User-Agent': 'ForetellMesh/0.1 public-data-research'}), timeout=25) as r:
            if r.url != url:raise ValidationError('unexpected discovery redirect')
            status, raw, date = r.status, r.read(4_000_001), r.headers.get('Date')
    except HTTPError as e:
        status, raw, date = e.code, e.read(4_000_001), e.headers.get('Date')
    except (URLError, TimeoutError, ConnectionError) as e:
        status, raw, date = 0, str(e).encode(), None
    if len(raw) > 4_000_000:raise ValidationError('discovery response too large')
    return status, raw, date


def capture(config, output):
    plan = load_plan(config)
    if output.exists():raise ValidationError('discovery archive exists')
    # Bind original search/calendar captures before any new requests.
    sources = []
    for ref in plan['source_refs']:
        raw = Path(ref['path']).read_bytes()
        if sha256_bytes(raw) != ref['sha256']:raise ValidationError('discovery source changed')
        sources.append((ref, raw))
    output.mkdir(parents=True); (output/'raw').mkdir(); (output/'sources').mkdir()
    archived_sources = []
    for n, (ref, raw) in enumerate(sources):
        name = f'sources/{n:03d}.bin'; (output/name).write_bytes(raw)
        archived_sources.append({**ref, 'file': name})
    requests = []
    for n, url in enumerate(request_urls(plan)):
        start = now(); status, raw, date = fetch(url); end = now()
        name = f'raw/{n:03d}.bin'; (output/name).write_bytes(raw)
        requests.append({'url': url, 'file': name, 'sha256': sha256_bytes(raw), 'status': status,
                         'started_at': start, 'completed_at': end, 'server_date': date})
    manifest = {'schema_version': '1', 'kind': 'macro_discovery_capture', 'plan': plan,
                'plan_sha256': canonical_hash(plan), 'sources': archived_sources,
                'requests': requests, 'requests_sha256': canonical_hash(requests)}
    (output/'manifest.json').write_text(json_text(manifest))
    return {'requests': len(requests), 'failures': sum(r['status'] != 200 for r in requests)}


def read_capture(root):
    m = strict_json((root/'manifest.json').read_text())
    if (m.get('schema_version') != '1' or m.get('kind') != 'macro_discovery_capture'
            or canonical_hash(m['plan']) != m['plan_sha256']
            or canonical_hash(m['requests']) != m['requests_sha256']):
        raise ValidationError('discovery manifest mismatch')
    validate_plan(m['plan'])
    if [r['url'] for r in m['requests']] != request_urls(m['plan']):
        raise ValidationError('missing or reordered discovery requests')
    expected = [{k: v for k, v in ref.items() if k != 'file'} for ref in m['sources']]
    if expected != m['plan']['source_refs']:raise ValidationError('unbound discovery source')
    result = {}
    for ref in m['sources'] + m['requests']:
        path = (root/ref['file']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != ref['sha256']:
            raise ValidationError('discovery body hash or path mismatch')
    for ref in m['requests']:
        if timestamp(ref['started_at'], 'start') > timestamp(ref['completed_at'], 'end'):
            raise ValidationError('invalid discovery request interval')
        result[ref['url']] = (ref, strict_json((root/ref['file']).read_text()) if ref['status'] == 200 else None)
    return m, result


def poly_inventory(group, event, states, ref, index):
    flags, markets, matches = set(), [], set()
    period = datetime.strptime(group['period'], '%Y-%m').strftime('%B %Y').lower()
    release = timestamp(group['calendar_reference_time'], 'calendar time')
    if states is None:flags.add('resolution_metadata_unavailable')
    state_rows = (states or {}).get('data', [])
    state_map = {s['condition_id'].lower(): s for s in state_rows}
    if len(state_map) != len(state_rows):flags.add('duplicate_resolution_identity')
    seen = set()
    for m in event.get('markets', []):
        condition = str(m.get('conditionId', '')).lower()
        if not re.fullmatch(r'0x[0-9a-f]{64}', condition) or condition in seen:
            raise ValidationError('invalid or duplicate condition identity')
        seen.add(condition)
        aliases = ['polymarket:'+str(m['id']), 'polymarket:'+condition]
        matches.update(exact_overlaps(aliases, [m['question']], index))
        text = (m['question']+' '+m.get('description', '')).lower()
        # Discovery guards only; full contract semantics and historical rules remain pending.
        if group['family'] != 'fomc' and period not in text:flags.add('period_metadata_mismatch')
        if group['family'] == 'fomc':
            day = datetime.strptime(group['period'], '%Y-%m')
            if day.strftime('%B').lower() not in text or str(day.year) not in text or 'fed' not in text:
                flags.add('period_metadata_mismatch')
        if group['family'] == 'cpi_yoy' and not all(s in text for s in ('12-month', 'bls')):
            flags.add('measurement_metadata_mismatch')
        if group['family'] == 'unemployment_u3' and not all(s in text for s in ('u-3', 'bls')):
            flags.add('measurement_metadata_mismatch')
        closed_time = m.get('closedTime')
        if closed_time:
            # Gamma uses both ISO T/Z and space-separated +00 representations.
            closed = datetime.fromisoformat(closed_time.replace('Z', '+00:00'))
            if closed.tzinfo is None:raise ValidationError('naive close time')
            if closed < release - timedelta(days=1):flags.add('closed_before_planned_observations')
        state = state_map.get(condition)
        if state and state.get('status') == 'resolved' and state.get('last_update_timestamp'):
            if datetime.fromtimestamp(int(state['last_update_timestamp']), timezone.utc) < release-timedelta(days=1):
                flags.add('resolved_status_predates_planned_observations')
        if not state:flags.add('resolution_metadata_missing_for_contract')
        if m.get('closed') is not True:flags.add('contract_not_closed')
        if m.get('endDate') and timestamp(m['endDate'], 'end date').date() != release.date():
            flags.add('market_end_date_differs_from_calendar')
        markets.append({'market_id': str(m['id']), 'condition_id': condition, 'aliases': aliases,
                        'question': m['question'], 'metadata_closed_time': closed_time,
                        'metadata_resolution_status': state.get('status') if state else None,
                        'metadata_last_update_timestamp': state.get('last_update_timestamp') if state else None})
    if not markets:flags.add('empty_event')
    quarantine = {'closed_before_planned_observations', 'resolved_status_predates_planned_observations',
                  'period_metadata_mismatch', 'measurement_metadata_mismatch', 'contract_not_closed',
                  'duplicate_resolution_identity', 'empty_event'}
    return {'platform': 'polymarket', 'native_event_id': str(event['id']), 'title': event.get('title'),
            'source_ref': ref, 'contracts': markets, 'flags': sorted(flags), 'exact_overlap_ids': sorted(matches),
            'disposition': 'quarantined' if flags & quarantine else 'pending_historical_evidence'}


def build(root, heldout_index, output):
    if output.exists():raise ValidationError('discovery inventory exists')
    manifest, bodies = read_capture(root); index = read_index(heldout_index)
    groups = []
    for spec in manifest['plan']['groups']:
        natives, matches, missing = [], set(), []
        for event_id in spec['polymarket_event_ids']:
            ref, event = bodies[GAMMA+event_id]; _, states = bodies[RESOLUTIONS+event_id]
            if event is None:
                missing.append({'platform': 'polymarket', 'id': event_id, 'reason': 'request_failed'}); continue
            if str(event.get('id')) != event_id:raise ValidationError('native event response mismatch')
            native = poly_inventory(spec, event, states, ref, index)
            natives.append(native); matches.update(native['exact_overlap_ids'])
        for ticker in spec['kalshi_event_tickers']:
            ref, event = bodies[KALSHI+'events/'+ticker+'?with_nested_markets=true']
            href, history = bodies[KALSHI+'historical/markets?event_ticker='+ticker+'&limit=1000']
            if event is None:
                missing.append({'platform': 'kalshi', 'id': ticker, 'reason': 'request_failed'}); continue
            if event.get('event', {}).get('event_ticker') != ticker:raise ValidationError('Kalshi event mismatch')
            flags = set(); markets = {}
            if history is None:flags.add('historical_contract_inventory_unavailable')
            if history and history.get('cursor'):flags.add('historical_contract_inventory_truncated')
            for m in event.get('markets', []) + event['event'].get('markets', []) + (history or {}).get('markets', []):
                if m.get('event_ticker') != ticker:raise ValidationError('cross-event Kalshi contract')
                alias = 'kalshi:'+m['ticker']
                found = exact_overlaps([alias], [m.get('title', '')], index); matches.update(found)
                # Endpoint copies may differ in optional fields; inventory identity is the ticker.
                markets[m['ticker']] = {'market_id': m['ticker'], 'aliases': [alias], 'question': m.get('title')}
            if not markets:flags.add('empty_event')
            natives.append({'platform': 'kalshi', 'native_event_id': ticker, 'source_ref': ref,
                            'historical_source_ref': href, 'contracts': list(markets.values()), 'flags': sorted(flags),
                            'disposition': 'quarantined' if flags else 'pending_historical_evidence'})
        if not spec['polymarket_event_ids']:missing.append({'platform': 'polymarket', 'reason': 'not_found_in_bounded_search'})
        if not spec['kalshi_event_tickers']:missing.append({'platform': 'kalshi', 'reason': 'native_series_mapping_not_verified'})
        # One exact identity hit excludes the *whole release*, across brackets and platforms.
        release = timestamp(spec['calendar_reference_time'], 'calendar time')
        groups.append({**spec, 'planned_observation_times': [iso(release-timedelta(days=d)) for d in (7, 1)],
                       'native_events': natives, 'missing': missing, 'exact_overlap_ids': sorted(matches),
                       'disposition': 'exclude_from_new_cohort' if matches else 'evaluation_only_pending_evidence',
                       'ready_for_scoring': False, 'ready_for_training': False})
    rows = [n for g in groups for n in g['native_events']]
    report = {'schema_version': '1', 'kind': 'macro_discovery_inventory', 'status': 'discovery_only',
              'plan_sha256': manifest['plan_sha256'], 'capture_manifest_sha256': sha256_file(root/'manifest.json'),
              'heldout_index_sha256': sha256_file(heldout_index), 'planned_groups': len(groups),
              'groups_by_family': dict(Counter(g['family'] for g in groups)),
              'groups_with_polymarket': sum(any(n['platform'] == 'polymarket' and n['contracts'] for n in g['native_events']) for g in groups),
              'groups_with_kalshi': sum(any(n['platform'] == 'kalshi' and n['contracts'] for n in g['native_events']) for g in groups),
              'pending_groups_after_identity_exclusions': sum(not g['exact_overlap_ids'] and any(
                  n['contracts'] and n['disposition'] != 'quarantined' for n in g['native_events']) for g in groups),
              'pending_contracts_after_identity_exclusions': sum(len(n['contracts']) for g in groups
                  if not g['exact_overlap_ids'] for n in g['native_events'] if n['disposition'] != 'quarantined'),
              'exact_overlap_excluded_groups': [g['event_group_id'] for g in groups if g['exact_overlap_ids']],
              'native_events_by_platform': dict(Counter(n['platform'] for n in rows)),
              'contracts_by_platform': dict(Counter(n['platform'] for n in rows for _ in n['contracts'])),
              'quarantined_native_events': [{'platform': n['platform'], 'id': n['native_event_id'], 'flags': n['flags']}
                                            for n in rows if n['disposition'] == 'quarantined'],
              'native_event_flag_counts': dict(Counter(f for n in rows for f in n['flags'])),
              'missing': [{'event_group_id': g['event_group_id'], **m} for g in groups for m in g['missing']],
              'admitted_rows': 0, 'model_calls': 0, 'score_metrics': None,
              'remaining_gates': ['dated_first_release_and_revision_audit', 'historical_contract_rules',
                                  'exact_final_settlement', 'pre_cutoff_quotes_and_evidence',
                                  'semantic_benchmark_review', 'chronological_split_and_checkpoint_freeze'],
              'limitations': ['Current calendar is a discovery reference, not proof of first public release or past-known schedule.',
                             'Search pagination/recall is not exhaustive; all planned missing slots remain in the denominator.',
                             'Resolution API status/update times only flag suspicious records; they never create outcome labels.',
                             'Macro releases may share drivers; group counts do not establish statistical independence.',
                             'All catalogue data are evaluation-only metadata; no forecast inputs or training rows are emitted.']}
    output.mkdir(parents=True)
    (output/'inventory.json').write_text(json_text({'groups': groups}))
    report['inventory_sha256'] = sha256_file(output/'inventory.json')
    (output/'report.json').write_text(json_text(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture'); c.add_argument('--config', type=Path, required=True); c.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('build'); b.add_argument('--capture', type=Path, required=True)
    b.add_argument('--heldout-index', type=Path, required=True); b.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    result = capture(a.config, a.output) if a.command == 'capture' else build(a.capture, a.heldout_index, a.output)
    print(json_text({k: v for k, v in result.items() if k not in ('missing', 'limitations', 'remaining_gates')}))


if __name__ == '__main__':main()
