"""Capture bounded historical evidence for the frozen macro inventory.

All reads are public. The inventory is rebuilt from its source archive before
selection; outcome values never select contracts. Failed requests stay archived.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import re
import tempfile
from urllib.parse import urlencode

from .data import sha256_bytes, sha256_file, strict_json
from .evaluation import json_text
from .historical_market import poly_price_url
from .historical_sources import allowed_url
from .macro_discovery import GAMMA, KALSHI, RESOLUTIONS, build as build_inventory, fetch, read_capture
from .market_dataset import now
from .macro_prices import clob_price_url
from .polymarket_dataset import mapping
from .schema import ValidationError, iso, timestamp
from .synthetic_sft import canonical_hash


def inputs(config_path):
    config = strict_json(config_path.read_text())
    if (config.get('schema_version') != '1' or config.get('purpose') != 'heldout_evidence_collection'
            or config.get('max_requests') != (2200 if config.get('polymarket_price_source') == 'clob' else 1600)
            or config.get('polymarket_price_source') not in (None, 'clob') or config.get('workers') != 4
            or any(config.get(k) is not False for k in ('training', 'prompt_tuning', 'reward_tuning', 'checkpoint_selection'))):
        raise ValidationError('invalid historical supplement policy')
    policy = config['price_policy']
    if (policy.get('price_window_hours') != 6 or policy.get('max_price_age_seconds') != 10800
            or policy.get('polymarket_bucket_seconds') not in (1800, 10800)
            or policy.get('kalshi_period_minutes') != 60
            or config.get('comparison_bucket_seconds') not in (None, 1800)):
        raise ValidationError('unsupported frozen historical price policy')
    if not 1 <= len(config['fed_releases']) <= 10:raise ValidationError('unbounded release sources')
    for release in config['fed_releases']:
        if not allowed_url(release['url']) or not release['url'].startswith('https://www.federalreserve.gov/'):
            raise ValidationError('unsupported dated release source')
        timestamp(release['published_at'], 'release')
    paths = {k: (config_path.parent/config[k]).resolve() for k in ('inventory', 'discovery', 'heldout_index', 'bls_archive')}
    expected = {'inventory': paths['inventory']/'inventory.json', 'discovery': paths['discovery']/'manifest.json',
                'heldout_index': paths['heldout_index'], 'bls_archive': paths['bls_archive']/'manifest.json'}
    if any(sha256_file(p) != config['source_hashes'][k] for k, p in expected.items()):
        raise ValidationError('frozen supplement source changed')
    with tempfile.TemporaryDirectory(prefix='foretellmesh-macro-verify-') as tmp:
        out = Path(tmp)/'inventory'; build_inventory(paths['discovery'], paths['heldout_index'], out)
        if (out/'inventory.json').read_bytes() != (paths['inventory']/'inventory.json').read_bytes():
            raise ValidationError('inventory does not rebuild from native sources')
    _, raw = read_capture(paths['discovery'])
    groups = strict_json((paths['inventory']/'inventory.json').read_text())['groups']
    selected = []
    for group in groups:
        if group['disposition'] == 'exclude_from_new_cohort':continue
        for native in group['native_events']:
            if native['disposition'] == 'quarantined':continue
            if native['platform'] == 'polymarket':
                ref, event = raw[GAMMA+native['native_event_id']]
                sref, states = raw[RESOLUTIONS+native['native_event_id']]
                by_condition = {s['condition_id']: s for s in states['data']}
                for market in event['markets']:
                    selected.append({'group': group, 'platform': 'polymarket', 'market': market, 'market_ref': ref,
                                     'state': by_condition.get(market['conditionId']), 'state_ref': sref})
            else:
                ref, event = raw[KALSHI+'events/'+native['native_event_id']+'?with_nested_markets=true']
                href, history = raw[KALSHI+'historical/markets?event_ticker='+native['native_event_id']+'&limit=1000']
                markets = {}
                for source, rows in ((ref, event.get('markets', [])+event['event'].get('markets', [])),
                                     (href, (history or {}).get('markets', []))):
                    for market in rows:markets[market['ticker']] = (source, market)
                for source, market in markets.values():
                    selected.append({'group': group, 'platform': 'kalshi', 'market': market, 'market_ref': source})
    return config, paths, groups, selected


def kalshi_url(market, observation, policy, cutoff):
    t = timestamp(observation, 'observation'); series = market['event_ticker'].split('-', 1)[0]
    if series not in ('KXFED', 'KXCPIYOY'):raise ValidationError('unsupported historical series')
    settled = market.get('settlement_ts')
    historical = settled and timestamp(settled, 'settlement') < timestamp(cutoff['market_settled_ts'], 'cutoff')
    prefix = 'historical/markets/' if historical else 'series/'+series+'/markets/'
    return KALSHI+prefix+market['ticker']+'/candlesticks?'+urlencode({
        'start_ts': int((t-timedelta(hours=policy['price_window_hours'])).timestamp()),
        'end_ts': int(t.timestamp()), 'period_interval': policy['kalshi_period_minutes']})


def request_plan(config, groups, selected, cutoff):
    urls = []
    for release in config['fed_releases']:urls.append(release['url'])
    urls.extend(KALSHI+'series/'+series for series in ('KXFED', 'KXCPIYOY'))
    policy = config['price_policy']
    for item in selected:
        market, group = item['market'], item['group']
        if item['platform'] == 'polymarket':
            tokens, _ = mapping(market)
            urls.append('https://clob.polymarket.com/clob-markets/'+market['conditionId'])
            tx = (item['state'] or {}).get('transaction_hash')
            if tx and re.fullmatch(r'0x[0-9a-fA-F]{64}', tx):urls.append('https://polygonscan.com/tx/'+tx)
        for observation in group['planned_observation_times']:
            if item['platform'] == 'polymarket':
                if config.get('polymarket_price_source') == 'clob':
                    urls.append(clob_price_url(tokens['Yes'], timestamp(observation, 'observation'), policy))
                urls.append(poly_price_url(tokens['Yes'], timestamp(observation, 'observation'), policy))
                if 'comparison_bucket_seconds' in config:
                    comparison = {**policy, 'polymarket_bucket_seconds': config['comparison_bucket_seconds']}
                    urls.append(poly_price_url(tokens['Yes'], timestamp(observation, 'observation'), comparison))
            else:urls.append(kalshi_url(market, observation, policy, cutoff))
    urls = list(dict.fromkeys(urls))
    if len(urls)+1 > config['max_requests']:raise ValidationError('historical evidence request budget exceeded')
    return urls


def fetched(url):
    started = now(); status, raw, date = fetch(url)
    return {'url': url, 'status': status, 'sha256': sha256_bytes(raw), 'started_at': started,
            'completed_at': now(), 'server_date': date}, raw


def capture(config_path, output, reuse=None):
    config, _, groups, selected = inputs(config_path)
    if output.exists():raise ValidationError('historical supplement exists')
    cached = {}; parent_hash = None
    if reuse is not None:
        parent_hash = sha256_file(reuse/'manifest.json')
        parent = strict_json((reuse/'manifest.json').read_text())
        if (parent.get('kind') != 'macro_history_capture' or parent['config']['source_hashes'] != config['source_hashes']
                or parent['requests_sha256'] != canonical_hash(parent['requests'])
                or parent['config_sha256'] != canonical_hash(parent['config'])):
            raise ValidationError('incompatible reusable evidence archive')
        for ref in parent['requests']:
            path = (reuse/ref['file']).resolve()
            if not path.is_relative_to(reuse.resolve()) or sha256_file(path) != ref['sha256'] or ref['url'] in cached:
                raise ValidationError('reusable evidence hash or identity changed')
            cached[ref['url']] = (ref, path.read_bytes())
    output.mkdir(parents=True); (output/'raw').mkdir()
    requests = []
    def save(result):
        ref, raw = result; ref['file'] = f'raw/{len(requests):04d}.bin'
        (output/ref['file']).write_bytes(raw); requests.append(ref)
        manifest = {'schema_version': '1', 'kind': 'macro_history_capture', 'config': config,
                    'config_sha256': canonical_hash(config), 'requests': requests,
                    'requests_sha256': canonical_hash(requests)}
        if reuse is not None:manifest['reused_archive_sha256'] = parent_hash
        (output/'manifest.json').write_text(json_text(manifest))
        return ref, raw
    def obtain(url):
        if url not in cached:return fetched(url)
        ref, raw = cached[url]
        return dict(ref), raw
    ref, raw = save(obtain(KALSHI+'historical/cutoff'))
    if ref['status'] != 200:raise ValidationError('historical routing cutoff unavailable; failure archived')
    cutoff = strict_json(raw.decode()); urls = request_plan(config, groups, selected, cutoff)
    (output/'request_plan.json').write_text(json_text(urls))
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for result in pool.map(obtain, urls):
            save(result)
            if len(requests) % 100 == 0:print(f'Archived {len(requests)}/{len(urls)+1} public responses', flush=True)
    return {'contracts': len(selected), 'requests': len(requests), 'reused_responses': sum(r['url'] in cached for r in requests),
            'failed_requests': sum(r['status'] != 200 for r in requests)}


def read_history(config_path, root):
    config, paths, groups, selected = inputs(config_path)
    manifest = strict_json((root/'manifest.json').read_text())
    if (manifest.get('kind') != 'macro_history_capture' or manifest.get('schema_version') != '1'
            or manifest['config'] != config or manifest['config_sha256'] != canonical_hash(config)
            or manifest['requests_sha256'] != canonical_hash(manifest['requests'])):
        raise ValidationError('historical supplement manifest changed')
    responses = {}
    for ref in manifest['requests']:
        path = (root/ref['file']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != ref['sha256'] or ref['url'] in responses:
            raise ValidationError('invalid supplement body reference')
        if timestamp(ref['started_at'], 'start') > timestamp(ref['completed_at'], 'end'):
            raise ValidationError('historical request clock backwards')
        responses[ref['url']] = (ref, path.read_bytes())
    first = KALSHI+'historical/cutoff'
    if first not in responses or responses[first][0]['status'] != 200:raise ValidationError('cutoff missing')
    cutoff = strict_json(responses[first][1].decode())
    expected = [first]+request_plan(config, groups, selected, cutoff)
    if [r['url'] for r in manifest['requests']] != expected:raise ValidationError('incomplete historical request coverage')
    return config, paths, groups, selected, cutoff, responses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse', type=Path)
    a = p.parse_args(); print(json_text(capture(a.config, a.output, a.reuse)))


if __name__ == '__main__':main()
