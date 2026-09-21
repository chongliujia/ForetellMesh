"""Capture a bounded, fixed list of public SPF HTML releases, without training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def now():
    return datetime.now(timezone.utc).isoformat()


def capture(spec, output):
    key, url = spec
    row = {'key': key, 'url': url, 'started_at': now()}
    try:
        request = Request(url, headers={'User-Agent': 'ForetellMesh public research provenance audit'})
        with urlopen(request, timeout=40) as response:
            content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError('source exceeds bounded capture size')
            if response.status != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                raise ValueError('not a successful HTML response')
            if not response.url.startswith('https://www.philadelphiafed.org/'):
                raise ValueError('redirect outside official source')
            row.update(status='captured', final_url=response.url,
                       content_type=response.headers.get('Content-Type'),
                       http_date=response.headers.get('Date'), bytes=len(content),
                       sha256=hashlib.sha256(content).hexdigest(), artifact=key+'.html')
            (output / row['artifact']).write_bytes(content)
    except Exception as exc:
        row.update(status='failed', error=type(exc).__name__+': '+str(exc))
    row['completed_at'] = now()
    (output / (key+'.capture.json')).write_text(json.dumps(row, indent=2)+'\n')
    print(key, row['status'], flush=True)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes(); c = json.loads(config_bytes)
    if c['years'] != list(range(2019, 2025)) or c['quarters'] != [1, 2, 3, 4] or c['collection_only'] is not True:
        raise ValueError('unsupported capture scope')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.json').write_bytes(config_bytes)
    prefix = 'https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/'
    specs = [(f'spf-q{q}-{y}', prefix+f'spf-q{q}-{y}') for y in c['years'] for q in c['quarters']]
    specs += [('spf-faqs', prefix+'spf-faqs'),
              ('spf-overview', prefix+'survey-of-professional-forecasters')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda spec: capture(spec, args.output), specs))
    manifest = {'kind': 'spf_supervision_capture', 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'captures': rows, 'training_performed': False, 'existing_holdouts_opened': False}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
