"""Download official PDF counterparts to review failed HTML releases."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen


ROOT = Path('data/raw/spf_forecast_pdf_review_20260920_v1')
BASE = 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/'
SPECS = [(f'spf-q{q}-2019', BASE+f'2019/spfq{q}19.pdf') for q in range(1, 5)] + [
    ('spf-q3-2020', BASE+'2020/spfq320.pdf'), ('spf-q4-2020', BASE+'2020/spfq420.pdf'),
    ('spf-errata', BASE+'spf-errata.pdf'), ('spf-documentation', BASE+'spf-documentation.pdf')]


def capture(spec):
    key, url = spec
    row = {'key': key, 'url': url, 'started_at': datetime.now(timezone.utc).isoformat()}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'ForetellMesh provenance audit'}), timeout=40) as response:
            data = response.read(8_000_001)
            if response.status != 200 or not data.startswith(b'%PDF-') or len(data) > 8_000_000:
                raise ValueError('invalid/big PDF response')
            if not response.url.startswith(BASE):
                raise ValueError('unexpected PDF redirect')
            row.update(final_url=response.url, http_date=response.headers.get('Date'))
        path = ROOT/(key+'.pdf'); path.write_bytes(data)
        subprocess.run(['pdftotext', '-layout', str(path), str(ROOT/(key+'.txt'))], check=True)
        row.update(status='captured', artifact=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                   text_sha256=hashlib.sha256((ROOT/(key+'.txt')).read_bytes()).hexdigest())
    except Exception as exc:
        row.update(status='failed', error=type(exc).__name__+': '+str(exc))
    row['completed_at'] = datetime.now(timezone.utc).isoformat()
    (ROOT/(key+'.capture.json')).write_text(json.dumps(row, indent=2)+'\n')
    print(key, row['status'], flush=True)
    return row


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT/'capture_source.py').write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(capture, SPECS))
    report = {'kind': 'spf_pdf_source_review_capture', 'captures': rows,
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'text_extractor': subprocess.run(['pdftotext', '-v'], capture_output=True, text=True).stderr.strip()}
    (ROOT/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
