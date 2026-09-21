"""Source-bound real forecast targets; deliberately not a trainable SFT bundle.

SPF panel probabilities are teacher judgments. They do not supply the original
information set, full project response schema, or event resolution vintage.
Keep these distinctions machine-readable instead of manufacturing completions.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

from .data import sha256_file, strict_json
from .evaluation import json_text
from .schema import ValidationError, timestamp
from .sft_data import jsonl


BLOCKERS = [
    'student_asof_evidence_not_built',
    'teacher_information_cutoff_not_documented',
    'archive_vintage_and_errata_review_required',
    'gdp_resolution_vintage_not_assigned',
    'benchmark_semantic_overlap_review_required',
    'new_development_cohort_not_frozen',
    'project_forecast_response_supervision_incomplete',
]


class ReleaseHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.titles, self.dates, self.tables = [], [], []
        self.heading = self.date = self.cell = None
        self.heading_tag = None
        self.last_heading = ''
        self.table = self.row = None
        self.skip = 0

    @staticmethod
    def clean(parts):
        return ' '.join(''.join(parts).split())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'):
            self.skip += 1
        if self.skip:
            return
        if tag in ('h1', 'h2', 'h3'):
            self.heading, self.heading_tag = [], tag
        if tag == 'p' and attrs.get('itemprop') == 'datePublished':
            self.date = []
        if tag == 'table':
            if self.table is not None:
                raise ValidationError('nested tables are unsupported')
            self.table = {'heading': self.last_heading, 'rows': []}
        if tag == 'tr' and self.table is not None:
            self.row = []
        if tag in ('th', 'td') and self.row is not None:
            self.cell = []
        if tag == 'br':
            self.handle_data(' ')

    def handle_data(self, data):
        if self.skip:
            return
        for parts in (self.heading, self.date, self.cell):
            if parts is not None:
                parts.append(data)

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == self.heading_tag:
            self.last_heading = self.clean(self.heading)
            if tag == 'h1':
                self.titles.append(self.last_heading)
            self.heading = self.heading_tag = None
        if tag == 'p' and self.date is not None:
            self.dates.append(self.clean(self.date)); self.date = None
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(self.clean(self.cell)); self.cell = None
        if tag == 'tr' and self.row is not None:
            self.table['rows'].append(self.row); self.row = None
        if tag == 'table' and self.table is not None:
            self.tables.append(self.table); self.table = None


def quarter_index(year, quarter):
    return year * 4 + quarter - 1


def quarter_name(index):
    year, q = divmod(index, 4)
    return f'{year}:Q{q+1}'


def parse_release(html, key):
    match = re.fullmatch(r'spf-q([1-4])-(20\d{2})', key)
    if not match:
        raise ValidationError('unsupported release identity')
    quarter, year = map(int, match.groups())
    parser = ReleaseHTML(); parser.feed(html); parser.close()
    title = f'{("First", "Second", "Third", "Fourth")[quarter-1]} Quarter {year} Survey of Professional Forecasters'
    if parser.titles != [title] or len(parser.dates) != 1:
        raise ValidationError('release title/date binding failed')
    day = None
    for fmt in ("%d %b '%y", '%B %d, %Y'):
        try:
            day = datetime.strptime(parser.dates[0].replace('’', "'"), fmt).date()
            break
        except ValueError:
            pass
    if day is None:
        raise ValidationError('unsupported publication date')
    if day.year != year or (day.month-1)//3+1 != quarter:
        raise ValidationError('publication date outside survey quarter')
    tables = [t for t in parser.tables if re.fullmatch(
        r'Risk of a Negative Quarter \(%\)(?:\s*Survey Means)?', t['heading'], re.I)]
    if len(tables) != 1:
        raise ValidationError('missing/ambiguous negative-growth survey table')
    rows = tables[0]['rows']
    if (len(rows) != 6 or rows[0] != ['Quarterly data:', 'Previous', 'New']
            or any(len(row) != 3 for row in rows)):
        raise ValidationError('unexpected probability table structure')
    start = quarter_index(year, quarter)
    if [r[0] for r in rows[1:]] != [quarter_name(start+h) for h in range(5)]:
        raise ValidationError('probability horizon sequence mismatch')
    forecasts = []
    for horizon, (target, previous, new) in enumerate(rows[1:]):
        try:
            value = Decimal(new)
        except InvalidOperation as exc:
            raise ValidationError('probability missing/not numeric') from exc
        if not value.is_finite() or not 0 <= value <= 100:
            raise ValidationError('probability outside percent bounds')
        forecasts.append({'target_quarter': target, 'horizon_quarters': horizon,
                          'published_percent': new, 'probability': float(value / 100),
                          'previous_percent_for_audit_only': previous})
    # This is an availability bound, NOT an invented precise publication time.
    bound = datetime.combine(day+timedelta(days=1), datetime.min.time(), ZoneInfo('America/New_York'))
    return {'release_date': day.isoformat(), 'publication_precision': 'day',
            'conservative_public_availability_bound': bound.isoformat(),
            'forecasts': forecasts}


def require_trainable(manifest):
    """Reject candidate target archives at any forecast SFT entry point."""
    if manifest.get('kind') != 'forecast_sft_dataset' or manifest.get('ready_for_sft') is not True:
        raise ValidationError('real forecast target inventory is not an admitted SFT dataset')


def parse_pdf_release(text, key):
    match = re.fullmatch(r'spf-q([1-4])-(20\d{2})', key)
    if not match:
        raise ValidationError('unsupported PDF identity')
    q, y = map(int, match.groups())
    ordinal = ('First', 'Second', 'Third', 'Fourth')[q-1]
    header = text.split('\f')[0]
    if not re.search(rf'\b{ordinal.upper()} QUARTER {y}\b', header):
        raise ValidationError('PDF survey identity differs')
    dates = re.findall(r'Release Date:\s*([A-Za-z]+ \d{1,2}, \d{4})', header)
    if len(dates) != 1:
        raise ValidationError('PDF release date missing/ambiguous')
    pages = [(i+1, p) for i, p in enumerate(text.split('\f'))
             if re.search(r'Risk of a Negative Quarter \(%\)\s+Survey Means', p)]
    if len(pages) != 1:
        raise ValidationError('PDF probability table missing/ambiguous')
    page_number, page = pages[0]
    table = re.split(r'Risk of a Negative Quarter \(%\)\s+Survey Means', page)[1]
    if not re.match(r'\s*Quarterly data:\s+Previous\s+New', table):
        raise ValidationError('PDF probability columns differ')
    rows = re.findall(r'^\s*(\d{4}:Q[1-4])\s+(N\.A\.|\d+\.\d+)\s+(\d+\.\d+)\s*$', table, re.M)
    if len(rows) != 5:
        raise ValidationError('PDF probability row count differs')
    # Reuse the same date, units and horizon validator on extracted cells.
    html = (f'<h1>{ordinal} Quarter {y} Survey of Professional Forecasters</h1>'
            f'<p itemprop="datePublished">{dates[0]}</p>'
            '<h3>Risk of a Negative Quarter (%) Survey Means</h3>'
            '<table><tr><th>Quarterly data:</th><th>Previous</th><th>New</th></tr>')
    for row in rows:
        html += '<tr>'+''.join('<td>'+cell+'</td>' for cell in row)+'</tr>'
    parsed = parse_release(html+'</table>', key)
    return {**parsed, 'pdf_page': page_number}


def load_pdf_review(path):
    review = strict_json(path.read_text()); root = Path(review['capture_root'])
    manifest = strict_json((root/'manifest.json').read_text())
    if sha256_file(root/'manifest.json') != review['capture_manifest_sha256']:
        raise ValidationError('PDF capture manifest changed')
    captures = {r['key']: r for r in manifest['captures']}
    result = {}
    for item in review['releases']:
        key = item['key']; row = captures[key]
        artifact = root/(key+'.pdf'); extracted = root/(key+'.txt')
        if (row['status'] != 'captured' or row['artifact'] != artifact.name
                or sha256_file(artifact) != row['sha256'] or row['sha256'] != item['pdf_sha256']
                or sha256_file(extracted) != row['text_sha256']):
            raise ValidationError('PDF review binding failed')
        actual = subprocess.run(['pdftotext', '-layout', str(artifact), '-'], check=True,
                                capture_output=True).stdout.decode()
        if actual != extracted.read_text():
            raise ValidationError('PDF extraction does not reproduce')
        parsed = parse_pdf_release(actual, key)
        if (parsed['pdf_page'] != item['table_page'] or parsed['release_date'] != item['release_date']
                or [f['published_percent'] for f in parsed['forecasts']] != item['visually_checked_new_percents']):
            raise ValidationError('PDF cells differ from visual review')
        q, y = re.fullmatch(r'spf-q([1-4])-(20\d{2})', key).groups()
        url = ('https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/'
               f'survey-of-professional-forecasters/{y}/spfq{q}{y[2:]}.pdf')
        if row['url'] != url or row['final_url'] != url:
            raise ValidationError('PDF source URL differs')
        if not timestamp(row['started_at'], 'start') <= timestamp(row['completed_at'], 'end'):
            raise ValidationError('PDF capture clock reversed')
        if parsed['release_date'] > timestamp(row['completed_at'], 'retrieval').date().isoformat():
            raise ValidationError('PDF publication after capture')
        result[key] = (parsed, row, artifact)
    return result


def build(config_path, capture_root, output, pdf_review=None):
    config = strict_json(config_path.read_text())
    if (config['collection_only'] is not True or config['automatic_training_admission'] is not False
            or config['years'] != list(range(2019, 2025)) or config['quarters'] != [1, 2, 3, 4]
            or config['horizons'] != [0, 1, 2, 3, 4] or config['extract_column'] != 'New'):
        raise ValidationError('unsupported supervision inventory policy')
    capture = strict_json((capture_root/'manifest.json').read_text())
    if capture['config_sha256'] != sha256_file(config_path):
        raise ValidationError('capture config mismatch')
    expected = {f'spf-q{q}-{y}' for y in config['years'] for q in config['quarters']}
    captures = {r['key']: r for r in capture['captures']}
    if len(captures) != len(capture['captures']) or set(captures) != expected | {'spf-faqs', 'spf-overview'}:
        raise ValidationError('capture scope differs')
    results, failures, releases, fallbacks = [], [], [], []
    pdfs = load_pdf_review(pdf_review) if pdf_review else {}
    if set(pdfs)-expected:
        raise ValidationError('PDF review outside configured releases')
    source_hashes = {'manifest.json': sha256_file(capture_root/'manifest.json')}
    for key in sorted(captures):
        row = captures[key]
        path = capture_root/(key+'.capture.json')
        if strict_json(path.read_text()) != row:
            raise ValidationError('capture record differs from manifest')
        source_hashes[path.name] = sha256_file(path)
        if row['status'] != 'captured':
            failures.append({'key': key, 'stage': 'capture', 'error': row['error']})
            if key not in pdfs:
                continue
        else:
            if row['artifact'] != key+'.html':
                raise ValidationError('unexpected capture artifact path')
            artifact = capture_root/row['artifact']
            if sha256_file(artifact) != row['sha256'] or artifact.stat().st_size != row['bytes']:
                raise ValidationError('source bytes changed')
            source_hashes[artifact.name] = row['sha256']
        expected_url = ('https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/'
                        + ('survey-of-professional-forecasters' if key == 'spf-overview' else key))
        if row['url'] != expected_url or row.get('final_url', expected_url) != expected_url:
            raise ValidationError('source URL binding differs')
        if timestamp(row['started_at'], 'start') > timestamp(row['completed_at'], 'end'):
            raise ValidationError('capture clock reversed')
        if key not in expected:
            continue
        try:
            if row['status'] != 'captured':
                raise ValidationError('HTML transport failed')
            parsed = parse_release(artifact.read_text(), key)
            if datetime.fromisoformat(parsed['release_date']).date() > timestamp(row['completed_at'], 'retrieval').date():
                raise ValidationError('publication after capture')
        except ValidationError as exc:
            if row['status'] == 'captured':
                failures.append({'key': key, 'stage': 'parse', 'error': str(exc)})
            if key not in pdfs:
                continue
            parsed, row, artifact = pdfs[key]
            fallbacks.append(key)
            source_hashes[str(artifact)] = row['sha256']
        releases.append({'key': key, **parsed})
        for f in parsed['forecasts']:
            group = 'macro:us:real_gdp_qoq_negative:'+f['target_quarter'].replace(':', '-')
            results.append({
                'sample_id': 'spf:'+key+':'+f['target_quarter'], 'event_group_id': group,
                'dataset_source': 'philadelphia_fed_spf', 'source_release': key,
                'question': f'Will U.S. real GDP growth from the preceding quarter be negative in {f["target_quarter"]}?',
                'question_origin': 'curator_normalization_of_published_survey_variable',
                'teacher': 'SPF panel mean, not a Federal Reserve institutional forecast',
                'teacher_probability': f['probability'], 'teacher_published_percent': f['published_percent'],
                'target_quarter': f['target_quarter'], 'horizon_quarters': f['horizon_quarters'],
                'provenance': {'kind': 'dated_official_forecast_archive_candidate', 'source_url': row['url'],
                    'raw_artifact': str(artifact), 'raw_sha256': row['sha256'],
                    'retrieved_at': row['completed_at'], 'published_date': parsed['release_date'],
                    'publication_precision': 'day', 'individual_forecast_issue_time': None,
                    'conservative_public_availability_bound': parsed['conservative_public_availability_bound']},
                'student_input': None, 'outcome': None, 'forecast_schema_completion': None,
                'split': 'unassigned', 'ready_for_sft': False, 'blockers': list(BLOCKERS),
            })
    if len({r['sample_id'] for r in results}) != len(results):
        raise ValidationError('duplicate forecast target')
    repeated = Counter(r['event_group_id'] for r in results)
    manifest = {'kind': 'real_forecast_target_inventory', 'schema_version': '1',
                'source': config['source'], 'config_sha256': sha256_file(config_path),
                'source_hashes': source_hashes, 'code_sha256': sha256_file(Path(__file__)),
                'planned_releases': len(expected), 'parsed_releases': len(releases),
                'teacher_target_count': len(results), 'unique_target_quarters': len(repeated),
                'forecasts_per_target_quarter': dict(sorted(repeated.items())),
                'ready_for_sft': False, 'ready_for_sft_count': 0, 'training_started': False,
                'holdouts_opened': False, 'outcome_labels_loaded': False,
                'status': 'candidate_targets_only', 'blockers': list(BLOCKERS),
                'limitations': config['limitations'], 'failures': failures,
                'pdf_fallbacks': fallbacks,
                'pdf_review_sha256': sha256_file(pdf_review) if pdf_review else None,
                'unresolved_releases': sorted(expected-{r['key'] for r in releases})}
    contents = {'targets.jsonl': jsonl(results), 'releases.jsonl': jsonl(releases),
                'review_queue.jsonl': jsonl([{'sample_id': r['sample_id'], 'blockers': r['blockers']} for r in results])}
    from .data import sha256_bytes
    manifest['artifact_hashes'] = {k: sha256_bytes(v.encode()) for k, v in contents.items()}
    contents['manifest.json'] = json_text(manifest)
    if output.exists():
        if any(not (output/k).is_file() or (output/k).read_text() != v for k, v in contents.items()):
            raise ValidationError('supervision inventory does not reproduce')
    else:
        output.mkdir(parents=True)
        for key, text in contents.items():
            (output/key).write_text(text)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'capture', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--pdf-review', type=Path)
    a = parser.parse_args()
    report = build(a.config, a.capture, a.output, a.pdf_review)
    print(json_text({k: v for k, v in report.items() if k not in ('source_hashes', 'forecasts_per_target_quarter')}))
