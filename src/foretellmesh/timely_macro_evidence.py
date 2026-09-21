"""Dated BLS narrative facts and causal additions to the fixed train pilot."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from .agent_baseline_data import input_context
from .data import sha256_file, strict_json
from .evaluation import json_text
from .market_development import rows
from .schema import ValidationError, iso, timestamp
from .sft_data import jsonl
from .synthetic_sft import canonical_hash


def one(pattern, text):
    found = re.findall(pattern, text, re.I)
    if len(found) != 1:
        raise ValidationError('ambiguous/missing narrative field: ' + pattern)
    return found[0]


def parse_capture(capture, spec):
    url = spec['url']; response = capture['response']
    if (capture['url'] != url or capture['key'] != spec['key'] or capture['status'] != 'fulfilled'
            or capture['capture_method'] != 'web_tool_extracted_text'
            or not re.fullmatch(r'https://www\.bls\.gov/news\.release/archives/(cpi|empsit)_\d{8}\.htm', url)
            or not timestamp(capture['started_at'], 'fetch start') <= timestamp(capture['completed_at'], 'fetch end')):
        raise ValidationError('invalid BLS capture binding')
    sections = [s.strip() for s in re.split(r'-{80}(?=\s*(?:Consumer Price Index|Employment Situation|Internal Error))', response)]
    opening = [s for s in sections if 'Source: open(' in s and url in s.split('\n', 1)[0]]
    if len(opening) != 1:
        raise ValidationError('missing/ambiguous source opening')
    text = re.sub(r'L\d+:[ \t]*', '', opening[0])
    if text.count('Transmission of material') != 1:
        raise ValidationError('missing publication header')
    body = text.split('Transmission of material', 1)[1]
    weekday, date = one(r'8:30 a\.m\. \(ET\)\s+([A-Za-z]+),\s+([A-Za-z]+ \d{1,2}, \d{4})', body[:500])
    published = datetime.strptime(date, '%B %d, %Y').replace(hour=8, minute=30, tzinfo=ZoneInfo('America/New_York'))
    if (published.strftime('%A') != weekday or published.strftime('%m%d%Y') not in url
            or published != timestamp(spec['expected_publication'], 'planned publication')
            or published > timestamp(capture['completed_at'], 'retrieval')):
        raise ValidationError('publication date/calendar binding differs')
    title = 'CONSUMER PRICE INDEX' if spec['family'] == 'cpi' else 'THE EMPLOYMENT SITUATION'
    period = one(title + r'\s*[-–—]+\s*([A-Za-z]+ \d{4})', body)
    period_dt = datetime.strptime(period.title(), '%B %Y')
    if (period_dt.year, period_dt.month) != ((published.replace(day=1)-timedelta(days=1)).year,
                                           (published.replace(day=1)-timedelta(days=1)).month):
        raise ValidationError('release period not preceding calendar month')
    available = published; review = 'no_correction_marker_found'
    for marker in ('corrected', 'correction', 'reissued'):
        negative = [s for s in sections if f'"pattern":"{marker}"' in s
                    and url in s.split('\n', 1)[0] and f'No matching text found for "{marker}"' in s]
        if len(negative) == 1:
            continue
        # A known same-day reissue is usable only after a conservative full-day delay.
        # Do not assert first-publication identity or silently remove its notice.
        if (spec['key'] == 'cpi_07112024' and marker == 'reissued'
                and 'This news release was reissued on July 11, 2024.' in body
                and 'These data have been removed from tables 2, 6, and 7' in ' '.join(body.split())):
            available = published.replace(hour=0, minute=0)+timedelta(days=1)
            review = 'dated_same_day_reissue_available_next_local_midnight'
        else:
            raise ValidationError('unreviewed correction marker/search failure: ' + marker)
    normalized = ' '.join(body.split())
    if review == 'no_correction_marker_found' and re.search(r'\b(corrected|correction|reissued)\b', normalized, re.I):
        raise ValidationError('capture contradicts negative correction search')
    start = re.search(title + r'\s*[-–—]+\s*[A-Za-z]+ \d{4}', normalized).end()
    narrative = normalized[start:]
    if spec['family'] == 'cpi':
        if 'Table A.' not in narrative:
            raise ValidationError('CPI narrative boundary missing')
        narrative = narrative.split('Table A.', 1)[0]
        change, headline = one(r'Over the last 12 months, the all items index (increased|decreased) (\d+\.\d+) percent before seasonal adjustment', narrative)
        core = one(r'all items less food and energy index (?:rose|increased) (\d+\.\d+) percent over the last 12 months', narrative)
        headline = ('-' if change.lower() == 'decreased' else '') + headline
        values = {'cpi_all_items_yoy_nsa': headline, 'cpi_core_yoy_nsa': core}
        fact = f'BLS dated release for {period_dt:%Y-%m}: all-items CPI-U year-over-year before seasonal adjustment {headline}%; all-items-less-food-and-energy CPI year-over-year {core}%.'
    else:
        if 'This news release presents statistics' not in narrative:
            raise ValidationError('employment headline boundary missing')
        narrative = narrative.split('This news release presents statistics', 1)[0]
        unchanged = re.findall(r'Total nonfarm payroll employment was (?:essentially|little) (?:unchanged|changed) in [A-Za-z]+ \(([+-][\d,]+)\)', narrative)
        if unchanged:
            if len(unchanged) != 1: raise ValidationError('ambiguous unchanged payroll headline')
            verb, count = 'increased', unchanged[0]
        else:
            verb, count = one(r'Total nonfarm payroll employment (rose|increased|declined|decreased)(?: by)? ([\d,]+)', narrative)
        # "rose 0.3 percentage point to 3.7 percent" reports a level of 3.7,
        # not 0.3. Do not confuse changes with levels or stop at a decimal point.
        rate = one(r'unemployment rate.{0,140}?(\d+\.\d+) percent\b', narrative)
        count = ('-' if verb.lower() in ('declined', 'decreased') else '')+count.replace(',', '')
        values = {'nonfarm_payroll_change_sa': count, 'unemployment_u3_sa': rate}
        fact = f'BLS dated release for {period_dt:%Y-%m}: reported current-month seasonally adjusted nonfarm payroll employment change {int(count):+,}; seasonally adjusted U-3 unemployment rate {rate}%. These are this release\'s current-period headline values, not subsequently revised historical series.'
    if available != published:
        fact += ' This version was reissued on the publication date and is conservatively treated as available from the next local midnight.'
    published_utc, available_utc = iso(published.astimezone(timezone.utc)), iso(available.astimezone(timezone.utc))
    evidence = {'evidence_id': 'bls-timely:'+canonical_hash({'url': url, 'values': values, 'available': available_utc})[:20],
                'source': url, 'text': fact, 'published_at': published_utc, 'available_at': available_utc}
    return {'family': spec['family'], 'period': period_dt.strftime('%Y-%m'), 'values': values,
            'review': review, 'narrative_excerpt': narrative, 'evidence': evidence}


def select_facts(facts, cutoff, max_age_days=60):
    chosen = []
    for family in ('cpi', 'empsit'):
        candidates = [f for f in facts if f['family'] == family
                      and timestamp(f['evidence']['available_at'], 'availability') <= cutoff
                      and cutoff-timestamp(f['evidence']['published_at'], 'publication') <= timedelta(days=max_age_days)]
        if candidates:
            chosen.append(max(candidates, key=lambda f: timestamp(f['evidence']['published_at'], 'publication')))
    return chosen


def build(config, raw, output):
    c = strict_json(config.read_text())
    reference = Path(c['reference_run'])
    if sha256_file(reference/'generation_report.json') != c['reference_report_sha256']:
        raise ValidationError('pilot reference changed')
    if c['training'] or c['final_test_opened'] or c['per_family'] != 1 or c['max_source_age_days'] != 60:
        raise ValidationError('unsupported evidence scope')
    facts, failures, bindings = [], [], {}
    for spec in c['sources']:
        path = raw / (spec['key']+'.json')
        bindings[str(path)] = sha256_file(path)
        try:
            facts.append({**parse_capture(strict_json(path.read_text()), spec), 'capture_file': str(path),
                          'capture_sha256': bindings[str(path)]})
        except ValidationError as exc:
            failures.append({'key': spec['key'], 'reason': str(exc)})
    selected = rows(reference/'selected.jsonl')
    members = {r['sample_id']: r for r in rows(Path('data/processed/pma_macro_training_extension_20260920_v2/membership.jsonl'))}
    packets = []
    for r in selected:
        if members[r['sample_id']]['split'] != 'train':
            raise ValidationError('non-training sample in pilot')
        cutoff = timestamp(r['input']['observation_time'], 'cutoff')
        picked = select_facts(facts, cutoff, c['max_source_age_days'])
        evidence = [f['evidence'] for f in picked]
        input_context({**r['input'], 'market': None, 'evidence': evidence})
        packets.append({'sample_id': r['sample_id'], 'observation_time': iso(cutoff), 'evidence': evidence,
                        'families': [f['family'] for f in picked],
                        'newest_age_days': min(((cutoff-timestamp(e['published_at'], 'publication')).total_seconds()/86400
                                               for e in evidence), default=None)})
    result = {'kind': 'timely_macro_evidence_v1', 'config_sha256': sha256_file(config),
              'reference_report_sha256': c['reference_report_sha256'],
              'selected_sha256': sha256_file(reference/'selected.jsonl'), 'source_bindings': bindings,
              'facts': facts, 'failures': failures, 'packets': packets, 'trust_scope': c['trust_scope']}
    if output.exists():
        if strict_json(output.read_text()) != result:
            raise ValidationError('evidence bundle does not reproduce')
    else:
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json_text(result))
    return result


def enrich_jobs(jobs, selected, reference):
    path = Path(reference['path'])
    if sha256_file(path) != reference['sha256']:
        raise ValidationError('external evidence bundle changed')
    bundle = build(Path(reference['config_path']), Path(reference['raw_root']), path)
    packets = {r['sample_id']: r for r in bundle['packets']}
    if set(packets) != {r['sample_id'] for r in selected}:
        raise ValidationError('external evidence sample scope differs')
    for file, digest in bundle['source_bindings'].items():
        if sha256_file(Path(file)) != digest:
            raise ValidationError('external capture changed')
    enriched = deepcopy(jobs)
    for job in enriched:
        if job['arm'] != 'refreshed_blind': continue
        packet = packets[job['original_sample_id']]
        if packet['observation_time'] != job['input']['observation_time']:
            raise ValidationError('external evidence cutoff differs')
        # Same release URL is one information source; richer fields replace old
        # narrow extracts rather than counting the same fact twice.
        evidence = {e['source']: e for e in job['input']['evidence']}
        for e in packet['evidence']: evidence[e['source']] = e
        job['input']['evidence'] = sorted(evidence.values(), key=lambda e: e['evidence_id'])
        input_context(job['input'])
    return enriched


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('config', 'raw', 'output'): parser.add_argument('--'+arg, type=Path, required=True)
    a = parser.parse_args(); r = build(a.config, a.raw, a.output)
    print(json_text({'facts': len(r['facts']), 'failures': r['failures'], 'packets': len(r['packets'])}))
