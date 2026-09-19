"""Parse a dated release's headline, never its later-vintage database series.

BLS evidence here is a frozen web-tool text extraction, not raw HTML. Its
embargo header and headline are publisher assertions, not a first-fetch proof.
"""
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from .data import sha256_file, strict_json
from .historical_market import fed_upper_bound
from .historical_sources import fed_statement, normalize
from .schema import ValidationError, iso, timestamp


def bls_headline(envelope):
    family = envelope['family']; period = datetime.strptime(envelope['period'], '%Y-%m')
    kind = {'cpi_yoy': 'cpi', 'unemployment_u3': 'empsit'}.get(family)
    url = envelope['url']
    if (kind is None or envelope.get('capture_method') != 'web_tool_extracted_text'
            or not re.fullmatch(r'https://www\.bls\.gov/news\.release/archives/'+kind+r'_\d{8}\.htm', url)):
        raise ValidationError('unsupported dated BLS source')
    response = envelope['response']
    if not isinstance(response, str) or url not in response.split('\n', 1)[0]:
        raise ValidationError('BLS source binding missing')
    # Strip line labels only; never reinterpret tool metadata as publication time.
    text = re.sub(r'(?m)^L\d+:[ \t]*', '', response)
    if text.count('Transmission of material') != 1:raise ValidationError('ambiguous BLS release header')
    body = text.split('Transmission of material', 1)[1]
    head = re.search(r'8:30 a\.m\. \(ET\) ([A-Za-z]+), ([A-Za-z]+ \d{1,2}, \d{4})', body[:500])
    if not head:raise ValidationError('BLS embargo clock missing')
    local = datetime.strptime(head[2], '%B %d, %Y').replace(hour=8, minute=30, tzinfo=ZoneInfo('America/New_York'))
    published = iso(local)
    if local.strftime('%A') != head[1] or local.strftime('%m%d%Y') not in url:
        raise ValidationError('BLS URL/date/weekday mismatch')
    if timestamp(published, 'release') != timestamp(envelope['expected_time'], 'expected'):
        raise ValidationError('BLS release differs from planned time')
    if not timestamp(envelope['started_at'], 'fetch start') <= timestamp(envelope['completed_at'], 'fetch end'):
        raise ValidationError('BLS capture clock backwards')
    if timestamp(published, 'release') > timestamp(envelope['completed_at'], 'fetch end'):
        raise ValidationError('BLS release after retrieval')
    title = ('CONSUMER PRICE INDEX' if kind == 'cpi' else 'THE EMPLOYMENT SITUATION')
    title_match = re.search(title+r'\s*[-–—]+\s*'+period.strftime('%B %Y'), body, re.I)
    if not title_match:raise ValidationError('BLS headline period mismatch')
    if re.search(r'\b(correction|corrected|reissued)\b', body[:title_match.end()], re.I):
        raise ValidationError('BLS correction requires separate vintage review')
    narrative = body[title_match.end():]
    # Only the headline paragraph is a fact in this release. Previous-month
    # revised tables and later publication schedules must not enter this fact.
    paragraphs = [normalize(p) for p in re.split(r'\n\s*\n', narrative) if p.strip()]
    if not paragraphs:raise ValidationError('BLS headline paragraph missing')
    first = paragraphs[0]
    if family == 'cpi_yoy':
        match = re.findall(r'Over the last 12 months, the all items index (increased|decreased) (\d+(?:\.\d+)?) percent before seasonal adjustment', first)
        if len(match) != 1:raise ValidationError('ambiguous all-items annual CPI headline')
        value = Decimal(match[0][1]) * (-1 if match[0][0] == 'decreased' else 1)
        fact = f'BLS dated release headline for {envelope["period"]}: US all-items CPI-U change over 12 months, before seasonal adjustment, {value}%.'
        unit = 'all_items_cpi_u_yoy_nsa_percent'
    else:
        match = re.findall(r'unemployment rate[^.]{0,100}?(\d+\.\d+) percent', first, re.I)
        if len(match) != 1:raise ValidationError('ambiguous headline unemployment rate')
        value = Decimal(match[0])
        if not 0 <= value <= 100:raise ValidationError('invalid unemployment percentage')
        fact = f'BLS dated release headline for {envelope["period"]}: US seasonally adjusted U-3 unemployment rate, {value}%.'
        unit = 'us_u3_sa_percent'
    return {'family': family, 'period': envelope['period'], 'published_at': published, 'source': url,
            'value': str(value), 'unit': unit, 'text': fact, 'headline_excerpt': first,
            'capture_method': 'web_tool_extracted_text',
            'trust_scope': 'Dated official archive header/headline via web-tool extraction; no independent first-publication snapshot or whole-page correction audit.'}


def load_bls(root: Path, refs: list[dict]) -> list[dict]:
    results = []
    for ref in refs:
        path = (root/ref['file']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != ref['sha256']:
            raise ValidationError('BLS extraction hash changed')
        result = bls_headline(strict_json(path.read_text()))
        result['capture_ref'] = ref
        results.append(result)
    identities = [(r['family'], r['period']) for r in results]
    if len(set(identities)) != len(identities):raise ValidationError('duplicate BLS release identity')
    return results


def fed_headline(raw, url, expected_time):
    statement = fed_statement(raw, url, expected_time)
    # Federal Reserve pages use U+2011 in mixed fractions such as 3‑3/4.
    high = fed_upper_bound(statement['text'].replace('\u2011', '-'))
    return {'family': 'fomc', 'period': expected_time[:7], 'published_at': statement['published_at'],
            'source': url, 'value': str(Decimal(high.numerator)/Decimal(high.denominator)),
            'unit': 'fed_target_upper_percent', 'text': statement['text'],
            'capture_method': 'http_raw', 'trust_scope': 'Dated official Federal Reserve archived statement.'}


def prior_release(releases, family, observation):
    cutoff = timestamp(observation, 'observation')
    prior = [r for r in releases if r['family'] == family and timestamp(r['published_at'], 'publication') <= cutoff]
    return max(prior, key=lambda r: timestamp(r['published_at'], 'publication')) if prior else None
