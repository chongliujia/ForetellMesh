"""Reviewed dated BLS headline facts and bounded Polymarket predicates.

Current databases and prior-month revised tables are never historical features.
Web-tool extractions attest publisher content; they are not original snapshots.
"""
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re

from .data import sha256_file, strict_json
from .historical_sources import normalize
from .macro_releases import bls_headline
from .schema import ValidationError, timestamp
from .synthetic_sft import canonical_hash


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def read_bound(root, ref):
    path = (root/ref['file']).resolve()
    require(path.is_relative_to(root.resolve()) and sha256_file(path) == ref['sha256'], 'BLS source artifact changed')
    return strict_json(path.read_text())


def reviewed_headline(envelope, decision, marker_search=None, errata=None):
    """Only two explicitly reviewed exception identities may carry notices."""
    family, period = envelope['family'], envelope['period']
    require((decision['family'], decision['period']) == (family, period), 'BLS review identity differs')
    text = re.sub(r'(?m)^L\d+:[ \t]*', '', envelope['response'])
    kind = decision['disposition']
    if kind == 'no_correction_marker_found':
        if family == 'cpi_yoy':
            search = envelope['correction_search']
            require(search.count(envelope['url']) >= 2 and all(
                f'No matching text found for "{term}"' in search for term in ('correct', 'reissu')),
                'CPI correction search incomplete or positive')
        else:
            require(marker_search and marker_search['url'] == envelope['url'] and all(
                f'No matching text found for "{term}"' in marker_search['response']
                for term in ('corrected', 'correction', 'reissued')), 'U-3 correction search incomplete or positive')
        # Detect notices in the captured headline region even if an indexed
        # search happened to differ. Never silently strip a newly found notice.
        headline_region = text.split('Transmission of material', 1)[-1]
        require(not re.search(r'\b(corrected|correction|reissued)\b', headline_region, re.I),
                'unreviewed correction in BLS excerpt')
    elif kind == 'april_2025_u3_unaffected_errata':
        require(family == 'unemployment_u3' and period in ('2025-04', '2025-05') and errata
                and errata['url'] == 'https://www.bls.gov/bls/errata/cps-corrections-april-2025.htm'
                and 'Major labor force measures, such as the unemployment rate' in errata['response']
                and 'were unaffected.' in errata['response']
                and 'will not be updated to reflect the corrected household survey estimates.' in errata['response'],
                'missing scoped BLS unaffected-headline attestation')
        if period == '2025-04':
            notes = re.findall(r'\(NOTE: BLS reissued this news release on June 3, 2025,.*?'
                               r'https://www\.bls\.gov/bls/errata/cps-corrections-april-2025\.htm\.\)', text, re.S)
            require(len(notes) == 1 and 'were unaffected.' in normalize(notes[0]), 'unrecognized BLS reissue notice')
            # This is a parsing view only. Original text/hash remain immutable;
            # later-added text and the errata are proof metadata, never evidence.
            text = text.replace(notes[0], '')
    else:
        raise ValidationError('unsupported BLS vintage review')
    result = bls_headline({**envelope, 'response': text})
    result['vintage_review'] = kind
    result['trust_scope'] = 'Dated official headline plus reviewed indexed correction search; no independent first-publication snapshot.'
    return result


def load_reviewed_bls(manifest_path, policy_path):
    manifest = strict_json(manifest_path.read_text()); policy = strict_json(policy_path.read_text())
    require(manifest.get('schema_version') == '1' and manifest.get('kind') == 'reviewed_bls_headline_captures'
            and canonical_hash(manifest['refs']) == manifest['refs_sha256'], 'BLS manifest changed')
    require(policy.get('schema_version') == '1' and policy.get('review_id') == 'pma_bls_headline_v1'
            and policy.get('archive_manifest_sha256') == sha256_file(manifest_path)
            and policy.get('historical_tables_allowed') is False and policy.get('whole_release_model_evidence_allowed') is False
            and policy.get('allowed_fields') == ['headline_cpi_yoy_nsa', 'headline_unemployment_u3_sa']
            and policy.get('trust_model') == 'dated_official_headline_and_indexed_correction_search_not_first_fetch_snapshot',
            'unsupported or changed BLS review')
    decisions = {(d['family'], d['period']): d for d in policy['releases']}
    require(len(decisions) == len(policy['releases']) == len(manifest['refs']), 'BLS review coverage differs')
    root = manifest_path.parent; errata = read_bound(root, manifest['errata']); result = {}
    for ref in manifest['refs']:
        envelope = read_bound(root, ref); key = (ref['family'], ref['period'])
        require(key not in result and key in decisions and decisions[key]['capture_sha256'] == ref['sha256']
                and decisions[key]['rationale'] and envelope['url'] == ref['url']
                and key == (envelope['family'], envelope['period']), 'BLS reviewed content or identity changed')
        markers = read_bound(root, ref['marker_search']) if 'marker_search' in ref else None
        fact = reviewed_headline(envelope, decisions[key], markers, errata)
        fact['capture_ref'] = ref
        result[key] = fact
    return result


EXPRESSION = (r'(?:(less than or equal to|greater than or equal to|exactly) )?'
              r'([≤≥]?)(\d+\.\d+)%(?: (or less|or lower|or more|or higher|or greater))?')


def comparator(match):
    words, symbol, number, suffix = match.groups()
    require(sum(bool(x) for x in (words, symbol, suffix)) <= 1, 'ambiguous bracket comparator')
    marker = words or symbol or suffix
    op = 'le' if marker in ('less than or equal to', '≤', 'or less', 'or lower') else (
        'ge' if marker in ('greater than or equal to', '≥', 'or more', 'or higher', 'or greater') else 'eq')
    return op, Decimal(number)


def expected_bls_outcome(market, group, result):
    family, period = group['family'], group['period']
    require(result['family'] == family and result['period'] == period
            and timestamp(result['published_at'], 'official release') == timestamp(group['release']['published_at'], 'planned release'),
            'official BLS event identity differs')
    dt = datetime.strptime(period, '%Y-%m'); month, full = dt.strftime('%B'), dt.strftime('%B %Y')
    question, rules = normalize(market['question']), normalize(market['description'])
    require(full in rules and 'Bureau of Labor Statistics' in rules, 'BLS rule period or source differs')
    if family == 'cpi_yoy':
        require(re.search(r'12[- ]month period ending(?: in)? '+full, rules)
                and 'before seasonal adjustment' in rules and result['unit'] == 'all_items_cpi_u_yoy_nsa_percent',
                'CPI annual unadjusted rule scope missing')
        prefix = r'Will (?:US annual |annual )?inflation (?:increase by|be) '
        matches = [re.fullmatch(prefix+EXPRESSION+' in '+month+r'\?', question)]
    elif family == 'unemployment_u3':
        require('seasonally adjusted' in rules and 'U-3' in rules and result['unit'] == 'us_u3_sa_percent',
                'US seasonally adjusted U-3 rule scope missing')
        matches = [re.fullmatch(pattern, question) for pattern in (
            'Will the '+month+r'(?: '+dt.strftime('%Y')+r')? unemployment rate be '+EXPRESSION+r'\?',
            'Will the unemployment rate for '+month+' be '+EXPRESSION+r'\?',
            'Will US unemployment be '+EXPRESSION+' in '+full+r'\?')]
    else:
        raise ValidationError('unsupported BLS outcome family')
    matches = [m for m in matches if m is not None]
    require(len(matches) == 1, 'unsupported or ambiguous BLS bracket title')
    op, threshold = comparator(matches[0]); value = Decimal(result['value'])
    require(value.is_finite() and value*10 == (value*10).to_integral_value(), 'BLS headline precision differs from rules')
    # Older bracket-specific descriptions must agree with the title; later
    # common descriptions delegate the bracket to the title instead.
    scope = (r'index increased by (\d+\.\d+)%? percent( or less| or lower| or more| or greater| or higher)? over'
             if family == 'cpi_yoy' else re.escape(full)+r' is (\d+\.\d+)%( or less| or lower| or more| or greater| or higher)?(?=[,.])')
    stated = re.findall(scope, rules)
    explicit = r'index increased by \d' if family == 'cpi_yoy' else re.escape(full)+r' is \d'
    require(not re.search(explicit, rules) or stated, 'unsupported bracket-specific rule predicate')
    if stated:
        require(len(stated) == 1, 'multiple bracket-specific rule predicates')
        number, tail = stated[0]; rule_op = {' or less': 'le', ' or lower': 'le', ' or more': 'ge', ' or greater': 'ge', ' or higher': 'ge', '': 'eq'}[tail]
        require((op, threshold) == (rule_op, Decimal(number)), 'BLS title and rule predicate disagree')
    return int(value <= threshold if op == 'le' else value >= threshold if op == 'ge' else value == threshold)
