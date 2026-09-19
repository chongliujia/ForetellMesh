"""Audit historical macro observations while isolating outcomes from inputs.

This supplement does not certify full contract rule histories or final CTF
payouts. It records positive evidence and exact remaining gates for every slot.
"""
import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import re

from .adapters import identifier
from .data import sha256_file, strict_json
from .evaluation import json_text
from .historical_market import historical_kalshi_label, historical_poly_label, kalshi_history_quote, poly_history_quote, poly_price_url
from .historical_sources import initialized_question, normalize
from .macro_history_capture import KALSHI, kalshi_url, read_history
from .macro_releases import fed_headline, load_bls, prior_release
from .macro_prices import clob_history_quote, clob_price_url
from .polymarket_dataset import mapping
from .schema import ValidationError, timestamp
from .sft_data import jsonl


def expected_outcome(item, result, previous=None):
    """Strictly supported predicates, with decimal arithmetic at boundaries."""
    m, group = item['market'], item['group']; family = group['family']
    value = Decimal(result['value'])
    period = datetime.strptime(group['period'], '%Y-%m').strftime('%B %Y').lower()
    rule_text = m['rules_primary'] if item['platform'] == 'kalshi' else m['question']+' '+m['description']
    if family != 'fomc' and period not in normalize(rule_text).lower():
        raise ValidationError('contract period differs from official release')
    if item['platform'] == 'kalshi':
        match = re.fullmatch(re.escape(m['event_ticker'])+r'-T(\d+(?:\.\d+)?)', m['ticker'])
        if (not match or m.get('strike_type') != 'greater' or m.get('market_type') != 'binary'
                or Decimal(str(m.get('notional_value_dollars'))) != 1
                or Decimal(str(m['floor_strike'])) != Decimal(match[1])):
            raise ValidationError('unsupported Kalshi threshold predicate')
        if family == 'cpi_yoy' and ('twelve months ending' not in m['rules_primary']
                or 'more than '+format(Decimal(match[1]), '.1f')+'%' not in m['rules_primary']):
            raise ValidationError('Kalshi CPI wording differs from threshold predicate')
        return int(value > Decimal(match[1]))
    question = m['question']
    if family == 'fomc':
        if previous is None:raise ValidationError('previous FOMC target missing')
        change = (value-Decimal(previous['value']))*100
        if change % 25:raise ValidationError('nonstandard FOMC increment requires contract rounding review')
        if re.match(r'(?:Will there be no change|No change) in Fed interest rates ', question, re.I):return int(change == 0)
        match = re.match(r'(?:Will the )?Fed (decreases?|increases?) interest rates by (25|50)(\+)? bps ', question)
        if not match:raise ValidationError('unsupported FOMC change predicate')
        signed = -change if match[1].startswith('decrease') else change
        return int(signed >= Decimal(match[2]) if match[3] else signed == Decimal(match[2]))
    if family == 'cpi_yoy':
        if '12-month' not in m['description'] or 'before seasonal adjustment' not in m['description']:
            raise ValidationError('CPI annual unadjusted rule scope missing')
        match = re.fullmatch(r'Will annual inflation (?:increase by|be) ([≤≥]?)(\d+\.\d+)%\s*(or less|or more)? in [A-Za-z]+\?', question)
    else:
        if 'U-3' not in m['description'] or 'seasonally adjusted' not in m['description']:
            raise ValidationError('US seasonally adjusted U-3 rule scope missing')
        match = re.fullmatch(r'Will the [A-Za-z]+ \d{4} unemployment rate be ([≤≥]?)(\d+\.\d+)%\s*(or less|or more)?\?', question)
    if not match:raise ValidationError('unsupported macro numeric predicate')
    sign, threshold, words = match.groups(); threshold = Decimal(threshold)
    if sign and words:raise ValidationError('ambiguous threshold comparator')
    if sign == '≤' or words == 'or less':return int(value <= threshold)
    if sign == '≥' or words == 'or more':return int(value >= threshold)
    return int(value == threshold)


def build(config_path, capture, output):
    if output.exists():raise ValidationError('historical macro audit exists')
    config, paths, groups, selected, cutoff, responses = read_history(config_path, capture)
    def get(url, binary=False):
        ref, raw = responses[url]
        if ref['status'] != 200:return ref, None
        return ref, raw if binary else strict_json(raw.decode())
    bls_manifest = strict_json((paths['bls_archive']/'manifest.json').read_text())
    releases = load_bls(paths['bls_archive'], bls_manifest['refs'])
    for source in config['fed_releases']:
        ref, raw = get(source['url'], binary=True)
        if raw is None:continue
        release = fed_headline(raw, source['url'], source['published_at']); release['capture_ref'] = ref
        releases.append(release)
    releases.sort(key=lambda r: timestamp(r['published_at'], 'publication'))
    by_release = {(r['family'], r['period']): r for r in releases}
    observations, outcomes, proofs = [], [], []
    for item in selected:
        market, group, platform = item['market'], item['group'], item['platform']
        mid = str(market['id']) if platform == 'polymarket' else market['ticker']
        result = by_release.get((group['family'], group['period']))
        previous = [r for r in releases if r['family'] == 'fomc' and result
                    and timestamp(r['published_at'], 'previous') < timestamp(result['published_at'], 'result')]
        previous = previous[-1] if previous else None
        proof, proof_error, expected, outcome_error = None, None, None, None
        if result:
            try:expected = expected_outcome(item, result, previous)
            except (ValidationError, KeyError) as exc:outcome_error = str(exc)
        else:outcome_error = 'dated_official_release_missing'
        if platform == 'polymarket':
            tokens, _ = mapping(market)
            _, clob = get('https://clob.polymarket.com/clob-markets/'+market['conditionId'])
            mapping_ok = bool(clob and clob.get('c') == market['conditionId']
                and [(x.get('o'), x.get('t')) for x in clob.get('t', [])] == list(tokens.items()))
            state = item['state']; tx = (state or {}).get('transaction_hash')
            try:
                if not tx:raise ValidationError('initialization transaction locator missing')
                ref, raw = get('https://polygonscan.com/tx/'+tx, binary=True)
                if raw is None:raise ValidationError('initialization archive fetch failed')
                try:
                    proof = initialized_question(raw, tx_hash=tx, request_id=market['negRiskRequestID'], adapter=market['resolvedBy'])
                except ValidationError as exc:
                    if b'QuestionReset' in raw:
                        raise ValidationError('API transaction locator is a reset, not a verified initialization') from exc
                    raise
                if state.get('question_id') != proof['question_id']:raise ValidationError('initialization identity differs from resolution metadata')
                prefix = 'q: title: '+market['question']+', description: '+market['description']
                if not normalize(proof['ancillary_data']).startswith(normalize(prefix)):
                    raise ValidationError('current rules differ from initialized text')
                proof['ref'] = ref
                proofs.append({'market_id': mid, 'event_group_id': group['event_group_id'], 'question_proof': proof})
            except (ValidationError, KeyError) as exc:proof = None; proof_error = str(exc)
        for observation in group['planned_observation_times']:
            t = timestamp(observation, 'observation'); blockers = {'semantic_benchmark_review_required', 'evaluation_only_reserved'}
            evidence = prior_release(releases, group['family'], observation)
            if evidence is None:blockers.add('pre_cutoff_evidence_missing')
            if result is None:blockers.add('dated_official_release_missing')
            elif not t < timestamp(result['published_at'], 'first public result'):blockers.add('observation_after_public_result')
            if group['family'] != 'fomc':blockers.add('bls_headline_vintage_review_required')
            label = None; ledger = {'status': 'official_predicate_unverified', 'error': outcome_error}
            if platform == 'polymarket':
                if config.get('polymarket_price_source') == 'clob':
                    quote_url = clob_price_url(tokens['Yes'], t, config['price_policy']); price_ref, price = get(quote_url)
                    quote, quality = clob_history_quote(price, t, config['price_policy'])
                    blockers.add('clob_price_semantics_review_required')
                else:
                    quote_url = poly_price_url(tokens['Yes'], t, config['price_policy']); price_ref, price = get(quote_url)
                    quote, quality = poly_history_quote([] if price is None else [price], t, config['price_policy'])
                if not mapping_ok:blockers.add('outcome_token_mapping_unverified')
                if proof is None:blockers.add('initial_question_unverified')
                elif timestamp(proof['published_at'], 'initialization') > t:blockers.add('question_initialized_after_observation')
                if timestamp(market['startDate'], 'open') > t:blockers.add('market_not_open_at_observation')
                blockers.add('rule_update_history_review_required')
                if proof and proof['adapter'] != '0x69c47de9d4d3dad79590d61b9e05918e03775f24':
                    blockers.add('different_adapter_source_review_required')
                if expected is not None:
                    label, ledger = historical_poly_label(market, item['state'], item['state_ref']['completed_at'], t, expected)
            else:
                quote_url = kalshi_url(market, observation, config['price_policy'], cutoff); price_ref, price = get(quote_url)
                if price is not None and price.get('ticker', mid) != mid:raise ValidationError('price history ticker mismatch')
                quote, quality = kalshi_history_quote(price, t, config['price_policy'], historical='/historical/markets/' in quote_url)
                if not timestamp(market['open_time'], 'open') <= t < timestamp(market['close_time'], 'close'):
                    blockers.add('market_not_open_at_observation')
                blockers.add('historical_market_rules_version_missing')
                if expected is not None:
                    label, ledger = historical_kalshi_label(market, item['market_ref']['completed_at'], t, expected)
            if quote is None:blockers.add('historical_price_unusable')
            if label and result and timestamp(label['resolution_time'], 'settlement') < timestamp(result['published_at'], 'official result'):
                label = None; ledger = {**ledger, 'status': 'settlement_precedes_official_release'}
            if label is None:blockers.add('exact_settlement_proof_missing')
            if ledger['status'] not in ('verified_settlement', 'outcome_cross_checked_time_unverified'):
                blockers.add('official_outcome_crosscheck_failed')
            sid = identifier('foretellmesh_macro_history_'+platform, mid, observation)
            observations.append({'sample_id': sid, 'event_group_id': group['event_group_id'], 'family': group['family'],
                'platform': platform, 'market_id': mid, 'observation_time': observation,
                'release_time_attestation': result['published_at'] if result else None,
                'market_probability': quote, 'price_ref': price_ref, 'quote_quality': quality,
                'initial_question_proof_id': proof['question_id'] if proof else None, 'initial_question_error': proof_error,
                'prior_evidence': None if evidence is None else {k: evidence[k] for k in
                    ('family', 'period', 'published_at', 'source', 'value', 'unit', 'text', 'capture_method', 'trust_scope')},
                'blockers': sorted(blockers), 'ready_for_scoring': False, 'ready_for_training': False})
            outcomes.append({'sample_id': sid, 'event_group_id': group['event_group_id'], 'platform': platform,
                             'market_id': mid, 'label': label, 'crosscheck': ledger,
                             'official_result_ref': result['capture_ref'] if result else None})
    counts = {'planned_groups': len(groups), 'selected_event_groups': len({r['group']['event_group_id'] for r in selected}),
              'contracts': len(selected), 'observation_rows': len(observations),
              'dated_releases': len(releases), 'initial_questions': len(proofs),
              'usable_historical_quotes': sum(r['market_probability'] is not None for r in observations),
              'groups_with_usable_quotes': len({r['event_group_id'] for r in observations if r['market_probability'] is not None}),
              'quotes_by_platform': dict(Counter(r['platform'] for r in observations if r['market_probability'] is not None)),
              'pre_cutoff_evidence_rows': sum(r['prior_evidence'] is not None for r in observations),
              'exact_settlement_labels': sum(r['label'] is not None for r in outcomes),
              'official_outcome_crosschecked_rows': sum(r['crosscheck']['status'] in ('verified_settlement', 'outcome_cross_checked_time_unverified') for r in outcomes),
              'blockers': dict(Counter(b for r in observations for b in r['blockers']))}
    group_quality = []
    for group in groups:
        rows = [r for r in observations if r['event_group_id'] == group['event_group_id']]
        group_quality.append({'event_group_id': group['event_group_id'], 'inventory_disposition': group['disposition'],
                              'rows': len(rows), 'usable_quotes': sum(r['market_probability'] is not None for r in rows),
                              'blockers': dict(Counter(b for r in rows for b in r['blockers'])), 'missing': group['missing']})
    output.mkdir(parents=True)
    artifacts = {'observations.jsonl': jsonl(observations), 'outcomes.jsonl': jsonl(outcomes),
                 'initial_questions.jsonl': jsonl(proofs), 'dated_releases.jsonl': jsonl(releases),
                 'group_quality.json': json_text(group_quality)}
    for name, content in artifacts.items():(output/name).write_text(content)
    report = {'schema_version': '1', 'kind': 'macro_historical_evidence_audit', 'status': 'staging_not_admitted',
              'config_sha256': sha256_file(config_path), 'capture_sha256': sha256_file(capture/'manifest.json'),
              'counts': counts, 'requests': len(responses), 'failed_requests': sum(ref['status'] != 200 for ref, _ in responses.values()),
              'artifact_hashes': {name: sha256_file(output/name) for name in artifacts},
              'model_calls': 0, 'score_metrics': None, 'admitted_rows': 0,
              'limitations': ['BLS header/headline attestation is not a whole-page correction or first-publication snapshot audit.',
                             'CLOB sample prices passing local cutoff checks still require price-semantics review before scoring.',
                             'Initial ancillary text is not proof of the complete rule-update history.',
                             'Kalshi settlement labels rely on timestamped native API finalization; historical rule versions remain unproven.',
                             'Polymarket API oracle outcomes are cross-checks, never final CTF payout timestamps.',
                             'No forecast inputs or training targets are exported; outcomes stay in a separate ledger.']}
    (output/'report.json').write_text(json_text(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'capture', 'output'):p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json_text(build(a.config, a.capture, a.output)))


if __name__ == '__main__':main()
