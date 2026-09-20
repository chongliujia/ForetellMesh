"""Time-bounded deterministic market features and two optional Base analyst roles."""
from collections import defaultdict
from copy import deepcopy
from datetime import timedelta
from fractions import Fraction
import json
import math
import statistics

from .data import strict_json
from .paper_trading import TradePrintFeed
from .schema import ValidationError, fields, iso, nonempty, timestamp
from .synthetic_sft import canonical_hash

SOURCE = 'foretellmesh:historical_market_features:v1'
FEATURE_ID = 'market_features:v1'
FEATURE_KEYS = {'last_probability', 'quote_age_seconds'} | {
    f'{metric}_{window}' for metric in ('change', 'trade_count', 'block_count', 'price_range', 'price_stddev')
    for window in ('1h', '24h', '7d')}
VIEWS = {'supports_yes', 'supports_no', 'mixed', 'no_independent_edge'}
MECHANISMS = {'information_aggregation', 'hedging_demand', 'liquidity_constraints', 'attention', 'strategic_policy'}
PROMPTS = {
    'market_quant': (
        'You are the quantitative prediction-market analyst. Interpret ONLY supplied deterministic market_features:v1. '
        'Arithmetic has already been performed by Python/SQL. Price changes are probability points, not returns. '
        'Trade counts are activity, not dollar volume or executable depth. Price dispersion is not event uncertainty. '
        'Momentum or disagreement alone does not establish mispricing or predict continuation. '
        'Return exactly {"signals":[{"feature":"an existing numeric feature key","interpretation":"brief interpretation",'
        '"evidence_ids":["market_features:v1"]}],"market_view":"supports_yes|supports_no|mixed|no_independent_edge",'
        '"unknowns":[],"observation_time":"copy input"}. At most 3 signals; use [] when no usable feature. '
        'Do not invent a calibrated probability, volume, order book, participant identity or future observation. '
        'Use no_independent_edge when statistics describe price but give no justified forecasting advantage.'
    ),
    'game_theory': (
        'You are the game-theoretic prediction-market analyst. Consider information aggregation, possible hedging demand, '
        'liquidity constraints, attention, and strategic policy incentives. Market traders and policy decision makers are '
        'different actors. Trader beliefs and prices do not cause the measured macroeconomic outcome. '
        'Only supplied evidence is factual; never infer insider knowledge, manipulation, positions, inventory, flow direction '
        'or trader identities from price/trade counts. Choose ONE main mechanism and ONE conditional hypothesis. '
        'Return a FLAT JSON object with exactly these seven fields: mechanism (one string), assumption (one string beginning '
        'with If), implication (one short conditional conclusion), evidence_ids (array of supplied IDs), '
        'market_view (one string), unknowns (array of strings), observation_time (copy input). '
        'Allowed mechanism values: information_aggregation, hedging_demand, liquidity_constraints, attention, strategic_policy. '
        'Allowed market_view values: supports_yes, supports_no, mixed, no_independent_edge. '
        'Do not output a scenarios array. Do not list every mechanism. Keep assumption and implication to one sentence each. '
        'Evidence IDs cite background context, not proof that the assumption holds; [] is allowed. '
        'Do not manufacture a contrarian trade or numerical edge. Use no_independent_edge unless available evidence justifies a direction.'
    ),
}


def feature_values(context):
    matches = [e for e in context.evidence if e.source == SOURCE]
    if len(matches) != 1 or matches[0].evidence_id != FEATURE_ID:
        raise ValidationError('exactly one deterministic market feature source required')
    value = strict_json(matches[0].text)
    fields(value, {'version', 'as_of', 'features', 'limitations'}, 'market features')
    if value['version'] != '1' or timestamp(value['as_of'], 'feature cutoff') != context.observation_time:
        raise ValidationError('market feature cutoff mismatch')
    fields(value['features'], FEATURE_KEYS, 'deterministic feature mapping')
    for key, number in value['features'].items():
        if number is None and key not in ('last_probability', 'quote_age_seconds') and not key.startswith(('trade_count_', 'block_count_')):
            continue
        if type(number) not in (int, float) or not math.isfinite(number):raise ValidationError('invalid numeric feature')
        if key.startswith(('trade_count_', 'block_count_')) and (type(number) is not int or number < 0):
            raise ValidationError('invalid feature count')
        if key.startswith('change_') and not -1 <= number <= 1:raise ValidationError('invalid price change')
        if key.startswith(('price_range_', 'price_stddev_')) and not 0 <= number <= 1:raise ValidationError('invalid price dispersion')
    if not 0 < value['features']['last_probability'] < 1 or not 0 <= value['features']['quote_age_seconds'] <= 10800:
        raise ValidationError('invalid feature price or age')
    return value['features']


def validate_output(role, value, context, refs, string_list):
    required = {'signals'} if role == 'market_quant' else {'mechanism', 'assumption', 'implication', 'evidence_ids'}
    fields(value, required | {'market_view', 'unknowns', 'observation_time'}, role)
    if timestamp(value['observation_time'], 'expert observation') != context.observation_time:
        raise ValidationError('expert observation changed')
    if value['market_view'] not in VIEWS:raise ValidationError('invalid market view')
    string_list(value['unknowns'], 'expert unknowns')
    items = value['signals'] if role == 'market_quant' else [value]
    if not isinstance(items, list) or len(items) > 3:raise ValidationError('expert summary exceeds budget')
    features = feature_values(context) if role == 'market_quant' else None
    for item in items:
        if role == 'market_quant':
            fields(item, {'feature', 'interpretation', 'evidence_ids'}, 'quant signal')
            key = nonempty(item['feature'], 'feature key')
            if key not in features or type(features[key]) not in (int, float):
                raise ValidationError('quant signal must reference an available numeric feature')
            nonempty(item['interpretation'], 'feature interpretation')
            if FEATURE_ID not in item['evidence_ids']:raise ValidationError('quant signal must cite deterministic feature source')
        else:
            if item['mechanism'] not in MECHANISMS:raise ValidationError('unknown game mechanism')
            nonempty(item['assumption'], 'conditional assumption');nonempty(item['implication'], 'conditional implication')
            if not item['assumption'].startswith('If '):raise ValidationError('game assumption must be explicitly conditional: If ...')
        refs(item['evidence_ids'])
    return deepcopy(value)


def build_features(feed: TradePrintFeed, market_id: str, observation):
    """Read only trades with block timestamps <= T. No outcomes or future prices."""
    latest = feed.latest(market_id, observation)
    if latest is None:raise ValidationError('no price at feature cutoff')
    q, qt = latest
    if (observation - qt).total_seconds() > 10800:raise ValidationError('stale feature reference')
    cutoff = int(observation.timestamp())
    raw = feed.connection.execute(
        'SELECT t.block_number,b.unix_time,t.numerator,t.denominator,t.tx,t.log_index '
        'FROM trades t JOIN blocks b USING(block_number) WHERE t.market_id=? AND b.unix_time>? '
        'AND b.unix_time<=? ORDER BY b.unix_time,t.block_number,t.tx,t.log_index',
        (market_id, cutoff - 7*86400, cutoff)).fetchall()
    blocks = defaultdict(list)
    for block, seconds, n, d, tx, idx in raw:
        p = Fraction(int(n), int(d))
        if not 0 < p < 1:raise ValidationError('invalid native feature price')
        blocks[(seconds, block)].append(p)
    means = [(seconds, float(sum(v, Fraction())/len(v))) for (seconds, _), v in sorted(blocks.items())]
    f = {'last_probability':float(q), 'quote_age_seconds':(observation-qt).total_seconds()}
    anchors = {}
    for hours, name in [(1, '1h'), (24, '24h'), (168, '7d')]:
        start = cutoff - hours*3600
        pp = [p for ts, p in means if ts > start]
        anchor = feed.latest(market_id, observation-timedelta(hours=hours))
        valid = anchor is not None and 0 <= (observation-timedelta(hours=hours)-anchor[1]).total_seconds() <= 10800
        anchors[name] = None if not valid else {'probability':str(anchor[0]), 'time':iso(anchor[1])}
        f.update({f'change_{name}':float(q-anchor[0]) if valid else None,
                  f'trade_count_{name}':sum(len(v) for (ts, _), v in blocks.items() if ts > start),
                  f'block_count_{name}':len(pp),
                  f'price_range_{name}':max(pp)-min(pp) if pp else None,
                  f'price_stddev_{name}':statistics.pstdev(pp) if len(pp)>1 else None})
    feature = {'version':'1', 'as_of':iso(observation), 'features':f,
               'limitations':['Historical normalized Yes trade prices; no order book, trader identities, flow direction or volume.',
                 'Block prices are unweighted means; dispersion describes irregular historical prints, not predictive uncertainty.',
                 'Changes use an as-of window-start anchor at most 3 hours old; missing or stale anchors yield null.',
                 'Timestamp availability is historical block-time reconstruction, not a captured live feed.']}
    audit = {'market_id':market_id, 'as_of':iso(observation), 'source_rows':len(raw),
             'selected_rows_sha256':canonical_hash(raw), 'anchors':anchors,
             'max_source_time':iso(qt), 'features_sha256':canonical_hash(feature)}
    return feature, audit


def enrich(row, feature):
    out = deepcopy(row);p = out['input'];t = p['observation_time']
    if any(e['evidence_id']==FEATURE_ID or e['source']==SOURCE for e in p['evidence']):
        raise ValidationError('duplicate derived features')
    p['evidence'].append({'evidence_id':FEATURE_ID, 'source':SOURCE,
        'text':json.dumps(feature, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
        'published_at':t, 'available_at':t})
    return out
