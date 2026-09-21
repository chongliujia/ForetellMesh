"""Bounded team exploration, causal tool access, paper feedback and memory.

The backend never receives labels before feedback. Model-generated lessons are
hypotheses, not admitted SFT answers. This module has no live exchange client.
"""
from copy import deepcopy
from datetime import timedelta
import json
import math
import statistics
import time
from typing import TypedDict

from .agent_baseline_data import input_context
from .agent_runtime import decode_agent_response
from .metrics import score_predictions
from .paper_trading import Signal, Settlement, decimal, simulate, validate_policy
from .schema import ValidationError, fields, iso, probability, timestamp
from .synthetic_sft import canonical_hash


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def strings(value, maximum=8):
    require(isinstance(value, list) and len(value) <= maximum
            and all(isinstance(s, str) and 0 < len(s) <= 600 for s in value)
            and len(value) == len(set(value)), 'invalid bounded string list')
    return value


def validate_context(context):
    fields(context, {'episode_id', 'observation_time', 'markets', 'account', 'memory'}, 'team context')
    at = timestamp(context['observation_time'], 'team cutoff')
    require(isinstance(context['episode_id'], str) and context['episode_id'], 'missing episode identity')
    require(isinstance(context['markets'], list) and 1 <= len(context['markets']) <= 8, 'invalid team market count')
    mids = set()
    for market in context['markets']:
        fields(market, {'market_id', 'event_group_id', 'input'}, 'team market')
        require(isinstance(market['market_id'], str) and market['market_id']
                and isinstance(market['event_group_id'], str) and market['event_group_id'], 'missing team market identity')
        require(market['market_id'] not in mids, 'duplicate team market')
        mids.add(market['market_id'])
        inp = input_context(market['input'])
        require(inp.observation_time == at and inp.market is not None, 'market cutoff/quote missing')
    require(context['account'] == {'initial_cash': '100', 'simulation_only': True}, 'invalid episode account')
    require(isinstance(context['memory'], list) and len(context['memory']) <= 8, 'unbounded memory')
    require(len({m['memory_id'] for m in context['memory']}) == len(context['memory']), 'duplicate context memory')
    for memory in context['memory']:
        validate_memory(memory)
        require(timestamp(memory['available_at'], 'memory availability') <= at, 'future memory')
        require(not set(memory['event_group_ids']) & {m['event_group_id'] for m in context['markets']},
                'same-event retrospective memory')
    return mids


def validate_memory(row):
    fields(row, {'memory_id', 'source_episode_id', 'event_group_ids', 'available_at', 'reflection',
                 'feedback_sha256', 'status'}, 'team memory')
    require(row['status'] == 'unverified_hypothesis', 'memory is not truth or training admission')
    timestamp(row['available_at'], 'memory availability')
    strings(row['event_group_ids'])
    expected = canonical_hash({k: v for k, v in row.items() if k != 'memory_id'})
    require(row['memory_id'] == expected, 'memory content hash changed')


class MemoryBank:
    def __init__(self, rows=()):
        self.rows = []
        for row in rows:
            self.add(row)

    def add(self, row):
        validate_memory(row)
        require(all(r['memory_id'] != row['memory_id'] for r in self.rows), 'duplicate memory')
        self.rows.append(deepcopy(row))

    def at(self, cutoff, *, excluded_groups=(), limit=4):
        require(type(limit) is int and 0 <= limit <= 8, 'invalid memory budget')
        at = timestamp(cutoff, 'memory cutoff')
        visible = [r for r in self.rows if timestamp(r['available_at'], 'memory availability') <= at
                   and not set(r['event_group_ids']) & set(excluded_groups)]
        return deepcopy(sorted(visible, key=lambda r: (timestamp(r['available_at'], 'availability'), r['memory_id']))[-limit:]) if limit else []


INSTRUCTIONS = {
    'research': ('You are the research member of a prediction-market team. Propose relationships or uncertainty to investigate; '
        'you are not restricted to a financial category or a prescribed trading rule. Use only the historical input. '
        'Return exactly {"queries":[{"tool":"history" or "pair_changes","market_ids":[IDs],"lookback_days":N}],'
        '"hypotheses":[strings]}. At most 2 queries and 3 hypotheses. history takes 1 ID, pair_changes takes 2 distinct IDs. '
        'N is an integer 1..30. An empty query list is allowed. Correlation does not establish causality.'),
    'forecast': ('You are the forecasting/decision member. Inspect the researcher hypotheses and deterministic tool results. '
        'Return exactly {"forecasts":[{"market_id":ID,"probability":NUMBER,"consider_trade":BOOLEAN}],'
        '"unknowns":[strings]}. Include every input market exactly once. Probabilities are event probabilities in [0,1]. '
        'A hypothesis is not an established fact. False consider_trade keeps cash; no required activity. '
        'Execution uses the supplied cost/risk policy and historical print assumptions. Do not invent evidence.'),
    'risk': ('You are the independent risk reviewer of the team. Challenge the proposed relations and decisions against '
        'the supplied evidence and tool coverage. Return exactly {"veto_markets":[IDs],"risks":[strings]}. '
        'You may veto any proposed entry. No required number of trades; do not veto solely to meet a trading cadence.'),
    'reflection': ('You are the retrospective learning member. This is explicitly AFTER feedback, not an ex-ante forecast. '
        'Return exactly {"fact_ids":[supplied fact IDs],"lessons":[strings],"next_experiments":[strings]}. '
        'At most 4 facts, 3 lessons and 3 experiments. Treat lessons as hypotheses for future checking. '
        'Profit is not proof of good forecasting, a loss is not proof that a probability was wrong. '
        'Do not rewrite the original decision or present outcomes as prior knowledge.')}


def validate_output(role, value, mids, fact_ids=()):
    if role == 'research':
        fields(value, {'queries', 'hypotheses'}, role)
        strings(value['hypotheses'], 3)
        require(isinstance(value['queries'], list) and len(value['queries']) <= 2, 'unbounded research queries')
        for query in value['queries']:
            fields(query, {'tool', 'market_ids', 'lookback_days'}, 'query')
            require(query['tool'] in ('history', 'pair_changes'), 'unknown team tool')
            strings(query['market_ids'], 2)
            require(set(query['market_ids']) <= mids and len(query['market_ids']) ==
                    (1 if query['tool'] == 'history' else 2), 'invalid tool market scope')
            require(type(query['lookback_days']) is int and 1 <= query['lookback_days'] <= 30, 'invalid lookback')
    elif role == 'forecast':
        fields(value, {'forecasts', 'unknowns'}, role); strings(value['unknowns'])
        require(isinstance(value['forecasts'], list) and len(value['forecasts']) == len(mids), 'forecast population differs')
        seen = set()
        for item in value['forecasts']:
            fields(item, {'market_id', 'probability', 'consider_trade'}, 'team forecast')
            require(isinstance(item['market_id'], str) and item['market_id'] in mids
                    and item['market_id'] not in seen, 'unknown/duplicate forecast identity')
            seen.add(item['market_id']); probability(item['probability'])
            require(type(item['consider_trade']) is bool, 'invalid trade intent')
    elif role == 'risk':
        fields(value, {'veto_markets', 'risks'}, role)
        require(set(strings(value['veto_markets'])) <= mids, 'risk veto outside scope'); strings(value['risks'])
    elif role == 'reflection':
        fields(value, {'fact_ids', 'lessons', 'next_experiments'}, role)
        require(set(strings(value['fact_ids'], 4)) <= set(fact_ids), 'unknown feedback fact')
        strings(value['lessons'], 3); strings(value['next_experiments'], 3)
    else:
        raise ValidationError('unknown team role')
    return deepcopy(value)


def run_query(query, feed, cutoff, max_age_seconds=10800):
    """Only daily as-of samples; never reads feed.data or future prices."""
    require(isinstance(query, dict) and isinstance(query.get('market_ids'), list), 'invalid query')
    validate_output('research', {'queries': [query], 'hypotheses': []}, set(strings(query['market_ids'], 2)))
    require(type(max_age_seconds) is int and max_age_seconds > 0, 'invalid query freshness')
    at = timestamp(cutoff, 'tool cutoff'); series = {}
    for mid in query['market_ids']:
        values = []
        for days in range(query['lookback_days'], -1, -1):
            point = at - timedelta(days=days); quote = feed.latest(mid, point)
            if quote is not None:
                p, qt = quote
                require(qt <= point and 0 < decimal(p) < 1, 'future/invalid tool quote')
                if (point-qt).total_seconds() <= max_age_seconds:
                    values.append({'at': iso(point), 'source_time': iso(qt), 'price': float(p)})
        series[mid] = values
    result = {'query': deepcopy(query), 'as_of': iso(at), 'series': series,
              'missing_is_not_zero': True, 'is_executable_quote': False}
    if query['tool'] == 'pair_changes':
        changes = []
        for values in series.values():
            by_time = {timestamp(v['at'], 'sample'): v['price'] for v in values}
            changes.append({t: p-by_time[t-timedelta(days=1)] for t, p in by_time.items()
                            if t-timedelta(days=1) in by_time})
        common = sorted(set(changes[0]) & set(changes[1]))
        x = [changes[0][t] for t in common]; y = [changes[1][t] for t in common]
        corr = statistics.correlation(x, y) if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else None
        result.update(paired_daily_changes=len(common), pearson_change_correlation=corr,
                      interpretation='descriptive_training_history_not_causal_or_validated_prediction')
    result['result_sha256'] = canonical_hash(result)
    return result


class TeamState(TypedDict):
    research: dict
    tools: list
    forecast: dict
    risk: dict


class TeamRunner:
    def __init__(self, backend, *, max_repairs=1, adapter=None, learning_artifacts=()):
        require(type(max_repairs) is int and 0 <= max_repairs <= 1, 'invalid repair budget')
        self.backend = backend; self.max_repairs = max_repairs; self.adapter = adapter; self.calls = []
        self.learning_artifacts = deepcopy(list(learning_artifacts))

    def call(self, role, context, upstream, mids, fact_ids=()):
        upstream = deepcopy(upstream)
        if self.learning_artifacts:
            # Offline fitted method state is distinct from historical evidence
            # and from MemoryBank's online replay memory. Never backdate it.
            at = timestamp(context['observation_time'], 'artifact cutoff')
            groups = {m['event_group_id'] for m in context.get('markets', [])}
            for artifact in self.learning_artifacts:
                fields(artifact, {'kind', 'built_at', 'source_public_through', 'source_event_group_ids',
                                  'source_episode_hashes', 'lessons', 'artifact_sha256'}, 'learning artifact')
                require(artifact['kind'] == 'offline_training_reflections_v1'
                        and artifact['artifact_sha256'] == canonical_hash({k: v for k, v in artifact.items() if k != 'artifact_sha256'}),
                        'unbound learning artifact')
                require(timestamp(artifact['source_public_through'], 'source facts') < at
                        and not set(artifact['source_event_group_ids']) & groups, 'learning artifact leaks target event or later facts')
                timestamp(artifact['built_at'], 'actual artifact creation')
            upstream['offline_learning_artifacts'] = self.learning_artifacts
        request = {'agent': role, 'instruction': INSTRUCTIONS[role], 'input': deepcopy(context),
                   'upstream': deepcopy(upstream), 'adapter': self.adapter}
        if role in ('research', 'risk', 'forecast'):
            request['instruction'] += (' Allowed market IDs for this call are exactly '
                + json.dumps(sorted(mids)) + '. Never invent or increment an ID. Use only these IDs in every field.')
        if role == 'research' and len(mids) == 1:
            request['instruction'] += (' Only one market is available: pair_changes is unavailable. '
                'Use history on that one ID or return an empty queries list. Unsupported external facts must remain unknown.')
        for attempt in range(self.max_repairs+1):
            started = time.perf_counter(); raw = None
            try:
                raw = self.backend.generate(deepcopy(request))
                value, _ = decode_agent_response(raw, 'single_json_fence')
                result = validate_output(role, value, mids, fact_ids)
                self.calls.append({'request': deepcopy(request), 'output': raw, 'error': None,
                                   'seconds': time.perf_counter()-started})
                return result
            except (ValueError, TypeError, KeyError) as exc:
                self.calls.append({'request': deepcopy(request), 'output': raw, 'error': str(exc),
                                   'seconds': time.perf_counter()-started})
                if attempt == self.max_repairs:
                    raise ValidationError(f'{role} failed: {exc}') from exc
                request['repair'] = {'error': str(exc), 'previous_response': raw}

    def decide(self, context, feed, policy):
        from langgraph.graph import END, START, StateGraph
        mids = validate_context(context); validate_policy(policy)
        graph = StateGraph(TeamState)
        graph.add_node('research', lambda _: {'research': self.call('research', context, {}, mids)})
        graph.add_node('tools', lambda s: {'tools': [run_query(q, feed, context['observation_time'], policy['max_quote_age_seconds'])
                                                   for q in s['research']['queries']]})
        graph.add_node('forecast', lambda s: {'forecast': self.call('forecast', context,
                       {'research': s['research'], 'tools': s['tools'], 'execution_policy': policy}, mids)})
        graph.add_node('risk', lambda s: {'risk': self.call('risk', context,
                       {'research': s['research'], 'tools': s['tools'], 'forecast': s['forecast'], 'execution_policy': policy}, mids)})
        nodes = [START, 'research', 'tools', 'forecast', 'risk', END]
        for left, right in zip(nodes, nodes[1:]):
            graph.add_edge(left, right)
        return graph.compile().invoke({}, {'max_concurrency': 1, 'recursion_limit': 8})


def feedback_for(context, decision, labels, feed, policy, elapsed_seconds):
    mids = validate_context(context)
    if decision is not None:
        validate_output('forecast', decision['forecast'], mids)
        validate_output('risk', decision['risk'], mids)
    require(type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds)
            and elapsed_seconds >= 0, 'invalid decision latency')
    at = timestamp(context['observation_time'], 'observation')
    ready = at + timedelta(seconds=elapsed_seconds)
    require(set(labels) == mids, 'feedback label population differs')
    settlements = []; feedback_times = []
    for mid, label in labels.items():
        fields(label, {'outcome', 'resolution_time', 'available_at'}, 'feedback label')
        require(type(label['outcome']) is int and label['outcome'] in (0, 1), 'invalid feedback outcome')
        resolution = timestamp(label['resolution_time'], 'resolution'); available = timestamp(label['available_at'], 'label availability')
        require(at < resolution <= available, 'invalid label chronology')
        settlements.append(Settlement(mid, resolution, label['outcome'])); feedback_times.append(available)
    forecasts = {} if decision is None else {f['market_id']: f for f in decision['forecast']['forecasts']}
    vetoes = set() if decision is None else set(decision['risk']['veto_markets'])
    signals = []; predictions = []; baselines = []; outcomes = []
    for market in context['markets']:
        mid = market['market_id']; inp = market['input']; q = inp['market']['probability']
        f = forecasts.get(mid); p = f['probability'] if f else None
        predictions.append(p); baselines.append(q); outcomes.append(labels[mid]['outcome'])
        signals.append(Signal(context['episode_id']+':'+mid, mid, market['event_group_id'], at, ready,
                              timestamp(inp['market']['observed_at'], 'quote'), decimal(q),
                              decimal(p) if f and f['consider_trade'] and mid not in vetoes else None))
    account = simulate(signals, settlements, feed, policy)
    scores = score_predictions(predictions, outcomes); baseline = score_predictions(baselines, outcomes)
    facts = {'paper_net_pnl': account['net_pnl'], 'paper_fees': account['fees_paid'],
             'filled_trades': account['filled_trades'], 'forecast_brier': scores['brier'],
             'market_brier': baseline['brier'], 'forecast_coverage': scores['coverage']}
    return {'available_at': iso(max([ready, *feedback_times])), 'decision_time': iso(ready),
            'facts': facts, 'scores': scores, 'market_scores': baseline, 'account': account,
            'label_sha256': canonical_hash(labels), 'outcomes': dict(zip([m['market_id'] for m in context['markets']], outcomes))}


def run_episode(context, labels, feed, policy, runner, *, recorded_latency=None):
    """Isolated $100 episodic account; not a continuous portfolio backtest."""
    validate_context(context); validate_policy(policy); runner.calls = []
    context = deepcopy(context); started = time.perf_counter(); decision = None; error = None
    try:
        decision = runner.decide(context, feed, policy)
    except (ValueError, TypeError, KeyError) as exc:
        error = str(exc)
    elapsed = time.perf_counter()-started if recorded_latency is None else recorded_latency
    decision_call_count = len(runner.calls)
    feedback = feedback_for(context, decision, labels, feed, policy, elapsed)
    reflection = None; reflection_error = None; memory = None
    if decision is not None:
        retrospective = {'episode_id': context['episode_id'], 'observation_time': feedback['available_at'],
                         'phase': 'retrospective', 'original_cutoff': context['observation_time'],
                         'decisions': decision['forecast'], 'feedback_facts': feedback['facts']}
        try:
            reflection_started = time.perf_counter()
            reflection = runner.call('reflection', retrospective, {}, set(), feedback['facts'])
            memory = {'source_episode_id': context['episode_id'],
                      'event_group_ids': sorted({m['event_group_id'] for m in context['markets']}),
                      'available_at': iso(timestamp(feedback['available_at'], 'feedback') +
                                          timedelta(seconds=time.perf_counter()-reflection_started)),
                      'reflection': reflection, 'feedback_sha256': canonical_hash(feedback),
                      'status': 'unverified_hypothesis'}
            memory['memory_id'] = canonical_hash(memory); validate_memory(memory)
        except (ValueError, TypeError, KeyError) as exc:
            reflection_error = str(exc)
    return {'kind': 'team_learning_episode_v1', 'episode_id': context['episode_id'],
            'context': context, 'context_sha256': canonical_hash(context), 'decision': decision,
            'decision_error': error, 'decision_seconds': elapsed, 'decision_call_count': decision_call_count,
            'feedback': feedback, 'reflection': reflection, 'reflection_error': reflection_error,
            'memory': memory, 'calls': deepcopy(runner.calls), 'execution_policy': deepcopy(policy),
            'training_admission': {'admitted': False, 'reason': 'generated_experience_requires_independent_task_specific_validation'},
            'simulation_only': True, 'continuous_portfolio': False}


def experience_candidates(episodes):
    """Export *candidates*, never silently convert profitable traces into truth."""
    result = []
    for episode in episodes:
        require(episode['kind'] == 'team_learning_episode_v1' and
                episode['context_sha256'] == canonical_hash(episode['context']), 'episode input changed')
        validate_context(episode['context'])
        for i, call in enumerate(episode['calls']):
            if call['output'] is None or call['error'] is not None:
                continue
            role = call['request']['agent']; retrospective = role == 'reflection'
            result.append({'candidate_id': episode['episode_id']+':'+str(i), 'role': role,
                'task_phase': 'retrospective' if retrospective else 'ex_ante',
                'request': deepcopy(call['request']), 'response': call['output'],
                'episode_sha256': canonical_hash(episode), 'feedback_sha256': canonical_hash(episode['feedback']),
                'ready_for_training': False, 'status': 'candidate_not_verified_teacher'})
    return result
