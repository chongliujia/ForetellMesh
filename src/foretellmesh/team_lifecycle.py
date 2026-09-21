"""Opt-in team position review and adaptive research timing; simulation only.

Historical labels belong to the replay environment, never to model requests.
The bounded observation window is a compute budget, not a liquidation rule.
"""
from copy import deepcopy
from datetime import timedelta
import time

from .paper_trading import Signal, Settlement, decimal, simulate, validate_policy
from .schema import fields, iso, timestamp, ValidationError
from .team_learning import require, validate_context, validate_output
from .team_memory_loop import MemoryRunner
from .team_trade_experience import trade_experiences, learn_available, validate_online_lessons


REVIEW = ('You manage the existing prediction-market portfolio using the research, forecast and risk review. '
    'Return exactly {"actions":[{"market_id":ID,"action":"buy|hold|sell",'
    '"position_id":null,"min_exit_price":null,"reason":"specific evidence and uncertainty"}],'
    '"signal_valid_for_seconds":INTEGER,"order_ttl_seconds":INTEGER,"next_review_after_seconds":INTEGER}. '
    'Include every input market exactly once. buy asks the executor to consider an entry using the supplied '
    'event probability and cost/risk limits. hold means keep the position or cash and cancel outstanding orders '
    'for that market. sell closes the entire existing position, with its exact position_id and a numeric '
    'min_exit_price strictly between 0 and 1 for the HELD side, after execution premium but before sale fees. '
    'Only sell has non-null position_id and min_exit_price. No shorting or partial exits. '
    'An entry veto blocks buying, not selling. No required trades, holding duration, stop-loss or profit target. '
    'Choose research timing independently from order lifetime and forecast lifetime within timing_limits. '
    'All intervals start at observation_time except order TTL, which starts after generation. '
    'A new valid review replaces outstanding intent. No new evidence is invented between reviews. '
    'Compare future holding value and uncertainty with net sale proceeds and capital waiting time; '
    'do not charge past entry costs again in that comparison. Price forecasts are not event probabilities. '
    'Trade prints are hypothetical execution references, not order-book bids or guaranteed liquidity.')


def validate_timing(value):
    fields(value, {'min_review_seconds','max_review_seconds','max_signal_valid_seconds',
                   'max_order_ttl_seconds','failure_review_seconds'}, 'lifecycle timing')
    require(all(type(v) is int and v>0 for v in value.values()), 'invalid timing bounds')
    require(value['min_review_seconds']<=value['failure_review_seconds']<=value['max_review_seconds'],
            'invalid failure review interval')
    return deepcopy(value)


def validate_review(value, context, account, timing):
    fields(value, {'actions','signal_valid_for_seconds','order_ttl_seconds','next_review_after_seconds'}, 'position review')
    mids=validate_context(context);validate_timing(timing)
    for name,low,high in [('signal_valid_for_seconds',1,timing['max_signal_valid_seconds']),
                          ('order_ttl_seconds',1,timing['max_order_ttl_seconds']),
                          ('next_review_after_seconds',timing['min_review_seconds'],timing['max_review_seconds'])]:
        require(type(value[name]) is int and low<=value[name]<=high, 'invalid '+name)
    require(isinstance(value['actions'],list) and len(value['actions'])==len(mids), 'review population differs')
    positions={p['position_id']:p for p in account['positions']};seen=set()
    for a in value['actions']:
        fields(a, {'market_id','action','position_id','min_exit_price','reason'}, 'position action')
        require(isinstance(a['market_id'],str) and a['market_id'] in mids and a['market_id'] not in seen, 'unknown/duplicate review market')
        seen.add(a['market_id'])
        require(a['action'] in ('buy','hold','sell'), 'invalid review action')
        require(isinstance(a['reason'],str) and 1<=len(a['reason'].strip())<=600, 'missing action reason')
        if a['action']=='sell':
            require(isinstance(a['position_id'],str) and a['position_id'] in positions and
                    positions[a['position_id']]['market_id']==a['market_id'], 'sell position not held')
            require(a['min_exit_price'] is not None and 0<decimal(a['min_exit_price'])<1, 'invalid sale price limit')
        else:
            require(a['position_id'] is None and a['min_exit_price'] is None, 'exit fields on non-sell review')
            if a['action']=='buy':
                require(not any(p['market_id']==a['market_id'] for p in positions.values()), 'buy on existing position')
    return deepcopy(value)


class LifecycleRunner(MemoryRunner):
    def __init__(self, backend, catalogue, timing, memory=None):
        super().__init__(backend,catalogue,memory);self.timing=validate_timing(timing)
        self.trade_reflections={}

    def with_memory(self, context, upstream):
        result=super().with_memory(context,upstream)
        rows=[r for r in self.trade_reflections.values() if r.get('admitted_to_exploratory_memory') is True]
        if rows:
            from .team_experience_gate import method_views
            result['online_trade_lessons']=method_views(rows[-8:],context['observation_time'])
        return result

    def decide(self, context, feed, policy, account):
        require(account['as_of']==context['observation_time'], 'portfolio snapshot cutoff differs')
        mids=validate_context(context)
        require({p['market_id'] for p in account['positions']}<=mids, 'held market omitted from review')
        decision=super().decide(context,feed,policy,account)
        upstream=self.with_memory(context,{'team_decision':decision,'account_state':account,
            'execution_policy':policy,'timing_limits':self.timing})
        decision['lifecycle']=self.structured('position_review',REVIEW,context,upstream,
            lambda v:validate_review(v,context,account,self.timing))
        return decision


def signals_for_episode(episode, timing):
    context=episode['context'];mids=validate_context(context)
    elapsed=decimal(episode['decision_seconds']);require(elapsed>=0, 'negative generation latency')
    if episode['decision'] is None:return []
    decision=episode['decision'];validate_output('forecast',decision['forecast'],mids)
    validate_output('risk',decision['risk'],mids)
    review=validate_review(decision['lifecycle'],context,episode['account_at_observation'],timing)
    at=timestamp(context['observation_time'],'cutoff');ready=at+timedelta(seconds=float(elapsed))
    expires=at+timedelta(seconds=review['signal_valid_for_seconds'])
    forecasts={r['market_id']:r for r in decision['forecast']['forecasts']}
    actions={r['market_id']:r for r in review['actions']};signals=[]
    for m in context['markets']:
        mid=m['market_id'];f=forecasts[mid];a=actions[mid];action=a['action'];quote=m['input']['market']
        if action=='buy' and (not f['consider_trade'] or mid in decision['risk']['veto_markets']):action='hold'
        signals.append(Signal(context['episode_id']+':'+mid,mid,m['event_group_id'],at,ready,
            timestamp(quote['observed_at'],'quote'),decimal(quote['probability']),decimal(f['probability']),
            action,expires,review['order_ttl_seconds'],a['position_id'],
            None if a['min_exit_price'] is None else decimal(a['min_exit_price'])))
    return signals


def replay(episodes, settlements, feed, policy, timing, *, through=None):
    signals=[];previous_ready=None
    for ep in episodes:
        at=timestamp(ep['context']['observation_time'],'cutoff')
        require(previous_ready is None or at>previous_ready,'overlapping team decisions')
        actual=simulate(signals,settlements,feed,policy,through=at,lifecycle=True)['snapshot']
        require(actual==ep['account_at_observation'],'archived portfolio snapshot differs')
        signals.extend(signals_for_episode(ep,timing))
        previous_ready=at+timedelta(seconds=float(ep['decision_seconds']))
    return simulate(signals,settlements,feed,policy,through=through,lifecycle=True)


def refresh_context(template, at, feed, resolved):
    """Retain proven rules/evidence; refresh only as-of prices. No future news."""
    context=deepcopy(template);context['observation_time']=iso(at)
    context['episode_id']=template['episode_id']+':review:'+iso(at)
    context['markets']=[m for m in context['markets'] if m['market_id'] not in resolved]
    for m in context['markets']:
        inp=m['input'];inp['observation_time']=iso(at);quote=feed.latest(m['market_id'],at)
        if quote is not None:
            q,qt=quote
            require(qt<=at and 0<=decimal(q)<=1,'invalid/future review quote')
            # Terminal boundary prints do not supply an admissible entry quote.
            # Preserve the last supplied quote; age validation prevents stale fills.
            if 0<decimal(q)<1:
                inp['market']={'probability':float(q),'observed_at':iso(qt),'available_at':iso(qt)}
    if context['markets']:validate_context(context)
    return context


def run_window(job, feed, policy, runner, *, end, max_reviews, recorded_latencies=None, learning=False):
    """One bounded research window, with a separately labelled residual settlement replay.

    Learning consumes every available closed trade in training only. Independent
    evaluation freezes lessons. end/max_reviews are declared compute limits.
    """
    validate_policy(policy);validate_context(job['context'])
    require(type(max_reviews) is int and max_reviews>0,'invalid review budget')
    require(type(learning) is bool and (not learning or job['partition']=='train'), 'online learning requires training partition')
    require(not runner.trade_reflections,'runner must start a fresh account/learning session')
    start=timestamp(job['context']['observation_time'],'start')
    require(end.tzinfo is not None and end>start,'invalid review window')
    require(set(job['labels'])=={m['market_id'] for m in job['context']['markets']},'label population differs')
    settlements=[Settlement(mid,timestamp(l['resolution_time'],'settlement'),l['outcome']) for mid,l in job['labels'].items()]
    episodes=[];at=start;timing=runner.timing
    while at<end and len(episodes)<max_reviews:
        prior=replay(episodes,settlements,feed,policy,timing,through=at);account=prior['snapshot']
        resolved={s.market_id for s in settlements if s.time<=at}
        context=refresh_context(job['context'],at,feed,resolved)
        if not context['markets']:break
        runner.calls=[];runner.discovery={};decision=None;error=None;began=time.perf_counter()
        cases=trade_experiences(episodes,prior,job['labels'],policy)
        learn_available(runner,cases,iso(at),enabled=learning)
        try:decision=runner.decide(context,feed,policy,account)
        except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        measured=time.perf_counter()-began
        elapsed=measured if recorded_latencies is None else recorded_latencies[len(episodes)]
        require(decimal(elapsed)>=0,'negative generation latency')
        ep={'context':context,'account_at_observation':account,'decision':decision,'decision_error':error,
            'decision_seconds':elapsed,'measured_seconds':elapsed,'calls':deepcopy(runner.calls),
            'discovery':deepcopy(runner.discovery),'fine_tuning_triggered':False}
        episodes.append(ep)
        interval=timing['failure_review_seconds'] if decision is None else decision['lifecycle']['next_review_after_seconds']
        # One shared executor: no overlapping calls or backdating a late decision.
        at=max(at+timedelta(seconds=interval),at+timedelta(seconds=float(elapsed),microseconds=1))
    observation=replay(episodes,settlements,feed,policy,timing,through=end)
    terminal=replay(episodes,settlements,feed,policy,timing)
    experiences=trade_experiences(episodes,observation,job['labels'],policy)
    runner.calls=[]
    learn_available(runner,experiences,iso(end),enabled=learning)
    closing_calls=deepcopy(runner.calls)
    return {'kind':'team_lifecycle_window_v1','episodes':episodes,'observation_account':observation,
        'trade_experiences':experiences,'terminal_experiences_without_further_reviews':
            trade_experiences(episodes,terminal,job['labels'],policy),
        'trade_reflections':deepcopy(list(runner.trade_reflections.values())),
        'closing_reflection_calls':closing_calls,'online_learning':learning,
        'terminal_without_further_reviews':terminal,'next_review_at':iso(at),
        'review_budget_exhausted':len(episodes)>=max_reviews,'review_count':len(episodes),
        'effectiveness_verified':False,'fine_tuning_triggered':False,'real_orders_sent':0,
        'interpretation':'Bounded interface experiment; residual holdings settle without additional model decisions. Not validated strategy performance.'}
