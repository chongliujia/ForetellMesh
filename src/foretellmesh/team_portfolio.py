"""One $100 paper account across team decisions; target reporting never fits weights."""
from datetime import timedelta
from .paper_trading import Signal, Settlement, decimal, simulate
from .schema import timestamp, iso
from .team_learning import require, validate_context
from copy import deepcopy


def trajectory(jobs, episodes):
    require(len(jobs) == len(episodes), 'portfolio episode population differs')
    signals = []; labels = {}
    for job, episode in zip(jobs, episodes):
        context = episode['context']; require(context == job['context'], 'portfolio context changed')
        validate_context(context); at = timestamp(context['observation_time'], 'at')
        ready = at+timedelta(seconds=episode['decision_seconds'])
        decision = episode['decision']
        forecasts = {} if decision is None else {r['market_id']: r for r in decision['forecast']['forecasts']}
        vetoes = set() if decision is None else set(decision['risk']['veto_markets'])
        for market in context['markets']:
            mid = market['market_id']; label = job['labels'][mid]
            require(mid not in labels or labels[mid] == label, 'inconsistent settlement across observations')
            labels[mid] = label; forecast = forecasts.get(mid); inp = market['input']
            p = forecast['probability'] if forecast and forecast['consider_trade'] and mid not in vetoes else None
            signals.append(Signal(context['episode_id']+':'+mid, mid, market['event_group_id'], at, ready,
                timestamp(inp['market']['observed_at'], 'quote'), decimal(inp['market']['probability']),
                decimal(p) if p is not None else None))
    settlements = [Settlement(mid, timestamp(l['resolution_time'], 'settlement'), l['outcome']) for mid,l in labels.items()]
    return signals, settlements


def replay_portfolio(jobs, episodes, feed, policy, *, through=None):
    signals, settlements = trajectory(jobs, episodes)
    return simulate(signals, settlements, feed, policy, through=through)


def continuous_feedback(prior_jobs, prior_episodes, job, decision, elapsed, feedback, feed, policy):
    """Retrospective fill attribution from actual shared-capital execution, not reset accounts."""
    draft={'context':job['context'],'decision':decision,'decision_seconds':elapsed}
    jobs=[*prior_jobs,job]; episodes=[*prior_episodes,draft]
    account=replay_portfolio(jobs,episodes,feed,policy)
    ids={job['context']['episode_id']+':'+m['market_id'] for m in job['context']['markets']}
    fills=[r for r in account['ledger'] if r['kind']=='fill' and r['order_id'] in ids]
    settlements=[r for r in account['ledger'] if r['kind']=='settle' and r['order_id'] in ids]
    result=deepcopy(feedback); result['account']=account; result['execution_scope']='single_continuous_account'
    result['available_at']=iso(max(timestamp(result['available_at'],'feedback'),
        *(timestamp(l['available_at'],'label availability') for j in jobs for l in j['labels'].values())))
    result['facts'].update(paper_net_pnl=str(sum((decimal(r['net_pnl']) for r in settlements),decimal(0))),
        paper_fees=str(sum((decimal(r['fee']) for r in fills),decimal(0))),filled_trades=len(fills),
        portfolio_terminal_cash=account['final_cash'])
    return result


def target_progress(account, objective):
    require(objective == {'initial_cash':'100','target_net_return':'0.10','target_terminal_cash':'110',
            'basis':'single_continuous_settled_account','deadline':None,'force_trades':False,
            'automatic_fine_tuning':False}, 'unsupported simulated account objective')
    require(account['initial_cash'] == '100' and 'snapshot' not in account, 'terminal single-account replay required')
    target = decimal(objective['target_terminal_cash'])
    flat_hits = [r['time'] for r in account['equity_curve'] if r['open_positions'] == 0
                 and decimal(r['reserved']) == 0 and decimal(r['cash']) >= target]
    return {'initial_cash':'100','target_terminal_cash':'110','target_net_return':'0.10',
        'final_cash':account['final_cash'],'net_return':account['return_fraction'],
        'terminal_target_reached':decimal(account['final_cash']) >= target,
        'first_observed_flat_target_time':flat_hits[0] if flat_hits else None,
        'shortfall_usd':str(max(decimal(0),target-decimal(account['final_cash']))),
        'sampled_equity_proxy_max_drawdown_usd':account['sampled_equity_proxy_max_drawdown_usd'],
        'effectiveness_verified':False,'fine_tuning_triggered':False,'simulation_only':True,
        'fees_already_in_ledger':True,'deadline':None}
