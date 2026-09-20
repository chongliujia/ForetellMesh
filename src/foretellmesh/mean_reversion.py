"""Causal short-horizon price mean reversion, with an explicit 7-day fill constraint.

Targets are future trade prices, not calibrated event probabilities. Every fill
uses the first strictly later historical trade print, under stated assumptions.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from fractions import Fraction
import heapq
import sqlite3
import statistics

from .paper_trading import decimal, Settlement
from .schema import ValidationError, binary_outcome, iso

D=Decimal
MICRO=D('.000001')


@dataclass(frozen=True)
class MarketWindow:
    market_id: str
    event_group_id: str
    initialized_at: datetime
    retire_at: datetime  # predeclared last approved observation, never settlement


class HistoricalBars:
    """Exact block averages in memory. Policy methods expose only as-of history."""
    def __init__(self, data):
        self.data={};self.times={};self.counts={}
        for market,items in data.items():
            ordered=sorted(items,key=lambda x:x[0])
            if any(t.tzinfo is None or not 0<decimal(p)<1 or type(n) is not int or n<1 for t,p,n in ordered):
                raise ValidationError('invalid historical bar')
            self.data[market]=[(t,decimal(p),n) for t,p,n in ordered]
            self.times[market]=[t for t,_,_ in ordered];acc=[0]
            for _,_,n in ordered:acc.append(acc[-1]+n)
            self.counts[market]=acc

    @classmethod
    def from_store(cls,path,markets,start,end):
        con=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON');data={}
        try:
            for market in sorted(markets):
                raw=con.execute('SELECT b.unix_time,t.block_number,t.numerator,t.denominator FROM trades t '
                    'JOIN blocks b USING(block_number) WHERE t.market_id=? AND b.unix_time>=? AND b.unix_time<=? '
                    'ORDER BY b.unix_time,t.block_number,t.tx,t.log_index',(market,start.timestamp(),end.timestamp()))
                items=[];key=None;total=Fraction();count=0
                def append():
                    if count:items.append((datetime.fromtimestamp(key[0],timezone.utc),decimal(float(total/count)),count))
                for ts,block,n,d in raw:
                    next_key=(ts,block)
                    if key!=next_key:
                        append();total=Fraction();count=0;key=next_key
                    price=Fraction(int(n),int(d))
                    if not 0<price<1:raise ValidationError('invalid native price')
                    total+=price;count+=1
                append();data[market]=items
        finally:con.close()
        return cls(data)

    def latest(self,market,t):
        i=bisect_right(self.times.get(market,[]),t)-1
        if i<0:return None
        ts,p,_=self.data[market][i];return p,ts

    def next(self,market,t,deadline):
        i=bisect_right(self.times.get(market,[]),t)
        if i>=len(self.times.get(market,[])):return None
        ts,p,_=self.data[market][i]
        return (p,ts) if ts<=deadline else None

    def count(self,market,start,end):
        tt=self.times.get(market,[]);acc=self.counts.get(market,[0])
        return acc[bisect_right(tt,end)]-acc[bisect_right(tt,start)]


def validate_config(c):
    fixed={'schema_version':'1','initial_cash':'100','step_seconds':3600,'lookback_hours':72,
        'minimum_history_points':36,'quote_max_age_seconds':10800,'fill_window_seconds':300,
        'cadence_seconds':604800,'cadence_lead_seconds':86400,'max_holding_seconds':172800,
        'mandatory_holding_seconds':86400,'cooldown_seconds':86400,'max_entries_per_day':2,
        'max_trade_usd':'2','mandatory_trade_usd':'1','max_event_usd':'5','max_portfolio_usd':'20',
        'minimum_trade_usd':'1','entry_z':'1.5','std_floor':'0.01','min_round_trip_edge':'0.01',
        'stop_probability_points':'0.05','entry_price_tolerance':'0.02','min_yes_price':'0.05',
        'max_yes_price':'0.95','max_24h_change':'0.15','simulation_only':True,'training':False}
    if c.get('schema_version')=='2':
        fixed.update(schema_version='2',mandatory_price_filter='feasible_side')
    if set(c)!=set(fixed) or any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):
        raise ValidationError('unsupported frozen mean-reversion policy')
    return c


def features(feed,market,t,c):
    quote=feed.latest(market,t)
    if quote is None:return {'status':'no_quote'}
    q,qt=quote
    if qt>t:raise ValidationError('future feature price')
    if (t-qt).total_seconds()>c['quote_max_age_seconds']:return {'status':'stale_quote'}
    history=[]
    for h in range(1,c['lookback_hours']+1):
        at=t-timedelta(hours=h);v=feed.latest(market,at)
        if v is not None:
            if v[1]>at:raise ValidationError('future history price')
            if (at-v[1]).total_seconds()<=c['quote_max_age_seconds']:history.append(float(v[0]))
    mean=statistics.fmean(history) if len(history)>=c['minimum_history_points'] else None
    sd=statistics.pstdev(history) if mean is not None else None
    old=feed.latest(market,t-timedelta(hours=24))
    if old and old[1]>t-timedelta(hours=24):raise ValidationError('future 24h anchor')
    change=float(q-old[0]) if old and (t-timedelta(hours=24)-old[1]).total_seconds()<=c['quote_max_age_seconds'] else None
    return {'status':'ready','as_of':iso(t),'quote_time':iso(qt),'yes_price':float(q),'history_points':len(history),
        'mean_yes_price':mean,'price_stddev':sd,'z_score':(float(q)-mean)/max(sd,float(c['std_floor'])) if mean is not None else None,
        'change_24h':change,'trade_count_24h':feed.count(market,t-timedelta(hours=24),t)}


def entry_choice(state,c,costs,mandatory=False):
    """No future fills, labels, or event probabilities are accepted by this function."""
    if state['status']!='ready':return None
    q=decimal(state['yes_price']);premium=decimal(costs['entry_price_premium']);fee=decimal(costs['fee_fraction'])
    expanded=mandatory and c.get('mandatory_price_filter')=='feasible_side'
    if not 0<q<1:raise ValidationError('invalid signal reference')
    if not expanded and not decimal(c['min_yes_price'])<=q<=decimal(c['max_yes_price']):return None
    if mandatory:
        # Liquidity/participation control: buy the higher-price side for lower
        # relative spread exposure; no claimed information advantage or target.
        sides=[('yes',q),('no',1-q)]
        if expanded:sides=[(s,p) for s,p in sides if (p+premium)*(1+fee)<1]
        if not sides:return None
        side=max(sides,key=lambda x:x[1])[0];target=None;reason='weekly_minimum'
    else:
        z=state['z_score'];change=state['change_24h']
        if z is None or abs(z)<float(c['entry_z']) or change is None or abs(change)>float(c['max_24h_change']):return None
        side='yes' if z<0 else 'no';target=decimal(state['mean_yes_price']);reason='mean_reversion'
    reference=q if side=='yes' else 1-q
    if reference+premium>=1:return None
    unit=(reference+premium)*(1+fee)
    target_side=(target if side=='yes' else 1-target) if target is not None else reference
    edge=max(D(0),target_side-premium)*(1-fee)-unit
    if not mandatory and edge<decimal(c['min_round_trip_edge']):return None
    return {'side':side,'target_yes_price':None if target is None else str(target),
        'expected_round_trip_edge':str(edge),'expected_return_on_cost':str(edge/unit),
        'reference_yes_price':str(q),'reason':reason,'features':state}


def cadence_report(ledger,start,end,seconds):
    fills=sorted({datetime.fromisoformat(r['time']) for r in ledger if r['kind'] in ('buy_fill','sell_fill')
                  and start<=datetime.fromisoformat(r['time'])<=end})
    points=[start,*fills,end];violations=[];maximum=0
    for i,(a,b) in enumerate(zip(points,points[1:])):
        gap=(b-a).total_seconds();maximum=max(maximum,gap)
        missed_at_end=i==len(points)-2 and b not in fills and gap>=seconds
        if gap>seconds or missed_at_end:violations.append({'from':iso(a),'to':iso(b),'gap_hours':gap/3600,
            'overdue_from':iso(a+timedelta(seconds=seconds))})
    return {'required_max_gap_hours':seconds/3600,'compliant':not violations,
        'actual_max_gap_hours':maximum/3600,'filled_transactions_in_period':len([r for r in ledger
            if r['kind'] in ('buy_fill','sell_fill') and start<=datetime.fromisoformat(r['time'])<=end]),
        'violations':violations,'settlements_count_as_trades':False,'orders_count_as_trades':False}


def simulate(markets,settlements,feed,start,end,c,costs,*,natural=True,weekly=True,states=None):
    validate_config(c)
    if start.tzinfo is None or end.tzinfo is None or end<=start:raise ValidationError('invalid replay interval')
    if type(natural) is not bool or type(weekly) is not bool:raise ValidationError('invalid strategy flags')
    if set(costs)!={'entry_price_premium','fee_fraction'} or any(not 0<=decimal(v)<1 for v in costs.values()):
        raise ValidationError('invalid execution costs')
    windows={m.market_id:m for m in markets};labels={s.market_id:s for s in settlements}
    if len(windows)!=len(markets) or len(labels)!=len(settlements) or set(windows)!=set(labels):raise ValidationError('market/settlement identity mismatch')
    for m in markets:
        if m.initialized_at.tzinfo is None or m.retire_at.tzinfo is None or not m.initialized_at<m.retire_at:raise ValidationError('invalid market window')
    for s in settlements:
        binary_outcome(s.outcome)
        if s.time.tzinfo is None or s.time<=windows[s.market_id].retire_at:raise ValidationError('invalid settlement boundary')
    premium=decimal(costs['entry_price_premium']);fee_rate=decimal(costs['fee_fraction'])
    cash=D(c['initial_cash']);reserved={};positions={};resolved=set();cooldown={};last_fill=start
    ledger=[];curve=[];decisions=[];serial=0;oid=0;events=[];entries_day=Counter();rejects=Counter()
    fees=D(0);realized=D(0);closed=0;wins=0;peak=cash;drawdown=D(0);drawdown_fraction=D(0)
    attribution=Counter();period_equity=None
    def push(t,priority,kind,value):
        nonlocal serial
        serial+=1;heapq.heappush(events,(t,priority,serial,kind,value))
    def log(t,kind,**kw):ledger.append({'time':iso(t),'kind':kind,**{k:str(v) if isinstance(v,Decimal) else v for k,v in kw.items()}})
    def exposure(group=None):
        return sum((x['budget'] for x in reserved.values() if x['kind']=='buy' and (group is None or x['group']==group)),D(0))+sum(
            (x['cost'] for x in positions.values() if group is None or x['group']==group),D(0))
    def pending(market):return any(o['market']==market for o in reserved.values())
    def queue_order(t,o):
        nonlocal oid
        oid+=1;o={**o,'order_id':str(oid)};reserved[str(oid)]=o
        log(t,o['kind']+'_order',**{k:v for k,v in o.items() if k not in ('features','kind')})
        quote=feed.next(o['market'],t,t+timedelta(seconds=c['fill_window_seconds']))
        if quote is not None and (not t<quote[1]<=t+timedelta(seconds=c['fill_window_seconds']) or not 0<quote[0]<1):
            raise ValidationError('invalid future execution reference')
        push(quote[1] if quote else t+timedelta(seconds=c['fill_window_seconds']),1,'fill',(str(oid),quote))
    t=start
    while t<=end:push(t,2,'tick',None);t+=timedelta(seconds=c['step_seconds'])
    if (end-start).total_seconds()%c['step_seconds']:push(end,2,'tick',None)
    for s in settlements:push(s.time,0,'settle',s)
    while events:
        t,_,_,kind,item=heapq.heappop(events)
        if kind=='settle':
            resolved.add(item.market_id)
            for key,o in list(reserved.items()):
                if o['market']==item.market_id:
                    if o['kind']=='buy':cash+=o['budget']
                    del reserved[key];log(t,'cancel',order_id=key,reason='settled_before_fill')
            if item.market_id in positions:
                pos=positions.pop(item.market_id);payout=pos['shares']*(item.outcome if pos['side']=='yes' else 1-item.outcome)
                pnl=payout-pos['cost'];cash+=payout;realized+=pnl;closed+=1;wins+=int(pnl>0);attribution[pos['reason']]+=pnl
                log(t,'settle',market=item.market_id,payout=payout,net_pnl=pnl,outcome=item.outcome,entry_reason=pos['reason'])
        elif kind=='fill':
            key,quote=item
            if key not in reserved:continue
            o=reserved.pop(key)
            if o['kind']=='buy':cash+=o['budget']
            if quote is None:log(t,'cancel',order_id=key,reason='no_post_decision_print')
            elif o['kind']=='buy':
                q,_=quote;side_price=q if o['side']=='yes' else 1-q;price=side_price+premium;unit=price*(1+fee_rate)
                reference=o['reference_yes_price'] if o['side']=='yes' else 1-o['reference_yes_price']
                reason='price_limit' if side_price>reference+decimal(c['entry_price_tolerance']) or price>=1 else None
                if o['reason']=='weekly_minimum' and c.get('mandatory_price_filter')=='feasible_side' and unit>=1:
                    reason='all_in_price_limit'
                if o['reason']=='mean_reversion':
                    target=o['target_yes_price'] if o['side']=='yes' else 1-o['target_yes_price']
                    if max(D(0),target-premium)*(1-fee_rate)-unit<decimal(c['min_round_trip_edge']):reason='edge_eroded'
                if reason:log(t,'cancel',order_id=key,reason=reason)
                else:
                    shares=(o['budget']/unit).quantize(MICRO,rounding=ROUND_DOWN)
                    notional=(shares*price).quantize(MICRO,rounding=ROUND_DOWN);fee=(notional*fee_rate).quantize(MICRO,rounding=ROUND_DOWN);cost=notional+fee
                    cash-=cost;fees+=fee;last_fill=t
                    positions[o['market']]={**o,'shares':shares,'cost':cost,'filled_at':t,'entry_reference':side_price}
                    log(t,'buy_fill',order_id=key,market=o['market'],side=o['side'],shares=shares,price=price,fee=fee,cost=cost,entry_reason=o['reason'])
            elif o['market'] in positions:
                pos=positions.pop(o['market']);q,_=quote;price=max(D(0),(q if pos['side']=='yes' else 1-q)-premium)
                notional=(pos['shares']*price).quantize(MICRO,rounding=ROUND_DOWN);fee=(notional*fee_rate).quantize(MICRO,rounding=ROUND_DOWN)
                proceeds=notional-fee;pnl=proceeds-pos['cost'];cash+=proceeds;fees+=fee;realized+=pnl;closed+=1;wins+=int(pnl>0)
                attribution[pos['reason']]+=pnl;last_fill=t;cooldown[o['market']]=t+timedelta(seconds=c['cooldown_seconds'])
                log(t,'sell_fill',order_id=key,market=o['market'],side=pos['side'],shares=pos['shares'],price=price,
                    fee=fee,proceeds=proceeds,net_pnl=pnl,exit_reason=o['reason'],entry_reason=pos['reason'])
        else:
            # Only this branch is the policy. It never reads settlement outcomes or future prints.
            for mid,pos in list(positions.items()):
                if pending(mid):continue
                quote=feed.latest(mid,t);reason=None
                if t>=min(end,windows[mid].retire_at):reason='scope_end'
                elif (t-pos['filled_at']).total_seconds()>=(c['mandatory_holding_seconds'] if pos['reason']=='weekly_minimum' else c['max_holding_seconds']):reason='holding_limit'
                if quote is not None and (t-quote[1]).total_seconds()<=c['quote_max_age_seconds']:
                    q=quote[0];side_price=q if pos['side']=='yes' else 1-q;target=pos['target_yes_price']
                    if side_price<=pos['entry_reference']-decimal(c['stop_probability_points']):reason='stop_loss'
                    elif target is not None and (q>=target if pos['side']=='yes' else q<=target):reason='mean_reached'
                if reason:queue_order(t,{'kind':'sell','market':mid,'reason':reason})
            due=weekly and (t-last_fill).total_seconds()>=c['cadence_seconds']-c['cadence_lead_seconds']
            candidates=[];eligible=[]
            for m in markets:
                if m.market_id in resolved or m.market_id in positions or pending(m.market_id) or t<cooldown.get(m.market_id,start):continue
                horizon=c['mandatory_holding_seconds'] if due else c['max_holding_seconds']
                if not m.initialized_at<=t or t+timedelta(seconds=horizon)>min(end,m.retire_at):continue
                state=features(feed,m.market_id,t,c) if states is None else states[(m.market_id,iso(t))]
                choice=entry_choice(state,c,costs)
                if natural and choice is not None:candidates.append((m,choice))
                forced=entry_choice(state,c,costs,mandatory=True)
                if forced is not None:eligible.append((m,forced))
            candidates.sort(key=lambda x:(-decimal(x[1]['expected_return_on_cost']),-x[1]['features']['trade_count_24h'],x[0].market_id))
            eligible.sort(key=lambda x:(-x[1]['features']['trade_count_24h'],abs(x[1]['features']['yes_price']-.5),x[0].market_id))
            if entries_day[t.date()]>=c['max_entries_per_day']:candidates=[]
            # A due cadence prefers a natural signal; otherwise $1 participation.
            choices=candidates+eligible if due else candidates
            selected=None
            for m,ch in choices:
                cap=decimal(c['mandatory_trade_usd'] if ch['reason']=='weekly_minimum' else c['max_trade_usd'])
                budget=min(cash,cap,decimal(c['max_event_usd'])-exposure(m.event_group_id),decimal(c['max_portfolio_usd'])-exposure())
                if budget<decimal(c['minimum_trade_usd']):continue
                selected=(m,ch,budget);break
            if selected:
                m,ch,budget=selected;cash-=budget
                if ch['reason']=='mean_reversion':entries_day[t.date()]+=1
                decisions.append({'time':iso(t),'market':m.market_id,'weekly_due':due,**ch})
                queue_order(t,{'kind':'buy','market':m.market_id,'group':m.event_group_id,'budget':budget,
                    'side':ch['side'],'reason':ch['reason'],'target_yes_price':None if ch['target_yes_price'] is None else decimal(ch['target_yes_price']),
                    'reference_yes_price':decimal(ch['reference_yes_price'])})
            elif due:
                reason='no_eligible_market' if not eligible and not candidates else 'capital_or_risk_limit'
                rejects[reason]+=1;log(t,'weekly_unfilled',reason=reason)
        if cash<0 or exposure()>decimal(c['max_portfolio_usd']) or any(exposure(m.event_group_id)>decimal(c['max_event_usd']) for m in markets):
            raise ValidationError('cash or exposure constraint violated')
        value=cash+sum((o['budget'] for o in reserved.values() if o['kind']=='buy'),D(0));stale=0
        for mid,pos in positions.items():
            quote=feed.latest(mid,t)
            if quote is None:q=pos['reference_yes_price'];stale+=1
            else:
                q,qt=quote
                if qt>t:raise ValidationError('future equity mark')
                stale+=int((t-qt).total_seconds()>c['quote_max_age_seconds'])
            value+=pos['shares']*max(D(0),(q if pos['side']=='yes' else 1-q)-premium)*(1-fee_rate)
        peak=max(peak,value);drawdown=max(drawdown,peak-value);drawdown_fraction=max(drawdown_fraction,(peak-value)/peak)
        curve.append({'time':iso(t),'cash':str(cash),'equity_proxy':str(value),'positions':len(positions),'stale_marks':stale})
        if t<=end:period_equity=value
    if positions or reserved or cash!=D(c['initial_cash'])+realized:raise ValidationError('terminal ledger does not reconcile')
    cadence=cadence_report(ledger,start,end,c['cadence_seconds'])
    return {'simulation_only':True,'training':False,'real_orders_sent':0,'initial_cash':c['initial_cash'],'final_cash':str(cash),
        'period_end_equity_proxy':str(period_equity),'terminal_after_settlements':True,'net_pnl':str(realized),'fees_paid':str(fees),
        'entry_count':sum(r['kind']=='buy_fill' for r in ledger),'exit_count':sum(r['kind']=='sell_fill' for r in ledger),
        'settlement_count':sum(r['kind']=='settle' for r in ledger),'closed_positions':closed,'winning_positions':wins,
        'win_rate':wins/closed if closed else None,'pnl_by_entry_reason':{k:str(v) for k,v in attribution.items()},
        'sampled_equity_proxy_max_drawdown_usd':str(drawdown),'sampled_equity_proxy_max_drawdown_fraction':str(drawdown_fraction),
        'cadence_required':weekly,'cadence':cadence,'policy_requirements_met':not weekly or cadence['compliant'],
        'weekly_blocked_ticks':dict(rejects),
        'ledger':ledger,'equity_curve':curve,'decisions':decisions}
