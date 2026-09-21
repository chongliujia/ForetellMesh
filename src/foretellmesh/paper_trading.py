"""USD paper ledger and causal event replay. No exchange client or live orders.

Trade prints are hypothetical fill references, not executable order books.
All monetary arithmetic uses Decimal; cost parameters are scenario assumptions.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from fractions import Fraction
from functools import lru_cache
import heapq
import sqlite3

from .pma_trades import quote_at
from .schema import ValidationError, binary_outcome, iso, timestamp

D = Decimal


def decimal(value) -> Decimal:
    if type(value) not in (str, int, float, Decimal):raise ValidationError('invalid decimal value')
    try:value=D(str(value))
    except InvalidOperation as exc:raise ValidationError('invalid decimal value') from exc
    if not value.is_finite():raise ValidationError('nonfinite decimal value')
    return value


@dataclass(frozen=True)
class Signal:
    sample_id: str
    market_id: str
    event_group_id: str
    observation_time: datetime
    decision_time: datetime
    quote_time: datetime
    market_probability: Decimal
    probability: Decimal | None
    action: str = 'auto'
    expires_at: datetime | None = None
    order_ttl_seconds: int | None = None
    position_id: str | None = None
    min_exit_price: Decimal | None = None


@dataclass(frozen=True)
class Settlement:
    market_id: str
    time: datetime
    outcome: int


def validate_policy(c: dict) -> dict:
    names={'initial_cash','max_trade_usd','max_event_usd','max_portfolio_usd','min_trade_usd','min_edge',
           'entry_price_premium','fee_fraction','max_quote_age_seconds','fill_window_seconds'}
    if set(c)!=names:raise ValidationError('paper policy fields differ')
    out={k:decimal(v) if k not in ('max_quote_age_seconds','fill_window_seconds') else v for k,v in c.items()}
    if (any(type(out[k]) is not int or out[k]<1 for k in ('max_quote_age_seconds','fill_window_seconds'))
            or not 0<out['min_trade_usd']<=out['max_trade_usd']<=out['max_event_usd']<=out['max_portfolio_usd']<=out['initial_cash']
            or not 0<out['min_edge']<1 or not 0<=out['entry_price_premium']<1 or not 0<=out['fee_fraction']<1):
        raise ValidationError('invalid paper risk/cost policy')
    return out


def choose_side(signal: Signal, policy: dict):
    """Outcome-free decision. Positive estimated edge after scenario costs only."""
    if signal.probability is None:return None
    p,q=decimal(signal.probability),decimal(signal.market_probability)
    if not 0<=p<=1 or not 0<q<1:raise ValidationError('invalid signal probability')
    choices=[]
    for side,belief,reference in [('yes',p,q),('no',1-p,1-q)]:
        unit=(reference+policy['entry_price_premium'])*(1+policy['fee_fraction'])
        edge=belief-unit
        if reference+policy['entry_price_premium']<1 and edge>=policy['min_edge']:
            choices.append({'side':side,'belief':belief,'estimated_edge':edge})
    return max(choices,key=lambda x:x['estimated_edge']) if choices else None


class TradePrintFeed:
    """Read-only PMA block-mean print replay; no depth/queue availability claim."""
    def __init__(self,path):
        self.connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        self.connection.execute('PRAGMA query_only=ON')

    def close(self):self.connection.close()

    @lru_cache(maxsize=20000)
    def latest(self,market_id,time):
        quote,_=quote_at(self.connection,market_id,iso(time),10**12)
        if quote is None:return None
        # The underlying dataset also uses the same block mean float encoding.
        return decimal(quote['probability']),datetime.fromtimestamp(quote['unix_time'],tz=time.tzinfo)

    @lru_cache(maxsize=20000)
    def next(self,market_id,time,deadline):
        row=self.connection.execute('SELECT t.block_number,b.unix_time FROM trades t JOIN blocks b USING(block_number) '
            'WHERE t.market_id=? AND b.unix_time>? AND b.unix_time<=? ORDER BY b.unix_time,t.block_number LIMIT 1',
            (market_id,time.timestamp(),deadline.timestamp())).fetchone()
        if row is None:return None
        block,seconds=row
        raw=self.connection.execute('SELECT numerator,denominator FROM trades WHERE market_id=? AND block_number=?',
                                    (market_id,block)).fetchall()
        price=sum((Fraction(int(n),int(d)) for n,d in raw),Fraction())/len(raw)
        return D(price.numerator)/D(price.denominator),datetime.fromtimestamp(seconds,tz=time.tzinfo)


def simulate(signals: list[Signal], settlements: list[Settlement], feed, config: dict, *, through=None, lifecycle=False) -> dict:
    c=validate_policy(config)
    if through is not None and through.tzinfo is None:raise ValidationError('snapshot cutoff must be timezone aware')
    if len({s.sample_id for s in signals})!=len(signals):raise ValidationError('duplicate paper signal')
    group_by_market={}
    for s in signals:
        extended = (s.action != 'auto' or any(v is not None for v in
                    (s.expires_at, s.order_ttl_seconds, s.position_id, s.min_exit_price)))
        if extended and not lifecycle:raise ValidationError('lifecycle signals require explicit opt-in')
        if s.action not in ('auto','buy','hold','sell'):raise ValidationError('invalid lifecycle action')
        if s.expires_at is not None and (s.expires_at.tzinfo is None or s.expires_at<=s.observation_time):
            raise ValidationError('invalid signal expiry')
        if s.order_ttl_seconds is not None and (type(s.order_ttl_seconds) is not int or s.order_ttl_seconds<1):
            raise ValidationError('invalid order lifetime')
        if s.action=='sell':
            if not s.position_id or s.min_exit_price is None or not 0<decimal(s.min_exit_price)<1:
                raise ValidationError('sell requires position identity and positive price limit')
        elif s.position_id is not None or s.min_exit_price is not None:
            raise ValidationError('exit fields on non-sell action')
        for t in (s.observation_time,s.decision_time,s.quote_time):
            if t.tzinfo is None:raise ValidationError('paper timestamps must be timezone aware')
        if not s.quote_time<=s.observation_time<=s.decision_time:raise ValidationError('future quote or decision before observation')
        if group_by_market.get(s.market_id,s.event_group_id)!=s.event_group_id:raise ValidationError('market group changed')
        group_by_market[s.market_id]=s.event_group_id
        if not 0<decimal(s.market_probability)<1 or s.probability is not None and not 0<=decimal(s.probability)<=1:
            raise ValidationError('invalid paper probability')
    settlement_by_market={}
    for s in settlements:
        binary_outcome(s.outcome)
        if s.time.tzinfo is None or s.market_id in settlement_by_market:raise ValidationError('invalid/duplicate settlement')
        settlement_by_market[s.market_id]=s
    if set(group_by_market)-set(settlement_by_market):raise ValidationError('missing terminal settlement')
    events=[];serial=0
    def push(t,priority,kind,payload):
        nonlocal serial
        serial+=1;heapq.heappush(events,(t,priority,serial,kind,payload))
    for s in sorted(signals,key=lambda s:(s.decision_time,s.sample_id)):push(s.decision_time,2,'decision',s)
    for s in sorted(settlements,key=lambda s:(s.time,s.market_id)):push(s.time,0,'settlement',s)
    cash=c['initial_cash'];pending={};positions={};exits={};resolved=set();ledger=[];curve=[];skips=Counter()
    fees=D(0);turnover=D(0);realized=D(0);wins=0;closed=0;peak=cash;max_drawdown=D(0);max_drawdown_fraction=D(0)
    def record(t,kind,**values):
        ledger.append({'time':iso(t),'kind':kind,**{k:str(v) if isinstance(v,Decimal) else v for k,v in values.items()}})
    def exposure(group=None):
        return sum((v['budget'] for v in pending.values() if group is None or v['group']==group),D(0))+sum(
            (v['cost'] for v in positions.values() if group is None or v['group']==group),D(0))
    def equity(t):
        nonlocal peak,max_drawdown,max_drawdown_fraction
        value=cash+sum((o['budget'] for o in pending.values()),D(0));stale=0
        for pos in positions.values():
            quote=feed.latest(pos['market'],t)
            if quote is None:q,qt=pos['reference'],pos['filled_at'];stale+=1
            else:
                q,qt=quote
                if qt>t or not 0<=q<=1:raise ValidationError('invalid/future equity mark')
                stale+=int((t-qt).total_seconds()>c['max_quote_age_seconds'])
            side_price=q if pos['side']=='yes' else 1-q
            liquidation=max(D(0),side_price-c['entry_price_premium'])*(1-c['fee_fraction'])
            value+=pos['shares']*liquidation
        peak=max(peak,value);max_drawdown=max(max_drawdown,peak-value)
        max_drawdown_fraction=max(max_drawdown_fraction,(peak-value)/peak)
        curve.append({'time':iso(t),'cash':str(cash),'reserved':str(sum((o['budget'] for o in pending.values()),D(0))),
                      'open_positions':len(positions),'equity_proxy':str(value),'stale_marks':stale})
    while events:
        t,_,_,kind,item=heapq.heappop(events)
        if through is not None and t>through:break
        if kind=='settlement':
            resolved.add(item.market_id)
            for oid,o in list(exits.items()):
                if o['market']==item.market_id:
                    del exits[oid];record(t,'cancel',order_id=oid,reason='settled_before_sell')
            for oid,o in list(pending.items()):
                if o['market']==item.market_id:
                    cash+=o['budget'];del pending[oid];record(t,'cancel',order_id=oid,reason='settled_before_fill')
            for oid,pos in list(positions.items()):
                if pos['market']!=item.market_id:continue
                payout=pos['shares']*(item.outcome if pos['side']=='yes' else 1-item.outcome)
                pnl=payout-pos['cost'];cash+=payout;realized+=pnl;closed+=1;wins+=int(pnl>0)
                record(t,'settle',order_id=oid,market_id=pos['market'],outcome=item.outcome,payout=payout,net_pnl=pnl)
                del positions[oid]
        elif kind=='decision':
            s=item;reason=None
            # A new explicit review replaces that market's outstanding intent.
            # Missing/invalid model output produces no signal and cancels nothing.
            if s.action!='auto' and (s.expires_at is None or t<s.expires_at):
                for oid,o in list(pending.items()):
                    if o['market']==s.market_id:
                        cash+=o['budget'];del pending[oid];record(t,'cancel',order_id=oid,reason='superseded_review')
                for oid,o in list(exits.items()):
                    if o['market']==s.market_id:
                        del exits[oid];record(t,'cancel',order_id=oid,reason='superseded_review')
            if s.action in ('hold','sell'):
                if s.market_id in resolved:reason='already_settled'
                elif s.expires_at is not None and t>=s.expires_at:reason='expired_decision'
                elif s.action=='hold':reason='team_hold'
                elif (t-s.quote_time).total_seconds()>c['max_quote_age_seconds']:reason='stale_decision_quote'
                elif s.position_id not in positions or positions[s.position_id]['market']!=s.market_id:reason='position_not_held'
                if reason:
                    skips[reason]+=1;record(t,'hold',sample_id=s.sample_id,reason=reason)
                else:
                    pos=positions[s.position_id]
                    deadline=t+timedelta(seconds=s.order_ttl_seconds or c['fill_window_seconds'])
                    if s.expires_at is not None:deadline=min(deadline,s.expires_at)
                    exits[s.sample_id]={'market':s.market_id,'position_id':s.position_id,
                        'min_exit_price':decimal(s.min_exit_price),'expires_at':deadline,'signal_expires_at':s.expires_at}
                    record(t,'sell_order',order_id=s.sample_id,position_id=s.position_id,market_id=s.market_id,
                        side=pos['side'],shares=pos['shares'],min_exit_price=decimal(s.min_exit_price),expires_at=iso(deadline))
                    quote=feed.next(s.market_id,t,deadline)
                    if quote is not None and (not t<quote[1]<=deadline or not 0<quote[0]<1):
                        raise ValidationError('invalid execution print timestamp/price')
                    push(deadline if quote is None else quote[1],1,'sell_fill',(s.sample_id,quote))
                equity(t)
                continue
            if s.market_id in resolved:reason='already_settled'
            elif s.expires_at is not None and t>=s.expires_at:reason='expired_decision'
            elif (s.decision_time-s.quote_time).total_seconds()>c['max_quote_age_seconds']:reason='stale_decision_quote'
            elif any(o['market']==s.market_id for o in [*pending.values(),*positions.values()]):reason='already_exposed_to_contract'
            choice=None if reason else choose_side(s,c)
            if not reason and choice is None:reason='missing_forecast' if s.probability is None else 'no_cost_adjusted_edge'
            budget=min(cash,c['max_trade_usd'],c['max_event_usd']-exposure(s.event_group_id),c['max_portfolio_usd']-exposure())
            if not reason and budget<c['min_trade_usd']:reason='capital_or_exposure_limit'
            if reason:
                skips[reason]+=1;record(t,'hold',sample_id=s.sample_id,reason=reason)
            else:
                cash-=budget
                order={'market':s.market_id,'group':s.event_group_id,'budget':budget,**choice}
                pending[s.sample_id]=order
                record(t,'order',order_id=s.sample_id,market_id=s.market_id,event_group_id=s.event_group_id,
                       side=choice['side'],budget=budget,estimated_edge=choice['estimated_edge'])
                deadline=t+timedelta(seconds=s.order_ttl_seconds or c['fill_window_seconds'])
                if s.expires_at is not None:deadline=min(deadline,s.expires_at)
                if lifecycle:order.update(expires_at=deadline,signal_expires_at=s.expires_at)
                quote=feed.next(s.market_id,t,deadline)
                if quote is None:push(deadline,1,'fill',(s.sample_id,None))
                else:
                    if not t<quote[1]<=deadline or not 0<quote[0]<1:raise ValidationError('invalid execution print timestamp/price')
                    push(quote[1],1,'fill',(s.sample_id,quote))
        elif kind=='sell_fill':
            oid,quote=item
            if oid not in exits:continue
            order=exits.pop(oid);pos=positions.get(order['position_id'])
            if pos is None:raise ValidationError('reserved sell position disappeared')
            reason='no_post_decision_print' if quote is None else None
            if order['signal_expires_at'] is not None and t>=order['signal_expires_at']:reason='expired_signal'
            if quote:
                q,_=quote;price=(q if pos['side']=='yes' else 1-q)-c['entry_price_premium']
                if reason is None and price<order['min_exit_price']:reason='sell_price_limit'
            if reason:record(t,'cancel',order_id=oid,reason=reason)
            else:
                gross=(pos['shares']*price).quantize(D('.000001'),rounding=ROUND_DOWN)
                fee=(gross*c['fee_fraction']).quantize(D('.000001'),rounding=ROUND_DOWN)
                proceeds=gross-fee;pnl=proceeds-pos['cost']
                cash+=proceeds;fees+=fee;realized+=pnl;closed+=1;wins+=int(pnl>0)
                record(t,'sell_fill',order_id=oid,position_id=order['position_id'],market_id=pos['market'],
                    side=pos['side'],shares=pos['shares'],price=price,reference_yes_price=q,
                    gross_proceeds=gross,fee=fee,proceeds=proceeds,net_pnl=pnl,
                    holding_seconds=(t-pos['filled_at']).total_seconds(),
                    fill_assumption='full_fill_at_next_block_mean_minus_premium')
                del positions[order['position_id']]
        else:
            oid,quote=item
            if oid not in pending:continue
            order=pending.pop(oid);cash+=order['budget']
            reason='no_post_decision_print' if quote is None else None
            if order.get('signal_expires_at') is not None and t>=order['signal_expires_at']:reason='expired_signal'
            if quote:
                q,qt=quote;reference=q if order['side']=='yes' else 1-q
                price=reference+c['entry_price_premium'];unit=price*(1+c['fee_fraction'])
                if reason is None and (price>=1 or order['belief']-unit<c['min_edge']):reason='price_limit'
            if reason:record(t,'cancel',order_id=oid,reason=reason)
            else:
                shares=(order['budget']/unit).quantize(D('.000001'),rounding=ROUND_DOWN)
                # Currency legs must have finite precision: repeated fractional
                # block means otherwise accumulate non-associative rounding dust.
                # This is the simulator's micro-USD floor, not an exchange fee rule.
                notional=(shares*price).quantize(D('.000001'),rounding=ROUND_DOWN)
                fee=(notional*c['fee_fraction']).quantize(D('.000001'),rounding=ROUND_DOWN);cost=notional+fee
                cash-=cost;fees+=fee;turnover+=notional
                positions[oid]={**order,'cost':cost,'shares':shares,'reference':q,'filled_at':t}
                record(t,'fill',order_id=oid,market_id=order['market'],side=order['side'],shares=shares,price=price,
                       reference_yes_price=q,fee=fee,cost=cost,fill_assumption='full_fill_at_next_block_mean_plus_premium')
        if cash<0 or exposure()>c['max_portfolio_usd']:raise ValidationError('paper accounting/risk invariant violated')
        equity(t)
    if through is None and (positions or pending or exits or cash!=c['initial_cash']+realized):raise ValidationError('paper terminal ledger does not reconcile')
    if through is not None:equity(through)
    result={'kind':'historical_trade_print_simulation','initial_cash':str(c['initial_cash']),'final_cash':str(cash),
            'net_pnl':str(realized),'return_fraction':str(realized/c['initial_cash']),'fees_paid':str(fees),
            'entry_turnover':str(turnover),'orders':sum(r['kind']=='order' for r in ledger),'filled_trades':closed,
            'winning_trades':wins,'win_rate':wins/closed if closed else None,'hold_reasons':dict(skips),
            'sampled_equity_proxy_max_drawdown_usd':str(max_drawdown),
            'sampled_equity_proxy_max_drawdown_fraction':str(max_drawdown_fraction),
            'equity_points_with_stale_marks':sum(r['stale_marks']>0 for r in curve),
            'currency_accounting':'micro_usd_floor_v1','share_decimal_places':6,
            'ledger':ledger,'equity_curve':curve,'real_orders_sent':0}
    if through is not None:
        result['snapshot']={'as_of':iso(through),'cash':str(cash),'reserved':curve[-1]['reserved'],
            'equity_proxy':curve[-1]['equity_proxy'],'stale_marks':curve[-1]['stale_marks'],
            'positions':[{'market_id':p['market'],'side':p['side'],'shares':str(p['shares']),
                          'cost':str(p['cost']),'filled_at':iso(p['filled_at'])} for p in positions.values()],
            'pending_orders':len(pending)+len(exits),'realized_net_pnl':str(realized),'fees_paid':str(fees)}
        if lifecycle:
            for row,oid in zip(result['snapshot']['positions'],positions):row['position_id']=oid
            result['snapshot']['open_orders']=[{'order_id':oid,'market_id':o['market'],'action':action,
                'expires_at':iso(o['expires_at'])} for action,orders in [('buy',pending),('sell',exits)] for oid,o in orders.items()]
    if lifecycle:
        result['execution_protocol']='team_lifecycle_v1'
        result['entry_count']=sum(r['kind']=='fill' for r in ledger)
        result['early_exit_count']=sum(r['kind']=='sell_fill' for r in ledger)
        result['settlement_count']=sum(r['kind']=='settle' for r in ledger)
        result['exit_turnover']=str(sum((decimal(r['gross_proceeds']) for r in ledger if r['kind']=='sell_fill'),D(0)))
    return result
