"""Gymnasium allocation environment: causal prints, masked actions, no live client.

An action selects a visible candidate and a $1/$2 buy or a partial/full exit.
The same risk/cadence executor is used by learned and deterministic policies.
"""
from collections import Counter
from datetime import timedelta
from decimal import Decimal as D, ROUND_DOWN
import heapq
import math

import gymnasium as gym
import numpy as np

from .mean_reversion import cadence_report
from .paper_trading import decimal
from .schema import ValidationError, iso
from .synthetic_sft import canonical_hash

MICRO=D('.000001')
OPS=('yes_1','yes_2','no_1','no_2','sell_half','sell_all')
GLOBAL_FEATURES=('cash','equity','exposure','hours_since_fill','weekly_due','entries_left','hours_to_end','position_count')
SLOT_FEATURES=('present','yes_price','mean','std','z','change24','history_fraction','activity','quote_age',
               'hours_to_retire','position_side','position_cost','unrealized_return','holding_age',
               'yes_edge','no_edge','yes_round_trip_cost','no_round_trip_cost','group_exposure','mean_known')


class AllocationEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,data,settlements,config,costs,*,record=False):
        super().__init__();self.data=data;self.p=config;self.costs=costs;self.record=record
        self.windows={m.market_id:m for m in data.windows};self.labels={s.market_id:s for s in settlements}
        if len(self.labels)!=len(settlements) or set(self.labels)!=set(self.windows):raise ValidationError('settlement identity mismatch')
        for s in settlements:
            if s.time<=self.windows[s.market_id].retire_at or type(s.outcome) is not int or s.outcome not in (0,1):
                raise ValidationError('invalid settlement')
        self.premium=decimal(costs['entry_price_premium']);self.fee=decimal(costs['fee_fraction'])
        if not (0<=self.premium<1 and 0<=self.fee<1):raise ValidationError('invalid cost scenario')
        self.k=self.p['candidate_slots'];self.action_space=gym.spaces.Discrete(1+self.k*len(OPS))
        self.observation_space=gym.spaces.Box(-5.,5.,shape=(len(GLOBAL_FEATURES)+self.k*len(SLOT_FEATURES),),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.i=0;self.cash=D(self.p['initial_cash']);self.positions={};self.pending={};self.resolved=set()
        self.cooldowns={};self.attempts=Counter();self.last_fill=self.data.start;self.events=[];self.serial=0;self.order_id=0
        self.ledger=[];self.curve=[];self.decisions=[];self.fills=[];self.realized=D(0);self.fees=D(0)
        self.peak=self.cash;self.max_dd=D(0);self.max_dd_fraction=D(0);self.reward_total=0.;self.done=False
        self.rejections=Counter();self.period_end_equity=None;self.auto_orders=0;self.invalid_actions=0
        self.attribution=Counter();self.settlement_count=0;self.cancellations=Counter()
        for s in self.labels.values():self._push(s.time,0,'settle',s)
        self._curve(self.data.start);self._refresh()
        return self.obs.copy(),{}

    @property
    def time(self):return self.data.ticks[self.i]

    @property
    def due(self):return (self.time-self.last_fill).total_seconds()>=self.p['cadence_seconds']-self.p['cadence_lead_seconds']

    def _push(self,t,priority,kind,item):
        self.serial+=1;heapq.heappush(self.events,(t,priority,self.serial,kind,item))

    def _log(self,t,kind,**values):
        row={'time':iso(t),'kind':kind,**{k:str(v) if isinstance(v,D) else v for k,v in values.items()}}
        if self.record:self.ledger.append(row)
        if kind in ('buy_fill','sell_fill'):self.fills.append(row);self.last_fill=t
        if kind in ('sell_fill','settle'):self.attribution[values['entry_reason']]+=decimal(values['net_pnl'])
        if kind=='settle':self.settlement_count+=1
        if kind=='cancel':self.cancellations[values['reason']]+=1

    def _exposure(self,group=None):
        return sum((v['cost'] for mid,v in self.positions.items() if group is None or self.windows[mid].event_group_id==group),D(0))+sum(
            (v['budget'] for v in self.pending.values() if v['kind']=='buy' and (group is None or self.windows[v['market']].event_group_id==group)),D(0))

    def _quote(self,mid,t):
        q=self.data.feed.latest(mid,t)
        if q and q[1]>t:raise ValidationError('future mark')
        return q

    def _side_price(self,p,side):return p if side=='yes' else 1-p

    def _mark(self,t):
        value=self.cash+sum((o['budget'] for o in self.pending.values() if o['kind']=='buy'),D(0));stale=0
        for mid,pos in self.positions.items():
            quote=self._quote(mid,t)
            if quote is None:q=pos['reference_yes'];stale+=1
            else:q,qt=quote;stale+=int((t-qt).total_seconds()>self.p['quote_max_age_seconds'])
            value+=pos['shares']*max(D(0),self._side_price(q,pos['side'])-self.premium)*(1-self.fee)
        return value,stale

    def _curve(self,t):
        v,stale=self._mark(t);self.peak=max(self.peak,v);self.max_dd=max(self.max_dd,self.peak-v)
        self.max_dd_fraction=max(self.max_dd_fraction,(self.peak-v)/self.peak)
        if self.record:self.curve.append({'time':iso(t),'cash':str(self.cash),'equity_proxy':str(v),'stale_marks':stale,'positions':len(self.positions)})

    def _check(self):
        if (self.cash<0 or self._exposure()>D(self.p['max_portfolio_usd'])
                or len(self.positions)>self.p['max_positions']
                or any(self._exposure(m.event_group_id)>D(self.p['max_event_usd']) for m in self.windows.values())
                or self.cash+self._exposure()!=D(self.p['initial_cash'])+self.realized):
            raise ValidationError('cash/cost-basis ledger or risk constraint failed')

    def _cost_fraction(self,q,side):
        p=self._side_price(decimal(q),side);unit=(p+self.premium)*(1+self.fee)
        return float((unit-max(D(0),p-self.premium)*(1-self.fee))/unit)

    def _edge(self,state,side):
        q=decimal(state['yes_price']);mean=decimal(state['mean_yes_price']) if state['mean_yes_price'] is not None else q
        return float(max(D(0),self._side_price(mean,side)-self.premium)*(1-self.fee)-
                     (self._side_price(q,side)+self.premium)*(1+self.fee))

    def _auto_reason(self,mid,pos):
        if self.time>=min(self.windows[mid].retire_at,self.data.end):return 'scope_end'
        age=(self.time-pos['filled_at']).total_seconds()
        if age>=(self.p['weekly_holding_seconds'] if pos['reason']=='weekly' else self.p['max_holding_seconds']):return 'holding_limit'
        quote=self._quote(mid,self.time)
        if quote and (self.time-quote[1]).total_seconds()<=self.p['quote_max_age_seconds']:
            if self._side_price(quote[0],pos['side'])<=pos['entry_reference']-D(self.p['stop_price_points']):return 'stop_loss'
        return None

    def _fresh(self,mid):
        quote=self._quote(mid,self.time)
        return quote is not None and (self.time-quote[1]).total_seconds()<=self.p['quote_max_age_seconds']

    def _can_buy(self,mid,state,side,budget):
        if mid in self.positions or mid in self.resolved or len(self.positions)>=self.p['max_positions']:return False
        if state.get('status')!='ready' or self.time<self.cooldowns.get(mid,self.data.start):return False
        if self.time+timedelta(seconds=self.p['weekly_holding_seconds'])>min(self.windows[mid].retire_at,self.data.end):return False
        if not self.due and self.attempts[self.time.date()]>=self.p['max_entries_per_day']:return False
        q=decimal(state['yes_price']);p=self._side_price(q,side)
        if self.due:
            # No tail-side reversal when the higher-probability side is too costly.
            if budget!=1 or side!=('yes' if q>=D('.5') else 'no'):return False
        elif not D(self.p['min_yes_price'])<=q<=D(self.p['max_yes_price']):return False
        if (p+self.premium)*(1+self.fee)>=1:return False
        group=self.windows[mid].event_group_id
        return min(self.cash,D(self.p['max_event_usd'])-self._exposure(group),D(self.p['max_portfolio_usd'])-self._exposure())>=budget

    def _refresh(self):
        state=self.data.states[self.i];pool=[]
        for mid,s in state.items():
            if mid in self.positions:continue
            if any(self._can_buy(mid,s,side,D(1)) for side in ('yes','no')):
                q=s['yes_price'];side='yes' if q>=.5 else 'no'
                rank=(self._cost_fraction(q,side),-s['trade_count_24h'],mid) if self.due else (
                    -max(self._edge(s,'yes'),self._edge(s,'no')),-s['trade_count_24h'],mid)
                pool.append((rank,mid))
        self.slots=sorted(self.positions)+[mid for _,mid in sorted(pool)][:max(0,self.k-len(self.positions))]
        self.slots=self.slots[:self.k]
        self.mask=np.zeros(self.action_space.n,dtype=bool);self.mask[0]=True
        self.auto={mid:self._auto_reason(mid,pos) for mid,pos in self.positions.items()}
        for j,mid in enumerate(self.slots):
            s=state.get(mid,{});pos=self.positions.get(mid)
            for op in range(4):self.mask[1+j*6+op]=self._can_buy(mid,s,'yes' if op<2 else 'no',D(1+op%2))
            if pos and not self.auto[mid] and self._fresh(mid):
                quote=self._quote(mid,self.time);unit=max(D(0),self._side_price(quote[0],pos['side'])-self.premium)*(1-self.fee)
                self.mask[1+j*6+5]=unit>0
                self.mask[1+j*6+4]=unit*pos['shares']/2>=D(self.p['minimum_partial_exit_usd'])
        # The guard requires an attempt ahead of the deadline, never invents a fill.
        if self.due and self.mask[1:].any() and not any(self.auto.values()):self.mask[0]=False
        equity,_=self._mark(self.time)
        obs=[float(self.cash)/100,float(equity)/100,float(self._exposure())/20,
             (self.time-self.last_fill).total_seconds()/self.p['cadence_seconds'],float(self.due),
             max(0,self.p['max_entries_per_day']-self.attempts[self.time.date()])/self.p['max_entries_per_day'],
             (self.data.end-self.time).total_seconds()/604800,len(self.positions)/self.p['max_positions']]
        for mid in self.slots:
            s=state.get(mid,{});quote=self._quote(mid,self.time);pos=self.positions.get(mid)
            q=float(quote[0]) if quote else float(pos['reference_yes']) if pos else .5
            mean=s.get('mean_yes_price');std=s.get('price_stddev') or 0.;mean=q if mean is None else mean
            entry_side=0 if pos is None else (1 if pos['side']=='yes' else -1)
            pnl=0 if pos is None else float((pos['shares']*max(D(0),self._side_price(D(str(q)),pos['side'])-self.premium)*(1-self.fee)-pos['cost'])/pos['cost'])
            ff=s if s.get('status')=='ready' else {'yes_price':q,'mean_yes_price':None}
            obs.extend([1,q,mean,std*10,math.tanh((s.get('z_score') or 0)/3),s.get('change_24h') or 0,
                s.get('history_points',0)/self.p['lookback_hours'],math.log1p(s.get('trade_count_24h',0))/math.log(10001),
                (self.time-quote[1]).total_seconds()/self.p['quote_max_age_seconds'] if quote else 5,
                (self.windows[mid].retire_at-self.time).total_seconds()/604800,entry_side,
                float(pos['cost'])/2 if pos else 0,pnl,(self.time-pos['filled_at']).total_seconds()/172800 if pos else 0,
                self._edge(ff,'yes'),self._edge(ff,'no'),self._cost_fraction(q,'yes'),self._cost_fraction(q,'no'),
                float(self._exposure(self.windows[mid].event_group_id))/5,float(s.get('mean_yes_price') is not None)])
        obs.extend([0.]*(len(SLOT_FEATURES)*(self.k-len(self.slots))))
        self.obs=np.clip(np.asarray(obs,dtype=np.float32),-5,5)
        if not np.isfinite(self.obs).all():raise ValidationError('nonfinite observation')

    def action_masks(self):return self.mask.copy()

    def fixed_action(self,weekly_only=False):
        """Corrected deterministic comparator; uses the identical candidate universe."""
        valid=[int(a) for a in np.flatnonzero(self.mask) if a]
        if self.due:
            if any(self.auto.values()) or not valid:return 0
            def cost(a):
                slot,op=divmod(a-1,6);mid=self.slots[slot];q=self._quote(mid,self.time)[0]
                if op>=4:
                    pos=self.positions[mid];price=self._side_price(q,pos['side'])
                    return float((self.premium+max(D(0),price-self.premium)*self.fee)/max(price,D('.000001'))),mid,op
                return self._cost_fraction(q,'yes' if op<2 else 'no'),mid,op
            return min(valid,key=cost)
        if weekly_only:return 0
        for a in valid:
            j,op=divmod(a-1,6);mid=self.slots[j]
            if op==5:
                pos=self.positions[mid];target=pos['target'];q=self._quote(mid,self.time)[0]
                if target is not None and (q>=target if pos['side']=='yes' else q<=target):return a
        candidates=[]
        for a in valid:
            j,op=divmod(a-1,6)
            if op>=4:continue
            s=self.data.states[self.i][self.slots[j]];z=s['z_score'];change=s['change_24h'];side='yes' if op<2 else 'no'
            if (z is None or abs(z)<self.p['entry_z'] or change is None or abs(change)>self.p['max_24h_change']
                or side!=('yes' if z<0 else 'no') or self._edge(s,side)<self.p['min_round_trip_edge']):continue
            candidates.append((self._edge(s,side),op%2,-a,a))
        return max(candidates)[-1] if candidates else 0

    def _order(self,t,kind,mid,*,side=None,budget=None,fraction=D(1),reason='policy'):
        self.order_id+=1;oid=str(self.order_id)
        if any(o['market']==mid for o in self.pending.values()):return
        quote=self._quote(mid,t)
        if quote is None:raise ValidationError('order without causal reference')
        q=quote[0];state=self.data.states[self.i].get(mid,{})
        o={'kind':kind,'market':mid,'side':side,'budget':budget,'fraction':fraction,'reason':reason,
           'reference_yes':q,'target':None if state.get('mean_yes_price') is None else decimal(state['mean_yes_price'])}
        if kind=='buy':self.cash-=budget;self.attempts[t.date()]+=1
        self.pending[oid]=o
        self._log(t,kind+'_order',order_id=oid,**{k:v for k,v in o.items() if k!='kind'})
        future=self.data.feed.next(mid,t,t+timedelta(seconds=self.p['fill_window_seconds']))
        if future and not t<future[1]<=t+timedelta(seconds=self.p['fill_window_seconds']):raise ValidationError('invalid execution time')
        self._push(future[1] if future else t+timedelta(seconds=self.p['fill_window_seconds']),1,'fill',(oid,future))

    def _fill(self,t,oid,quote):
        o=self.pending.pop(oid,None)
        if o is None:return
        mid=o['market']
        if o['kind']=='buy':self.cash+=o['budget']
        if quote is None:self._log(t,'cancel',order_id=oid,reason='no_post_decision_print');return
        q=quote[0]
        if not 0<q<1:raise ValidationError('invalid execution price')
        if o['kind']=='buy':
            p=self._side_price(q,o['side']);price=p+self.premium;unit=price*(1+self.fee)
            if unit>=1 or p>self._side_price(o['reference_yes'],o['side'])+D(self.p['entry_price_tolerance']):
                self._log(t,'cancel',order_id=oid,reason='price_limit');return
            shares=(o['budget']/unit).quantize(MICRO,rounding=ROUND_DOWN)
            notional=(shares*price).quantize(MICRO,rounding=ROUND_DOWN);fee=(notional*self.fee).quantize(MICRO,rounding=ROUND_DOWN)
            cost=notional+fee;self.cash-=cost;self.fees+=fee
            self.positions[mid]={**o,'shares':shares,'cost':cost,'entry_reference':p,'filled_at':t}
            self._log(t,'buy_fill',order_id=oid,market=mid,side=o['side'],shares=shares,price=price,cost=cost,fee=fee,entry_reason=o['reason'])
        elif mid in self.positions:
            pos=self.positions[mid];price=self._side_price(q,pos['side'])-self.premium
            if price<=0:self._log(t,'cancel',order_id=oid,reason='nonpositive_synthetic_bid');return
            shares=(pos['shares']*o['fraction']).quantize(MICRO,rounding=ROUND_DOWN)
            full=shares==pos['shares'];basis=pos['cost'] if full else (pos['cost']*shares/pos['shares']).quantize(MICRO,rounding=ROUND_DOWN)
            notional=(shares*price).quantize(MICRO,rounding=ROUND_DOWN);fee=(notional*self.fee).quantize(MICRO,rounding=ROUND_DOWN)
            proceeds=notional-fee;pnl=proceeds-basis;self.cash+=proceeds;self.fees+=fee;self.realized+=pnl
            if full:del self.positions[mid];self.cooldowns[mid]=t+timedelta(seconds=self.p['cooldown_seconds'])
            else:pos['shares']-=shares;pos['cost']-=basis
            self._log(t,'sell_fill',order_id=oid,market=mid,side=pos['side'],shares=shares,price=price,fee=fee,
                      proceeds=proceeds,net_pnl=pnl,full_exit=full,exit_reason=o['reason'],entry_reason=pos['reason'])

    def _advance(self,until):
        while self.events and self.events[0][0]<=until:
            t,_,_,kind,item=heapq.heappop(self.events)
            if kind=='fill':self._fill(t,*item)
            else:
                mid=item.market_id;self.resolved.add(mid)
                for oid,o in list(self.pending.items()):
                    if o['market']==mid:
                        if o['kind']=='buy':self.cash+=o['budget']
                        del self.pending[oid];self._log(t,'cancel',order_id=oid,reason='settled_before_fill')
                pos=self.positions.pop(mid,None)
                if pos:
                    payout=pos['shares']*(item.outcome if pos['side']=='yes' else 1-item.outcome)
                    pnl=payout-pos['cost'];self.cash+=payout;self.realized+=pnl
                    self._log(t,'settle',market=mid,outcome=item.outcome,payout=payout,net_pnl=pnl,entry_reason=pos['reason'])
            self._check();self._curve(t)

    def step(self,action):
        if self.done:raise ValidationError('step after terminal state')
        action=int(action)
        if not self.action_space.contains(action):raise ValidationError('action out of range')
        before,_=self._mark(self.time);requested=action;due=self.due
        if not self.mask[action]:self.invalid_actions+=1;action=0
        if self.record:self.decisions.append({'time':iso(self.time),'action':requested,'executed_action':action,
            'weekly_due':due,'slots':list(self.slots),'mask':self.mask.tolist(),'observation_sha256':canonical_hash(self.obs.tolist())})
        for mid,reason in self.auto.items():
            if reason:self._order(self.time,'sell',mid,reason=reason);self.auto_orders+=1
        if action:
            j,op=divmod(action-1,6);mid=self.slots[j]
            if op<4:self._order(self.time,'buy',mid,side='yes' if op<2 else 'no',budget=D(1+op%2),reason='weekly' if due else 'policy')
            else:self._order(self.time,'sell',mid,fraction=D('.5') if op==4 else D(1),reason='weekly' if due else 'policy_exit')
        elif due and not any(self.auto.values()):
            self.rejections['no_eligible_action' if not self.mask[1:].any() else 'invalid_action']+=1
            self._log(self.time,'weekly_unfilled',reason='no_eligible_action' if not self.mask[1:].any() else 'invalid_action')
        self.i+=1;self._advance(self.time);self._check();self._curve(self.time)
        self.done=self.i==len(self.data.ticks)-1
        if self.done:
            self.period_end_equity=self._mark(self.time)[0]
            for mid in list(self.positions):self._order(self.time,'sell',mid,reason='calendar_end')
            until=max([self.data.end+timedelta(seconds=self.p['fill_window_seconds'])]+[s.time for s in self.labels.values()])
            self._advance(until)
            if self.pending or self.positions:raise ValidationError('terminal positions not resolved')
            after=self.cash
        else:after=self._mark(self.time)[0]
        reward=float(after-before);self.reward_total+=reward
        self._refresh()
        info={'cash':float(self.cash),'cadence_due':due}
        if self.done:info['episode_summary']=self.summary()
        return self.obs.copy(),reward,self.done,False,info

    def summary(self):
        if not self.done:raise ValidationError('episode incomplete')
        if abs(self.reward_total-float(self.cash-D(self.p['initial_cash'])))>1e-7:raise ValidationError('reward does not telescope to net PnL')
        cadence=cadence_report(self.fills,self.data.start,self.data.end,self.p['cadence_seconds'])
        return {'initial_cash':self.p['initial_cash'],'final_cash':str(self.cash),'net_pnl':str(self.realized),
            'period_end_equity_proxy':str(self.period_end_equity),'reward_sum_usd':self.reward_total,
            'fees_paid':str(self.fees),'entry_count':sum(x['kind']=='buy_fill' for x in self.fills),
            'exit_count':sum(x['kind']=='sell_fill' for x in self.fills),'cadence':cadence,'policy_requirements_met':cadence['compliant'],
            'max_drawdown_usd':str(self.max_dd),'max_drawdown_fraction':str(self.max_dd_fraction),
            'invalid_actions':self.invalid_actions,'automatic_exit_orders':self.auto_orders,'weekly_blocked_ticks':dict(self.rejections),
            'weekly_entries':sum(x['kind']=='buy_fill' and x['entry_reason']=='weekly' for x in self.fills),
            'settlement_count':self.settlement_count,'cancellations':dict(self.cancellations),
            'pnl_by_entry_reason':{k:str(v) for k,v in self.attribution.items()},
            'simulation_only':True,'real_orders_sent':0}
