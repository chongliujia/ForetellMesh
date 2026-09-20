from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
from importlib.util import find_spec
from pathlib import Path
import unittest

from foretellmesh.data import strict_json
from foretellmesh.mean_reversion import MarketWindow,HistoricalBars,features
from foretellmesh.paper_trading import Settlement
from foretellmesh.schema import ValidationError
from foretellmesh.trading_rl_data import ReplayData

AVAILABLE=find_spec('gymnasium') is not None and find_spec('numpy') is not None
if AVAILABLE:from foretellmesh.trading_rl_env import AllocationEnv
ROOT=Path(__file__).resolve().parents[1]
CONFIG=strict_json((ROOT/'configs/allocation_ppo_v1.json').read_text())
T=datetime(2025,2,1,tzinfo=timezone.utc);H=timedelta(hours=1)
FREE=CONFIG['scenarios']['frictionless_reference'];COST=CONFIG['scenarios']['cost_assumption']


def fixture(prices=lambda h:.5,days=10,markets=1,prints=True):
    p=CONFIG['environment'];ticks=[T+h*H for h in range(days*24+1)];windows=[];raw={}
    for i in range(markets):
        mid=str(i);windows.append(MarketWindow(mid,'g',T-4*24*H,ticks[-1]))
        bars=[]
        for h in range(-76,days*24+1):
            q=D(str(prices(h)))
            bars.append((T+h*H,q,1))
            if prints:bars.append((T+h*H+timedelta(seconds=60),q,1))
        raw[mid]=bars
    feed=HistoricalBars(raw);states=[{m.market_id:features(feed,m.market_id,t,p) for m in windows} for t in ticks]
    data=ReplayData('train',windows,T,ticks[-1],ticks,feed,states,[],{}, {})
    labels=[Settlement(m.market_id,ticks[-1]+H,1) for m in windows]
    return data,labels


def env_for(data,labels,costs=COST):return AllocationEnv(data,labels,CONFIG['environment'],costs,record=True)


def finish(env,weekly_only=False):
    while not env.done:env.step(env.fixed_action(weekly_only=weekly_only))
    return env.summary()


@unittest.skipUnless(AVAILABLE,'optional Gymnasium/numpy dependencies unavailable')
class AllocationEnvTests(unittest.TestCase):
    def test_gym_contract_and_observations(self):
        from gymnasium.utils.env_checker import check_env
        data,labels=fixture(days=3);env=env_for(data,labels)
        check_env(env,skip_render_check=True)
        obs,_=env.reset(seed=12)
        self.assertTrue(env.observation_space.contains(obs));self.assertTrue(env.action_masks().any())

    def test_reward_telescope_both_sides_fees_and_partial_exit(self):
        for price,action,side in [(.4,2,'yes'),(.6,4,'no')]:
            data,labels=fixture(lambda h:.5 if h<0 else price if h==0 else .5,days=3)
            env=env_for(data,labels);env.reset();env.step(action)
            buy=next(x for x in env.ledger if x['kind']=='buy_fill');self.assertEqual(buy['side'],side)
            self.assertTrue(env.action_masks()[5]);env.step(5)  # half of >$2 liquidation value
            self.assertEqual(len(env.positions),1);env.step(6)
            r=finish(env);self.assertGreater(D(r['net_pnl']),0)
            self.assertEqual(D(r['final_cash']),D('100')+D(r['net_pnl']))
            self.assertAlmostEqual(r['reward_sum_usd'],float(r['net_pnl']),places=8)
            sells=[x for x in env.ledger if x['kind']=='sell_fill']
            self.assertFalse(sells[0]['full_exit']);self.assertTrue(sells[1]['full_exit'])
            self.assertEqual(sum(D(x['shares']) for x in sells),D(buy['shares']))

    def test_no_tail_reversal_for_cadence_and_no_fake_fills(self):
        data,labels=fixture(lambda h:.995)
        env=env_for(data,labels);env.reset()
        for _ in range(144):env.step(0)
        self.assertTrue(env.due);self.assertFalse(env.action_masks()[1:].any())
        r=finish(env,weekly_only=True)
        self.assertFalse(r['cadence']['compliant']);self.assertEqual(r['entry_count'],0)
        self.assertEqual(r['final_cash'],'100');self.assertEqual(r['reward_sum_usd'],0)
        free=env_for(data,labels,FREE);free.reset()
        for _ in range(144):free.step(0)
        self.assertFalse(free.action_masks()[0]);self.assertTrue(free.action_masks()[1])
        self.assertFalse(free.action_masks()[3])

    def test_weekly_control_retries_and_counts_actual_fills(self):
        data,labels=fixture(days=16);env=env_for(data,labels);env.reset();r=finish(env,True)
        self.assertTrue(r['policy_requirements_met']);self.assertGreaterEqual(r['entry_count'],2)
        self.assertLess(D(r['net_pnl']),0);self.assertEqual(r['invalid_actions'],0)
        self.assertTrue(all(D(x['cost'])<=1 for x in env.ledger if x['kind']=='buy_fill'))
        data,labels=fixture(days=10,prints=False);env=env_for(data,labels);env.reset();r=finish(env,True)
        self.assertFalse(r['policy_requirements_met']);self.assertEqual(r['entry_count'],0)
        self.assertGreater(r['cancellations']['no_post_decision_print'],0)

    def test_zero_synthetic_exit_price_cancels_until_settlement(self):
        data,labels=fixture(lambda h:.1 if h<=0 else .001,days=3)
        env=env_for(data,labels);env.reset();env.step(1)
        r=finish(env,True)
        self.assertEqual(r['entry_count'],1);self.assertEqual(r['exit_count'],0)
        self.assertEqual(r['settlement_count'],1)
        self.assertGreater(r['cancellations']['nonpositive_synthetic_bid'],0)
        self.assertEqual(r['cadence']['filled_transactions_in_period'],1)
        self.assertAlmostEqual(r['reward_sum_usd'],float(r['net_pnl']),places=8)

    def test_future_prices_and_outcomes_do_not_change_prior_policy_observation(self):
        data,labels=fixture(days=3)
        future=HistoricalBars({'0':[(t,D('.8') if t>T+10*H else q,n) for t,q,n in data.feed.data['0']]})
        other=replace(data,feed=future,states=[{'0':features(future,'0',t,CONFIG['environment'])} for t in data.ticks])
        a=env_for(data,labels);b=env_for(other,[Settlement('0',labels[0].time,0)])
        a.reset();b.reset()
        for i in range(10):
            self.assertEqual(a.obs.tolist(),b.obs.tolist());self.assertEqual(a.action_masks().tolist(),b.action_masks().tolist())
            a.step(1 if i==0 else 0);b.step(1 if i==0 else 0)
        self.assertEqual(a.decisions,b.decisions)

    def test_first_adverse_print_cancels_without_using_later_better_quote(self):
        data,labels=fixture(lambda h:.4 if h==0 else .5,days=3)
        raw=[(t,D('.49') if t==T+timedelta(seconds=60) else q,n) for t,q,n in data.feed.data['0']]
        raw.append((T+timedelta(seconds=120),D('.4'),1));data=replace(data,feed=HistoricalBars({'0':raw}))
        env=env_for(data,labels);env.reset();env.step(1)
        self.assertFalse(env.positions);self.assertEqual(env.cash,100)
        self.assertEqual(env.cancellations['price_limit'],1)

    def test_shared_group_cash_limits_and_masking(self):
        data,labels=fixture(days=3,markets=8);env=env_for(data,labels);env.reset()
        for _ in range(30):
            valid=[i for i,x in enumerate(env.action_masks()) if x and i and (i-1)%6<4]
            env.step(valid[-1] if valid else 0)
            self.assertLessEqual(env._exposure('g'),5);self.assertGreaterEqual(env.cash,95)
        r=finish(env);self.assertEqual(r['invalid_actions'],0)
        with self.assertRaises(ValidationError):env.step(0)


if __name__=='__main__':unittest.main()
