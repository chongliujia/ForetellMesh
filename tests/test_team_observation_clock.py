from copy import deepcopy
from datetime import timedelta
import unittest
from foretellmesh.mean_reversion import HistoricalBars
from foretellmesh.schema import iso, ValidationError
from foretellmesh.team_observation_clock import input_schedule,live_points
from test_team_executable_methods import AT


class ClockTests(unittest.TestCase):
    def setUp(self):
        self.cat={k:{'initialized_at':iso(AT)} for k in ('a','b')}
        self.binding={'target':'a','peer':'b'};self.refs=[('target',0),('peer',0)]
    def schedule(self,feed,**kwargs):
        return input_schedule(self.binding,self.cat,feed,iso(AT+timedelta(days=110)),mode='trade_clock',refs=self.refs,**kwargs)
    def test_real_trade_observations_recover_midnight_gaps_and_late_lifecycle(self):
        feed=HistoricalBars({k:[(AT+timedelta(days=d,hours=12),.4,1) for d in (0,1,100)] for k in self.cat})
        daily=input_schedule(self.binding,self.cat,feed,iso(AT+timedelta(days=110)),mode='daily_90',refs=self.refs)
        real=self.schedule(feed)
        self.assertEqual(daily['points'],[]);self.assertEqual(len(real['points']),3)
        self.assertEqual(real['points'][-1]['observation_time'],iso(AT+timedelta(days=100,hours=12)))
    def test_missing_peer_is_not_filled_from_future_or_stale_prices(self):
        feed=HistoricalBars({'a':[(AT+timedelta(hours=h),.5,1) for h in (1,5,10)],
                             'b':[(AT+timedelta(hours=9),.4,1)]})
        points=self.schedule(feed)['points'];self.assertEqual(len(points),1)
        self.assertEqual(points[0]['observation_time'],iso(AT+timedelta(hours=10)))
        self.assertLessEqual(points[0]['evidence'][0]['source_time'],points[0]['observation_time'])
    def test_adding_future_prices_does_not_change_schedule_prefix(self):
        data={k:[(AT+timedelta(days=d,hours=12),.4,1) for d in range(5)] for k in self.cat}
        first=self.schedule(HistoricalBars(data));later=deepcopy(data)
        for k in later:later[k].append((AT+timedelta(days=6),.9,1))
        second=self.schedule(HistoricalBars(later))
        self.assertEqual(first['points'],second['points'][:5])
    def test_lag_read_is_bounded_by_its_own_asof_and_initialization(self):
        data={k:[(AT+timedelta(days=d,hours=12),.4,1) for d in range(5)] for k in self.cat}
        result=input_schedule(self.binding,self.cat,HistoricalBars(data),iso(AT+timedelta(days=5)),mode='trade_clock',refs=[('peer',1)])
        self.assertEqual(len(result['points']),4)
        for row in result['points']:
            self.assertEqual(next(e for e in row['evidence'] if e['role']=='peer')['lag_days'],1)
        self.assertEqual(result['failures']['peer:before_initialization'],1)
    def test_spacing_and_cap_never_select_best_future_returns(self):
        data={k:[(AT+timedelta(hours=h),.3+h/1000,1) for h in range(80)] for k in self.cat}
        points=self.schedule(HistoricalBars(data),max_points=2)['points']
        self.assertEqual([r['observation_time'] for r in points],[iso(AT),iso(AT+timedelta(days=1))])
    def test_settlement_censoring_does_not_replace_chosen_points(self):
        feed=HistoricalBars({k:[(AT+timedelta(days=d),.4,1) for d in range(5)] for k in self.cat})
        schedule=self.schedule(feed);before=deepcopy(schedule)
        live=live_points(schedule,self.binding,{'a':iso(AT+timedelta(days=3)),'b':iso(AT+timedelta(days=2))})
        self.assertEqual(len(live),2);self.assertEqual(schedule,before)
        self.assertFalse(schedule['schedule_uses_labels'])
    def test_invalid_grid_controls_rejected(self):
        feed=HistoricalBars({})
        for kw in [{'max_points':0},{'max_points':121},{'spacing_days':0},{'max_age_seconds':10801}]:
            with self.assertRaises(ValidationError): self.schedule(feed,**kw)

if __name__=='__main__':unittest.main()
