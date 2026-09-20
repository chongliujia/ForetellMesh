"""Partition-bound historical data for the allocation-policy RL prototype."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .agent_baseline_data import input_context
from .data import sha256_file, strict_json
from .market_development import rows
from .mean_reversion import HistoricalBars, MarketWindow, features
from .paper_trading import Settlement
from .schema import ValidationError, binary_outcome, iso, timestamp
from .synthetic_sft import canonical_hash


@dataclass
class ReplayData:
    partition: str
    windows: list[MarketWindow]
    start: datetime
    end: datetime
    ticks: list[datetime]
    feed: HistoricalBars
    states: list[dict]
    catalog: list[dict]
    bindings: dict
    sample_markets: dict

    def feature_rows(self):
        return [{'time':iso(t),'markets':state} for t,state in zip(self.ticks,self.states)]


def prepare(dataset: Path, store: Path, partition: str, config: dict) -> ReplayData:
    """Parse only the requested development input partition. Never parse labels here."""
    if partition not in ('train','validation'):raise ValidationError('final test is sealed')
    if sha256_file(dataset/'report.json')!=config['dataset_report_sha256']:raise ValidationError('dataset report changed')
    report=strict_json((dataset/'report.json').read_text());bindings={'report.json':config['dataset_report_sha256']}
    for name in ('membership.jsonl',f'partitions/{partition}.inputs.jsonl',f'partitions/{partition}.labels.jsonl'):
        digest=sha256_file(dataset/name)
        if digest!=report['artifact_hashes'][name]:raise ValidationError('partition artifact changed')
        bindings[name]=digest
    membership={};groups={};events={}
    for r in rows(dataset/'membership.jsonl'):
        sid,split=r['sample_id'],r['split']
        if (sid in membership or split not in ('train','validation','test')
                or r['dataset_version']!=report['dataset_version']
                or groups.get(r['event_group_id'],split)!=split or events.get(r['event_id'],split)!=split):
            raise ValidationError('duplicate identity or cross-partition event')
        membership[sid]=r;groups[r['event_group_id']]=split;events[r['event_id']]=split
    inputs=rows(dataset/f'partitions/{partition}.inputs.jsonl');seen=set();grouped=defaultdict(list);sample_markets={}
    for r in inputs:
        sid=r['sample_id']
        if sid in seen or sid not in membership or membership[sid]['split']!=partition:raise ValidationError('invalid partition identity')
        input_context(r['input']);seen.add(sid);m=membership[sid]
        if not m['event_id'].startswith('polymarket:'):raise ValidationError('wrong market platform')
        mid=m['event_id'].split(':',1)[1]
        if not mid.isdigit():raise ValidationError('invalid market identity')
        sample_markets[sid]=mid;grouped[mid].append(r)
    if not seen or seen!={sid for sid,m in membership.items() if m['split']==partition}:raise ValidationError('incomplete partition')
    proofs={}
    for name,digest in report['artifact_hashes'].items():
        if not (name.startswith('audit/') and name.endswith('/contracts.jsonl')):continue
        if sha256_file(dataset/name)!=digest:raise ValidationError('rule audit changed')
        bindings[name]=digest
        for r in rows(dataset/name):
            if r['market_id'] not in grouped:continue
            if not r.get('proof') or r['proof_errors'] or r['proof']['creator_update_count']!=0:raise ValidationError('invalid immutable rule proof')
            # The audit file includes resolution metadata: project it away.
            pr={k:r['proof'][k] for k in ('initialized_at','rule_valid_through','historical_question')}
            pr['event_group_id']=r['event_group_id'];mid=r['market_id']
            if mid in proofs and proofs[mid]!=pr:raise ValidationError('conflicting historical rules')
            proofs[mid]=pr
    if set(proofs)!=set(grouped):raise ValidationError('missing historical rules')
    windows=[];catalog=[]
    for mid,rr in sorted(grouped.items()):
        pr=proofs[mid];initial=timestamp(pr['initialized_at'],'initialization')
        retire=max(timestamp(x['input']['observation_time'],'observation') for x in rr)
        if initial>=retire or timestamp(pr['rule_valid_through'],'rule coverage')<retire:raise ValidationError('invalid rule interval')
        if any(x['input']['question']!=pr['historical_question'] or membership[x['sample_id']]['event_group_id']!=pr['event_group_id'] for x in rr):
            raise ValidationError('question/group differs from proof')
        windows.append(MarketWindow(mid,pr['event_group_id'],initial,retire))
        catalog.append({'market_id':mid,'event_group_id':pr['event_group_id'],'initialized_at':iso(initial),
                        'retire_at':iso(retire),'question_sha256':canonical_hash(pr['historical_question'])})
    start=min(timestamp(r['input']['observation_time'],'start') for r in inputs)
    end=max(timestamp(r['input']['observation_time'],'end') for r in inputs)
    if end<=start:raise ValidationError('empty replay calendar')
    tr=strict_json((store/'report.json').read_text());n=tr['counts'];digest=sha256_file(store/'prices.sqlite')
    if (tr['kind']!='pma_historical_trade_store' or not n['unique_valid_selected_trades']
        or n['unique_valid_selected_trades']!=n['trades_with_block_time']
        or n['required_distinct_blocks']!=n['matched_block_times'] or digest!=tr['artifact_hashes']['prices.sqlite']):
        raise ValidationError('invalid native trade store')
    bindings.update(trade_store_sha256=digest,trade_report_sha256=sha256_file(store/'report.json'))
    p=config['environment'];feed=HistoricalBars.from_store(store/'prices.sqlite',grouped,
        start-timedelta(hours=p['lookback_hours']+3),end+timedelta(seconds=p['fill_window_seconds']))
    ticks=[];states=[];t=start
    while t<end:ticks.append(t);t+=timedelta(seconds=p['step_seconds'])
    ticks.append(end)
    for t in ticks:
        states.append({m.market_id:features(feed,m.market_id,t,p) for m in windows if m.initialized_at<=t<=m.retire_at})
    return ReplayData(partition,windows,start,end,ticks,feed,states,catalog,bindings,sample_markets)


def settlements(dataset: Path, data: ReplayData) -> list[Settlement]:
    path=dataset/f'partitions/{data.partition}.labels.jsonl'
    if sha256_file(path)!=data.bindings[f'partitions/{data.partition}.labels.jsonl']:raise ValidationError('labels changed')
    rr=rows(path);seen=set();result={};windows={m.market_id:m for m in data.windows}
    for r in rr:
        sid=r['sample_id']
        if sid in seen or sid not in data.sample_markets:raise ValidationError('duplicate/foreign label')
        seen.add(sid);mid=data.sample_markets[sid];label=r['label']
        s=Settlement(mid,timestamp(label['resolution_time'],'onchain settlement'),binary_outcome(label['outcome']))
        if s.time<=windows[mid].retire_at or (mid in result and result[mid]!=s):raise ValidationError('inconsistent settlement')
        result[mid]=s
    if seen!=set(data.sample_markets):raise ValidationError('incomplete labels')
    return list(result.values())
