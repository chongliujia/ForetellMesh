"""Frozen-model, as-of catalogue search and preregistered relationship research.

Tools supply observations, not invented relationships or accepted training truth.
The search universe contains admitted learning inputs only, never held-out jobs.
"""
from collections import Counter
from copy import deepcopy
import json
import math
import re
import time

from .agent_runtime import decode_agent_response
from .schema import fields, iso, timestamp, ValidationError
from .synthetic_sft import canonical_hash
from .team_learning import TeamRunner, require, strings, validate_context, run_query, feedback_for


class HistoricalCatalogue:
    def __init__(self, jobs, *, feed=None, max_age_seconds=10800, partition="train"):
        require(partition in ('train','development'), 'invalid catalogue partition')
        self.feed = feed; self.max_age_seconds = max_age_seconds
        self.rows = {}
        for job in jobs:
            require(job['partition'] == partition, 'catalogue cannot index held-out inputs across partitions')
            validate_context(job['context'])
            for m in job['context']['markets']:
                mid = m['market_id']
                row = {'market_id':mid,'event_group_id':m['event_group_id'],'question':m['input']['question'],
                       'initialized_at':job['provenance'][mid]['initialized_at']}
                row['record_sha256'] = canonical_hash(row)
                require(mid not in self.rows or self.rows[mid] == row, 'catalogue rules changed')
                self.rows[mid] = row

    def visible(self, cutoff):
        at = timestamp(cutoff, 'catalogue cutoff')
        return {k:deepcopy(v) for k,v in self.rows.items() if timestamp(v['initialized_at'], 'initialization') <= at}

    def search(self, query, cutoff, targets=()):
        fields(query, {'text','limit','offset'}, 'catalogue query')
        require(isinstance(query['text'], str) and len(query['text']) <= 200
                and type(query['limit']) is int and 1 <= query['limit'] <= 6
                and type(query['offset']) is int and 0 <= query['offset'] <= 10000, 'invalid catalogue query')
        visible = self.visible(cutoff)
        terms = set(re.findall(r'\w+', query['text'].casefold()))
        docs = {mid:set(re.findall(r'\w+', row['question'].casefold())) for mid,row in visible.items()}
        df = Counter(t for words in docs.values() for t in words)
        scored = []
        for mid,row in visible.items():
            overlap = terms & docs[mid]
            if terms and not overlap and query['text'] != mid: continue
            score = math.fsum(math.log(1+(len(docs)+1)/(df[t]+1)) for t in sorted(overlap))
            if query['text'] == mid: score += 100
            scored.append((score, mid, row))
        scored.sort(key=lambda v:(-v[0],v[1]))
        offset = query['offset']; end = offset+query['limit']
        result = {'query':deepcopy(query),'as_of':cutoff,'visible_count':len(visible),'matched_count':len(scored),
                  'next_offset':end if end < len(scored) else None,
                  'records':[row for _,_,row in scored[offset:end]],
                  'ranking':'as_of_lexical_overlap_no_category_or_outcome_filter'}
        if self.feed is not None:
            result['coverage'] = self.coverage(result['records'], cutoff, targets)
        result['result_sha256'] = canonical_hash(result); return result

    def coverage(self, rows, cutoff, targets):
        """Describe only as-of daily samples, using the identical study freshness rule."""
        visible = self.visible(cutoff)
        require(set(targets) <= set(visible), 'future/unknown coverage target')
        coverage = {}
        for row in rows:
            mid = row['market_id']; windows = {}
            for days in (7, 14, 30):
                result = run_query({'tool':'history','market_ids':[mid],'lookback_days':days},
                                   self.feed, cutoff, self.max_age_seconds)
                series = result['series'][mid]
                overlaps = {}
                for target in sorted(set(targets)-{mid}):
                    pair = run_query({'tool':'pair_changes','market_ids':[target,mid],'lookback_days':days},
                                     self.feed, cutoff, self.max_age_seconds)
                    a,b = pair['series'].values()
                    overlaps[target] = {'paired_daily_samples':len({x['at'] for x in a}&{x['at'] for x in b}),
                                        'paired_daily_changes':pair['paired_daily_changes']}
                windows[str(days)] = {'daily_samples':len(series),
                    'first_source_time':series[0]['source_time'] if series else None,
                    'last_source_time':series[-1]['source_time'] if series else None,
                    'overlap_with_targets':overlaps}
            coverage[mid] = {'as_of':cutoff,'max_quote_age_seconds':self.max_age_seconds,'windows_days':windows}
        return coverage


SEARCH = ('Choose a historical catalogue search. Return exactly {"text":"search words", "limit":6, "offset":0}. '
    'Never copy a previous tool result into the answer: return only the three query arguments text, limit, offset. '
    'Use title words, entities or an exact market ID. Empty text browses ALL visible admitted markets, with pagination. '
    'No categories are imposed. Previously returned results are supplied; refine or browse a different page if useful. '
    'Do not invent a market ID or assume missing evidence. Only contracts initialized by the cutoff can be returned.')
STUDY = ('Plan up to 2 tests using the target markets and retrieved catalogue records. Return exactly '
    '{"investigations":[{"market_ids":["ID1","ID2"],"lookback_days":14,"claim":"tentative relationship",'
    '"falsifier":"observation that would count against it","expected_direction":"positive"}],"no_finding_reason":null}. '
    'Each investigation uses 1 or 2 DISTINCT supplied IDs, at least one target ID, and 1..30 lookback days. '
    'Two IDs test paired daily price changes (positive/negative correlation); one ID tests net price change '
    '(positive means last sampled price above first, negative means below). expected_direction is positive, negative or unspecified. '
    'Each text is at most 600 characters. These are hypotheses, not facts. '
    'The claim must refer to the selected contracts by ID or unambiguous description and state a concrete testable price pattern. '
    'The falsifier must describe an observable result that contradicts that specific pattern. '
    'Never copy schema placeholder text such as tentative relationship or observation that would count against it. '
    'If no worthwhile test is possible, return an empty investigations list and a specific nonempty no_finding_reason. '
    'Correlation alone cannot prove causality or profitable forecasting. Choose what to test yourself.')
REVIEW = ('Review every preregistered investigation against its actual tool result. Return exactly '
    '{"reviews":[{"investigation_id":"supplied ID","verdict":"insufficient_data",'
    '"evidence_refs":["supplied result_sha256"],"reason":"what the data show",'
    '"next_check":"specific independent test"}]}. '
    'Verdict must be retain_for_testing, reject or insufficient_data. Include each investigation exactly once. '
    'Use only its exact tool result hash. Each reason/next_check at most 600 characters. '
    'Claims about future outcomes, causality or external asset prices cannot be established by historical contract-price samples; mark insufficient_data if the required evidence is absent. '
    'Retaining a hypothesis does NOT mean it is validated. If no investigations, return {"reviews":[]}.')
RETROSPECTIVE = ('Review the actual outcome feedback AFTER the original decision. Return exactly '
    '{"observations":[{"fact_id":"supplied feedback fact ID","lesson":"tentative lesson",'
    '"next_check":"how to test it on independent cases"}],"no_lesson_reason":null}. '
    'At most 3 observations, each text at most 600 characters. Reference exact supplied feedback fact IDs, not invented IDs. '
    'Use the preregistered claim, falsifier and tool findings. Profit alone does not validate a relationship; '
    'one binary outcome does not show that a probability should have been 0 or 1. '
    'If nothing can be learned, return observations=[] and explain the concrete evidence gap in no_lesson_reason. '
    'Do not silently return all empty fields or rewrite the ex-ante record.')


def explanation(value, label):
    require(isinstance(value,str) and 0 < len(value.strip()) <= 600, 'missing/bounded '+label)


def catalogue_query_payload(value):
    """Accept one explicit transport envelope; never extract prose or fix arguments."""
    if isinstance(value,dict) and set(value) == {'output'}:
        wrapped=value['output']
        if isinstance(wrapped,dict) and set(wrapped) == {'search'}:
            return deepcopy(wrapped['search'])
    return deepcopy(value)


def study_quality_issues(spec):
    """Minimum semantic binding; passing is NOT proof of a sensible or true claim."""
    placeholders={'tentative relationship','observation that would count against it',
                  'testable statement naming the chosen ids','observable price result that would contradict the claim'}
    issues=[]
    if any(spec[k].strip().casefold() in placeholders for k in ('claim','falsifier')):
        issues.append('schema placeholder copied as hypothesis/falsifier')
    return issues


def validate_study(value, allowed, targets):
    fields(value, {'investigations','no_finding_reason'}, 'study')
    require(isinstance(value['investigations'],list) and len(value['investigations']) <= 2, 'study budget exceeded')
    for r in value['investigations']:
        fields(r, {'market_ids','lookback_days','claim','falsifier','expected_direction'}, 'investigation')
        ids = strings(r['market_ids'],2)
        require(1 <= len(ids) <= 2 and set(ids) <= allowed and set(ids) & targets, 'study market scope differs')
        require(type(r['lookback_days']) is int and 1 <= r['lookback_days'] <= 30, 'study window out of bounds')
        explanation(r['claim'],'claim'); explanation(r['falsifier'],'falsifier')
        require(not study_quality_issues(r), '; '.join(study_quality_issues(r)))
        require(r['expected_direction'] in ('positive','negative','unspecified'),
                'expected_direction must be positive, negative or unspecified')
    if not value['investigations']: explanation(value['no_finding_reason'],'no finding reason')
    elif value['no_finding_reason'] is not None: explanation(value['no_finding_reason'],'no finding reason')
    return deepcopy(value)


def directional_test(spec, tool):
    """Deterministic descriptive check; never a significance or profitability test."""
    pair = len(spec['market_ids']) == 2
    if pair:
        measure = tool['pearson_change_correlation']
        count = tool['paired_daily_changes']
    else:
        series = tool['series'][spec['market_ids'][0]]
        count = len(series)
        measure = series[-1]['price']-series[0]['price'] if count >= 2 else None
    expected = spec['expected_direction']
    status = ('insufficient_data' if measure is None else 'descriptive_only' if expected == 'unspecified'
              else 'direction_supported_in_window' if (measure > 0 if expected == 'positive' else measure < 0)
              else 'direction_not_supported_in_window')
    return {'kind':'paired_daily_change_correlation' if pair else 'single_market_net_price_change',
            'observations':count,'measure':measure,'expected_direction':expected,'status':status,
            'independent_validation':False,'predictive_effectiveness_verified':False}


def validate_reviews(value, records):
    fields(value, {'reviews'}, 'review'); require(isinstance(value['reviews'],list), 'invalid reviews')
    indexed = {r['investigation_id']:r for r in records}; seen=set()
    for r in value['reviews']:
        fields(r, {'investigation_id','verdict','evidence_refs','reason','next_check'}, 'review record')
        key=r['investigation_id']; require(key in indexed and key not in seen, 'unknown/repeated investigation')
        seen.add(key); require(r['verdict'] in ('retain_for_testing','reject','insufficient_data'), 'invalid verdict')
        require(r['evidence_refs'] == [indexed[key]['tool_result']['result_sha256']], 'invented evidence reference')
        explanation(r['reason'],'review reason'); explanation(r['next_check'],'next check')
    require(seen == set(indexed), 'missing investigation review'); return deepcopy(value)


def validate_reflection(value, facts):
    fields(value, {'observations','no_lesson_reason'}, 'discovery reflection')
    require(isinstance(value['observations'],list) and len(value['observations']) <= 3, 'reflection budget exceeded')
    for r in value['observations']:
        fields(r, {'fact_id','lesson','next_check'}, 'reflection observation')
        require(r['fact_id'] in facts,'invented feedback fact'); explanation(r['lesson'],'lesson'); explanation(r['next_check'],'next check')
    if not value['observations']: explanation(value['no_lesson_reason'],'no lesson reason')
    elif value['no_lesson_reason'] is not None: explanation(value['no_lesson_reason'],'no lesson reason')
    return deepcopy(value)


def compact_coverage(coverage):
    """Lossless prompt table: retain every count/time while avoiding repeated field names."""
    if not coverage: return {}
    clocks={(v['as_of'],v['max_quote_age_seconds']) for v in coverage.values()}
    require(len(clocks)==1,'mixed coverage clocks')
    at,age=next(iter(clocks)); windows=[]; pairs=[]
    for mid,meta in sorted(coverage.items()):
        for days,window in sorted(meta['windows_days'].items(),key=lambda v:int(v[0])):
            windows.append([mid,int(days),window['daily_samples'],window['first_source_time'],window['last_source_time']])
            for target,overlap in sorted(window['overlap_with_targets'].items()):
                pairs.append([mid,int(days),target,overlap['paired_daily_samples'],overlap['paired_daily_changes']])
    return {'as_of':at,'max_quote_age_seconds':age,
        'window_columns':['market_id','lookback_days','daily_samples','first_source_time','last_source_time'],
        'windows':windows,'pair_columns':['market_id','lookback_days','target_market_id','paired_daily_samples','paired_daily_changes'],
        'pairs':pairs,'encoding':'lossless_coverage_tables_v1'}


def search_for_prompt(result):
    value=deepcopy(result)
    if 'coverage' in value: value['coverage']=compact_coverage(value['coverage'])
    # Hash still identifies the archived raw search result, not its prompt encoding.
    value['archived_result_sha256']=value.pop('result_sha256')
    return value


def compact_tool(result):
    value={k:deepcopy(v) for k,v in result.items() if k != 'series'}
    value['series_summary']={mid:{'samples':len(rows),'first':rows[0] if rows else None,
        'last':rows[-1] if rows else None} for mid,rows in result['series'].items()}
    value['raw_series_archived']=True; return value


class DiscoveryRunner(TeamRunner):
    def __init__(self, backend, catalogue):
        super().__init__(backend); self.catalogue=catalogue; self.discovery={}

    def structured(self, role, instruction, context, upstream, validator):
        request={'agent':role,'instruction':instruction,'input':deepcopy(context),'upstream':deepcopy(upstream),'adapter':None}
        for attempt in range(2):
            raw=None; started=time.perf_counter()
            try:
                raw=self.backend.generate(deepcopy(request)); value,_=decode_agent_response(raw,'single_json_fence')
                result=validator(value)
                self.calls.append({'request':deepcopy(request),'output':raw,'error':None,'seconds':time.perf_counter()-started})
                return result
            except (ValueError,TypeError,KeyError) as exc:
                self.calls.append({'request':deepcopy(request),'output':raw,'error':str(exc),'seconds':time.perf_counter()-started})
                if attempt: raise ValidationError(role+' failed: '+str(exc)) from exc
                request['repair']={'error':str(exc),'previous_response':raw}

    study_instruction = STUDY

    def validate_investigations(self, value, allowed, targets):
        return validate_study(value, allowed, targets)

    def investigate(self, spec, feed, cutoff, policy):
        query={'tool':'pair_changes' if len(spec['market_ids']) == 2 else 'history',
               'market_ids':spec['market_ids'],'lookback_days':spec['lookback_days']}
        tool=run_query(query,feed,cutoff,policy['max_quote_age_seconds'])
        return tool,directional_test(spec,tool)

    def decide(self, context, feed, policy, account):
        targets=validate_context(context); cutoff=context['observation_time']; searches=[]; pool={}
        self.discovery={'searches':searches,'investigations':[],'study':None,'review':None,'errors':[]}
        # Each search is selected after seeing the previous results; not a fixed peer shortlist.
        for _ in range(2):
            try:
                def query_ok(value):
                    payload=catalogue_query_payload(value); self.catalogue.search(payload,cutoff,targets); return payload
                q=self.structured('catalogue_search',SEARCH,context,
                    {'visible_contracts':len(self.catalogue.visible(cutoff)),'previous_searches':[search_for_prompt(s) for s in searches],'account':account},query_ok)
                result=self.catalogue.search(q,cutoff,targets); searches.append(result)
                pool.update({r['market_id']:r for r in result['records']})
            except (ValueError,TypeError,KeyError) as exc: self.discovery['errors'].append(str(exc))
        allowed=set(pool)|targets
        study=self.structured('investigation',self.study_instruction+' Allowed IDs: '+json.dumps(sorted(allowed)),context,
            {'retrieved_markets':list(pool.values()),'search_coverage':[compact_coverage(s.get('coverage',{})) for s in searches],'account':account},lambda v:self.validate_investigations(v,allowed,targets))
        self.discovery['study']=study; records=[]
        for i,spec in enumerate(study['investigations']):
            tool,test=self.investigate(spec,feed,cutoff,policy)
            record={'investigation_id':context['episode_id']+':study:'+str(i),'observation_time':cutoff,
                'registered_call_index':len(self.calls)-1,'registration_basis':'model_call_recorded_before_tool_execution',
                'specification':deepcopy(spec),'specification_sha256':canonical_hash(spec),
                'descriptive_test':test,
                'tool_result':tool,'status':'unvalidated_hypothesis','predictive_effectiveness_verified':False}
            records.append(record)
        self.discovery['investigations']=records
        reviewed=self.structured('research_critic',REVIEW,context,
            {'investigations':[{**r,'tool_result':compact_tool(r['tool_result'])} for r in records]},
            lambda v:validate_reviews(v,records))
        self.discovery['review']=reviewed
        selected={mid for r in records for mid in r['specification']['market_ids']} - targets
        retrieved=[pool[mid] for mid in sorted(selected)]
        research={'queries':[r['tool_result']['query'] for r in records],
                  'hypotheses':[r['specification']['claim'] for r in records]}
        upstream={'research':research,'tools':[r['tool_result'] for r in records],
            'retrieved_markets':retrieved,'execution_policy':policy,'research_review':reviewed,'account_state':account}
        forecast=self.call('forecast',context,upstream,targets)
        risk=self.call('risk',context,{**upstream,'forecast':forecast},targets)
        return {'research':research,'tools':upstream['tools'],'forecast':forecast,'risk':risk}


def explore_episode(context, labels, feed, policy, runner, account, *, recorded_latency=None, feedback_transform=None):
    runner.calls=[]; runner.discovery={}; started=time.perf_counter(); decision=None; error=None
    try: decision=runner.decide(context,feed,policy,account)
    except (ValueError,TypeError,KeyError) as exc: error=str(exc)
    elapsed=time.perf_counter()-started if recorded_latency is None else recorded_latency
    count=len(runner.calls); feedback=feedback_for(context,decision,labels,feed,policy,elapsed)
    if feedback_transform is not None: feedback=feedback_transform(decision,elapsed,feedback)
    reflection=None; reflection_error=None
    retrospective={'phase':'retrospective','observation_time':feedback['available_at'],
        'original_cutoff':context['observation_time'],'feedback_facts':feedback['facts'],
        'realized_outcomes':feedback['outcomes'], 'original_forecast':decision['forecast'] if decision else None,
        'decision_error':error, 'investigations':[{**r,'tool_result':compact_tool(r['tool_result'])}
            for r in runner.discovery.get('investigations',[])]}
    retrospective['execution_scope']=feedback.get('execution_scope','isolated_diagnostic_account')
    try:
        reflection=runner.structured('discovery_reflection',RETROSPECTIVE+' Allowed fact_id values are exactly '
                                     +json.dumps(sorted(feedback['facts']))+'. Market IDs are NOT fact IDs.',retrospective,{},
                                     lambda v:validate_reflection(v,feedback['facts']))
    except (ValueError,TypeError,KeyError) as exc: reflection_error=str(exc)
    return {'kind':'team_discovery_episode_v1','episode_id':context['episode_id'],'context':deepcopy(context),
        'account_at_observation':deepcopy(account),'decision':decision,'decision_error':error,'decision_seconds':elapsed,
        'decision_call_count':count,'discovery':deepcopy(runner.discovery),'feedback':feedback,
        'reflection':reflection,'reflection_error':reflection_error,'calls':deepcopy(runner.calls),
        'fine_tuning_triggered':False,'effectiveness_verified':False,'simulation_only':True}
