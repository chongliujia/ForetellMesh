"""Selectable historical research tools and frozen, cited method-memory injection."""
from copy import deepcopy
from .schema import fields, timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require, strings
from .team_discovery import DiscoveryRunner, explanation, study_quality_issues, directional_test
from .team_learning import run_query
from .team_method_memory import validate_memory

STUDY = ('Choose up to 2 investigations yourself. Return exactly an object with investigations (list) and '
    'no_finding_reason (null, or a specific reason if empty). Each investigation has exactly: '
    'tool, market_ids, lookback_days, claim, falsifier, expected_direction. '
    'Available tools: rules_snapshot reads immutable resolution rules and fresh as-of prices for 1 or 2 contracts '
    '(lookback_days=0, expected_direction=unspecified); history reads daily samples for 1 contract; '
    'pair_changes compares daily changes for 2 contracts (both use lookback_days=1..30). '
    'expected_direction is positive, negative or unspecified; it means price change/correlation direction for the price tools. '
    'Use DISTINCT supplied market_ids, at least one target. Write a concrete claim and an observable falsifier '
    'for these contracts, each at most 600 characters. Do not copy placeholder text. '
    'Explore semantic dependencies, shared conditions, timing or prices as appropriate; no relationship is presumed. '
    'A price sum alone does not prove exclusivity, implication, arbitrage or causality. '
    'Rules alone cannot establish future outcomes. No external news tool is available; leave missing evidence missing. '
    'An empty investigations list is valid when no worthwhile check is possible.')


def validate_study(value, allowed, targets):
    fields(value,{'investigations','no_finding_reason'},'tool study')
    require(isinstance(value['investigations'],list) and len(value['investigations'])<=2,'study budget exceeded')
    for row in value['investigations']:
        fields(row,{'tool','market_ids','lookback_days','claim','falsifier','expected_direction'},'tool investigation')
        ids=strings(row['market_ids'],2)
        require(1<=len(ids)<=2 and set(ids)<=allowed and set(ids)&targets,'study market scope differs')
        require(row['tool'] in ('rules_snapshot','history','pair_changes'),'unknown study tool')
        require(type(row['lookback_days']) is int,'invalid lookback')
        require(row['expected_direction'] in ('positive','negative','unspecified'),'invalid direction')
        if row['tool']=='rules_snapshot':
            require(row['lookback_days']==0 and row['expected_direction']=='unspecified','rules do not measure price direction')
        else:
            require(len(ids)==(1 if row['tool']=='history' else 2) and 1<=row['lookback_days']<=30,'price query shape differs')
        explanation(row['claim'],'claim'); explanation(row['falsifier'],'falsifier')
        require(not study_quality_issues(row),'placeholder investigation')
    if not value['investigations']: explanation(value['no_finding_reason'],'no finding reason')
    elif value['no_finding_reason'] is not None: explanation(value['no_finding_reason'],'no finding reason')
    return deepcopy(value)


def rules_snapshot(query,catalogue,feed,cutoff,max_age_seconds):
    fields(query,{'tool','market_ids','lookback_days'},'rules query')
    require(query['tool']=='rules_snapshot' and type(query['lookback_days']) is int and query['lookback_days']==0,'invalid rules query')
    ids=strings(query['market_ids'],2); visible=catalogue.visible(cutoff)
    require(1<=len(ids)<=2 and set(ids)<=set(visible),'future/unknown rules contract')
    at=timestamp(cutoff,'cutoff'); series={}; records=[]
    for mid in ids:
        records.append(visible[mid]); quote=feed.latest(mid,at)
        require(quote is None or quote[1]<=at,'future rules quote')
        series[mid]=[] if quote is None or (at-quote[1]).total_seconds()>max_age_seconds else [
            {'at':cutoff,'source_time':quote[1].isoformat(),'price':float(quote[0])}]
    result={'query':deepcopy(query),'as_of':cutoff,'series':series,'rule_records':records,
        'relationship_verified':False,
        'interpretation':'Immutable historical rules and fresh trade prints only. Shared labels or price sums do not establish logical relations or profitable execution.'}
    result['result_sha256']=canonical_hash(result);return result


class MemoryRunner(DiscoveryRunner):
    study_instruction=STUDY
    def __init__(self,backend,catalogue,memory=None):
        super().__init__(backend,catalogue); self.memory=deepcopy(memory)

    def with_memory(self,context,upstream):
        result=deepcopy(upstream)
        if self.memory is not None:
            result['offline_team_memory']=validate_memory(self.memory,context,
                {r['event_group_id'] for r in self.catalogue.rows.values()})
        return result

    def structured(self,role,instruction,context,upstream,validator):
        if role in ('catalogue_search','investigation','research_critic'):
            upstream=self.with_memory(context,upstream)
            if self.memory is not None:
                instruction+=' Consult offline_team_memory as candidate research procedures, never as current evidence or verified trading rules.'
        return super().structured(role,instruction,context,upstream,validator)

    def call(self,role,context,upstream,mids,fact_ids=()):
        return super().call(role,context,self.with_memory(context,upstream),mids,fact_ids)

    def validate_investigations(self,value,allowed,targets):
        return validate_study(value,allowed,targets)

    def investigate(self,spec,feed,cutoff,policy):
        query={k:spec[k] for k in ('tool','market_ids','lookback_days')}
        if spec['tool']=='rules_snapshot':
            tool=rules_snapshot(query,self.catalogue,feed,cutoff,policy['max_quote_age_seconds'])
            test={'kind':'rule_evidence_snapshot','observations':len(tool['rule_records']),
                'status':'semantic_hypothesis_unverified','independent_validation':False,'predictive_effectiveness_verified':False}
        else:
            tool=run_query(query,feed,cutoff,policy['max_quote_age_seconds']);test=directional_test(spec,tool)
        return tool,test
