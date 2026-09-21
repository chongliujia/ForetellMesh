"""Training-only, input-metadata discovery before a team designs its variables.

The source menu is program-owned. Queries reveal no prices or outcome labels;
coverage at the research cutoff is NOT a historical forecasting feature.
"""
from copy import deepcopy

from .schema import fields,timestamp,iso
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text
from .team_variable_contracts import validate_registry

QUERY = '''You are the research team's data-catalogue member, before task negotiation and hypothesis design.
Inspect the registered source menu and admitted training market index. Return ONLY JSON with exactly source_ids,
market_ids, as_of, reason. Choose 1..4 distinct STRING market IDs and 1..4 distinct source IDs from the supplied menus.
as_of must be an ISO timestamp no later than research_cutoff. reason is under 400 characters: what metadata you need.
This tool reports exact measurement names, units, timestamp semantics and past input coverage. It returns NO prices,
outcomes, inferred relationships or strategies. Select the contracts yourself; supplied prior failure is experience.
Do not invent a source for a desired variable. Unregistered evidence remains unavailable, not disproven.'''
GUIDANCE = '''\nYou now have an explicit data_catalogue result before choosing the task or variables. Use it to decide what
can actually be measured, without a prescribed strategy. You may propose a NEW hypothesis about the registered
measurements, request clearly defined missing external observations, or abstain. A new price relationship is a new
hypothesis: never rename revenue, sports outcomes or other external quantities as prices while keeping the old claim.
For a registered measurement, copy its exact quantity and unit if that is what you genuinely propose to measure.
Inventory counts and date ranges are retrospective TRAINING design metadata, not features at earlier observation times.
Listed source support does not establish joint coverage, future target availability, predictive edge or trade liquidity.
A request for external observations needs actual measurement identity/unit and source publication/as-of time; contract
initialization is NOT an observation's publication time. Model critic opinions are unverified. Different topics are allowed.'''


def menu(registry):
    validate_registry(registry)
    # Metadata only; do not expose paths, arbitrary new source schemas or labels.
    return [{k:deepcopy(s[k]) for k in ('source_id','quantity','unit','measurement','time_field','asof_supported',
        'artifact_sha256','source_report_sha256')} for s in registry]


def validate_query(value,catalogue,registry,cutoff):
    fields(value,{'source_ids','market_ids','as_of','reason'},'catalogue query');bounded_text(value['reason'],400)
    for key,allowed in [('market_ids',set(catalogue)),('source_ids',{s['source_id'] for s in registry})]:
        ids=value[key]
        require(isinstance(ids,list) and 1<=len(ids)<=4 and all(isinstance(k,str) for k in ids)
            and len(set(ids))==len(ids) and set(ids)<=allowed,'query requires 1..4 distinct admitted '+key)
    require(timestamp(value['as_of'],'query cutoff')<=timestamp(cutoff,'research cutoff'),'query cutoff exceeds research cutoff')
    return deepcopy(value)


def query_catalogue(query,catalogue,registry,feed,cutoff):
    sources=menu(registry);query=validate_query(query,catalogue,registry,cutoff)
    at=timestamp(query['as_of'],'query cutoff');rows=[]
    for mid in query['market_ids']:
        initialized=timestamp(catalogue[mid]['initialized_at'],'init')
        require(initialized<=at,'queried contract not initialized at query cutoff')
        # Project timestamps/counts only. No price statistics, label object or resolution time.
        times=[(t,n) for t,_,n in feed.data.get(mid,[]) if initialized<=t<=at]
        rows.append({'market_id':mid,'initialized_at':iso(initialized),'historical_blocks':len(times),
            'historical_trade_rows':sum(n for _,n in times),
            'first_source_time':iso(min(t for t,_ in times)) if times else None,
            'last_source_time':iso(max(t for t,_ in times)) if times else None})
    value={'kind':'training_input_catalogue_result_v1','query':query,'research_cutoff':cutoff,
        'sources':[s for s in sources if s['source_id'] in query['source_ids']], 'markets':rows,
        'time_semantics':{'source_time':'Historical trade-block timestamp; not external evidence publication time.',
            'feature_cutoff':'At each forecast T, input source_time <= T - declared lag; freshness and initialization enforced separately.',
            'initialization':'Contract initialization does not timestamp a variable observation.'},
        'limitations':['Offline training design metadata, never backfilled into earlier prediction features.',
            'Marginal coverage is not joint coverage, future target coverage or execution liquidity.',
            'Registered Yes trade prices are not revenues, sports results or true event probabilities.',
            'Missing external sources remain missing; no source conversion or proxy substitution.'],
        'prices_exposed':False,'outcome_labels_exposed':False,'strategy_recommended':False}
    value['result_sha256']=canonical_hash(value);return value


def discover(runner,catalogue,registry,feed,protocol,index,history,task):
    record={'query':None,'result':None,'error':None,'status':'data_catalogue_failed'}
    try:
        query=runner.structured('research_data_catalogue',QUERY,
            {'phase':'offline_training_data_discovery','source_menu':menu(registry),'market_index':index,
             'research_cutoff':protocol['as_of'],'pending_task':deepcopy(task)},
            {'training_history':deepcopy(history)},lambda v:validate_query(v,catalogue,registry,protocol['as_of']))
        record['query']=query
        record['result']=query_catalogue(query,catalogue,registry,feed,protocol['as_of'])
        record['status']='data_catalogue_read'
    except (ValueError,TypeError,KeyError) as exc:record['error']=str(exc)
    record['discovery_sha256']=canonical_hash(record);return record
