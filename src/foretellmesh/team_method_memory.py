"""Cited offline method memory; fitted hypotheses never become historical evidence."""
from copy import deepcopy
from .schema import fields, timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require, strings
from .team_discovery import explanation, compact_tool


def learning_evidence(jobs, episodes):
    require(len(jobs)==len(episodes) and all(j['partition']=='train' for j in jobs),'memory requires learning episodes')
    cases=[]; facts={}
    for i,(job,ep) in enumerate(zip(jobs,episodes)):
        require(ep['context']==job['context'],'memory source context differs')
        require(ep['feedback']['label_sha256']==canonical_hash(job['labels']),'memory feedback labels differ')
        eid='e'+str(i)
        for key,value in ep['feedback']['facts'].items(): facts[eid+'.feedback.'+key]=deepcopy(value)
        for k,r in enumerate(ep['discovery'].get('investigations',[])):
            tool=compact_tool(r['tool_result'])
            if 'rule_records' in tool:
                tool['archived_rule_record_hashes']={x['market_id']:x['record_sha256'] for x in tool.pop('rule_records')}
            facts[eid+'.study'+str(k)]={'specification':r['specification'],'tool_result':tool,
                                       'descriptive_test':r['descriptive_test']}
        facts[eid+'.decision_error']=ep['decision_error']
        cases.append({'case_id':eid,'source_episode_sha256':canonical_hash(ep),
            'observation_time':job['context']['observation_time'],
            'fact_refs':sorted(key for key in facts if key.startswith(eid+'.')),
            'reflection':ep['reflection'],'reflection_error':ep['reflection_error']})
    return {'phase':'offline_learning_only','cases':cases,'facts':facts,
            'interpretation':'Observations are checked tool/account feedback; all proposed lessons remain unvalidated.'}


def validate_evidence(jobs, episodes, evidence):
    # v1 runtime enumerated a dict into fact_refs before sorted-key JSON storage.
    # Accept ONLY a permutation of the same unique references; replay the exact
    # archived request order. All source hashes, values and other fields must match.
    expected=learning_evidence(jobs,episodes);normalized=deepcopy(evidence)
    require(isinstance(normalized,dict) and isinstance(normalized.get('cases'),list),'invalid learning evidence')
    for case in normalized['cases']:
        refs=case['fact_refs'];require(isinstance(refs,list) and all(isinstance(k,str) for k in refs)
            and len(refs)==len(set(refs)),'duplicate/invalid fact references')
        case['fact_refs']=sorted(refs)
    require(normalized==expected,'learning facts changed')
    return deepcopy(evidence)


EDITOR=('Create up to 3 reusable METHOD lessons from the supplied learning cases and checked facts. '
    'Return a JSON object with lessons (a list) and no_lesson_reason (null, or a nonempty reason when the list is empty). '
    'Each lesson must have exactly fact_refs (1..3 exact supplied fact IDs), when_applicable (string), '
    'action (specific instruction for the next research attempt), and failure_condition (how to detect that the lesson does not help). '
    'Each text is 1..350 characters. Learn which evidence/tools/decisions to change from these experiences; '
    'do not restate a particular resolved answer, prescribe a market ID to buy, or claim an observed profit proves a trading rule. '
    'Keep failures and missing information. Lessons are candidate procedures, not verified facts or probability targets. '
    'Do not copy generic placeholder text. An empty list with a concrete explanation is valid.')


def validate_lessons(value, facts):
    fields(value,{'lessons','no_lesson_reason'},'method editor')
    require(isinstance(value['lessons'],list) and len(value['lessons'])<=3,'unbounded lessons')
    for row in value['lessons']:
        fields(row,{'fact_refs','when_applicable','action','failure_condition'},'method lesson')
        refs=strings(row['fact_refs'],3)
        require(refs and set(refs)<=set(facts),'invented memory evidence reference')
        for key in ('when_applicable','action','failure_condition'):
            explanation(row[key],key);require(len(row[key])<=350,'method lesson too long')
            require(row[key].strip().casefold() not in {'specific instruction','tentative lesson','when applicable','failure condition'},'placeholder method lesson')
    if not value['lessons']: explanation(value['no_lesson_reason'],'no lesson reason')
    elif value['no_lesson_reason'] is not None: explanation(value['no_lesson_reason'],'no lesson reason')
    return deepcopy(value)


def freeze_memory(jobs, episodes, evidence, edited, built_at, catalogue):
    validate_evidence(jobs,episodes,evidence)
    edited=validate_lessons(edited,evidence['facts'])
    refs={k for lesson in edited['lessons'] for k in lesson['fact_refs']}
    memory={'kind':'offline_team_method_memory_v1','built_at':built_at,
        'source_public_through':max(l['resolution_time'] for j in jobs for l in j['labels'].values()),
        'source_feedback_available_at':max(e['feedback']['available_at'] for e in episodes),
        'source_event_group_ids':sorted({r['event_group_id'] for r in catalogue.rows.values()}),
        'source_episode_hashes':[canonical_hash(e) for e in episodes],
        'source_evidence_sha256':canonical_hash(evidence),
        'lessons':edited['lessons'],'no_lesson_reason':edited['no_lesson_reason'],
        'cited_facts':{k:evidence['facts'][k] for k in sorted(refs)},
        'status':'unvalidated_offline_fitted_method','effectiveness_verified':False,'fine_tuning_admitted':False}
    memory['memory_sha256']=canonical_hash(memory);return memory


def validate_memory(memory, context, catalogue_groups=()):
    fields(memory,{'kind','built_at','source_public_through','source_feedback_available_at','source_event_group_ids',
        'source_episode_hashes','source_evidence_sha256','lessons','no_lesson_reason','cited_facts','status',
        'effectiveness_verified','fine_tuning_admitted','memory_sha256'},'method memory')
    require(memory['kind']=='offline_team_method_memory_v1' and memory['status']=='unvalidated_offline_fitted_method'
        and memory['effectiveness_verified'] is False and memory['fine_tuning_admitted'] is False,'unqualified memory status')
    require(memory['memory_sha256']==canonical_hash({k:v for k,v in memory.items() if k!='memory_sha256'}),'memory hash changed')
    require(timestamp(memory['built_at'],'built')>=timestamp(memory['source_feedback_available_at'],'actual feedback capture'),
            'memory creation predates actual feedback')
    require(timestamp(memory['source_public_through'],'source settlement')<timestamp(context['observation_time'],'target cutoff'),
            'later source facts leak into target')
    groups={m['event_group_id'] for m in context['markets']}|set(catalogue_groups)
    require(not groups & set(memory['source_event_group_ids']),'source event overlaps target catalogue')
    validate_lessons({'lessons':memory['lessons'],'no_lesson_reason':memory['no_lesson_reason']},memory['cited_facts'])
    return deepcopy(memory)
