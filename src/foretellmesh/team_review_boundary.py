"""Machine-checkable execution contracts; fallible language review is advisory.

No topic classification, inference of missing sources, or semantic-truth claim.
"""
from copy import deepcopy

from .schema import timestamp
from .synthetic_sft import canonical_hash
from .team_learning import require

POLICY = {
    'kind': 'review_authority_boundary_v1',
    'model_review_authority': 'advisory_only',
    'probe_usage': 'diagnostic_only_not_execution_permission',
    'blocking_checks': ['admitted_task_ids', 'target_kind_and_horizon', 'variable_identity_and_lag',
        'exact_source_quantity_and_unit', 'historical_input_availability', 'bounded_calculation',
        'freeze_before_score', 'event_group_and_time_evaluation'],
    'arbitrary_prose_truth_verified': False,
    'predictive_relation_requires_empirical_test': True,
    'fine_tuning_admitted': False,
}
LEARNING = '''\nModel semantic reviews in this protocol are UNVERIFIED advisory opinions, not deterministic failures.
Do not treat different topics as proof that a predictive relationship cannot exist. Separate tool-verified routing,
source, unit and timing failures from reviewer opinions. A source missing means untestable with current inputs,
not a disproven relation. Do not copy an unsupported critic claim into a factual lesson or supervision answer.'''


def execution_contract(relation, catalogue, protocol):
    from .team_multi_input import validate_relation,used_markets
    from .team_executable_methods import validate_protocol
    validate_relation({'relation':relation,'reason':None},catalogue);validate_protocol(protocol)
    for mid in used_markets(relation):
        require(timestamp(catalogue[mid]['initialized_at'],'initialization') <= timestamp(protocol['as_of'],'as of'),
            'contract not initialized at research cutoff')
    price=relation['forecast_target']=='future_yes_price'
    result={'kind':'declared_execution_contract_v1','relation_sha256':canonical_hash(relation),
        'protocol_sha256':canonical_hash(protocol),
        'targets':[{'market_id':p['target'],'historical_rule_sha256':canonical_hash(catalogue[p['target']]['question']),
            'output_quantity':'future_contract_yes_trade_price' if price else 'event_resolution_probability',
            'output_unit':'USD_per_YES_share' if price else 'probability',
            'horizon_days':relation['horizon_days'],'input_ids':deepcopy(p['input_ids'])} for p in relation['predictions']],
        'variables':deepcopy(relation['variables']),
        'input_rule':'source_time <= observation_time - lag_days; after contract initialization; within freshness limit',
        'source_binding_pending':True,'source_values_checked':False,
        'narrative_alignment_verified':False,'empirical_relationship_verified':False}
    result['contract_sha256']=canonical_hash(result);return result


def advisory_result(relation_value, contract, review):
    require(review['relation_sha256']==canonical_hash(relation_value['relation']), 'review bound to different relation')
    require(contract['relation_sha256']==review['relation_sha256'],'execution contract bound to different relation')
    return {'relation':deepcopy(relation_value),'execution_contract':deepcopy(contract),
        'review_boundary':deepcopy(POLICY),
        'semantic_reviews':[{'relation':deepcopy(relation_value['relation']),'review':deepcopy(review)}],
        'semantic_revision_error':None,'status':'semantic_advice_recorded','stop_reason':None}


def observe(runner, relation_value, catalogue, protocol):
    from .team_semantic_consistency import review_relation
    # Deterministic validation must happen BEFORE the optional model opinion.
    contract=execution_contract(relation_value['relation'],catalogue,protocol)
    review=review_relation(runner,relation_value['relation'],catalogue)
    return advisory_result(relation_value,contract,review)


def audit_reference(config_path, source, source_report_sha256, output):
    """Post-hoc CPU boundary regression, not an alternative model trajectory."""
    from pathlib import Path
    from .data import strict_json,sha256_file
    from .evaluation import json_text
    from .team_research_loop import prepare
    from .team_semantic_consistency import review_relation
    from .team_lifecycle_experiment import RecordedBackend,AuditRunner
    from .team_multi_input import check_sources
    config=strict_json(config_path.read_text())
    require(config['handoff_protocol']=='executable_review_boundary_v1','wrong boundary protocol')
    require(not output.exists(),'reference audit output already exists')
    report=strict_json((source/'report.json').read_text())
    require(sha256_file(source/'report.json')==source_report_sha256,'reference report changed')
    require(report['status']=='completed' and report['development_opened'] is False and report['final_test_opened'] is False,
        'reference must be completed training research')
    for name in ('attempt_01.json','calls.json','catalogue.json'):
        require(sha256_file(source/name)==report['artifact_hashes'][name],'reference artifact changed')
    _,_,cat,_,_,protocol,context,_=prepare(config)
    attempt=strict_json((source/'attempt_01.json').read_text())
    history=attempt['workflow']['semantic_reviews'];require(len(history)==1,'expected one preserved review')
    relation=history[0]['relation'];archive=strict_json((source/'catalogue.json').read_text())
    from .team_multi_input import used_markets
    require(all(cat[k]==archive[k] for k in used_markets(relation)),'historical rules changed')
    calls=[c for c in strict_json((source/'calls.json').read_text())
           if c['request']['agent']=='semantic_consistency_reviewer' and c['request']['input']['hypothesis']==relation['hypothesis']]
    backend=RecordedBackend(calls);runner=AuditRunner(backend,None,{'min_review_seconds':3600,'max_review_seconds':2592000,
        'max_signal_valid_seconds':604800,'max_order_ttl_seconds':86400,'failure_review_seconds':86400})
    review=review_relation(runner,relation,cat)
    require(review==history[0]['review'] and backend.index==len(calls) and runner.calls==calls,'preserved reviews differ')
    contract=execution_contract(relation,cat,protocol)
    result=advisory_result({'relation':relation,'reason':None},contract,review)
    registry=context['variable_sources'];bindings=[]
    for var in relation['variables']:
        matches=[s for s in registry if s['quantity']==var['quantity'] and s['unit']==var['unit']]
        require(not matches,'this fixed regression expects unsupported archived variables')
        bindings.append({'variable_id':var['variable_id'],'source_id':None,'missing_reason':'No exact quantity/unit in registered sources.'})
    class NoPrices:
        def latest(self,*args):raise AssertionError('archived missing-source check must not read prices')
    check=check_sources({'bindings':bindings},relation,registry,cat,NoPrices(),protocol)
    value={'kind':'archived_review_boundary_regression_v1','source_report_sha256':source_report_sha256,
        'config_sha256':sha256_file(config_path),'old_status':attempt['workflow']['status'],
        'review':review,'advisory_result':result,'deterministic_missing_source_check':check,
        'model_reviews_replayed':len(calls),'new_model_calls':0,'new_predictions_scored':0,
        'source_mapping_is_deterministic_diagnostic_not_model_output':True,
        'original_run_reinterpreted':False,'alternative_model_trajectory_claimed':False,
        'implementation_hashes':{p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')}}
    output.mkdir(parents=True);(output/'assessment.json').write_text(json_text(value));return value


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--source-report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=audit_reference(args.config,args.source,args.source_report_sha256,args.output)
    print({k:value[k] for k in ('old_status','model_reviews_replayed','new_model_calls','new_predictions_scored')})
    print('Advisory stop:',value['advisory_result']['stop_reason'],
          'source check:',value['deterministic_missing_source_check']['status'])

if __name__=='__main__':main()
