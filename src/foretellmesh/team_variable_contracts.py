"""Typed research variables and exact, program-owned data-source bindings.

Checks the declared quantity/unit, not the truth of arbitrary natural language.
No unit conversion, proxy substitution, external retrieval or new training data.
"""
from copy import deepcopy
from pathlib import Path
import re

from .data import strict_json, sha256_file
from .schema import fields
from .synthetic_sft import canonical_hash
from .team_learning import require
from .team_executable_methods import bounded_text, validate_protocol
from .team_staged_methods import validate_relation as validate_base_relation, check_data


BASE_KEYS = {'hypothesis', 'forecast_target', 'horizon_days', 'bindings'}
SOURCE = {'source_id': 'pma_yes_trade_price', 'quantity': 'contract_yes_trade_price',
          'unit': 'USD_per_YES_share', 'reader': 'HistoricalBars.latest',
          'value_field': 'price', 'time_field': 'source_time', 'asof_supported': True,
          'measurement': 'historical Yes trade block mean; not an executable order-book quote'}

RELATION = '''You are the research member. Design one tentative relationship and explicitly name its required
variables BEFORE another member chooses data. You choose the relationship and contracts; no strategy category is prescribed.
Return one JSON object with exactly relation and reason. If abstaining, relation=null and reason is a short explanation.
Otherwise reason=null and relation has exactly hypothesis, forecast_target, horizon_days, bindings, variables.
hypothesis is a concrete tentative relation under 600 characters, not a profitability claim.
forecast_target is future_yes_price or resolves_yes. horizon_days is integer 1..14 for future_yes_price, null for resolves_yes.
bindings is a list of 1..4 objects with target and peer using supplied IDs. Targets are distinct, peers differ from targets.
Either all bindings have a peer or all use peer=null. Never pair adjacent catalogue entries without a reason.
variables is a list of 1..8 objects, each exactly variable_id, definition, quantity, unit, role, lag_days.
variable_id is a short unique identifier starting with a letter. definition is a precise description under 240 characters.
quantity is a short stable name of the actual measured quantity; unit is its actual unit. role is target or peer.
lag_days is integer 0..30. Include peer variables for a cross-contract relation, and no peer variable without a bound peer.
Name each quantity from the hypothesis itself, independently of which sources might be available.
Movie revenue, tweet counts, contract prices and event outcomes are different measurements with different units.
Declare external quantities accurately even when no source is currently available. Missing evidence is a valid result.
Do not relabel the quantity or unit merely to make a source fit. Market price may be a proxy only in a separately stated
price-based hypothesis, not a silent replacement for measured event data. List only variables the formula would actually use.
A data member will bind your exact variable IDs; it cannot rewrite your quantity, unit, role or lag.
No formula or test threshold is required in this step. Rules and quantities do not establish a predictive relationship.'''

DATA = '''You are the data member. Bind EVERY declared variable to an exact source from the supplied source_registry.
Return ONLY one JSON object with exactly bindings. bindings is a list with each variable exactly once.
Each object has exactly variable_id, source_id, missing_reason. Do not redefine variables or add quantities/units.
A source is usable ONLY when BOTH its quantity and unit exactly equal the variable's declared quantity and unit.
Use source_id=null and a short missing_reason when no exact source exists. Otherwise missing_reason=null.
Never convert units, rename a quantity, invent a source or replace an external observation with a price proxy.
Definitions remain research hypotheses; source binding is not proof of semantic correctness or prediction skill.
Use the supplied variable IDs exactly. Do not write a formula or retrieve external data.'''


def source_registry(base):
    root = Path(base['price_store']); report_path = root/'report.json'
    require(sha256_file(report_path) == base['price_store_report_sha256'], 'source registry report changed')
    report = strict_json(report_path.read_text())
    digest = report['artifact_hashes']['prices.sqlite']
    require(sha256_file(root/'prices.sqlite') == digest, 'source registry data changed')
    return [{**SOURCE, 'artifact_sha256': digest, 'source_report_sha256': base['price_store_report_sha256']}]


def validate_registry(registry):
    require(isinstance(registry, list) and len(registry) == 1, 'unsupported source registry')
    source = registry[0]
    fields(source, set(SOURCE) | {'artifact_sha256', 'source_report_sha256'}, 'source registry record')
    require(all(type(source[k]) is type(v) and source[k] == v for k, v in SOURCE.items()), 'source metadata changed')
    for key in ('artifact_sha256', 'source_report_sha256'):
        require(isinstance(source[key], str) and re.fullmatch(r'[0-9a-f]{64}', source[key]), 'invalid source digest')


def project_relation(relation):
    return {k: deepcopy(relation[k]) for k in sorted(BASE_KEYS)}


def research_context(context):
    # Keep source names/type tags out of initial variable declaration. The data
    # member receives the registry only AFTER that declaration is immutable.
    result = deepcopy(context)
    result.pop('variable_sources', None)
    result.pop('price_coverage', None)
    # Routing/grouping metadata belongs to the evaluator, not the tool-argument
    # namespace. Retain only the exact callable ID, rules and initialization.
    if isinstance(result.get('catalogue'), list):
        result['catalogue'] = [{k: row[k] for k in ('market_id', 'question', 'initialized_at') if k in row}
                               if isinstance(row, dict) else row for row in result['catalogue']]
    result['source_availability_not_provided'] = 'data_catalogue' not in result
    if 'data_catalogue' in result:
        result['source_values_not_provided'] = True
        result['joint_availability_not_established'] = True
    return result


def validate_relation(value, catalogue):
    fields(value, {'relation', 'reason'}, 'typed relation design')
    if value['relation'] is None: return validate_base_relation(value, catalogue)
    relation = value['relation']
    fields(relation, BASE_KEYS | {'variables'}, 'typed relation')
    if isinstance(relation['bindings'], list):
        for binding in relation['bindings']:
            if not isinstance(binding, dict): continue
            for role in ('target', 'peer'):
                key = binding.get(role)
                if key is not None:
                    require(isinstance(key, str) and key in catalogue,
                            role+' requires an exact market_id, not an event_group_id. Got '+str(key)+
                            '; allowed market_ids: '+', '.join(sorted(catalogue)))
    validate_base_relation({'relation': project_relation(relation), 'reason': value['reason']}, catalogue)
    variables = relation['variables']
    require(isinstance(variables, list) and 1 <= len(variables) <= 8, 'unbounded research variables')
    has_peer = relation['bindings'][0]['peer'] is not None
    ids = set(); measures = set()
    for v in variables:
        fields(v, {'variable_id', 'definition', 'quantity', 'unit', 'role', 'lag_days'}, 'research variable')
        key = v['variable_id']
        require(isinstance(key, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,39}', key)
                and key not in ids, 'invalid/duplicate variable ID'); ids.add(key)
        bounded_text(v['definition'], 240); bounded_text(v['quantity'], 80); bounded_text(v['unit'], 80)
        require(v['role'] in ('target', 'peer') and (has_peer or v['role'] != 'peer'), 'unbound variable role')
        require(type(v['lag_days']) is int and 0 <= v['lag_days'] <= 30, 'future/unbounded variable lag')
        measure = (v['quantity'], v['unit'], v['role'], v['lag_days'])
        require(measure not in measures, 'duplicate research measurement'); measures.add(measure)
        if v['quantity'] == SOURCE['quantity']:
            require(v['unit'] == SOURCE['unit'], 'incorrect unit for contract_yes_trade_price')
    require(not has_peer or any(v['role'] == 'peer' for v in variables), 'cross-contract relation lacks peer variable')
    return deepcopy(value)


def validate_mapping(value, relation, registry):
    validate_registry(registry); fields(value, {'bindings'}, 'variable-source mapping')
    value = deepcopy(value)
    variables = {v['variable_id']: v for v in relation['variables']}
    sources = {s['source_id']: s for s in registry}
    require(isinstance(value['bindings'], list) and len(value['bindings']) == len(variables), 'mapping population differs')
    seen = set()
    for binding in value['bindings']:
        fields(binding, {'variable_id', 'source_id', 'missing_reason'}, 'variable binding')
        key = binding['variable_id']
        require(isinstance(key, str) and key in variables and key not in seen, 'unknown/duplicate bound variable'); seen.add(key)
        source_id = binding['source_id']
        # Explicit transport compatibility: the literal string "null" can only
        # mean missing, never a usable source, and requires a missing reason.
        # structured() archives the original model output before validation.
        if source_id == 'null':
            bounded_text(binding['missing_reason'])
            binding['source_id'] = None; source_id = None
        if source_id is None:
            bounded_text(binding['missing_reason']); continue
        require(isinstance(source_id, str) and source_id in sources, 'unknown data source')
        require(binding['missing_reason'] is None, 'bound source has missing reason')
        v, source = variables[key], sources[source_id]
        require(v['quantity'] == source['quantity'] and v['unit'] == source['unit'],
                'source quantity/unit mismatch for '+key+': required '+v['quantity']+'/'+v['unit']+
                '; source measures '+source['quantity']+'/'+source['unit']+'. Use null source if unavailable.')
    return deepcopy(value)


def compiler_request(mapping, relation, registry):
    mapping = validate_mapping(mapping, relation, registry)
    bindings = {b['variable_id']: b for b in mapping['bindings']}; inputs = []; missing = []
    for variable in relation['variables']:
        bound = bindings[variable['variable_id']]
        if bound['source_id'] is None:
            missing.append(variable['variable_id']+': '+bound['missing_reason']); continue
        inputs.append({'role': variable['role'], 'quantity': 'yes_price', 'lag_days': variable['lag_days']})
    # This internal request is only sent to the old interpreter after all typed
    # bindings pass. Never reduce missing quantities to an available price.
    require(not missing, 'unbound variable cannot enter calculation')
    return {'inputs': inputs, 'missing_data': []}


def check_sources(mapping, relation, registry, catalogue, feed, protocol):
    validate_protocol(protocol)
    mapping = validate_mapping(mapping, relation, registry)
    variables = {v['variable_id']: v for v in relation['variables']}
    missing = [{'variable': deepcopy(variables[b['variable_id']]), 'reason': b['missing_reason']}
               for b in mapping['bindings'] if b['source_id'] is None]
    if missing:
        result = {'kind': 'typed_method_input_check_v1', 'status': 'unavailable_variables', 'jointly_available': 0,
                  'bindings': [], 'observations': [], 'unavailable_variables': missing,
                  'availability_not_attempted': True, 'interpretation': 'Declared variable has no bound source; no proxy substitution.'}
    else:
        result = check_data(compiler_request(mapping, relation, registry), project_relation(relation), catalogue, feed, protocol)
        result.pop('check_sha256')
        result.update(kind='typed_method_input_check_v1', unavailable_variables=[], availability_not_attempted=False)
    result.update(variable_contract_sha256=canonical_hash(relation), mapping_sha256=canonical_hash(mapping),
                  source_registry_sha256=canonical_hash(registry))
    result['check_sha256'] = canonical_hash(result); return result
