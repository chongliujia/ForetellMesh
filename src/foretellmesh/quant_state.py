"""Deterministic Quant stage over an explicit, time-bound numerical contract.

No text extraction, retrieval, judge access, default probabilities or generated
unknowns answer. The caller must supply one unambiguous structured model record.
"""
from fractions import Fraction

from .data import strict_json
from .probability_tools import execute_probability_tool
from .schema import ValidationError, fields, iso, probability
from .synthetic_sft import canonical_hash

SOURCE = 'structured://probability-model/v1'
VERSION = 'probability_state_v1'
INPUTS = {
    'mixture': ('high_rate', 'low_rate', 'weight'),
    'bayes': ('prior', 'sensitivity', 'false_positive_rate'),
    'complement': ('hit_probability', 'miss_probability'),
    'sampling': ('successes', 'trials', 'population_probability'),
}
ASSUMPTIONS = {
    'mixture': 'Exactly one of two exhaustive scenarios is selected with the stated weight.',
    'bayes': 'A positive binary signal is observed under the stated likelihoods.',
    'complement': 'Hit and miss are mutually exclusive and exhaustive.',
    'sampling': 'Finite IID Bernoulli observations; sample frequency does not identify population_probability.',
}


def validate_spec(spec: dict) -> dict:
    fields(spec, {'schema_version', 'model', 'values'}, 'numerical model specification')
    model, values = spec['model'], spec['values']
    if spec['schema_version'] != '1' or not isinstance(model, str) or model not in INPUTS:
        raise ValidationError('unsupported numerical model')
    if not isinstance(values, dict) or set(values) - set(INPUTS[model]):
        raise ValidationError('unknown numerical input fields')
    for key, value in values.items():
        if key in ('successes', 'trials'):
            if type(value) is not int or not 0 <= value <= 10**12:raise ValidationError('invalid observation count')
        else:probability(value, key)
    if model == 'sampling' and {'successes', 'trials'} <= values.keys() and values['successes'] > values['trials']:
        raise ValidationError('success count exceeds trial count')
    if model == 'complement' and set(values) == set(INPUTS[model]):
        if sum(Fraction(str(values[k])) for k in INPUTS[model]) != 1:
            raise ValidationError('inconsistent complementary probabilities')
    return spec


def compute_state(spec: dict) -> dict:
    validate_spec(spec)
    model, values = spec['model'], spec['values']
    calculations = []
    def calculate(quantity, required, call):
        missing = sorted(set(required) - values.keys())
        row = {'quantity': quantity, 'required_inputs': list(required), 'missing_inputs': missing,
               'status': 'missing_inputs', 'value': None, 'exact_fraction': None, 'error': None}
        if not missing:
            try:
                answer = execute_probability_tool(call())
                row.update(status='computed', value=answer['result'], exact_fraction=answer['exact_fraction'])
            except ValidationError as exc:
                # Undefined conditioning events and empty samples are not missing data.
                row.update(status='undefined', error=str(exc))
        calculations.append(row)
    if model == 'mixture':
        calculate('success_probability', INPUTS[model], lambda: {'name': 'binary_mixture',
            'arguments': dict(values)})
    elif model == 'bayes':
        calculate('posterior_probability', INPUTS[model], lambda: {'name': 'bayes_binary', 'arguments': dict(values)})
    elif model == 'sampling':
        calculate('empirical_frequency', ('successes', 'trials'), lambda: {'name': 'empirical_frequency',
            'arguments': {k: values[k] for k in ('successes', 'trials')}})
    else:
        for given, derived in (('hit_probability', 'miss_probability'), ('miss_probability', 'hit_probability')):
            if derived not in values:
                calculate(derived, (given,), lambda key=given: {'name': 'complement_probability',
                    'arguments': {'probability': values[key]}})
    return {'tool': VERSION, 'model': model, 'assumptions': ASSUMPTIONS[model],
            'inputs': dict(values), 'calculations': calculations}


def build_quant_state(context) -> dict:
    # Local import keeps the opt-in stage independent of AgentRunner construction.
    from .agent_runtime import validated_input
    context = validated_input(context)
    sources = [e for e in context.evidence if e.source == SOURCE]
    if len(sources) != 1:
        raise ValidationError('Quant requires exactly one structured numerical model source')
    evidence = sources[0]
    result = compute_state(strict_json(evidence.text))
    return {**result, 'observation_time': iso(context.observation_time),
            'source_evidence_ids': [evidence.evidence_id], 'source_published_at': iso(evidence.published_at),
            'source_available_at': iso(evidence.available_at), 'source_text_sha256': canonical_hash(evidence.text)}
