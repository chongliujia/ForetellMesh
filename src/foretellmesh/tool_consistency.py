"""Opt-in checks against explicit input scope and deterministic Quant facts.

This is not a general uncertainty judge. In particular, a calculator's missing
dependency does not prove its output unidentifiable in every degenerate case.
No outcomes, teacher answers, or full expected unknown sets enter this guard.
"""
from .data import strict_json
from .quant_state import INPUTS
from .schema import ValidationError, fields

VERSION = 'requested_inputs_v1'
SOURCE = 'structured://information-scope/v1'
DERIVED = {'mixture': {'success_probability'}, 'bayes': {'posterior_probability'},
           'complement': set(), 'sampling': {'empirical_frequency'}}


def requested_fields(context, quant):
    sources = [e for e in context.evidence if e.source == SOURCE]
    if len(sources) != 1:
        raise ValidationError('consistency guard requires exactly one information scope')
    spec = fields(strict_json(sources[0].text), {'schema_version', 'requested_fields'}, 'information scope')
    names = spec['requested_fields']
    allowed = set(INPUTS[quant['model']]) | DERIVED[quant['model']] | {'future_outcome'}
    if (spec['schema_version'] != '1' or not isinstance(names, list)
            or any(not isinstance(n, str) for n in names) or len(names) != len(set(names))
            or set(names) - allowed):
        raise ValidationError('invalid information scope')
    return set(names)


def validate_consistency(output, context, quant):
    requested = requested_fields(context, quant)
    unknown = set(output['unknowns'])
    known = set(quant['inputs'])
    missing = set()
    for calculation in quant['calculations']:
        if calculation['status'] == 'computed':
            known.add(calculation['quantity'])
        missing.update(calculation['missing_inputs'])
    # Complementary parameters can be derived despite being absent as inputs.
    missing -= known
    omitted = (missing & requested) - unknown
    wrongly_missing = known & unknown
    outside = unknown - requested
    errors = []
    if omitted:errors.append('requested input parameters absent from supplied model must be acknowledged: ' + ', '.join(sorted(omitted)))
    if wrongly_missing:errors.append('supplied or computed values cannot be listed as unknown: ' + ', '.join(sorted(wrongly_missing)))
    if outside:errors.append('unknowns outside requested vocabulary: ' + ', '.join(sorted(outside)))
    if errors:raise ValidationError('tool consistency: ' + '; '.join(errors))
