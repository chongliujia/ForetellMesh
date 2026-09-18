"""Small deterministic tools. No shell, code execution, network, or label access."""
from fractions import Fraction

from .schema import ValidationError, fields, probability


def execute_probability_tool(call: dict) -> dict:
    fields(call, {"name", "arguments"}, "tool call")
    args = call["arguments"]
    if call["name"] == "bayes_binary":
        fields(args, {"prior", "sensitivity", "false_positive_rate"}, "Bayes arguments")
        prior, sensitivity, false_positive = [Fraction(str(probability(args[k], k)))
                                             for k in ("prior", "sensitivity", "false_positive_rate")]
        mass = prior * sensitivity + (1 - prior) * false_positive
        if mass == 0:
            raise ValidationError("conditioning event has zero probability")
        result = prior * sensitivity / mass
    elif call["name"] == "weighted_probability":
        fields(args, {"probabilities", "weights"}, "mixture arguments")
        ps, ws = args["probabilities"], args["weights"]
        if not isinstance(ps, list) or not isinstance(ws, list) or not 1 <= len(ps) == len(ws) <= 100:
            raise ValidationError("mixture arrays must have equal bounded lengths")
        ps = [Fraction(str(probability(p))) for p in ps]
        ws = [Fraction(str(probability(w, "weight"))) for w in ws]
        if sum(ws) != 1:
            raise ValidationError("mixture weights must sum to one")
        result = sum(p * w for p, w in zip(ps, ws))
    else:
        raise ValidationError("unknown probability tool")
    return {"name": call["name"], "arguments": args, "result": float(result),
            "exact_fraction": f"{result.numerator}/{result.denominator}"}
