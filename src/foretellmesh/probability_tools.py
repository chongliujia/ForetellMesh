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
    elif call["name"] == "binary_mixture":
        fields(args, {"high_rate", "low_rate", "weight"}, "binary mixture arguments")
        high, low, weight = [Fraction(str(probability(args[k], k)))
                             for k in ("high_rate", "low_rate", "weight")]
        result = weight * high + (1 - weight) * low
    elif call["name"] == "complement_probability":
        fields(args, {"probability"}, "complement arguments")
        result = 1 - Fraction(str(probability(args["probability"])))
    elif call["name"] == "empirical_frequency":
        fields(args, {"successes", "trials"}, "frequency arguments")
        s, n = args["successes"], args["trials"]
        if type(s) is not int or type(n) is not int or not 0 <= s <= n <= 10**12:
            raise ValidationError("counts must be integers with 0 <= successes <= trials <= 1e12")
        if n == 0:
            raise ValidationError("empirical frequency is undefined for zero trials")
        result = Fraction(s, n)
    else:
        raise ValidationError("unknown probability tool")
    return {"name": call["name"], "arguments": args, "result": float(result),
            "exact_fraction": f"{result.numerator}/{result.denominator}"}
