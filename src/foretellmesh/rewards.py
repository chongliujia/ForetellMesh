"""Label-side proper scoring. Never serialize this judge into an agent request."""
from types import SimpleNamespace

from .agent_runtime import validated_input
from .schema import ForecastInput, ValidationError, binary_outcome, probability
from .sft_data import validate_response

REWARD_VERSION = "binary_brier_v1"


def brier_reward(p: float, outcome: int) -> float:
    return -(probability(p) - binary_outcome(outcome)) ** 2


def score_forecast_response(response: dict, context: ForecastInput, outcome: int) -> dict:
    """Invalid output gets the score floor; valid probability uses only -Brier."""
    context = validated_input(context)
    binary_outcome(outcome)
    try:
        prediction = validate_response(response, SimpleNamespace(forecast_input=context))
    except ValidationError as exc:
        return {"version": REWARD_VERSION, "valid": False, "reward": -1.0, "error": str(exc)}
    return {"version": REWARD_VERSION, "valid": True,
            "reward": brier_reward(prediction["probability"], outcome), "error": None}


def critic_score_delta(original: float, revised: float, outcome: int) -> float:
    """Offline marginal score diagnostic, not evidence that a critic is causal."""
    return brier_reward(revised, outcome) - brier_reward(original, outcome)
