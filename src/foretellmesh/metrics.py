"""Proper scores and fixed-bin event-probability calibration."""

import math
from collections.abc import Sequence

from .schema import ValidationError, binary_outcome, probability


def score_predictions(
    predictions: Sequence[float | None], outcomes: Sequence[int], *,
    ece_bins: int = 10, log_loss_epsilon: float = 1e-15,
) -> dict:
    if len(predictions) != len(outcomes):
        raise ValidationError("predictions and outcomes must have equal length")
    if type(ece_bins) is not int or not 1 <= ece_bins <= 1000:
        raise ValidationError("ece_bins: expected integer in [1, 1000]")
    epsilon = probability(log_loss_epsilon, "log_loss_epsilon")
    if not 0 < epsilon < 0.5:
        raise ValidationError("log_loss_epsilon: require 0 < epsilon < 0.5")
    pairs = []
    for prediction, outcome in zip(predictions, outcomes):
        binary_outcome(outcome)
        if prediction is not None:
            pairs.append((probability(prediction), outcome))
    count, eligible = len(pairs), len(outcomes)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(ece_bins)]
    for p, y in pairs:
        bins[min(int(p * ece_bins), ece_bins - 1)].append((p, y))
    curve = []
    weighted_error = []
    for index, entries in enumerate(bins):
        mean_p = math.fsum(p for p, _ in entries) / len(entries) if entries else None
        frequency = math.fsum(y for _, y in entries) / len(entries) if entries else None
        curve.append({"lower": index / ece_bins, "upper": (index + 1) / ece_bins,
                      "count": len(entries), "mean_probability": mean_p,
                      "event_frequency": frequency})
        if entries:
            weighted_error.append(len(entries) * abs(mean_p - frequency))
    losses = []
    for p, y in pairs:
        # Clip the probability assigned to the observed outcome directly. This
        # remains finite even when 1 - epsilon rounds to 1 in floating point.
        correct_probability = p if y else 1 - p
        losses.append(-math.log(max(epsilon, correct_probability)))
    return {
        "eligible_count": eligible,
        "prediction_count": count,
        "missing_count": eligible - count,
        "coverage": count / eligible if eligible else None,
        "brier": math.fsum((p - y) ** 2 for p, y in pairs) / count if count else None,
        "log_loss": math.fsum(losses) / count if count else None,
        "ece": math.fsum(weighted_error) / count if count else None,
        "calibration_curve": curve,
    }
