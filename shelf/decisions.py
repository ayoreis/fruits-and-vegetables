"""Turning model probabilities into answers that score the most points under the challenge's rules."""

import numpy as np

from shelf.score import POINTS, STOCK_VALUES, area_cases


def expected_points(prediction, compliant_probability, stock_probabilities, empty_is_compliant=True):
    """Expected points of a prediction, given how likely each true answer is.

    The organisers count an empty area as compliant, so only the other stock levels can be the wrong product.
    """
    empty = stock_probabilities.get("empty", 0) if empty_is_compliant else 0
    # The compliance chance covers all areas; take out the empty ones to get the chance for the rest.
    otherwise = compliant_probability if empty >= 1 else min(1, max(0, (compliant_probability - empty) / (1 - empty)))
    total = 0
    for actual_level, stock_chance in stock_probabilities.items():
        compliant_chance = 1.0 if empty_is_compliant and actual_level == "empty" else otherwise
        for actual_compliant, chance in ((True, compliant_chance), (False, 1 - compliant_chance)):
            annotation = {"planogram_compliant": actual_compliant, "stock_value": STOCK_VALUES[actual_level]}
            total += chance * stock_chance * sum(POINTS[case] for case in area_cases(prediction, annotation))
    return total


def adjusted(stock_probabilities, lower_bias=0.0, half_bias=0.0):
    """Reweight the stock probabilities before deciding.

    lower_bias shifts toward lower levels: each level down is weighted e^lower_bias more (negative shifts up).
    half_bias widens "half" from both sides: half is weighted e^half_bias more (negative narrows it).
    shelf/pages/decisions.js does the same for the pages; keep the two in step.
    """
    levels = list(STOCK_VALUES)
    weights = {
        level: stock_probabilities[level] * np.exp(lower_bias * (len(levels) - 1 - i) + (half_bias if level == "half" else 0))
        for i, level in enumerate(levels)
    }
    total = sum(weights.values())
    return {level: weight / total for level, weight in weights.items()}


def lowered(stock_probabilities, lower_bias):
    """Shift probability toward lower stock levels (see adjusted)."""
    return adjusted(stock_probabilities, lower_bias)


def decide(compliant_probability, stock_probabilities, lower_bias=0.0, half_bias=0.0, empty_is_compliant=True):
    """Pick the answer with the most expected points; the uneven penalties favour lower stock levels.

    lower_bias and half_bias reweight the stock probabilities first (see adjusted): for when the labels may run
    higher than the organisers' ("when in doubt, under-estimate"), or call half more or less often than they do.
    An empty answer is never paired with "wrong product".
    """
    if lower_bias or half_bias:
        stock_probabilities = adjusted(stock_probabilities, lower_bias, half_bias)
    candidates = [
        {"planogram_compliant": compliant, "stock_level": level, "stock_value": value}
        for compliant in (True, False)
        for level, value in STOCK_VALUES.items()
        if compliant or level != "empty" or not empty_is_compliant
    ]
    return max(
        candidates,
        key=lambda prediction: expected_points(prediction, compliant_probability, stock_probabilities, empty_is_compliant),
    )
