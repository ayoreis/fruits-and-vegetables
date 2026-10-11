"""Score predictions against annotations with the challenge's points rules.

Each annotated area earns points for identification (planogram_compliant) and for stock level (stock_value).
Areas missing from the predictions count as a wrong identification and a stock level above the real stock.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

POINTS = {
    "identification_correct": 1,
    "identification_wrong": -2,
    "stock_correct": 1,
    "stock_below": -2,
    "stock_above": -3,
}
LABELS = {
    "identification_correct": "Identification correct",
    "identification_wrong": "Identification wrong",
    "stock_correct": "Stock level correct",
    "stock_below": "Stock level below the real stock",
    "stock_above": "Stock level above the real stock",
}
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}


def identification_case(predicted, actual):
    return "identification_correct" if predicted == actual else "identification_wrong"


def stock_case(predicted, actual):
    if predicted == actual:
        return "stock_correct"
    return "stock_below" if predicted < actual else "stock_above"


def area_cases(prediction, annotation):
    """The two scoring cases for one area; a missing prediction or field scores as the worst case."""
    prediction = prediction or {}
    compliant = prediction.get("planogram_compliant")
    stock = prediction.get("stock_value")
    return (
        identification_case(compliant, annotation["planogram_compliant"]) if compliant is not None else "identification_wrong",
        stock_case(stock, annotation["stock_value"]) if stock is not None else "stock_above",
    )


def score(predictions, annotations):
    """Count the scoring cases per camera, plus how many annotated areas had no prediction."""
    cases = defaultdict(Counter)
    for camera, images in annotations.items():
        for file, areas in images.items():
            for area_id, annotation in areas.items():
                prediction = predictions.get(camera, {}).get(file, {}).get(area_id)
                cases[camera].update(area_cases(prediction, annotation))
                if prediction is None:
                    cases[camera]["missing"] += 1
    return cases


def points(counts):
    return sum(POINTS[case] * counts[case] for case in POINTS)


def summary(counts):
    """Points as a share of the maximum, and how many answers were right, overall and per part."""
    areas = counts["identification_correct"] + counts["identification_wrong"]
    if not areas:
        return "no annotated areas"
    total = points(counts)
    identification = counts["identification_correct"] / areas
    stock = counts["stock_correct"] / areas
    return (
        f"{total:+} / {2 * areas} points ({total / (2 * areas):.1%} of maximum) · "
        f"{(identification + stock) / 2:.1%} correct "
        f"(identification {identification:.1%}, stock level {stock:.1%})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="?", default=Path("output/annotations.json"))
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--by-camera", action="store_true", help="also print the total for each camera")
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    annotations = json.loads(args.annotations.read_text())
    cases = score(predictions, annotations)
    total = sum(cases.values(), Counter())
    areas = total["identification_correct"] + total["identification_wrong"]

    print(f"{'Case':<34}{'Areas':>7}{'Each':>7}{'Points':>8}")
    for case, label in LABELS.items():
        print(f"{label:<34}{total[case]:>7}{POINTS[case]:>+7}{POINTS[case] * total[case]:>+8}")
    print(f"{'Total':<48}{points(total):>+8}")
    print(f"\nOver {areas} areas: {summary(total)}")
    if total["missing"]:
        print(f"{total['missing']} annotated areas were missing from the predictions")

    if args.by_camera:
        print()
        for camera in sorted(cases):
            print(f"Camera {camera}: {summary(cases[camera])}")


if __name__ == "__main__":
    main()
