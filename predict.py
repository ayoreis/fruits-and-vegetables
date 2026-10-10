"""Predict compliance and stock level for every planogram area of every photo.

Writes the challenge's submission format: {camera: {image file: {area_id: {planogram_compliant, stock_level, stock_value}}}}.
Every area of every image is included, since a missing area scores worst of all.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cut_areas import cut_area
from planogram import area_quad, load_cameras
from score import POINTS, STOCK_VALUES, area_cases


def predict_area(crop, area):
    """Probabilities for one area: (chance it is compliant, {stock level: chance}).

    Baseline: ignores the image and returns the same guess for every area. Replace this with a real model.
    """
    return 0.9, {"empty": 1 / 3, "half": 1 / 3, "full": 1 / 3}


def expected_points(prediction, compliant_probability, stock_probabilities):
    """Expected points of a prediction, given how likely each true answer is."""
    total = 0
    for actual_compliant, compliant_chance in ((True, compliant_probability), (False, 1 - compliant_probability)):
        for actual_level, stock_chance in stock_probabilities.items():
            annotation = {"planogram_compliant": actual_compliant, "stock_value": STOCK_VALUES[actual_level]}
            total += compliant_chance * stock_chance * sum(POINTS[case] for case in area_cases(prediction, annotation))
    return total


def decide(compliant_probability, stock_probabilities):
    """Pick the answer with the most expected points; the uneven penalties favour lower stock levels."""
    candidates = [
        {"planogram_compliant": compliant, "stock_level": level, "stock_value": value}
        for compliant in (True, False)
        for level, value in STOCK_VALUES.items()
    ]
    return max(candidates, key=lambda prediction: expected_points(prediction, compliant_probability, stock_probabilities))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--output", type=Path, default=Path("output/annotations.json"))
    args = parser.parse_args()

    predictions = {}
    for camera, camera_data in sorted(load_cameras(args.planogram).items()):
        for image_path in sorted((args.cameras / camera).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            areas = predictions.setdefault(camera, {}).setdefault(image_path.name, {})
            for area in camera_data["products"]:
                crop = cut_area(image, np.array(area_quad(area), dtype=np.float32)) if image is not None else None
                areas[area["area_id"]] = decide(*predict_area(crop, area))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2))
    count = sum(len(areas) for images in predictions.values() for areas in images.values())
    print(f"Wrote {count} area predictions for {sum(len(images) for images in predictions.values())} images to {args.output}")


if __name__ == "__main__":
    main()
