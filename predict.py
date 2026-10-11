"""Predict compliance and stock level for every planogram area of every photo.

Trains the shelf model (see model.py) on the manual annotations, then writes the challenge's submission format:
{camera: {image file: {area_id: {planogram_compliant, stock_level, stock_value}}}}. Every area of every image is
included, since a missing area scores worst of all.

With --evaluate, it instead estimates the validation score: it repeatedly hides 1-2 labelled photos per camera,
trains on the rest and scores the hidden ones.

With --out-of-fold, labelled photos are predicted by models that never saw their labels (the photos are split into
folds, each predicted by a model trained on the others). Comparing those predictions with the annotations, for
example in the annotator's compare view, then shows the model's real mistakes rather than what it memorised.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from cut_areas import cut_area
from model import FeatureExtractor, ShelfModel
from planogram import area_quad, load_cameras
from score import POINTS, STOCK_VALUES, area_cases, points, score

VALIDATION_AREAS = 289


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


def area_crops(planogram, cameras_dir, photos):
    """Crops of every area in the given (camera, file) photos, keyed by (camera, file, area id); unreadable photos are skipped."""
    crops = {}
    for camera, file in photos:
        image = cv2.imread(str(cameras_dir / camera / file))
        if image is None:
            print(f"Skipping unreadable photo {cameras_dir / camera / file}")
            continue
        for area in planogram[camera]["products"]:
            crops[camera, file, area["area_id"]] = cut_area(image, np.array(area_quad(area), dtype=np.float32))
    return crops


def nested(keys, values):
    result = {}
    for (camera, file, area_id), value in zip(keys, values):
        result.setdefault(camera, {}).setdefault(file, {})[area_id] = value
    return result


def subset(features, rows):
    return {name: value[rows] for name, value in features.items()}


def out_of_fold(features, keys, labels, folds):
    """Answers for every labelled area from a model that didn't see its photo's labels.

    Each camera's photos are dealt round-robin into the folds, so every fold has every camera.
    """
    photos = {}
    for camera, file, _ in keys:
        photos.setdefault(camera, set()).add(file)
    fold_of = {}
    for camera, files in photos.items():
        shuffled = sorted(files)
        random.Random(camera).shuffle(shuffled)
        fold_of.update({(camera, file): i % folds for i, file in enumerate(shuffled)})
    fold = np.array([fold_of[key[:2]] for key in keys])

    answers = {}
    for number in range(folds):
        test = fold == number
        print(f"  fold {number + 1} of {folds}: predicting {test.sum()} areas")
        model = ShelfModel().fit(subset(features, ~test), [l for l, t in zip(labels, test) if not t])
        test_keys = [key for key, t in zip(keys, test) if t]
        answers.update(zip(test_keys, (decide(*p) for p in model.predict(subset(features, test)))))
    return answers


def evaluate(features, keys, labels, rounds):
    """Average held-out score over several random picks of 1-2 hidden photos per camera."""
    photos = {}
    for camera, file, _ in keys:
        photos.setdefault(camera, set()).add(file)

    scaled = []
    for round_number in range(rounds):
        rng = random.Random(round_number)
        hidden = {
            (camera, file)
            for camera, files in photos.items()
            for file in rng.sample(sorted(files), 2 if len(files) >= 3 else 1)
        }
        test = np.array([(camera, file) in hidden for camera, file, _ in keys])
        model = ShelfModel().fit(subset(features, ~test), [l for l, t in zip(labels, test) if not t])
        answers = [decide(*p) for p in model.predict(subset(features, test))]
        test_keys = [key for key, t in zip(keys, test) if t]
        truth = nested(test_keys, [l for l, t in zip(labels, test) if t])
        total = points(sum(score(nested(test_keys, answers), truth).values(), Counter()))
        scaled.append(VALIDATION_AREAS * total / len(test_keys))
        print(f"  round {round_number + 1}: {scaled[-1]:+.0f} / {2 * VALIDATION_AREAS}")

    mean = np.mean(scaled)
    print(f"\nEstimated validation score: {mean:+.0f} / {2 * VALIDATION_AREAS} ({mean / (2 * VALIDATION_AREAS):.0%}), "
          f"from {min(scaled):+.0f} to {max(scaled):+.0f} across rounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"), help="photos to predict")
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"), help="labels to train on")
    parser.add_argument("--train-cameras", type=Path, default=Path("input/cameras"), help="photos the labels refer to")
    parser.add_argument("--output", type=Path, default=Path("output/annotations.json"))
    parser.add_argument("--evaluate", type=int, nargs="?", const=20, metavar="ROUNDS",
                        help="estimate the validation score instead of predicting")
    parser.add_argument("--out-of-fold", type=int, nargs="?", const=10, metavar="FOLDS",
                        help="predict labelled photos with models that didn't see their labels")
    args = parser.parse_args()

    planogram = load_cameras(args.planogram)
    annotations = json.loads(args.annotations.read_text())
    extract = FeatureExtractor(planogram)

    train_crops = area_crops(planogram, args.train_cameras, [(c, f) for c, images in annotations.items() for f in images])
    train_keys = [
        (camera, file, area_id)
        for camera, images in annotations.items()
        for file, areas in images.items()
        for area_id in areas
        if (camera, file, area_id) in train_crops
    ]
    labels = [annotations[camera][file][area_id] for camera, file, area_id in train_keys]
    train_features = extract(train_keys, [train_crops[key] for key in train_keys])
    print(f"Training on {len(labels)} labelled areas from {len({k[:2] for k in train_keys})} photos")

    if args.evaluate:
        evaluate(train_features, train_keys, labels, args.evaluate)
        return

    model = ShelfModel().fit(train_features, labels)

    photos = [(camera, path.name) for camera in sorted(planogram) for path in sorted((args.cameras / camera).glob("*.jpg"))]
    crops = area_crops(planogram, args.cameras, photos)
    keys = list(crops)
    answers = [decide(*p) for p in model.predict(extract(keys, [crops[key] for key in keys]))]
    if args.out_of_fold:
        held_out = out_of_fold(train_features, train_keys, labels, args.out_of_fold)
        answers = [held_out.get(key, answer) for key, answer in zip(keys, answers)]
        print(f"{sum(key in held_out for key in keys)} labelled areas predicted out of fold")
    predictions = nested(keys, answers)

    # A photo that can't be read still gets an answer for every area: the cautious "compliant, empty".
    cautious = decide(1.0, {level: 1 / 3 for level in STOCK_VALUES})
    for camera, file in photos:
        for area in planogram[camera]["products"]:
            predictions.setdefault(camera, {}).setdefault(file, {}).setdefault(area["area_id"], cautious)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, indent=2))
    count = sum(len(areas) for images in predictions.values() for areas in images.values())
    print(f"Wrote {count} area predictions for {len(photos)} photos to {args.output}")


if __name__ == "__main__":
    main()
