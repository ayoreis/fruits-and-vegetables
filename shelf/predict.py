"""Predict compliance and stock level for every planogram area of every photo.

Trains a model (see shelf/models.py for the choices) on the manual annotations and writes the challenge's
submission format: {camera: {image file: {area_id: {planogram_compliant, stock_level, stock_value}}}}. Every area of
every image is included, since a missing area scores worst of all.

The predictions go to output/predictions/<model>.json, where the staff and admin pages can pick them, and to
output/annotations.json, the file to submit.

    --evaluate          estimate the validation score instead (compare several models with --model a b c)
    --split cameras     with --evaluate, test on cameras the model never saw labelled instead of hidden photos
    --out-of-fold       predict labelled photos with models that never saw their labels, to review mistakes
                        honestly in the admin compare view (saved as output/predictions/<model>.out-of-fold.json)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shelf.crops import cut_area
from shelf.decisions import decide
from shelf.evaluation import evaluate, nested, out_of_fold
from shelf.features import FeatureExtractor
from shelf.models import DEFAULT_MODEL, MODELS
from shelf.planogram import area_quad, load_cameras
from shelf.score import STOCK_VALUES

PREDICTIONS = Path("output/predictions")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], choices=list(MODELS),
                        help=f"which model (default {DEFAULT_MODEL}); several to compare with --evaluate")
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"), help="photos to predict")
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"), help="labels to train on")
    parser.add_argument("--train-cameras", type=Path, default=Path("input/cameras"), help="photos the labels refer to")
    parser.add_argument("--output", type=Path, default=Path("output/annotations.json"), help="the file to submit")
    parser.add_argument("--evaluate", type=int, nargs="?", const=20, metavar="ROUNDS",
                        help="estimate the validation score instead of predicting")
    parser.add_argument("--split", choices=["photos", "cameras"], default="photos", help="what --evaluate hides")
    parser.add_argument("--out-of-fold", type=int, nargs="?", const=10, metavar="FOLDS",
                        help="predict labelled photos with models that didn't see their labels")
    parser.add_argument("--lower-bias", type=float, default=0.0,
                        help="lean stock answers further toward lower levels (0 = none)")
    parser.add_argument("--biases", type=float, nargs="+", default=[0.0],
                        help="lower biases to compare with --evaluate")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="processes for --evaluate and --out-of-fold (-1 = all cores)")
    parser.add_argument("--device", default="auto",
                        help="where to run the image models: auto (a GPU if available), cpu, cuda or mps")
    args = parser.parse_args()

    planogram = load_cameras(args.planogram)
    annotations = json.loads(args.annotations.read_text())
    extract = FeatureExtractor(planogram, device=args.device)
    needs = tuple(dict.fromkeys(name for model in args.model for name in MODELS[model]().needs))

    train_crops = area_crops(planogram, args.train_cameras, [(c, f) for c, images in annotations.items() for f in images])
    train_keys = [
        (camera, file, area_id)
        for camera, images in annotations.items()
        for file, areas in images.items()
        for area_id in areas
        if (camera, file, area_id) in train_crops
    ]
    labels = [annotations[camera][file][area_id] for camera, file, area_id in train_keys]
    train_features = extract(train_keys, [train_crops[key] for key in train_keys], needs)
    print(f"Training on {len(labels)} labelled areas from {len({k[:2] for k in train_keys})} photos")

    if args.evaluate:
        evaluate(args.model, train_features, train_keys, labels, args.evaluate, args.biases, args.split, args.jobs)
        return

    name = args.model[0]
    model = MODELS[name]().fit(train_features, labels)
    photos = [(camera, path.name) for camera in sorted(planogram) for path in sorted((args.cameras / camera).glob("*.jpg"))]
    crops = area_crops(planogram, args.cameras, photos)
    keys = list(crops)
    answers = [decide(*p, args.lower_bias) for p in model.predict(extract(keys, [crops[key] for key in keys], needs))]
    if args.out_of_fold:
        held_out = out_of_fold(name, train_features, train_keys, labels, args.out_of_fold, args.lower_bias, args.jobs)
        answers = [held_out.get(key, answer) for key, answer in zip(keys, answers)]
        print(f"{sum(key in held_out for key in keys)} labelled areas predicted out of fold")
    predictions = nested(keys, answers)

    # A photo that can't be read still gets an answer for every area: the cautious "compliant, empty".
    cautious = decide(1.0, {level: 1 / 3 for level in STOCK_VALUES})
    for camera, file in photos:
        for area in planogram[camera]["products"]:
            predictions.setdefault(camera, {}).setdefault(file, {}).setdefault(area["area_id"], cautious)

    saved = PREDICTIONS / f"{name}{'.out-of-fold' if args.out_of_fold else ''}.json"
    for path in (saved, args.output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(predictions, indent=2))
    count = sum(len(areas) for images in predictions.values() for areas in images.values())
    print(f"Wrote {count} area predictions for {len(photos)} photos to {saved} and {args.output}")


if __name__ == "__main__":
    main()
