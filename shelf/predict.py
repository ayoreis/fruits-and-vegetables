"""Predict compliance and stock level for every planogram area of every photo.

Trains a model (see shelf/models.py for the choices) on the manual annotations and writes the challenge's
submission format: {camera: {image file: {area_id: {planogram_compliant, stock_level, stock_value}}}}. Every area of
every image is included, since a missing area scores worst of all.

The predictions go to output/predictions/<model>.json, where the staff and admin pages can pick them, and to
output/annotations.json, the file to submit.

    --output-only       write only --output (one versioned file per run) instead of both files above

    --evaluate          estimate the validation score instead (compare several models with --model a b c)
    --split cameras     with --evaluate, test on cameras the model never saw labelled instead of hidden photos
    --out-of-fold       predict labelled photos with models that never saw their labels, to review mistakes
                        honestly in the admin compare view (saved as output/predictions/<model>.out-of-fold.json)

Photos come from the training folder and input/uploads (the uploaded validation photos), so every run covers both;
with --evaluate, the uploads are also scored against their solution (input/uploads/annotations.json) if there is one.

Biases reweight the model's stock probabilities before each answer is chosen: --lower-bias leans answers lower
(negative: higher) and --half-bias widens "half" from both sides (negative: narrower). Per-area biases from
--area-biases ({camera: {area_id: {"lower": x, "half": y}}}, or a plain number for lower) add to those for every
photo of that area; tune them in the admin annotate view.

Next to each answers file, <name>.probabilities.json keeps the model's raw probabilities, so the pages can apply
changed biases straight away without running the model again.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shelf.crops import cut_area
from shelf.decisions import decide
from shelf.evaluation import biased_answers, evaluate, nested, out_of_fold, score_solution
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


def load_area_biases(path):
    """Per-area (lower, half) biases, {(camera, area id): (lower, half)}; a missing file or entry means (0, 0).

    Each entry is {"lower": x, "half": y}, or a plain number meaning a lower bias only (the older format).
    """
    if not path.exists():
        return {}
    biases = {}
    for camera, areas in json.loads(path.read_text()).items():
        for area_id, bias in areas.items():
            if isinstance(bias, dict):
                biases[camera, area_id] = (float(bias.get("lower", 0)), float(bias.get("half", 0)))
            else:
                biases[camera, area_id] = (float(bias), 0.0)
    return biases


def find_photos(planogram, folders):
    """{(camera, file): folder} for every photo in the folders; a name already seen in an earlier folder is skipped,
    since features are cached by camera, file and area."""
    folder_of = {}
    for folder in folders:
        for camera in sorted(planogram):
            for path in sorted((folder / camera).glob("*.jpg")):
                if (camera, path.name) in folder_of:
                    print(f"Skipping {path}: {camera}/{path.name} is already in {folder_of[camera, path.name]}")
                    continue
                folder_of[camera, path.name] = folder
    return folder_of


def crops_of(planogram, folder_of, photos):
    crops = {}
    for folder in dict.fromkeys(folder_of.values()):
        crops.update(area_crops(planogram, folder, [photo for photo in photos if folder_of[photo] == folder]))
    return crops


def probabilities_path(path):
    """Where a predictions file's raw probabilities live: <name>.probabilities.json next to it."""
    return path.with_name(f"{path.stem}.probabilities.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], choices=list(MODELS),
                        help=f"which model (default {DEFAULT_MODEL}); several to compare with --evaluate")
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, nargs="+", default=[Path("input/cameras"), Path("input/uploads")],
                        help="folders of photos to predict, each laid out <camera>/<photo>.jpg (default: the training "
                             "photos and the uploads)")
    parser.add_argument("--solution", type=Path, default=Path("input/uploads/annotations.json"),
                        help="the uploaded photos' right answers; --evaluate scores the uploads against it")
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"), help="labels to train on")
    parser.add_argument("--train-cameras", type=Path, default=Path("input/cameras"), help="photos the labels refer to")
    parser.add_argument("--output", type=Path, default=Path("output/annotations.json"), help="the file to submit")
    parser.add_argument("--output-only", action="store_true",
                        help="write only --output, skipping the predictions-dir copy (one file per run)")
    parser.add_argument("--evaluate", type=int, nargs="?", const=20, metavar="ROUNDS",
                        help="estimate the validation score instead of predicting")
    parser.add_argument("--split", choices=["photos", "cameras"], default="photos", help="what --evaluate hides")
    parser.add_argument("--out-of-fold", type=int, nargs="?", const=10, metavar="FOLDS",
                        help="predict labelled photos with models that didn't see their labels")
    parser.add_argument("--lower-bias", type=float, default=0.0,
                        help="lean stock answers toward lower levels (0 = none, negative = higher)")
    parser.add_argument("--half-bias", type=float, default=0.0,
                        help="widen half from both sides (0 = none, negative = narrower)")
    parser.add_argument("--area-biases", type=Path, default=Path("input/area-biases.json"),
                        help="per-area lower and half biases, added to the run's (tuned in the admin annotate view)")
    parser.add_argument("--biases", type=float, nargs="+", default=[0.0],
                        help="lower biases to compare with --evaluate")
    parser.add_argument("--half-biases", type=float, nargs="+", default=[0.0],
                        help="half biases to compare with --evaluate (every pairing with --biases is scored)")
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
    area_biases = load_area_biases(args.area_biases)
    if area_biases:
        print(f"Loaded biases for {len(area_biases)} areas from {args.area_biases}")
    folder_of = find_photos(planogram, args.cameras)

    if args.evaluate:
        biases = [(lower, half) for lower in args.biases for half in args.half_biases]
        evaluate(args.model, train_features, train_keys, labels, args.evaluate, biases, args.split, args.jobs,
                 area_biases)
        solution = json.loads(args.solution.read_text()) if args.solution.exists() else {}
        wanted = [(camera, file) for camera, files in solution.items() for file in files if (camera, file) in folder_of]
        upload_crops = crops_of(planogram, folder_of, wanted)
        upload_keys = [(camera, file, area_id) for camera, file in wanted for area_id in solution[camera][file]
                       if (camera, file, area_id) in upload_crops]
        if upload_keys:
            upload_features = extract(upload_keys, [upload_crops[key] for key in upload_keys], needs)
            score_solution(args.model, train_features, labels, upload_features, upload_keys,
                           [solution[c][f][a] for c, f, a in upload_keys], biases, area_biases)
        return

    name = args.model[0]
    model = MODELS[name]().fit(train_features, labels)
    photos = list(folder_of)
    crops = crops_of(planogram, folder_of, photos)
    keys = list(crops)
    probabilities = model.predict(extract(keys, [crops[key] for key in keys], needs))
    if args.out_of_fold:
        held_out = out_of_fold(name, train_features, train_keys, labels, args.out_of_fold, args.jobs)
        probabilities = [held_out.get(key, p) for key, p in zip(keys, probabilities)]
        print(f"{sum(key in held_out for key in keys)} labelled areas predicted out of fold")
    answers = biased_answers(probabilities, keys, (args.lower_bias, args.half_bias), area_biases)
    predictions = nested(keys, answers)

    # A photo that can't be read still gets an answer for every area: the cautious "compliant, empty".
    cautious = decide(1.0, {level: 1 / 3 for level in STOCK_VALUES})
    for camera, file in photos:
        for area in planogram[camera]["products"]:
            predictions.setdefault(camera, {}).setdefault(file, {}).setdefault(area["area_id"], cautious)

    saved = PREDICTIONS / f"{name}{'.out-of-fold' if args.out_of_fold else ''}.json"
    targets = [args.output] if args.output_only else [saved, args.output]
    # The raw probabilities, before any bias, so changed biases can be applied without running the model again.
    raw = {
        "model": name, "lower_bias": args.lower_bias, "half_bias": args.half_bias,
        "areas": nested(keys, [{"compliant": float(c), "stock": {level: float(v) for level, v in stock.items()}}
                               for c, stock in probabilities]),
    }
    for path in dict.fromkeys(targets):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(predictions, indent=2))
        probabilities_path(path).write_text(json.dumps(raw))
    count = sum(len(areas) for images in predictions.values() for areas in images.values())
    per_folder = ", ".join(f"{sum(f == folder for f in folder_of.values())} from {folder}" for folder in args.cameras)
    print(f"Wrote {count} area predictions for {len(photos)} photos ({per_folder}) "
          f"to {', '.join(str(p) for p in dict.fromkeys(targets))}")


if __name__ == "__main__":
    main()
