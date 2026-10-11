"""Estimate the validation score with a held-out test set that mimics the hidden validation set.

Run from the repository root inside `nix develop`:

    python -m experiments.holdout

For each round, 1–2 labelled photos per camera are set aside as a test set, like the organisers' 1–2 validation
photos per camera. Models are trained only on the remaining photos and scored on the held-out ones with score.py's
rules. Repeating with different random picks shows how much the score depends on which photos happen to be hidden.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.compare_classifiers import CLIP_MODELS, LEVELS, Clip, histogram_features, load_crops, product_logits
from planogram import load_cameras
from predict import decide
from score import points, score

VALIDATION_AREAS = 289


def held_out_photos(annotations, rng):
    """Two photos per camera, one when the camera has few labelled photos (even if that leaves none to train on)."""
    test = set()
    for camera, images in annotations.items():
        files = sorted(images)
        count = 2 if len(files) >= 3 else 1
        test.update((camera, file) for file in rng.sample(files, count))
    return test


def fit_predict(features, targets, train, test, classes):
    """Probabilities for the test rows from a logistic regression trained on the train rows."""
    probabilities = np.zeros((len(test), len(classes)))
    if len(set(targets[train])) == 1:
        probabilities[:, classes.index(targets[train][0])] = 1
        return probabilities
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    model.fit(features[train], targets[train])
    probabilities[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(features[test])
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    planogram = load_cameras(args.planogram)
    annotations = json.loads(args.annotations.read_text())
    samples = [
        (camera, file, area_id, label)
        for camera, images in annotations.items()
        for file, areas in images.items()
        for area_id, label in areas.items()
    ]
    keys = [sample[:3] for sample in samples]
    stock_targets = np.array([label["stock_level"] for *_, label in samples])
    compliance_targets = np.array([label["planogram_compliant"] for *_, label in samples])
    print(f"{len(samples)} labelled areas, {sum(map(len, annotations.values()))} photos, {len(annotations)} cameras")

    print("Cutting crops and embedding them with CLIP ViT-B/32…")
    crops = load_crops(planogram, args.cameras, annotations)
    labelled_crops = [crops[key] for key in keys]
    clip = Clip(*CLIP_MODELS["CLIP ViT-B/32"])
    embeddings = clip.images(labelled_crops)
    histograms = np.array([histogram_features(c) for c in labelled_crops])
    product_scores = product_logits(clip, embeddings, samples, planogram)

    # Each pipeline: (stock features or None for label frequencies, compliance features or None for always compliant).
    pipelines = {
        "Baseline: label frequencies, always compliant": (np.zeros((len(samples), 1)), None),
        "Colour histogram stock, always compliant": (histograms, None),
        "CLIP stock, always compliant": (embeddings, None),
        "CLIP stock, calibrated CLIP compliance": (embeddings, product_scores),
    }
    results = {name: [] for name in pipelines}
    unseen = Counter()

    for round_number in range(args.rounds):
        test_photos = held_out_photos(annotations, random.Random(round_number))
        test = np.array([i for i, (camera, file, _) in enumerate(keys) if (camera, file) in test_photos])
        train = np.setdiff1d(np.arange(len(keys)), test)
        trained_cameras = {keys[i][0] for i in train}
        unseen.update(camera for camera, _ in test_photos if camera not in trained_cameras)

        truth = {}
        for i in test:
            camera, file, area_id = keys[i]
            truth.setdefault(camera, {}).setdefault(file, {})[area_id] = samples[i][3]

        for name, (stock_features, compliance_features) in pipelines.items():
            stock = fit_predict(stock_features, stock_targets, train, test, LEVELS)
            compliant = (
                np.ones(len(test))
                if compliance_features is None
                else fit_predict(compliance_features, compliance_targets, train, test, [False, True])[:, 1]
            )
            predictions = {}
            for row, i in enumerate(test):
                camera, file, area_id = keys[i]
                answer = decide(compliant[row], dict(zip(LEVELS, stock[row])))
                predictions.setdefault(camera, {}).setdefault(file, {})[area_id] = answer
            total = sum(score(predictions, truth).values(), Counter())
            results[name].append((points(total), len(test), total))

    print(f"\nHeld-out score over {args.rounds} rounds, scaled to the validation set's {VALIDATION_AREAS} areas "
          f"(maximum {2 * VALIDATION_AREAS}):")
    print(f"{'Pipeline':<50}{'mean':>8}{'worst':>8}{'best':>8}   stock correct  compliance correct")
    for name, rounds in results.items():
        scaled = [VALIDATION_AREAS * p / n for p, n, _ in rounds]
        stock_correct = sum(t["stock_correct"] for *_, t in rounds) / sum(n for _, n, _ in rounds)
        compliance_correct = sum(t["identification_correct"] for *_, t in rounds) / sum(n for _, n, _ in rounds)
        print(f"{name:<50}{np.mean(scaled):>+8.0f}{min(scaled):>+8.0f}{max(scaled):>+8.0f}"
              f"   {stock_correct:>12.0%}  {compliance_correct:>17.0%}")
    held_out_areas = [n for _, n, _ in next(iter(results.values()))]
    print(f"\nAbout {np.mean(held_out_areas):.0f} held-out areas per round.")
    if unseen:
        print(f"Cameras with no training photos left when held out: {', '.join(sorted(unseen))}")


if __name__ == "__main__":
    main()
