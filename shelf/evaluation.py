"""Honest scores for models: each prediction comes from a model that never saw that photo's labels.

- photos split: hide 1-2 labelled photos per camera (like the hidden validation set), train on the rest, score the
  hidden ones; repeat with different picks.
- cameras split: hide every photo of one camera at a time, a test of how a model copes with a camera it has
  never seen labelled.

Rounds run side by side in separate processes.
"""

import random
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from shelf.decisions import decide
from shelf.models import MODELS
from shelf.score import points, score

VALIDATION_AREAS = 289


def subset(features, rows):
    return {name: value[rows] for name, value in features.items()}


def nested(keys, values):
    result = {}
    for (camera, file, area_id), value in zip(keys, values):
        result.setdefault(camera, {}).setdefault(file, {})[area_id] = value
    return result


def photos_by_camera(keys):
    photos = {}
    for camera, file, _ in keys:
        photos.setdefault(camera, set()).add(file)
    return photos


def hidden_photo_masks(keys, rounds):
    """For each round, which keys belong to the 1-2 hidden photos per camera."""
    photos = photos_by_camera(keys)
    masks = []
    for number in range(rounds):
        rng = random.Random(number)
        hidden = {
            (camera, file) for camera, files in photos.items()
            for file in rng.sample(sorted(files), 2 if len(files) >= 3 else 1)
        }
        masks.append(np.array([key[:2] in hidden for key in keys]))
    return masks


def hidden_camera_masks(keys):
    """One mask per camera, hiding all of its photos."""
    return [np.array([key[0] == camera for key in keys]) for camera in sorted(photos_by_camera(keys))]


def fit_and_predict(model_name, features, labels, test):
    model = MODELS[model_name]().fit(subset(features, ~test), [l for l, t in zip(labels, test) if not t])
    return model.predict(subset(features, test))


def score_round(model_name, features, keys, labels, test, biases):
    """The round's score for each lower bias, scaled to the validation set's size, and the share of answers right."""
    probabilities = fit_and_predict(model_name, features, labels, test)
    test_keys = [key for key, t in zip(keys, test) if t]
    truth = nested(test_keys, [l for l, t in zip(labels, test) if t])
    results = {}
    for bias in biases:
        counts = sum(score(nested(test_keys, [decide(*p, bias) for p in probabilities]), truth).values(), Counter())
        correct = (counts["identification_correct"] + counts["stock_correct"]) / (2 * len(test_keys))
        results[bias] = (VALIDATION_AREAS * points(counts) / len(test_keys), correct)
    return results


def jobs_label(jobs):
    return "all cores" if jobs == -1 else f"{jobs} process{'es' if jobs != 1 else ''}"


def evaluate(model_names, features, keys, labels, rounds=20, biases=(0.0,), split="photos", jobs=-1):
    """Print each model's estimated validation score, for each lower bias."""
    masks = hidden_photo_masks(keys, rounds) if split == "photos" else hidden_camera_masks(keys)
    tasks = [(name, mask) for name in model_names for mask in masks]
    print(f"Scoring {len(model_names)} model(s) on {len(masks)} {split} splits, on {jobs_label(jobs)}…")
    outcomes = Parallel(n_jobs=jobs)(
        delayed(score_round)(name, features, keys, labels, mask, biases) for name, mask in tasks
    )

    maximum = 2 * VALIDATION_AREAS
    print(f"\nEstimated validation score out of {maximum} ({split} split; range across splits in brackets):")
    width = max(len(name) for name in model_names) + 2
    header = "".join(f"{f'bias {bias}':>26}" for bias in biases)
    print(f"{'model':<{width}}{header}")
    for name in model_names:
        rows = [outcome for (task_name, _), outcome in zip(tasks, outcomes) if task_name == name]
        correct = np.mean([row[biases[0]][1] for row in rows])
        cells = []
        for bias in biases:
            scores = [row[bias][0] for row in rows]
            cells.append(f"{np.mean(scores):+.0f} {np.mean(scores) / maximum:5.1%} ({min(scores):+.0f}..{max(scores):+.0f})")
        print(f"{name:<{width}}" + "".join(f"{cell:>26}" for cell in cells) + f"   {correct:.1%} answers right (bias {biases[0]})")


def out_of_fold(model_name, features, keys, labels, folds=10, lower_bias=0.0, jobs=-1):
    """Answers for every labelled area from a model that didn't see its photo's labels.

    Each camera's photos are dealt round-robin into the folds, so every fold has every camera.
    """
    fold_of = {}
    for camera, files in photos_by_camera(keys).items():
        shuffled = sorted(files)
        random.Random(camera).shuffle(shuffled)
        fold_of.update({(camera, file): i % folds for i, file in enumerate(shuffled)})
    fold = np.array([fold_of[key[:2]] for key in keys])

    def predict_fold(number):
        test = fold == number
        probabilities = fit_and_predict(model_name, features, labels, test)
        test_keys = [key for key, t in zip(keys, test) if t]
        return list(zip(test_keys, (decide(*p, lower_bias) for p in probabilities)))

    print(f"Predicting {folds} folds on {jobs_label(jobs)}…")
    answers = {}
    for fold_answers in Parallel(n_jobs=jobs)(delayed(predict_fold)(number) for number in range(folds)):
        answers.update(fold_answers)
    return answers
