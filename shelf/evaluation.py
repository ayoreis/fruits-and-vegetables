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


def area_bias(area_biases, key):
    """A planogram area's own (lower, half) bias; areas without one count as (0, 0)."""
    return area_biases.get((key[0], key[2]), (0.0, 0.0))


def biased_answers(probabilities, keys, bias, area_biases):
    """Answers with a run-wide (lower, half) bias plus each area's own bias."""
    answers = []
    for p, key in zip(probabilities, keys):
        lower, half = area_bias(area_biases, key)
        answers.append(decide(*p, bias[0] + lower, bias[1] + half))
    return answers


def scored(keys, answers, truth_labels):
    """Points scaled to the validation set's size, and the share of answers right."""
    counts = sum(score(nested(keys, answers), nested(keys, truth_labels)).values(), Counter())
    correct = (counts["identification_correct"] + counts["stock_correct"]) / (2 * len(keys))
    return VALIDATION_AREAS * points(counts) / len(keys), correct


def score_round(model_name, features, keys, labels, test, biases, area_biases=None):
    """The round's score for each (lower, half) bias, scaled to the validation set's size, and the share right."""
    area_biases = area_biases or {}
    probabilities = fit_and_predict(model_name, features, labels, test)
    test_keys = [key for key, t in zip(keys, test) if t]
    test_labels = [l for l, t in zip(labels, test) if t]
    return {bias: scored(test_keys, biased_answers(probabilities, test_keys, bias, area_biases), test_labels)
            for bias in biases}


def bias_label(bias):
    return f"lower {bias[0]:g}, half {bias[1]:g}"


def jobs_label(jobs):
    return "all cores" if jobs == -1 else f"{jobs} process{'es' if jobs != 1 else ''}"


def evaluate(model_names, features, keys, labels, rounds=20, biases=((0.0, 0.0),), split="photos", jobs=-1,
             area_biases=None):
    """Print each model's estimated validation score, for each (lower, half) bias."""
    masks = hidden_photo_masks(keys, rounds) if split == "photos" else hidden_camera_masks(keys)
    tasks = [(name, mask) for name in model_names for mask in masks]
    print(f"Scoring {len(model_names)} model(s) on {len(masks)} {split} splits, on {jobs_label(jobs)}…")
    outcomes = Parallel(n_jobs=jobs)(
        delayed(score_round)(name, features, keys, labels, mask, biases, area_biases) for name, mask in tasks
    )

    maximum = 2 * VALIDATION_AREAS
    print(f"\nEstimated validation score out of {maximum} ({split} split; range across splits in brackets):")
    # One row per model and bias, so a grid of lower x half biases stays readable.
    width = max(len(name) for name in model_names) + 2
    print(f"{'model':<{width}}{'bias':<22}{'score':>8}{'of max':>8}  {'range':<14}{'answers right':>14}")
    for name in model_names:
        rows = [outcome for (task_name, _), outcome in zip(tasks, outcomes) if task_name == name]
        for bias in biases:
            scores = [row[bias][0] for row in rows]
            correct = np.mean([row[bias][1] for row in rows])
            print(f"{name:<{width}}{bias_label(bias):<22}{np.mean(scores):>+8.0f}{np.mean(scores) / maximum:>8.1%}  "
                  f"{f'{min(scores):+.0f}..{max(scores):+.0f}':<14}{correct:>14.1%}")


def score_solution(model_names, features, labels, upload_features, upload_keys, solution_labels, biases=((0.0, 0.0),),
                   area_biases=None):
    """Train each model on all the labels and score the uploaded photos against their solution, for each bias."""
    area_biases = area_biases or {}
    maximum = 2 * len(upload_keys)
    width = max(len(name) for name in model_names) + 2
    print(f"\nUploaded photos against the uploaded solution ({len(upload_keys)} areas, maximum {maximum}):")
    print(f"{'model':<{width}}{'bias':<22}{'score':>8}{'of max':>8}{'answers right':>16}")
    for name in model_names:
        probabilities = MODELS[name]().fit(features, labels).predict(upload_features)
        for bias in biases:
            answers = biased_answers(probabilities, upload_keys, bias, area_biases)
            counts = sum(score(nested(upload_keys, answers), nested(upload_keys, solution_labels)).values(), Counter())
            correct = (counts["identification_correct"] + counts["stock_correct"]) / maximum
            print(f"{name:<{width}}{bias_label(bias):<22}{points(counts):>+8}{points(counts) / maximum:>8.1%}{correct:>16.1%}")


def out_of_fold(model_name, features, keys, labels, folds=10, jobs=-1):
    """Probabilities for every labelled area from a model that didn't see its photo's labels.

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
        return list(zip([key for key, t in zip(keys, test) if t], probabilities))

    print(f"Predicting {folds} folds on {jobs_label(jobs)}…")
    held_out = {}
    for fold_probabilities in Parallel(n_jobs=jobs)(delayed(predict_fold)(number) for number in range(folds)):
        held_out.update(fold_probabilities)
    return held_out
