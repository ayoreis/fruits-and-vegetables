"""Try different stock-level approaches on the held-out test from experiments/holdout.py.

Run from the repository root inside `nix develop`:

    python -m experiments.approaches

The labels are snapshotted once at the start, so annotating while this runs doesn't change the comparison.
Cameras with fewer than --min-photos labelled photos are left out. Image embeddings are cached in
output/experiments/, so re-runs only embed newly labelled crops.

Every approach is scored on the same held-out rounds. Stock and compliance are compared separately; the stock
table's total adds the best compliance result.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from experiments.compare_classifiers import CLIP_MODELS, LEVELS, Clip, histogram_features, load_crops, product_logits
from experiments.holdout import VALIDATION_AREAS, held_out_photos
from planogram import load_cameras
from predict import decide
from score import POINTS, STOCK_VALUES, identification_case, stock_case

CACHE = Path("output/experiments")
PAD_COLOUR = (124, 116, 104)  # roughly CLIP's mean pixel, so padding looks neutral


# ---------- Preparing crops for the image models ----------


def to_pil(crop):
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def padded(crop):
    """The whole crop, letterboxed to a square so a centre crop loses nothing."""
    image = to_pil(crop)
    side = max(image.size)
    square = Image.new("RGB", (side, side), PAD_COLOUR)
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


# ---------- Embedding models ----------


class Dino:
    """DINOv2 ViT-S/14 at 224 px: self-supervised, good at visual similarity, smaller than CLIP B/32."""

    def __init__(self):
        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        self.model.eval()
        config = timm.data.resolve_model_data_config(self.model)
        config.update(input_size=(3, 224, 224), crop_pct=1.0)  # the pretrained default is 518 px
        self.transform = timm.data.create_transform(**config, is_training=False)

    def images(self, images):
        embeddings = []
        for start in range(0, len(images), 32):
            batch = self.torch.stack([self.transform(image) for image in images[start:start + 32]])
            with self.torch.no_grad():
                embeddings.append(self.model(batch))
        embeddings = self.torch.cat(embeddings)
        return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()


def clip_images(clip, images):
    embeddings = []
    for start in range(0, len(images), 32):
        batch = clip.torch.stack([clip.preprocess(image) for image in images[start:start + 32]])
        with clip.torch.no_grad():
            embeddings.append(clip.model.encode_image(batch))
    embeddings = clip.torch.cat(embeddings)
    return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()


def cached_embeddings(name, keys, crops, embed):
    """Embeddings for every key, computing only the ones not cached from an earlier run."""
    path = CACHE / f"{name}.npz"
    cache = dict(np.load(path)) if path.exists() else {}
    ids = ["|".join(key) for key in keys]
    missing = [i for i, id_ in enumerate(ids) if id_ not in cache]
    if missing:
        print(f"  {name}: embedding {len(missing)} crops…")
        for i, embedding in zip(missing, embed([crops[keys[i]] for i in missing])):
            cache[ids[i]] = embedding
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cache)
    return np.array([cache[id_] for id_ in ids])


# ---------- Stock approaches: each returns test probabilities from train rows ----------


def fit_logistic(train_features, train_targets, test_features, classes, regularisation=1.0):
    probabilities = np.zeros((len(test_features), len(classes)))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=regularisation, max_iter=5000))
    model.fit(train_features, train_targets)
    probabilities[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(test_features)
    return probabilities


def logistic(features, regularisation=1.0, classes=LEVELS):
    def run(targets, train, test):
        return fit_logistic(features[train], targets[train], features[test], classes, regularisation)

    return run


def area_one_hot(areas):
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(areas.reshape(-1, 1))


def logistic_with_area(features, areas, regularisation=1.0, classes=LEVELS):
    """Logistic regression that also knows which area each crop is, so it can learn per-area habits."""
    return logistic(np.hstack([features, area_one_hot(areas)]), regularisation, classes)


class TimeContext:
    """What the labelled photos of the same area just before and after a photo say about it.

    Only photos in the training pool are used as neighbours, and never the photo itself.
    """

    def __init__(self, embeddings, areas, times, stock_targets, compliance_targets, past_only=False):
        self.embeddings, self.areas, self.times, self.past_only = embeddings, areas, times, past_only
        self.stock, self.compliant = stock_targets, compliance_targets
        self.by_area = {}
        for i in np.argsort(times, kind="stable"):
            self.by_area.setdefault(areas[i], []).append(i)

    def neighbours(self, i, pool):
        previous = following = None
        for j in self.by_area[self.areas[i]]:
            if j == i or not pool[j]:
                continue
            if self.times[j] < self.times[i]:
                previous = j
            elif self.times[j] > self.times[i] and following is None:
                following = j
        return previous, None if self.past_only else following

    def features(self, rows, train):
        pool = np.zeros(len(self.times), bool)
        pool[train] = True
        table = []
        for i in rows:
            row = []
            for j in self.neighbours(i, pool):
                if j is None:
                    row += [0] * 7
                else:
                    level = [float(self.stock[j] == l) for l in LEVELS]
                    gap = abs(self.times[i] - self.times[j]) / 3600
                    row += [1, *level, float(self.compliant[j]), gap, float(self.embeddings[i] @ self.embeddings[j])]
            table.append(row)
        return np.array(table)

    def copy_nearest(self):
        """Baseline without the image: copy the stock level of the nearest labelled photo in time."""

        def run(targets, train, test):
            pool = np.zeros(len(self.times), bool)
            pool[train] = True
            probabilities = np.full((len(test), len(LEVELS)), 1 / 3)
            for row, i in enumerate(test):
                found = [j for j in self.neighbours(i, pool) if j is not None]
                if found:
                    nearest = min(found, key=lambda j: abs(self.times[j] - self.times[i]))
                    probabilities[row] = [0.8 if targets[nearest] == l else 0.1 for l in LEVELS]
            return probabilities

        return run

    def logistic(self, base_features, classes=LEVELS):
        """Image features plus the time context, in one logistic regression."""

        def run(targets, train, test):
            train_features = np.hstack([base_features[train], self.features(train, train)])
            test_features = np.hstack([base_features[test], self.features(test, train)])
            return fit_logistic(train_features, targets[train], test_features, classes)

        return run


def ensemble(*runs):
    """Average the probabilities of several approaches."""

    def run(targets, train, test):
        return np.mean([approach(targets, train, test) for approach in runs], axis=0)

    return run


def tempered(approach, temperature):
    """Soften (temperature > 1) or sharpen (< 1) an approach's probabilities."""

    def run(targets, train, test):
        probabilities = approach(targets, train, test) ** (1 / temperature)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    return run


def svm(features):
    def run(targets, train, test):
        probabilities = np.zeros((len(test), len(LEVELS)))
        model = make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=0))
        model.fit(features[train], targets[train])
        probabilities[:, [LEVELS.index(c) for c in model.classes_]] = model.predict_proba(features[test])
        return probabilities

    return run


def same_area_neighbours(features, areas, k=3, fallback=None):
    """Vote among the k most similar labelled photos of the same area; areas with no labels fall back."""

    def run(targets, train, test):
        probabilities = fallback(targets, train, test) if fallback else np.full((len(test), 3), 1 / 3)
        for row, i in enumerate(test):
            same = train[areas[train] == areas[i]]
            if not len(same):
                continue
            nearest = same[np.argsort(features[same] @ features[i])[::-1][:k]]
            votes = Counter(targets[nearest])
            # A little smoothing so a unanimous vote isn't treated as certainty.
            counts = np.array([votes[level] for level in LEVELS]) + 0.5
            probabilities[row] = counts / counts.sum()
        return probabilities

    return run


# ---------- Evaluation ----------


def stock_points(probabilities, labels):
    cases = Counter(
        stock_case(decide(1.0, dict(zip(LEVELS, p)))["stock_value"], label["stock_value"])
        for p, label in zip(probabilities, labels)
    )
    return sum(POINTS[case] * count for case, count in cases.items()), cases["stock_correct"]


def compliance_points(compliant_probabilities, labels):
    uniform = {level: 1 / 3 for level in LEVELS}
    cases = Counter(
        identification_case(decide(p, uniform)["planogram_compliant"], label["planogram_compliant"])
        for p, label in zip(compliant_probabilities, labels)
    )
    return sum(POINTS[case] * count for case, count in cases.items()), cases["identification_correct"]


def print_results(title, results, extra=None):
    """One row per approach, best first, scaled to the validation set's size."""
    print(f"\n{title}")
    print(f"{'Approach':<52}{'points':>7}{'worst':>7}{'best':>7}{'correct':>9}" + (f"{'total':>8}" if extra else ""))
    for name, rounds in sorted(results.items(), key=lambda item: -np.mean([p / n for p, _, n in item[1]])):
        scaled = [VALIDATION_AREAS * p / n for p, _, n in rounds]
        accuracy = sum(c for _, c, _ in rounds) / sum(n for *_, n in rounds)
        print(f"{name:<52}{np.mean(scaled):>+7.0f}{min(scaled):>+7.0f}{max(scaled):>+7.0f}{accuracy:>9.0%}"
              + (f"{np.mean(scaled) + extra:>+8.0f}" if extra else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--min-photos", type=int, default=3, help="leave out cameras with fewer labelled photos")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--large", action="store_true", help="also try CLIP ViT-L/14 (about 20x slower)")
    parser.add_argument("--quick", action="store_true", help="only the stock approaches that did best so far")
    parser.add_argument("--only", nargs="*", help="only run approaches whose name contains one of these")
    args = parser.parse_args()

    planogram = load_cameras(args.planogram)
    snapshot = json.loads(args.annotations.read_text())
    annotations = {camera: images for camera, images in snapshot.items() if len(images) >= args.min_photos}
    left_out = sorted(set(snapshot) - set(annotations))
    samples = [
        (camera, file, area_id, label)
        for camera, images in annotations.items()
        for file, areas in images.items()
        for area_id, label in areas.items()
    ]
    keys = [sample[:3] for sample in samples]
    labels = [sample[3] for sample in samples]
    stock_targets = np.array([label["stock_level"] for label in labels])
    compliance_targets = np.array([label["planogram_compliant"] for label in labels])
    areas = np.array([f"{camera}/{area_id}" for camera, _, area_id in keys])
    print(f"{len(samples)} labelled areas, {sum(map(len, annotations.values()))} photos, {len(annotations)} cameras"
          + (f" (left out: {', '.join(left_out)})" if left_out else ""))

    print("Cutting crops and computing features…")
    crops = load_crops(planogram, args.cameras, annotations)
    histograms = np.array([histogram_features(crops[key]) for key in keys])
    clip_b32 = Clip(*CLIP_MODELS["CLIP ViT-B/32"])
    features = {
        "CLIP B/32": cached_embeddings("clip-b32-centre", keys, crops, lambda c: clip_images(clip_b32, [to_pil(x) for x in c])),
        "CLIP B/32 padded": cached_embeddings("clip-b32-padded", keys, crops, lambda c: clip_images(clip_b32, [padded(x) for x in c])),
    }
    dino = None

    def embed_dino(c):
        nonlocal dino
        dino = dino or Dino()
        return dino.images([padded(x) for x in c])

    features["DINOv2 S/14"] = cached_embeddings("dinov2-s14-padded", keys, crops, embed_dino)
    if args.large:
        clip_l14 = Clip(*CLIP_MODELS["CLIP ViT-L/14"])
        features["CLIP L/14 padded"] = cached_embeddings(
            "clip-l14-padded", keys, crops, lambda c: clip_images(clip_l14, [padded(x) for x in c])
        )
    product_scores = product_logits(clip_b32, features["CLIP B/32"], samples, planogram)

    best = "CLIP B/32 padded"
    approaches = {
        "Colour histogram, logistic (reference)": logistic(histograms),
        "CLIP B/32, logistic (current best)": logistic(features["CLIP B/32"]),
    }
    for name, embedding in features.items():
        if name != "CLIP B/32":
            approaches[f"{name}, logistic"] = logistic(embedding)
    for regularisation in (0.01, 0.1, 10):
        approaches[f"{best}, logistic C={regularisation}"] = logistic(features[best], regularisation)
    approaches[f"{best} + histogram, logistic"] = logistic(np.hstack([features[best], histograms]))
    approaches[f"{best} + DINOv2, logistic"] = logistic(np.hstack([features[best], features["DINOv2 S/14"]]))
    approaches[f"{best} + area identity, logistic"] = logistic_with_area(features[best], areas)
    approaches["DINOv2 S/14 + area identity, logistic"] = logistic_with_area(features["DINOv2 S/14"], areas)
    approaches[f"{best} + DINOv2 + area identity, logistic"] = logistic_with_area(
        np.hstack([features[best], features["DINOv2 S/14"]]), areas
    )
    approaches[f"{best} + histogram + area identity, logistic"] = logistic_with_area(
        np.hstack([features[best], histograms]), areas
    )
    if args.quick:
        approaches = {name: run for name, run in approaches.items() if "area identity" in name or "(" in name}
    else:
        approaches[f"{best}, SVM"] = svm(features[best])
        for name in ("CLIP B/32 padded", "DINOv2 S/14"):
            for k in (1, 3, 5):
                approaches[f"{name}, same-area {k} nearest"] = same_area_neighbours(
                    features[name], areas, k, fallback=logistic(features[best])
                )

    # Time context: the labelled photos of the same area just before and after.
    times = np.array([int(Path(file).stem) for _, file, _ in keys])
    image = np.hstack([features[best], histograms, area_one_hot(areas)])
    time_context = TimeContext(features[best], areas, times, stock_targets, compliance_targets)
    past_context = TimeContext(features[best], areas, times, stock_targets, compliance_targets, past_only=True)
    approaches["Copy nearest labelled photo in time (no image)"] = time_context.copy_nearest()
    approaches["Copy previous labelled photo in time (no image)"] = past_context.copy_nearest()
    approaches[f"{best} + histogram + area + time context"] = time_context.logistic(image)
    approaches[f"{best} + histogram + area + past-only time context"] = past_context.logistic(image)

    clip_area = logistic_with_area(np.hstack([features[best], histograms]), areas)
    dino_area = logistic_with_area(features["DINOv2 S/14"], areas)
    approaches["Ensemble: CLIP + histogram + area, DINOv2 + area"] = ensemble(clip_area, dino_area)
    for temperature in (1.5, 2.0):
        approaches[f"Ensemble, tempered T={temperature}"] = tempered(ensemble(clip_area, dino_area), temperature)
        approaches[f"{best} + histogram + area identity, tempered T={temperature}"] = tempered(clip_area, temperature)
    for regularisation in (0.3, 3.0):
        approaches[f"{best} + histogram + area identity, C={regularisation}"] = logistic_with_area(
            np.hstack([features[best], histograms]), areas, regularisation
        )

    compliance_classes = [False, True]
    compliance_approaches = {
        "Always compliant": lambda targets, train, test: np.tile([0.0, 1.0], (len(test), 1)),
        "Calibrated CLIP product scores (current)": logistic(product_scores, classes=compliance_classes),
        "CLIP product scores + area identity": logistic_with_area(product_scores, areas, classes=compliance_classes),
        f"{best} + product scores, logistic": logistic(
            np.hstack([features[best], product_scores]), classes=compliance_classes
        ),
        f"{best} + product scores + area identity": logistic_with_area(
            np.hstack([features[best], product_scores]), areas, classes=compliance_classes
        ),
        "DINOv2 S/14 + product scores + area identity": logistic_with_area(
            np.hstack([features["DINOv2 S/14"], product_scores]), areas, classes=compliance_classes
        ),
        f"{best} + product scores + area + time context": time_context.logistic(
            np.hstack([image, product_scores]), compliance_classes
        ),
        f"{best} + product scores + area + past-only time context": past_context.logistic(
            np.hstack([image, product_scores]), compliance_classes
        ),
    }

    if args.only:
        keep = lambda name: any(part in name for part in args.only)
        approaches = {name: run for name, run in approaches.items() if keep(name)}
        compliance_approaches = {name: run for name, run in compliance_approaches.items() if keep(name)}

    print(f"Scoring {len(approaches)} stock and {len(compliance_approaches)} compliance approaches "
          f"over {args.rounds} held-out rounds…")
    results = {name: [] for name in approaches}
    compliance_results = {name: [] for name in compliance_approaches}
    for round_number in range(args.rounds):
        test_photos = held_out_photos(annotations, random.Random(round_number))
        test = np.array([i for i, (camera, file, _) in enumerate(keys) if (camera, file) in test_photos])
        train = np.setdiff1d(np.arange(len(keys)), test)
        test_labels = [labels[i] for i in test]

        for name, approach in compliance_approaches.items():
            compliant = approach(compliance_targets, train, test)[:, 1]
            points, correct = compliance_points(compliant, test_labels)
            compliance_results[name].append((points, correct, len(test)))

        for name, approach in approaches.items():
            points, correct = stock_points(approach(stock_targets, train, test), test_labels)
            results[name].append((points, correct, len(test)))

    best_compliance = max(
        np.mean([VALIDATION_AREAS * p / n for p, _, n in rounds]) for rounds in compliance_results.values()
    )
    print(f"\nScaled to the validation set's {VALIDATION_AREAS} areas: each part is out of {VALIDATION_AREAS}, "
          f"the total out of {2 * VALIDATION_AREAS}.")
    print_results("Compliance", compliance_results)
    print_results(f"Stock level (total adds the best compliance, {best_compliance:+.0f})", results, best_compliance)


if __name__ == "__main__":
    main()
