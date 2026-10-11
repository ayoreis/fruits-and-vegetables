"""Compare ways of classifying stock level and compliance, scored with the challenge's points.

Run from the repository root inside `nix develop`:

    python -m experiments.compare_classifiers

Every method outputs probabilities, predict.decide turns them into answers, and score.py's rules turn those into
points. Methods that learn from labels are cross-validated two ways:

- photo: each photo is predicted by a model trained on the other photos, so it has seen the same areas labelled
  at other times of day. This is close to the validation set, which uses the same cameras.
- area: each area is predicted by a model that never saw that area labelled.
- camera: each camera is predicted by a model that never saw any of its labels, a stand-in for unlabelled cameras.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cut_areas import cut_area
from planogram import area_quad, load_cameras
from predict import decide
from products import english_name, is_packaged
from score import POINTS, STOCK_VALUES, identification_case, stock_case

LEVELS = list(STOCK_VALUES)
GROUPINGS = ("photo", "area", "camera")
CLIP_MODELS = {
    "CLIP ViT-B/32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "CLIP ViT-L/14": ("ViT-L-14", "laion2b_s32b_b82k"),
}
PRODUCT_TEMPLATES = ["a photo of {}.", "a supermarket crate of {}.", "{} on display in a supermarket."]
STOCK_TEMPLATES = {
    "empty": ["a photo of an empty supermarket crate.", "an empty shelf with no {} left."],
    "half": ["a photo of a half-empty crate with a few {}.", "a shelf partly stocked with {}."],
    "full": ["a photo of a supermarket crate full of {}.", "a shelf fully stocked with {}."],
}


# ---------- Data ----------


def load_crops(planogram, cameras_dir, cameras):
    """Straightened crops of every area in every photo of the given cameras."""
    crops = {}
    for camera in cameras:
        for image_path in sorted((cameras_dir / camera).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            for area in planogram[camera]["products"]:
                quad = np.array(area_quad(area), dtype=np.float32)
                crops[camera, image_path.name, area["area_id"]] = cut_area(image, quad)
    return crops


# ---------- Hand-made colour features ----------


def visible_mask(crop):
    """Pixels inside the photo; parts of an area beyond the photo's edge come out pure black."""
    return crop.any(axis=2)


def produce_fraction(crop, min_saturation=90):
    """Share of visible pixels that look like produce: saturated, not dark, and not crate teal or blue."""
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    crate = (hue > 80) & (hue < 130)
    produce = (saturation > min_saturation) & (value > 60) & ~crate
    visible = visible_mask(crop)
    return (produce & visible).sum() / max(visible.sum(), 1)


def colour_features(crop):
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2][visible_mask(crop)]
    return [produce_fraction(crop, s) for s in (60, 90, 120)] + [value.mean() / 255, (value < 50).mean()]


def histogram_features(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = visible_mask(crop).astype(np.uint8)
    histogram = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return histogram / max(histogram.sum(), 1)


def per_area_features(crops):
    """Features that compare each crop with the same area across the whole day (no labels needed).

    - normalised fill: the produce fraction rescaled between that area's emptiest and fullest photo.
    - change from empty: how much of the crop differs from a reference built from the area's two emptiest photos.
    """
    by_area = defaultdict(list)
    for key in crops:
        by_area[key[0], key[2]].append(key)

    features = {}
    for keys in by_area.values():
        fills = {key: produce_fraction(crops[key]) for key in keys}
        low, high = min(fills.values()), max(fills.values())
        emptiest = sorted(keys, key=fills.get)[:2]
        reference = cv2.cvtColor(np.median([crops[k] for k in emptiest], axis=0).astype(np.uint8), cv2.COLOR_BGR2LAB)
        for key in keys:
            lab = cv2.cvtColor(crops[key], cv2.COLOR_BGR2LAB).astype(int)
            changed = (np.abs(lab - reference).max(axis=2) > 30) & visible_mask(crops[key])
            features[key] = [(fills[key] - low) / max(high - low, 1e-6), changed.mean()]
    return features


# ---------- CLIP ----------


class Clip:
    def __init__(self, name, pretrained):
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(name)

    def images(self, crops):
        from PIL import Image

        embeddings = []
        for start in range(0, len(crops), 32):
            batch = [self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops[start:start + 32]]
            with self.torch.no_grad():
                embeddings.append(self.model.encode_image(self.torch.stack(batch)))
        embeddings = self.torch.cat(embeddings)
        return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()

    def text(self, prompts):
        """One embedding for a list of prompts, averaged."""
        with self.torch.no_grad():
            embeddings = self.model.encode_text(self.tokenizer(prompts))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean = embeddings.mean(dim=0)
        return (mean / mean.norm()).numpy()


def product_label(product):
    name = english_name(product["product_name"])
    return f"packs of {name}" if is_packaged(product["product_name"]) else name


def product_logits(clip, image_embeddings, samples, planogram):
    """For each sample: CLIP's score for the planned product, and its best score for any other product on that camera."""
    text_cache = {}
    rows = []
    for embedding, (camera, _, area_id, _) in zip(image_embeddings, samples):
        products = planogram[camera]["products"]
        planned = english_name(next(p for p in products if p["area_id"] == area_id)["product_name"])
        names = sorted({english_name(p["product_name"]) for p in products})
        for name in names:
            if name not in text_cache:
                text_cache[name] = clip.text([t.format(name) for t in PRODUCT_TEMPLATES])
        logits = {name: 100 * embedding @ text_cache[name] for name in names}
        others = [logit for name, logit in logits.items() if name != planned]
        rows.append([logits[planned], max(others) if others else logits[planned] - 10])
    return np.array(rows)


def stock_logits(clip, image_embeddings, samples, planogram, generic):
    text_cache = {}
    rows = []
    for embedding, (camera, _, area_id, _) in zip(image_embeddings, samples):
        product = next(p for p in planogram[camera]["products"] if p["area_id"] == area_id)
        name = "fruit and vegetables" if generic else english_name(product["product_name"])
        for level in LEVELS:
            if (name, level) not in text_cache:
                text_cache[name, level] = clip.text([t.format(name) for t in STOCK_TEMPLATES[level]])
        rows.append([100 * embedding @ text_cache[name, level] for level in LEVELS])
    return np.array(rows)


def softmax(logits):
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


# ---------- Evaluation ----------


def cross_validated(features, targets, groups, classes, regularisation=1.0):
    """Out-of-fold probabilities from a logistic regression, one group held out at a time."""
    probabilities = np.zeros((len(targets), len(classes)))
    for train, test in LeaveOneGroupOut().split(features, targets, groups):
        if len(set(targets[train])) == 1:
            probabilities[np.ix_(test, [classes.index(targets[train][0])])] = 1
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=regularisation, max_iter=5000))
        model.fit(features[train], targets[train])
        columns = [classes.index(c) for c in model.classes_]
        probabilities[np.ix_(test, columns)] = model.predict_proba(features[test])
    return probabilities


def stock_result(probabilities, labels):
    choices = [decide(1.0, dict(zip(LEVELS, p)))["stock_value"] for p in probabilities]
    cases = Counter(stock_case(choice, label["stock_value"]) for choice, label in zip(choices, labels))
    return cases, sum(POINTS[case] * count for case, count in cases.items())


def compliance_result(probabilities, labels):
    uniform = {level: 1 / 3 for level in LEVELS}
    choices = [decide(p, uniform)["planogram_compliant"] for p in probabilities]
    cases = Counter(identification_case(choice, label["planogram_compliant"]) for choice, label in zip(choices, labels))
    return cases, sum(POINTS[case] * count for case, count in cases.items())


def print_table(title, rows, total):
    print(f"\n{title} (out of {total} points; accuracy in brackets)")
    print(f"{'Method':<44}" + "".join(f"{'by ' + grouping:>17}" for grouping in GROUPINGS))
    for name, *cells in rows:
        print(f"{name:<44}" + "".join(f"{f'{points:+5d} ({correct / total:4.0%})':>17}" for points, correct in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--annotations", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--models", nargs="*", default=list(CLIP_MODELS), choices=list(CLIP_MODELS))
    args = parser.parse_args()

    planogram = load_cameras(args.planogram)
    annotations = json.loads(args.annotations.read_text())
    samples = [
        (camera, file, area_id, label)
        for camera, images in annotations.items()
        for file, areas in images.items()
        for area_id, label in areas.items()
    ]
    labels = [sample[3] for sample in samples]
    keys = [sample[:3] for sample in samples]
    groups = {
        "photo": np.array([f"{c}/{f}" for c, f, _ in keys]),
        "area": np.array([f"{c}/{a}" for c, _, a in keys]),
        "camera": np.array([c for c, _, _ in keys]),
    }
    stock_targets = np.array([label["stock_level"] for label in labels])
    compliance_targets = np.array([label["planogram_compliant"] for label in labels])
    total = len(samples)
    print(f"{total} labelled areas from {len(annotations)} camera(s); cutting crops…")

    crops = load_crops(planogram, args.cameras, annotations)
    labelled_crops = [crops[key] for key in keys]
    area_features = per_area_features(crops)

    stock_features = {
        "No image: label frequencies": np.zeros((total, 1)),
        "Colour: produce fraction": np.array([colour_features(c) for c in labelled_crops]),
        "Colour: HSV histogram": np.array([histogram_features(c) for c in labelled_crops]),
        "Per area: normalised fill + change from empty": np.array([area_features[k] for k in keys]),
        "All colour features": np.array([colour_features(c) + area_features[k] for c, k in zip(labelled_crops, keys)]),
    }
    compliance_features = {"No image: label frequencies": np.zeros((total, 1))}
    zero_shot_stock, zero_shot_compliance = {}, {}

    for model_name in args.models:
        print(f"Embedding crops with {model_name}…")
        clip = Clip(*CLIP_MODELS[model_name])
        embeddings = clip.images(labelled_crops)
        stock_features[f"{model_name} embedding"] = embeddings
        compliance_features[f"{model_name} embedding"] = embeddings

        logits = product_logits(clip, embeddings, samples, planogram)
        # Compliant when the planned product beats every other product on the camera.
        zero_shot_compliance[f"{model_name} zero-shot product"] = 1 / (1 + np.exp(-(logits[:, 0] - logits[:, 1])))
        compliance_features[f"{model_name} product scores, calibrated"] = logits

        for generic in (True, False):
            wording = "generic" if generic else "product"
            stock = stock_logits(clip, embeddings, samples, planogram, generic)
            zero_shot_stock[f"{model_name} zero-shot ({wording} prompts)"] = softmax(stock)
            stock_features[f"{model_name} zero-shot ({wording}), calibrated"] = stock

    stock_rows = []
    for name, features in stock_features.items():
        cells = []
        for grouping in GROUPINGS:
            probabilities = cross_validated(features, stock_targets, groups[grouping], LEVELS)
            cases, points = stock_result(probabilities, labels)
            cells.append((points, cases["stock_correct"]))
        stock_rows.append((name, *cells))
    for name, probabilities in zero_shot_stock.items():
        cases, points = stock_result(probabilities, labels)
        stock_rows.append((name, *[(points, cases["stock_correct"])] * len(GROUPINGS)))

    cases, points = compliance_result(np.ones(total), labels)
    compliance_rows = [("Always compliant", *[(points, cases["identification_correct"])] * len(GROUPINGS))]
    for name, features in compliance_features.items():
        cells = []
        for grouping in GROUPINGS:
            probabilities = cross_validated(features, compliance_targets, groups[grouping], [False, True])[:, 1]
            cases, points = compliance_result(probabilities, labels)
            cells.append((points, cases["identification_correct"]))
        compliance_rows.append((name, *cells))
    for name, probabilities in zero_shot_compliance.items():
        cases, points = compliance_result(probabilities, labels)
        compliance_rows.append((name, *[(points, cases["identification_correct"])] * len(GROUPINGS)))

    print_table("Stock level", stock_rows, total)
    print_table("Compliance", compliance_rows, total)


if __name__ == "__main__":
    main()
