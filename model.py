"""The shelf model: features for each area crop, and the classifiers for stock level and compliance.

- Stock level averages two logistic regressions: one on the CLIP ViT-B/32 embedding, a colour histogram and which
  area the crop is, and one on the DINOv2 ViT-S/14 embedding and which area it is.
- Compliance is a logistic regression on the CLIP embedding plus CLIP's zero-shot scores for the planned product
  against the camera's other products.

experiments/approaches.py has the comparison these were chosen from.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from products import english_name
from score import STOCK_VALUES

LEVELS = list(STOCK_VALUES)
CLIP_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")
DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
PAD_COLOUR = (124, 116, 104)  # roughly CLIP's mean pixel, so padding looks neutral
PRODUCT_TEMPLATES = ["a photo of {}.", "a supermarket crate of {}.", "{} on display in a supermarket."]


# ---------- Features ----------


def padded(crop):
    """The whole crop as an RGB image, letterboxed to a square so the models' centre crop loses nothing."""
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    side = max(image.size)
    square = Image.new("RGB", (side, side), PAD_COLOUR)
    square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return square


def histogram_features(crop):
    """HSV colour histogram of the pixels inside the photo (areas past its edge come out black)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = crop.any(axis=2).astype(np.uint8)
    histogram = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return histogram / max(histogram.sum(), 1)


class Embedder:
    """CLIP and DINOv2 image embeddings and CLIP text embeddings, all normalised to unit length."""

    def __init__(self):
        import open_clip
        import timm
        import torch

        self.torch = torch
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL[0])
        self.dino = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0, img_size=224)
        self.dino.eval()
        config = timm.data.resolve_model_data_config(self.dino)
        config.update(input_size=(3, 224, 224), crop_pct=1.0)  # the pretrained default is 518 px
        self.dino_preprocess = timm.data.create_transform(**config, is_training=False)

    def _normalised(self, embeddings):
        return (embeddings / embeddings.norm(dim=-1, keepdim=True)).numpy()

    def _images(self, model, preprocess, images):
        embeddings = []
        for start in range(0, len(images), 32):
            batch = self.torch.stack([preprocess(image) for image in images[start:start + 32]])
            with self.torch.no_grad():
                embeddings.append(model(batch))
        return self._normalised(self.torch.cat(embeddings))

    def images(self, crops):
        images = [padded(crop) for crop in crops]
        return (
            self._images(self.clip.encode_image, self.clip_preprocess, images),
            self._images(self.dino, self.dino_preprocess, images),
        )

    def text(self, name):
        """One embedding for a product name, averaged over a few prompt wordings."""
        with self.torch.no_grad():
            embeddings = self._normalised(self.clip.encode_text(self.tokenizer([t.format(name) for t in PRODUCT_TEMPLATES])))
        mean = embeddings.mean(axis=0)
        return mean / np.linalg.norm(mean)


class FeatureExtractor:
    """Features for area crops, keyed by (camera, photo file, area id).

    Embeddings are cached in `cache`, so only crops not seen before are run through the models, and the models are
    only loaded when something needs embedding.
    """

    def __init__(self, planogram, cache=Path("output/cache")):
        self.planogram = planogram
        self.cache = cache
        self._embedder = None

    @property
    def embedder(self):
        self._embedder = self._embedder or Embedder()
        return self._embedder

    def _cached(self, name, ids, compute):
        path = self.cache / f"{name}.npz"
        stored = dict(np.load(path)) if path.exists() else {}
        missing = [id_ for id_ in dict.fromkeys(ids) if id_ not in stored]
        if missing:
            stored.update(zip(missing, compute(missing)))
            self.cache.mkdir(parents=True, exist_ok=True)
            np.savez(path, **stored)
        return np.array([stored[id_] for id_ in ids])

    def __call__(self, keys, crops):
        ids = ["|".join(key) for key in keys]
        crop_by_id = dict(zip(ids, crops))
        embedded = {}

        def embed(model):
            def compute(missing):
                if not embedded.keys() >= set(missing):
                    print(f"Embedding {len(missing)} crops…")
                    clip, dino = self.embedder.images([crop_by_id[id_] for id_ in missing])
                    for id_, c, d in zip(missing, clip, dino):
                        embedded[id_] = {"clip": c, "dino": d}
                return [embedded[id_][model] for id_ in missing]

            return compute

        clip = self._cached("clip-b32", ids, embed("clip"))
        dino = self._cached("dinov2-s14", ids, embed("dino"))
        names = sorted({english_name(p["product_name"]) for c in self.planogram.values() for p in c["products"]})
        text = dict(zip(names, self._cached("clip-b32-text", names, lambda missing: [self.embedder.text(n) for n in missing])))

        return {
            "clip": clip,
            "dino": dino,
            "histogram": np.array([histogram_features(crop) for crop in crops]),
            "product_scores": self.product_scores(clip, keys, text),
            "areas": np.array([f"{camera}/{area_id}" for camera, _, area_id in keys]).reshape(-1, 1),
        }

    def product_scores(self, clip, keys, text):
        """CLIP's score for the planned product, and its best score for any other product on the same camera."""
        rows = []
        for embedding, (camera, _, area_id) in zip(clip, keys):
            products = self.planogram[camera]["products"]
            planned = english_name(next(p for p in products if p["area_id"] == area_id)["product_name"])
            names = {english_name(p["product_name"]) for p in products}
            others = [100 * embedding @ text[name] for name in names if name != planned]
            planned_score = 100 * embedding @ text[planned]
            rows.append([planned_score, max(others) if others else planned_score - 10])
        return np.array(rows)


# ---------- Classifiers ----------


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def probabilities(model, features, classes):
    """Class probabilities in the order of `classes`; classes the model never saw get zero."""
    result = np.zeros((len(features), len(classes)))
    result[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(features)
    return result


class ShelfModel:
    def fit(self, features, labels):
        stock = np.array([label["stock_level"] for label in labels])
        compliant = np.array([label["planogram_compliant"] for label in labels])
        # Areas seen in training get their own learned adjustment; new areas are left neutral.
        self.area_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(features["areas"])
        self.stock_models = [classifier().fit(inputs, stock) for inputs in self._stock_inputs(features)]
        self.only_compliance = compliant[0] if len(set(compliant)) == 1 else None
        if self.only_compliance is None:
            self.compliance_model = classifier().fit(self._compliance_inputs(features), compliant)
        return self

    def _stock_inputs(self, features):
        areas = self.area_encoder.transform(features["areas"])
        return [
            np.hstack([features["clip"], features["histogram"], areas]),
            np.hstack([features["dino"], areas]),
        ]

    def _compliance_inputs(self, features):
        return np.hstack([features["clip"], features["product_scores"]])

    def predict(self, features):
        """For each crop: the chance it is compliant, and {stock level: chance}."""
        stock = np.mean(
            [probabilities(m, inputs, LEVELS) for m, inputs in zip(self.stock_models, self._stock_inputs(features))],
            axis=0,
        )
        if self.only_compliance is None:
            compliant = probabilities(self.compliance_model, self._compliance_inputs(features), [False, True])[:, 1]
        else:
            compliant = np.full(len(stock), float(self.only_compliance))
        return [(c, dict(zip(LEVELS, s))) for c, s in zip(compliant, stock)]
