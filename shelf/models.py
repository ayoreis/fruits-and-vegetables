"""Models for stock level and compliance, built from small parts and registered by name.

A model is a list of stock-level parts, whose probabilities are averaged, plus a compliance part. Each part declares
the features it needs (see shelf/features.py). To try a new idea, write a part (or reuse one), add a function to
MODELS, and run:

    python -m shelf.predict --model <name> --evaluate                 # estimate its validation score
    python -m shelf.predict --model <name> --out-of-fold              # honest predictions to look at in the admin
    python -m shelf.predict --model ensemble patch-fill --evaluate    # compare models in one run

Each model's predictions are saved under output/predictions/, where the staff and admin pages can switch between them.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from shelf.score import STOCK_VALUES

LEVELS = list(STOCK_VALUES)


def logistic(regularisation=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=regularisation, max_iter=5000))


def probabilities(model, inputs, classes):
    """Class probabilities in the order of `classes`; classes the model never saw get zero."""
    result = np.zeros((len(inputs), len(classes)))
    result[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(inputs)
    return result


class AreaIdentity:
    """One-hot "which area is this", so a classifier can learn per-area habits; unseen areas stay neutral."""

    def fit(self, features):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(features["areas"])
        return self

    def __call__(self, features):
        return self.encoder.transform(features["areas"])


# ---------- Stock-level parts ----------


class LogisticStock:
    """Stock level from a logistic regression on the named features, plus the area identity."""

    def __init__(self, *names, area=True):
        self.names = names
        self.area = area
        self.needs = names + (("areas",) if area else ())

    def _inputs(self, features):
        parts = [features[name].reshape(len(features[name]), -1).astype(np.float32) for name in self.names]
        if self.area:
            parts.append(self.areas(features))
        return np.hstack(parts)

    def fit(self, features, stock):
        if self.area:
            self.areas = AreaIdentity().fit(features)
        self.model = logistic().fit(self._inputs(features), stock)
        return self

    def predict(self, features):
        return probabilities(self.model, self._inputs(features), LEVELS)


class PatchFillStock:
    """Stock level from how much of the crop looks like produce, judged patch by patch.

    Every other model looks at a crop as one picture. This one asks the organisers' question directly, "how much
    of the area is covered?":

    1. A patch classifier learns what produce looks like from DINOv2 patch embeddings, with weak labels: patches
       of crops labelled full count as produce, patches of crops labelled empty as bare crate.
    2. Each crop is summarised by how its patches score: the mean, quantiles and share above one half.
    3. A small regression maps that summary, plus the area identity, to a stock level.
    """

    needs = ("dino_patches", "areas")

    def __init__(self, patches_per_crop=16, seed=0):
        self.patches_per_crop = patches_per_crop
        self.seed = seed

    def fill_summary(self, features):
        patches = features["dino_patches"].astype(np.float32)
        crops, cells, dimensions = patches.shape
        produce = self.patch_model.predict_proba(patches.reshape(-1, dimensions))[:, 1].reshape(crops, cells)
        quantiles = np.percentile(produce, [10, 25, 50, 75, 90], axis=1).T
        return np.column_stack([produce.mean(axis=1), quantiles, (produce > 0.5).mean(axis=1)])

    def fit(self, features, stock):
        patches = features["dino_patches"].astype(np.float32)
        rng = np.random.default_rng(self.seed)
        clear = np.flatnonzero(stock != "half")  # half-full crates mix both kinds of patch, so leave them out
        cells = rng.integers(0, patches.shape[1], size=(len(clear), self.patches_per_crop))
        sample = patches[clear[:, None], cells].reshape(-1, patches.shape[2])
        is_produce = np.repeat(stock[clear] == "full", self.patches_per_crop)
        self.patch_model = logistic(0.1).fit(sample, is_produce)

        self.areas = AreaIdentity().fit(features)
        self.model = logistic().fit(np.hstack([self.fill_summary(features), self.areas(features)]), stock)
        return self

    def predict(self, features):
        return probabilities(self.model, np.hstack([self.fill_summary(features), self.areas(features)]), LEVELS)


# ---------- Compliance parts ----------


class LogisticCompliance:
    """Chance an area holds the planned product, from a logistic regression on the named features."""

    def __init__(self, *names):
        self.names = names
        self.needs = names

    def _inputs(self, features):
        return np.hstack([features[name].reshape(len(features[name]), -1).astype(np.float32) for name in self.names])

    def fit(self, features, compliant):
        # Labels with only one answer (e.g. a camera that's always right) can't train a classifier.
        self.only = compliant[0] if len(set(compliant)) == 1 else None
        if self.only is None:
            self.model = logistic().fit(self._inputs(features), compliant)
        return self

    def predict(self, features):
        if self.only is not None:
            return np.full(len(next(iter(features.values()))), float(self.only))
        return probabilities(self.model, self._inputs(features), [False, True])[:, 1]


# ---------- Whole models ----------


class ShelfModel:
    """Averages the stock-level parts' probabilities and adds the compliance part."""

    def __init__(self, stock, compliance):
        self.stock_parts = stock
        self.compliance = compliance
        self.needs = tuple(dict.fromkeys(n for part in [*stock, compliance] for n in part.needs))

    def fit(self, features, labels):
        stock = np.array([label["stock_level"] for label in labels])
        compliant = np.array([label["planogram_compliant"] for label in labels])
        # These models are small enough that one thread fits them about 10x faster than sixteen; evaluation runs
        # several fits side by side in separate processes instead.
        with threadpool_limits(limits=1):
            for part in self.stock_parts:
                part.fit(features, stock)
            self.compliance.fit(features, compliant)
        return self

    def predict(self, features):
        """For each crop: the chance it is compliant, and {stock level: chance}."""
        with threadpool_limits(limits=1):
            stock = np.mean([part.predict(features) for part in self.stock_parts], axis=0)
            compliant = self.compliance.predict(features)
        return [(c, dict(zip(LEVELS, s))) for c, s in zip(compliant, stock)]


def clip_compliance():
    return LogisticCompliance("clip", "product_scores")


def ensemble():
    """The current best: CLIP + colour histogram and DINOv2, each with the area identity."""
    return ShelfModel([LogisticStock("clip", "histogram"), LogisticStock("dino")], clip_compliance())


def clip_only():
    """Cheapest: one image model (about 1.1 s of CPU per photo instead of 2.7 s)."""
    return ShelfModel([LogisticStock("clip", "histogram")], clip_compliance())


def patch_fill():
    """New: how much of each crop looks like produce, patch by patch."""
    return ShelfModel([PatchFillStock()], clip_compliance())


def dino_grid():
    """DINOv2 features of each quarter of the crop, so the classifier sees where the produce is."""
    return ShelfModel([LogisticStock("dino_grid")], clip_compliance())


def ensemble_patch_fill():
    """The ensemble with patch fill as a third opinion."""
    return ShelfModel(
        [LogisticStock("clip", "histogram"), LogisticStock("dino"), PatchFillStock()], clip_compliance()
    )


MODELS = {
    "ensemble": ensemble,
    "clip-only": clip_only,
    "patch-fill": patch_fill,
    "dino-grid": dino_grid,
    "ensemble+patch-fill": ensemble_patch_fill,
}
DEFAULT_MODEL = "ensemble"
