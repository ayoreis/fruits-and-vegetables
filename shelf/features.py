"""Features of area crops, computed on demand by name and cached.

Models declare the features they need (see shelf/models.py); only those are computed. Image-model features are
cached in output/cache, so each crop goes through each model once, and the models only load when something is missing.

    clip            CLIP ViT-B/32 embedding of the whole crop
    dino            DINOv2 ViT-S/14 embedding of the whole crop
    dino_patches    DINOv2 embeddings of an 8x8 grid of patches across the crop
    dino_grid       the patch embeddings averaged into a 2x2 grid
    histogram       HSV colour histogram
    product_scores  CLIP's score for the planned product, and its best score for the camera's other products
    areas           which area the crop is (camera/area id), for per-area adjustments
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from shelf.products import english_name

FEATURES = ("clip", "dino", "dino_patches", "dino_grid", "histogram", "product_scores", "areas")
CLIP_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")
DINO_MODEL = "vit_small_patch14_dinov2.lvd142m"
PATCH_GRID = 8
PAD_COLOUR = (124, 116, 104)  # roughly CLIP's mean pixel, so padding looks neutral
PRODUCT_TEMPLATES = ["a photo of {}.", "a supermarket crate of {}.", "{} on display in a supermarket."]


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


def pick_device(name="auto"):
    """The device to run the image models on: a GPU when PyTorch can use one, otherwise the CPU (all cores).

    "cuda" covers NVIDIA GPUs and AMD GPUs with a ROCm build of PyTorch; "mps" covers Apple silicon.
    """
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """CLIP and DINOv2 embeddings, all normalised to unit length."""

    def __init__(self, device="auto"):
        import open_clip
        import timm
        import torch

        self.torch = torch
        self.device = pick_device(device)
        print(f"Running the image models on {self.device}")
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
        self.clip.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL[0])
        self.dino = timm.create_model(DINO_MODEL, pretrained=True, num_classes=0, img_size=224)
        self.dino.eval().to(self.device)
        config = timm.data.resolve_model_data_config(self.dino)
        config.update(input_size=(3, 224, 224), crop_pct=1.0)  # the pretrained default is 518 px
        self.dino_preprocess = timm.data.create_transform(**config, is_training=False)

    def _normalised(self, embeddings):
        return (embeddings / embeddings.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def _batches(self, preprocess, crops, run):
        images = [padded(crop) for crop in crops]
        outputs = []
        for start in range(0, len(images), 32):
            batch = self.torch.stack([preprocess(image) for image in images[start:start + 32]]).to(self.device)
            with self.torch.no_grad():
                outputs.append(run(batch))
        return self._normalised(self.torch.cat(outputs))

    def clip_images(self, crops):
        return self._batches(self.clip_preprocess, crops, self.clip.encode_image)

    def dino_images(self, crops):
        return self._batches(self.dino_preprocess, crops, self.dino)

    def dino_patches(self, crops):
        """An embedding for each cell of a PATCH_GRID x PATCH_GRID grid over the crop: (crops, cells, dimensions)."""

        def run(batch):
            tokens = self.dino.forward_features(batch)[:, self.dino.num_prefix_tokens:]
            side = int(tokens.shape[1] ** 0.5)
            grid = tokens.reshape(len(batch), side, side, -1).permute(0, 3, 1, 2)
            pooled = self.torch.nn.functional.adaptive_avg_pool2d(grid, PATCH_GRID)
            return pooled.flatten(2).permute(0, 2, 1)

        return self._batches(self.dino_preprocess, crops, run).astype(np.float16)

    def text(self, name):
        """One CLIP embedding for a product name, averaged over a few prompt wordings."""
        with self.torch.no_grad():
            tokens = self.tokenizer([t.format(name) for t in PRODUCT_TEMPLATES]).to(self.device)
            embeddings = self._normalised(self.clip.encode_text(tokens))
        mean = embeddings.mean(axis=0)
        return mean / np.linalg.norm(mean)


class FeatureExtractor:
    """Named features for area crops keyed by (camera, photo file, area id); see the module docstring."""

    def __init__(self, planogram, cache=Path("output/cache"), device="auto"):
        self.planogram = planogram
        self.cache = cache
        self.device = device
        self._embedder = None

    @property
    def embedder(self):
        self._embedder = self._embedder or Embedder(self.device)
        return self._embedder

    def _cached(self, name, ids, compute):
        path = self.cache / f"{name}.npz"
        stored = dict(np.load(path)) if path.exists() else {}
        missing = [id_ for id_ in dict.fromkeys(ids) if id_ not in stored]
        if missing:
            print(f"Computing {name} for {len(missing)} crops…")
            stored.update(zip(missing, compute(missing)))
            self.cache.mkdir(parents=True, exist_ok=True)
            np.savez(path, **stored)
        return np.array([stored[id_] for id_ in ids])

    def __call__(self, keys, crops, names=FEATURES):
        ids = ["|".join(key) for key in keys]
        crop_by_id = dict(zip(ids, crops))
        crops_for = lambda missing: [crop_by_id[id_] for id_ in missing]
        features = {}

        def need(name):
            if name in features:
                return features[name]
            if name == "clip":
                value = self._cached("clip-b32", ids, lambda m: self.embedder.clip_images(crops_for(m)))
            elif name == "dino":
                value = self._cached("dinov2-s14", ids, lambda m: self.embedder.dino_images(crops_for(m)))
            elif name == "dino_patches":
                value = self._cached("dinov2-s14-patches", ids, lambda m: self.embedder.dino_patches(crops_for(m)))
            elif name == "dino_grid":
                patches = need("dino_patches").astype(np.float32)
                grid = patches.reshape(len(patches), PATCH_GRID, PATCH_GRID, -1)
                half = PATCH_GRID // 2
                quarters = [grid[:, r:r + half, c:c + half].mean(axis=(1, 2)) for r in (0, half) for c in (0, half)]
                value = np.concatenate(quarters, axis=1)
            elif name == "histogram":
                value = np.array([histogram_features(crop) for crop in crops])
            elif name == "product_scores":
                value = self.product_scores(need("clip"), keys)
            elif name == "areas":
                value = np.array([f"{camera}/{area_id}" for camera, _, area_id in keys]).reshape(-1, 1)
            else:
                raise ValueError(f"Unknown feature {name!r}; known: {', '.join(FEATURES)}")
            features[name] = value
            return value

        for name in names:
            need(name)
        return {name: features[name] for name in names}

    def product_scores(self, clip, keys):
        """CLIP's score for the planned product, and its best score for any other product on the same camera."""
        names = sorted({english_name(p["product_name"]) for c in self.planogram.values() for p in c["products"]})
        text = dict(zip(names, self._cached("clip-b32-text", names, lambda missing: [self.embedder.text(n) for n in missing])))
        rows = []
        for embedding, (camera, _, area_id) in zip(clip, keys):
            products = self.planogram[camera]["products"]
            planned = english_name(next(p for p in products if p["area_id"] == area_id)["product_name"])
            others = [100 * embedding @ text[name] for name in {english_name(p["product_name"]) for p in products} if name != planned]
            planned_score = 100 * embedding @ text[planned]
            rows.append([planned_score, max(others) if others else planned_score - 10])
        return np.array(rows)
