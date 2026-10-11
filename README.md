# Shock - Flock, but for Shelves!

Shock is a computer-vision shelf-monitoring tool for the fruits and vegetables section. It analyses camera photos against a planogram and predicts, for every product area:

- whether the correct product is in the planned location;
- whether the area is empty, half-stocked, or full; and
- the corresponding stock value used by the challenge scoring rules.

This also includes a small local web application. Store staff can use it to review the latest predictions and mark tasks as completed, while administrators can annotate photos and compare model predictions with the labels.

## Tech Stack

- Python
- OpenCV and Pillow for image loading, perspective crops, and annotations
- NumPy and scikit-learn for feature processing and evaluation
- PyTorch, OpenCLIP, timm, and sentence-transformers for image features
- A dependency-free Python HTTP server for the staff and admin pages
- Nix flakes for a reproducible development environment

## How It Works

1. Camera photos are stored under `input/cameras/<camera>/`.
2. `input/planogram.json` describes the product areas and their quadrilateral coordinates in each camera view.
3. Human labels in `input/annotations.json` are used to train the available models.
4. Each model predicts product compliance and stock level for every planogram area.
5. Predictions are written to `output/annotations.json` and can be reviewed in the local web app.
6. Predictions can be scored against the annotations or evaluated using held-out photos/cameras.

The image features are cached in `output/cache`, so later runs do not need to recompute features that are already available.

## Installation

### Nix (recommended)

The repository includes a `flake.nix` that creates a Python development shell and installs the pckages from `requirements.txt`:

```bash
nix develop
```

The Nix development shell is configured for Linux and Apple Silicon macOS. It creates or refreshes the local `.venv` automatically when the Python version or requirements change.

### Python virtual environment

If Nix is not available, use Python and pip directly:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The default dependencies use CPU-only PyTorch wheels. A compatible GPU can be selected with `--device` when the installed PyTorch build supports it.

## Usage

Run commands from the repository root.

### Generate predictions

Train the default model and write predictions for all available camera photos:

```bash
python -m shelf.predict
```

This writes:

- `output/annotations.json` — the prediction file;
- `output/predictions/<model>.json` — predictions for the selected model; and
- cached image features under `output/cache`.

Choose a specific model with `--model`. The available model names are defined in [`shelf/models.py`](shelf/models.py):

```bash
python -m shelf.predict --model ensemble
```

Use `--device cpu`, `--device cuda`, or `--device mps` to select the image-model device. The default `--device auto` chooses an available accelerator when possible.

### Evaluate models

Estimate a model's validation score without writing a submission:

```bash
python -m shelf.predict --model ensemble --evaluate
```

Compare several models in one run:

```bash
python -m shelf.predict --model ensemble patch-fill --evaluate
```

By default evaluation hides photos. To evaluate against cameras the model has not seen labelled, use:

```bash
python -m shelf.predict --model ensemble --evaluate --split cameras
```

For an honest review of labelled photos, generate out-of-fold predictions:

```bash
python -m shelf.predict --model ensemble --out-of-fold
```

### Score predictions

Score the generated file against the labelled annotations:

```bash
python -m shelf.score
```

Include a per-camera breakdown:

```bash
python -m shelf.score --by-camera
```

To score a different prediction file:

```bash
python -m shelf.score path/to/predictions.json
```

### Start the local web app

Start the staff and admin pages:

```bash
python -m shelf.app
```

Open [http://localhost:8000/staff](http://localhost:8000/staff) for the staff view or [http://localhost:8000/admin](http://localhost:8000/admin) for the annotation and comparison tools.

The server uses these default paths:

| Option | Default |
| --- | --- |
| Planogram | `input/planogram.json` |
| Camera photos | `input/cameras` |
| Annotations | `input/annotations.json` |
| Latest predictions | `output/annotations.json` |
| Model predictions | `output/predictions` |
| Completed staff tasks | `output/staff-done.json` |
| Port | `8000` |

For example, to use another prot:

```bash
python -m shelf.app --port 8080
```

### Generate crops and outlined images

Create perspective-corrected crops for every planogram area:

```bash
python -m shelf.crops
```

Create copies of the camera photos with planogram areas and product names outlined:

```bash
python -m shelf.outlines
```

Both commands write generated files below `output/`.

## Project Layout

```text
input/
  cameras/             Camera photos grouped by camera id
  annotations.json     Human labels used for training and scoring
  planogram.json       Product areas and coordinates for each camera
shelf/
  app.py               Local staff and admin web application
  crops.py             Perspective-corrected area crops
  decisions.py         Converts model outputs into scored answers
  evaluation.py        Held-out and out-of-fold evaluation
  features.py          Cached image feature extraction
  models.py            Model implementations and model registry
  outlines.py          Planogram overlays for camera photos
  predict.py           Training and prediction command
  products.py          Product metadata
  score.py             Challenge scoring command
output/
  cache/               Cached image features
  predictions/         Per-model prediction files
```

Generated output is intentionally kept separate from the source data so that predictions, caches, and staff progress can be regenerated without changing the input annotations or planogram.

## Data and Prediction Format

Annotations and predictions are nested by camera, image filename, and planogram area:

```json
{
  "6215": {
    "1751001268.jpg": {
      "area-1": {
        "planogram_compliant": true,
        "stock_level": "half",
        "stock_value": 0.5
      }
    }
  }
}
```

The scorer awards points for both product identification and stock level. Predicting a stock level above the actual level is penalised more heavily than predicting a level below it, so the decision layer deliberately accounts for that bias.

## License

No license has been specified for this project yet.
