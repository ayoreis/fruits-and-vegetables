"""Shelf monitoring for the fruits and vegetables section: compliance and stock level per planogram area.

Run from the repository root inside `nix develop`:

    python -m shelf.predict      train on input/annotations.json and write output/annotations.json
                                 (--model picks one of shelf/models.py's models; --evaluate estimates the validation
                                 score; --out-of-fold gives honest predictions for the admin compare view)
    python -m shelf.score        score output/annotations.json against input/annotations.json
    python -m shelf.app          staff page and admin pages (annotate, compare) at http://localhost:8000
    python -m shelf.crops        write every area's straightened crop to output/areas
    python -m shelf.outlines     write every photo with its areas outlined to output/outlined

Modules: features.py (named, cached crop features), models.py (model parts and the registry of models),
decisions.py (answers that score the most points), evaluation.py (honest held-out scores), app.py (website).
"""
