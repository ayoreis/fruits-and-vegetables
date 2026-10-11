"""
Shelf fullness for a layout like:
    output\areas\6386\1751017235\6386_1.jpg
    output\areas\6386\1751017900\6386_1.jpg
    output\areas\6385\1751016383\6385_1.jpg
Each timestamp folder holds one <area>_1.jpg. Every area is scored on its own.

Usage:
    python prototypes/shelf_fullness.py                 # all areas under output\areas
    python prototypes/shelf_fullness.py 6385 6386       # only these areas
"""
import csv
import datetime as dt
import glob
import os
import sys

import cv2
import numpy as np

# ---------------- settings ----------------
DATA_ROOT = os.path.join("output", "areas")
IMAGE_SUFFIX = "_1.jpg"              # file is <area><suffix>, e.g. 6386_1.jpg
OUT_DIR = os.path.join("output", "shelf_fullness")

# Optional: a known 100%-full image per area, e.g. {"6386": r"output\areas\6386\1751017235\6386_1.jpg"}
# Areas not listed use their fullest-looking image as the 100% reference.
REFERENCES = {}

# Shelf area as fractions of (width, height): top-left, top-right, bottom-right, bottom-left.
# Optional per-area overrides, e.g. {"6386": [(0.1, 0), (0.95, 0), (0.95, 0.9), (0.05, 0.9)]}
DEFAULT_POLY = [(0.20, 0.00), (0.98, 0.00), (0.98, 0.91), (0.06, 0.91)]
SHELF_POLYS = {}

# Apple colour (OpenCV HSV, H 0-179): red / orange / yellow
MIN_SAT = 45
MIN_VAL = 25
# ------------------------------------------


def apple_mask(im):
    H, S, V = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    warm = (H <= 35) | (H >= 155)
    crate = (H >= 70) & (H <= 100) & (S >= 60)           # teal crate
    m = (warm & (S >= MIN_SAT) & (V >= MIN_VAL) & ~crate).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def coverage(im, poly):
    h, w = im.shape[:2]
    pts = np.array([(int(x * w), int(y * h)) for x, y in poly], np.int32)
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [pts], 1)
    m = apple_mask(im) * roi
    return m.sum() / roi.sum(), m, pts


def ts_label(folder):
    try:
        return dt.datetime.fromtimestamp(int(folder)).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ""


def process_area(area):
    poly = SHELF_POLYS.get(area, DEFAULT_POLY)
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, area, "*", area + IMAGE_SUFFIX)))
    if not paths:
        print(f"[{area}] no images found")
        return []

    covs = {}
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            print(f"[{area}] unreadable: {p}")
            continue
        covs[p] = coverage(im, poly)

    ref_path = REFERENCES.get(area)
    if ref_path:
        ref_cov = coverage(cv2.imread(ref_path), poly)[0]
    else:
        ref_path = max(covs, key=lambda p: covs[p][0])
        ref_cov = covs[ref_path][0]
    print(f"[{area}] {len(covs)} images, 100% reference: {os.path.basename(os.path.dirname(ref_path))}")

    out = os.path.join(OUT_DIR, area)
    os.makedirs(out, exist_ok=True)
    rows = []
    for p, (cov, m, pts) in covs.items():
        pct = min(100.0, cov / ref_cov * 100)
        folder = os.path.basename(os.path.dirname(p))
        im = cv2.imread(p)
        ov = im.copy()
        ov[m == 0] = (ov[m == 0] * 0.2).astype(np.uint8)
        cv2.polylines(ov, [pts], True, (0, 255, 255), 1)
        ov = cv2.resize(np.hstack([im, ov]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        label = f"{pct:.0f}% full"
        cv2.putText(ov, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(ov, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(out, folder + ".png"), ov)
        rows.append((area, folder, ts_label(folder), round(pct, 1)))
        print(f"  {folder}  {pct:5.1f}%")
    return rows


def main():
    areas = sys.argv[1:] or sorted(d for d in os.listdir(DATA_ROOT)
                                   if os.path.isdir(os.path.join(DATA_ROOT, d)))
    all_rows = []
    for a in areas:
        all_rows += process_area(a)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["area", "timestamp", "datetime", "percent_full"])
        wr.writerows(all_rows)
    print(f"saved {os.path.join(OUT_DIR, 'results.csv')}")


if __name__ == "__main__":
    main()