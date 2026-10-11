r"""
Shelf monitoring for the "Fresh eyes on fresh produce" challenge.

Input layout (one crop per planogram area, named after its area_id):
    data\areas\<camera>\<timestamp>\<area_id>.jpg      e.g. data\areas\6386\1751017235\6386_1.jpg
plus planogram.json.

Output: submission.json in exactly the annotation format of the challenge:
    { "<camera>": { "<timestamp>.jpg": { "<area_id>": {
          "planogram_compliant": bool, "stock_level": "empty|half|full", "stock_value": 0|0.5|1 } } } }
and diagnostics.csv + overlay images to check what the code saw.

Usage:
    python shelf_monitor.py
    python shelf_monitor.py --cameras 6385 6386
    python shelf_monitor.py --areas 6386_1            # only analyse (and output) these areas
"""
import argparse
import base64
import csv
import datetime as dt
import html
import json
import os
import sys
import webbrowser

import cv2
import numpy as np

# ------------------------------------------------------------------ settings
# Colour of "product" pixels (OpenCV HSV, H 0-179). Anything saturated that is not the teal crate.
MIN_SAT = 45
MIN_VAL = 25
NON_PRODUCT_H = (70, 125)      # teal crates and blue reflections on dark metal shelves; no produce in the planogram has these hues
                               # (purple produce such as aubergine / red cabbage / red onion is H > 125 and is kept)

# Stock level: coverage of product pixels relative to a "full" reference coverage of that area.
DEFAULT_FULL_COVERAGE = 0.70   # whole-crop coverage measured on a full apple shelf; used when an area has too few images
MIN_IMAGES_FOR_AREA_REF = 8    # with at least this many images an area gets its own reference (90th percentile)
REF_FLOOR = 0.80               # ...but never below this fraction of DEFAULT_FULL_COVERAGE (avoids calling a half shelf "full")
LEVEL_CENTRES = {0.0: 0.10, 0.5: 0.5, 1.0: 1.0}  # expected relative coverage of empty / half / full.
                               # "empty" sits above 0 because staff call a shelf empty when only a few items are left
                               # (few enough for the next customer to clear them)
LEVEL_SIGMA = 0.20             # how fuzzy the link between coverage and label is
LEVEL_PRIOR = {0.0: 1 / 3, 0.5: 1 / 3, 1.0: 1 / 3}

# Scoring from the challenge PDF (points per area).
ID_RIGHT, ID_WRONG = +1, -2
STOCK_RIGHT, STOCK_BELOW, STOCK_ABOVE = +1, -2, -3

# Compliance (is the product the planogram expects?): flag only when the colour distribution of the
# area is clearly unlike what that product normally looks like in the same camera that day.
# A wrong flag costs -2 and a right one gains +1, so we only flag when very confident.
MIN_SAMPLES_FOR_COMPLIANCE = 6
COMPLIANCE_MIN_OVERLAP = 0.25  # histogram overlap below this -> wrong product
MIN_VISIBLE_FOR_COMPLIANCE = 0.30   # needs at least this relative fill to judge
HUE_BINS = 18
# ---------------------------------------------------------------------------

LEVEL_NAME = {0.0: "empty", 0.5: "half", 1.0: "full"}


def product_mask(im):
    H, S, V = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
    not_product = (H >= NON_PRODUCT_H[0]) & (H <= NON_PRODUCT_H[1])
    m = ((S >= MIN_SAT) & (V >= MIN_VAL) & ~not_product).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def hue_hist(im, mask):
    H = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 0][mask > 0]
    if H.size < 50:
        return None
    h, _ = np.histogram(H, bins=HUE_BINS, range=(0, 180))
    return h / h.sum()


def best_level(rel_cov):
    """Pick the stock level that maximises expected points, given the PDF's asymmetric penalties."""
    levels = sorted(LEVEL_CENTRES)
    like = np.array([np.exp(-((rel_cov - LEVEL_CENTRES[l]) ** 2) / (2 * LEVEL_SIGMA ** 2)) * LEVEL_PRIOR[l]
                     for l in levels])
    post = like / like.sum()

    def pay(g, t):
        return STOCK_RIGHT if g == t else (STOCK_BELOW if g < t else STOCK_ABOVE)

    ev = [sum(post[j] * pay(g, t) for j, t in enumerate(levels)) for g in levels]
    return levels[int(np.argmax(ev))]


def load_planogram(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["Frutas e Legumes"]["cameras"]


def timestamp_folders(cam_dir):
    return sorted(d for d in os.listdir(cam_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(cam_dir, d)))


def analyse(data_root, planogram, cameras, only_areas):
    """Return a list of records, one per (camera, timestamp, area)."""
    recs = []
    for cam in cameras:
        cam_dir = os.path.join(data_root, cam)
        if cam not in planogram:
            print(f"[{cam}] not in planogram.json, skipped")
            continue
        if not os.path.isdir(cam_dir):
            print(f"[{cam}] no folder {cam_dir}, skipped")
            continue
        for ts in timestamp_folders(cam_dir):
            for area in planogram[cam]["products"]:
                aid = area["area_id"]
                if only_areas and aid not in only_areas:
                    continue
                path = os.path.join(cam_dir, ts, aid + ".jpg")
                rec = dict(camera=cam, timestamp=ts, area_id=aid, product_id=area["product_id"],
                           product_name=area["product_name"], path=path, cov=None, hist=None, im=None, mask=None)
                im = cv2.imread(path) if os.path.isfile(path) else None
                if im is not None:
                    m = product_mask(im)
                    rec.update(cov=float(m.mean()), hist=hue_hist(im, m), im=im, mask=m)
                recs.append(rec)
    return recs


def decide(recs):
    # ---- reference "full" coverage per (camera, area)
    by_area = {}
    for r in recs:
        if r["cov"] is not None:
            by_area.setdefault((r["camera"], r["area_id"]), []).append(r["cov"])
    ref = {}
    for k, covs in by_area.items():
        if len(covs) >= MIN_IMAGES_FOR_AREA_REF:
            ref[k] = max(float(np.percentile(covs, 90)), REF_FLOOR * DEFAULT_FULL_COVERAGE)
        else:
            ref[k] = DEFAULT_FULL_COVERAGE

    # ---- stock level
    for r in recs:
        if r["cov"] is None:
            r.update(rel=None, level=0.0, note="no image: safe default")
        else:
            rel = r["cov"] / ref[(r["camera"], r["area_id"])]
            r.update(rel=rel, level=best_level(min(rel, 1.3)), note="")

    # ---- compliance: typical colour histogram per (camera, product) over the day
    samples = {}
    for r in recs:
        if r["hist"] is not None and r["rel"] is not None and r["rel"] >= MIN_VISIBLE_FOR_COMPLIANCE:
            samples.setdefault((r["camera"], r["product_id"]), []).append(r["hist"])
    typical = {}
    for k, hs in samples.items():
        if len(hs) >= MIN_SAMPLES_FOR_COMPLIANCE:
            t = np.median(np.stack(hs), axis=0)
            typical[k] = t / t.sum()
    for r in recs:
        r.update(compliant=True, overlap=None)
        k = (r["camera"], r["product_id"])
        if (r["hist"] is not None and r["rel"] is not None and r["rel"] >= MIN_VISIBLE_FOR_COMPLIANCE
                and k in typical):
            ov = float(np.minimum(r["hist"], typical[k]).sum())
            r["overlap"] = ov
            if ov < COMPLIANCE_MIN_OVERLAP:
                r["compliant"] = False
    return recs


def build_submission(recs):
    out = {}
    for r in recs:
        lvl = r["level"]
        out.setdefault(r["camera"], {}).setdefault(r["timestamp"] + ".jpg", {})[r["area_id"]] = {
            "planogram_compliant": bool(r["compliant"]),
            "stock_level": LEVEL_NAME[lvl],
            "stock_value": int(lvl) if lvl in (0.0, 1.0) else lvl,
        }
    return out


def write_outputs(recs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "submission.json"), "w", encoding="utf-8") as f:
        json.dump(build_submission(recs), f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "timestamp", "area_id", "product_name", "coverage", "relative_fill",
                    "stock_level", "hue_overlap", "compliant", "note"])
        for r in recs:
            fmt = lambda v: "" if v is None else round(v, 3)
            w.writerow([r["camera"], r["timestamp"], r["area_id"], r["product_name"], fmt(r["cov"]),
                        fmt(r["rel"]), LEVEL_NAME[r["level"]], fmt(r["overlap"]), r["compliant"], r["note"]])
    for r in recs:
        if r["im"] is None:
            continue
        d = os.path.join(out_dir, "overlays", r["camera"])
        os.makedirs(d, exist_ok=True)
        ov = r["im"].copy()
        ov[r["mask"] == 0] = (ov[r["mask"] == 0] * 0.2).astype(np.uint8)
        img = cv2.resize(np.hstack([r["im"], ov]), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        txt = f'{LEVEL_NAME[r["level"]]}' + ("" if r["compliant"] else "  WRONG PRODUCT")
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(d, f'{r["timestamp"]}_{r["area_id"]}.png'), img)


def _data_uri(im):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _when(ts):
    try:
        return dt.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError, OSError):
        return ts


REPORT_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--card:#f5f5f7;--line:#d2d2d7;--acc:#d70015}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#f5f5f7;--mut:#a1a1a6;--card:#2c2c2e;--line:#3a3a3c;--acc:#ff6961}}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 20px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px;min-width:110px}
.stat b{display:block;font-size:24px}.stat span{color:var(--mut);font-size:13px}
.head{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:14px;
 border-top:1px solid var(--line);padding-top:20px}
h2{font-size:18px;margin:0}
.switch{display:inline-flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.switch input{position:absolute;opacity:0}
.track{width:44px;height:26px;border-radius:13px;background:var(--line);position:relative;transition:background .15s}
.track::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:#fff;
 transition:transform .15s;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.switch input:checked+.track{background:var(--acc)}.switch input:checked+.track::after{transform:translateX(18px)}
.switch input:focus-visible+.track{outline:2px solid var(--fg);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{display:block;width:100%;height:150px;object-fit:contain;background:#000}
.card div{padding:8px 12px 10px}.card b{display:block}.card small{color:var(--mut);display:block}
.none{color:var(--mut)}
"""


def write_report(recs, out_dir):
    """Self-contained HTML page: summary, then all empty shelves at the end behind a show/hide toggle."""
    seen = [r for r in recs if r["im"] is not None]
    empty = sorted((r for r in seen if r["level"] == 0.0), key=lambda r: (r["camera"], r["area_id"], r["timestamp"]))
    count = lambda lvl: sum(r["level"] == lvl for r in seen)
    wrong = sum(not r["compliant"] for r in seen)
    cards = "".join(
        '<figure class="card"><img alt="{a}" src="{src}"><div><b>{a}</b><small>{name}</small>'
        '<small>{when} &middot; {fill}% of full</small></div></figure>'.format(
            a=html.escape(r["area_id"]), name=html.escape(r["product_name"]), when=html.escape(_when(r["timestamp"])),
            fill=round(100 * r["rel"]), src=_data_uri(r["im"]))
        for r in empty)
    body = cards or '<p class="none">No empty shelves found.</p>'
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Shelf report</title>'
        f"<style>{REPORT_CSS}</style></head><body>"
        "<h1>Shelf report</h1>"
        f'<p class="sub">{len(seen)} areas analysed</p>'
        '<div class="stats">'
        f'<div class="stat"><b>{count(0.0)}</b><span>empty</span></div>'
        f'<div class="stat"><b>{count(0.5)}</b><span>half</span></div>'
        f'<div class="stat"><b>{count(1.0)}</b><span>full</span></div>'
        f'<div class="stat"><b>{wrong}</b><span>wrong product</span></div></div>'
        '<div class="head"><h2>Empty shelves (' + str(len(empty)) + ")</h2>"
        '<label class="switch"><input type="checkbox" id="toggle" checked><span class="track"></span>'
        "<span>Show empty shelves</span></label></div>"
        f'<section class="grid" id="empty">{body}</section>'
        "<script>var t=document.getElementById('toggle'),s=document.getElementById('empty');"
        "t.addEventListener('change',function(){s.hidden=!t.checked;});</script>"
        "</body></html>")
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "areas"))
    ap.add_argument("--planogram", default="planogram.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cameras", nargs="*", help="camera ids (default: all folders under --data)")
    ap.add_argument("--areas", nargs="*", help="only these area ids, e.g. 6386_1")
    ap.add_argument("--no-open", action="store_true", help="do not open report.html in the browser at the end")
    a = ap.parse_args()

    planogram = load_planogram(a.planogram)
    cams = a.cameras or sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    recs = decide(analyse(a.data, planogram, cams, set(a.areas or [])))
    if not recs:
        sys.exit("nothing found: check --data and --planogram")
    write_outputs(recs, a.out)

    n_img = sum(r["im"] is not None for r in recs)
    print(f"{len(recs)} areas written, {n_img} with a crop image, {len(recs) - n_img} filled with the safe default")
    for lvl in (0.0, 0.5, 1.0):
        print(f"  {LEVEL_NAME[lvl]:5s}: {sum(r['level'] == lvl for r in recs if r['im'] is not None)}")
    print(f"  wrong product: {sum(not r['compliant'] for r in recs)}")
    print(f"saved {os.path.join(a.out, 'submission.json')}")
    report_path = write_report(recs, a.out)
    print(f"saved {report_path}")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(report_path))


if __name__ == "__main__":
    main()