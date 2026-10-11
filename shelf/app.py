"""Local website with a staff page and admin pages.

Run inside `nix develop` with `python -m shelf.app`, then open http://localhost:8000.

- /staff: what store staff need to do, from the latest predictions for each camera.
- /admin/annotate/<camera>/<photo>/<area>: label photos by hand.
- /admin/compare/<camera>/<photo>/<area>: compare predictions with the labels.

Filters and the selected task live in the URL, so reloading or sharing a link keeps your place.
"""

import argparse
import json
import os
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from shelf.crops import cut_area
from shelf.planogram import area_quad, load_cameras

# The staff and admin pages.
ROOT = Path(__file__).parent / "pages"
IMAGE_PATH = re.compile(r"^/image/(\d+)/(\d+)\.jpg$")
CROP_PATH = re.compile(r"^/crop/(\d+)/(\d+)/(\w+)\.jpg$")
# The admin page at /admin[/<view>[/<camera>[/<photo>[/<area>]]]], and its old address without /admin.
ADMIN_PATH = re.compile(r"^/admin(/(annotate|compare)(/\d+(/\d+(/\w+)?)?)?)?/?$")
OLD_ADMIN_PATH = re.compile(r"^/(annotate|compare)(/\d+(/\d+(/\w+)?)?)?/?$")
STAFF_PATH = re.compile(r"^/staff/?$")
# Tasks staff marked done, one per area of a specific photo.
done_lock = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, options, planogram, **kwargs):
        self.options = options
        self.planogram = planogram
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data):
        self.send(200, json.dumps(data).encode(), "application/json")

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def page(self, name):
        self.path = f"/{name}"
        return super().do_GET()

    def read_done(self):
        path = self.options.done
        return set(json.loads(path.read_text())) if path.exists() else set()

    def write_json(self, path, data):
        temporary = path.with_suffix(".json.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "w") as file:
            json.dump(data, file, indent=2)
        # Replace in one step so a crash mid-write never leaves a broken file.
        os.replace(temporary, path)

    def image_path(self, camera, timestamp):
        path = self.options.cameras / camera / f"{timestamp}.jpg"
        return path if camera in self.planogram and path.is_file() else None

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path == "/":
            return self.redirect("/staff")
        if STAFF_PATH.match(path):
            return self.page("staff.html")
        if ADMIN_PATH.match(path):
            return self.page("admin.html")
        if OLD_ADMIN_PATH.match(path):
            return self.redirect("/admin" + path + (f"?{url.query}" if url.query else ""))

        if path == "/api/state":
            return self.send_json(self.state(parse_qs(url.query).get("predictions", ["latest"])[0]))

        if match := IMAGE_PATH.match(path):
            if path := self.image_path(*match.groups()):
                return self.send(200, path.read_bytes(), "image/jpeg")

        if match := CROP_PATH.match(path):
            camera, timestamp, area_id = match.groups()
            path = self.image_path(camera, timestamp)
            area = next((a for a in self.planogram.get(camera, {}).get("products", []) if a["area_id"] == area_id), None)
            if path and area:
                quad = np.array(area_quad(area), dtype=np.float32)
                _, crop = cv2.imencode(".jpg", cut_area(cv2.imread(str(path)), quad))
                return self.send(200, crop.tobytes(), "image/jpeg")

        self.send(404, b"Not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/annotations":
            self.write_json(self.options.output, body)
            return self.send_json({"saved": True})
        if self.path == "/api/done":
            # body: {"task": "<camera>|<photo file>|<area id>", "done": true or false}
            with done_lock:
                done = self.read_done()
                (done.add if body["done"] else done.discard)(body["task"])
                self.write_json(self.options.done, sorted(done))
            return self.send_json({"saved": True})
        self.send(404, b"Not found", "text/plain")

    def prediction_files(self):
        """The predictions to choose from: the latest run, and each model's saved predictions."""
        files = {"latest": self.options.predictions}
        files.update({path.stem: path for path in sorted(self.options.predictions_dir.glob("*.json"))})
        return files

    def state(self, chosen="latest"):
        files = self.prediction_files()
        chosen = chosen if chosen in files else "latest"
        output, predictions = self.options.output, files[chosen]
        return {
            "planogram": {
                camera: [
                    {
                        "area_id": area["area_id"],
                        "product_id": area["product_id"],
                        "product_name": area["product_name"],
                        "quad": area_quad(area),
                    }
                    for area in camera_data["products"]
                ]
                for camera, camera_data in self.planogram.items()
            },
            "images": {
                camera: sorted(path.name for path in (self.options.cameras / camera).glob("*.jpg"))
                for camera in sorted(self.planogram)
            },
            "annotations": json.loads(output.read_text()) if output.exists() else {},
            "predictions": json.loads(predictions.read_text()) if predictions.exists() else {},
            "predictionsFile": str(predictions),
            "predictionsName": chosen,
            "predictionOptions": list(files),
            "done": sorted(self.read_done()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--output", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("output/predictions"),
                        help="each model's predictions, selectable on the pages (from python -m shelf.predict --model)")
    parser.add_argument("--predictions", type=Path, default=Path("output/annotations.json"),
                        help="predictions to compare with (from python -m shelf.predict; use --out-of-fold for honest ones)")
    parser.add_argument("--done", type=Path, default=Path("output/staff-done.json"), help="tasks staff marked done")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()

    handler = partial(Handler, options=options, planogram=load_cameras(options.planogram))
    server = ThreadingHTTPServer(("127.0.0.1", options.port), handler)
    print(f"Staff page: http://localhost:{options.port}/staff · admin: http://localhost:{options.port}/admin "
          f"(labels saved to {options.output})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
