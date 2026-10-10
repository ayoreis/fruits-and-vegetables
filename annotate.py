"""Local website for manually annotating the camera photos into an annotations.json file.

Run inside nix-shell with `python annotate.py`, then open http://localhost:8000
"""

import argparse
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from cut_areas import cut_area
from planogram import area_quad, load_cameras

ROOT = Path(__file__).parent
IMAGE_PATH = re.compile(r"^/image/(\d+)/(\d+)\.jpg$")
CROP_PATH = re.compile(r"^/crop/(\d+)/(\d+)/(\w+)\.jpg$")


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

    def image_path(self, camera, timestamp):
        path = self.options.cameras / camera / f"{timestamp}.jpg"
        return path if camera in self.planogram and path.is_file() else None

    def do_GET(self):
        if self.path == "/":
            self.path = "/annotate.html"
            return super().do_GET()

        if self.path == "/api/state":
            return self.send_json(self.state())

        if match := IMAGE_PATH.match(self.path):
            if path := self.image_path(*match.groups()):
                return self.send(200, path.read_bytes(), "image/jpeg")

        if match := CROP_PATH.match(self.path):
            camera, timestamp, area_id = match.groups()
            path = self.image_path(camera, timestamp)
            area = next((a for a in self.planogram.get(camera, {}).get("products", []) if a["area_id"] == area_id), None)
            if path and area:
                quad = np.array(area_quad(area), dtype=np.float32)
                _, crop = cv2.imencode(".jpg", cut_area(cv2.imread(str(path)), quad))
                return self.send(200, crop.tobytes(), "image/jpeg")

        self.send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/annotations":
            return self.send(404, b"Not found", "text/plain")

        annotations = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        output = self.options.output
        temporary = output.with_suffix(".json.tmp")
        with open(temporary, "w") as file:
            json.dump(annotations, file, indent=2)
        # Replace in one step so a crash mid-write never leaves a broken file.
        os.replace(temporary, output)
        self.send_json({"saved": True})

    def state(self):
        output = self.options.output
        annotations = json.loads(output.read_text()) if output.exists() else {}
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
            "annotations": annotations,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--output", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()

    handler = partial(Handler, options=options, planogram=load_cameras(options.planogram))
    server = ThreadingHTTPServer(("127.0.0.1", options.port), handler)
    print(f"Annotating into {options.output} — open http://localhost:{options.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
