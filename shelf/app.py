"""Local website with a staff page and admin pages.

Run inside `nix develop` with `python -m shelf.app`, then open http://localhost:8000.

- /staff: what store staff need to do, from the latest predictions for each camera.
- /admin/annotate/<camera>/<photo>/<area>: label photos by hand.
- /admin/compare/<camera>/<photo>/<area>: compare predictions with the labels.
- /admin/run: run shelf.predict on the server, browse past runs, and see each run's stats.

Filters and the selected task live in the URL, so reloading or sharing a link keeps your place.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from shelf.crops import cut_area
from shelf.models import DEFAULT_MODEL, MODELS
from shelf.planogram import area_quad, load_cameras
from shelf.score import STOCK_VALUES

# The staff and admin pages.
ROOT = Path(__file__).parent / "pages"
# The repository root: predict runs here so its relative output/cache paths land in the right place.
REPO = Path(__file__).resolve().parent.parent
IMAGE_PATH = re.compile(r"^/image/(\d+)/(\d+)\.jpg$")
CROP_PATH = re.compile(r"^/crop/(\d+)/(\d+)/(\w+)\.jpg$")
# The admin page at /admin[/<view>[/<camera>[/<photo>[/<area>]]]], and its old address without /admin.
# (/admin/stats is kept as a redirect to /admin/run, which now shows each run's stats.)
ADMIN_PATH = re.compile(r"^/admin(/(annotate|compare|run)(/\d+(/\d+(/\w+)?)?)?)?/?$")
OLD_ADMIN_PATH = re.compile(r"^/(annotate|compare|run)(/\d+(/\d+(/\w+)?)?)?/?$")
STAFF_PATH = re.compile(r"^/staff/?$")
# Tasks staff marked done, one per area of a specific photo.
done_lock = threading.Lock()
# Background predict runs, newest last; at most one runs at a time.
runs_lock = threading.Lock()
runs = {}
run_number = [0]
running_run = [None]
MAX_RUN_LOG = 100_000
MAX_RUNS = 20
DEVICES = ("auto", "cpu", "cuda", "mps")
UPLOAD_NAME = re.compile(r"^[\w][\w.\-]{0,100}\.jpg$")
MAX_UPLOAD = 50 * 1024 * 1024


def run_worker(job_id, command, runs_file, versioned_stem):
    """Run predict in the repository root, streaming its output into the job's log.

    Predict modes write straight to their versioned file (see --output-only), so later runs
    never rewrite history; job["file"] names the run's own output (None when it kept nothing).
    """
    try:
        process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as error:
        with runs_lock:
            job = runs[job_id]
            job["log"] += f"Failed to start: {error}\n"
            job["status"], job["finished"] = "error", time.time()
            if running_run[0] == job_id:
                running_run[0] = None
        save_runs(runs_file)
        return
    with process.stdout:
        for line in process.stdout:
            with runs_lock:
                job = runs[job_id]
                job["log"] += line
                job["log"] = job["log"][-MAX_RUN_LOG:]
    process.wait()
    with runs_lock:
        job = runs[job_id]
        job["status"] = "done" if process.returncode == 0 else "error"
        job["returncode"] = process.returncode
        job["finished"] = time.time()
        if job["status"] == "done":
            job["file"] = versioned_stem
        if running_run[0] == job_id:
            running_run[0] = None
    save_runs(runs_file)


def load_runs(runs_file):
    """Past runs back into memory; anything still marked running died with the last server."""
    try:
        stored = json.loads(Path(runs_file).read_text()).get("runs", [])
    except (OSError, ValueError):
        return
    with runs_lock:
        for job in stored[-MAX_RUNS:]:
            if job.get("status") == "running":
                job["status"] = "error"
                job["log"] = (job.get("log") or "") + "\nServer restarted while this run was going.\n"
                job["finished"] = job.get("finished") or time.time()
            runs[job["id"]] = job
        if runs:
            run_number[0] = max(runs)


def save_runs(runs_file):
    """Past runs to disk, so the history sidebar survives a restart; drops the oldest beyond MAX_RUNS."""
    with runs_lock:
        finished = sorted(job_id for job_id in runs if runs[job_id]["status"] != "running")
        for job_id in finished[:-MAX_RUNS + 1] if len(finished) >= MAX_RUNS else []:
            del runs[job_id]
        payload = {"runs": [runs[job_id] for job_id in sorted(runs)]}
    try:
        path = Path(runs_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        os.replace(temporary, path)
    except OSError:
        pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, options, planogram, **kwargs):
        self.options = options
        self.planogram = planogram
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def send(self, status, body, content_type, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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
        for base in (self.options.cameras, self.options.uploads):
            path = Path(base) / camera / f"{timestamp}.jpg"
            if camera in self.planogram and path.is_file():
                return path
        return None

    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path == "/":
            return self.redirect("/staff")
        if STAFF_PATH.match(path):
            return self.page("staff.html")
        if path == "/admin/stats" or path.startswith("/admin/stats/"):
            return self.redirect("/admin/run" + (f"?{url.query}" if url.query else ""))
        if path == "/stats" or path == "/stats/":
            return self.redirect("/admin/run" + (f"?{url.query}" if url.query else ""))
        if ADMIN_PATH.match(path):
            return self.page("admin.html")
        if OLD_ADMIN_PATH.match(path):
            return self.redirect("/admin" + path + (f"?{url.query}" if url.query else ""))

        if path == "/api/state":
            return self.send_json(self.state(parse_qs(url.query).get("predictions", ["latest"])[0]))

        if path == "/api/runs":
            with runs_lock:
                return self.send_json({"runs": [runs[job_id] for job_id in sorted(runs, reverse=True)]})

        if path.startswith("/api/predictions/"):
            # One run's predictions by selection name (see prediction_files); unknown names 404.
            # ?download=1 serves it as a file attachment instead of inline JSON.
            name = path[len("/api/predictions/"):]
            file = self.prediction_files().get(name)
            if file is not None and file.exists():
                download = parse_qs(url.query).get("download")
                return self.send(200, file.read_bytes(), "application/json",
                                 filename=f"{name}.json" if download else None)
            return self.send(404, b"No such predictions", "text/plain")

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
        if self.path.startswith("/api/upload"):
            return self.upload(length)
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
        if self.path == "/api/biases":
            # body: {camera: {area id: bias}}; per-area lower biases tuned in the annotate view.
            try:
                cleaned = {camera: {area_id: float(bias) for area_id, bias in areas.items()}
                           for camera, areas in body.items()}
            except (AttributeError, TypeError, ValueError):
                return self.send(400, json.dumps({"error": "Biases must be {camera: {area: number}}"}).encode(),
                                 "application/json")
            self.write_json(self.options.area_biases, cleaned)
            return self.send_json({"saved": True})
        if self.path == "/api/solutions":
            # body: {camera: {photo file: {area id: label}}}; ground truth for the uploaded photos.
            # Scored on the Runs page only — predict trains on --annotations alone, never this file.
            try:
                cleaned = self.clean_solutions(body)
            except ValueError as error:
                return self.send(400, json.dumps({"error": str(error)}).encode(), "application/json")
            self.write_json(Path(self.options.uploads) / "annotations.json", cleaned)
            return self.send_json({"saved": True})
        if self.path == "/api/latest":
            # body: {"name": "<predictions selection>"}; copies that file to --predictions ("Latest run").
            # Runs never touch the latest file themselves; this is the only way it changes.
            name = body.get("name") if isinstance(body, dict) else None
            file = self.prediction_files().get(name) if isinstance(name, str) else None
            if file is None or not file.exists():
                return self.send(400, json.dumps({"error": f"No predictions file {name!r}"}).encode(),
                                 "application/json")
            dest = Path(self.options.predictions)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".json.tmp")
            temporary.write_bytes(file.read_bytes())
            os.replace(temporary, dest)
            return self.send_json({"saved": True})
        if self.path == "/api/run":
            # body: predict.py options (see build_command); starts a background run unless one is going.
            try:
                command, label, run_mode, predictions_name = self.build_command(body)
            except ValueError as error:
                return self.send(400, json.dumps({"error": str(error)}).encode(), "application/json")
            with runs_lock:
                if running_run[0] is not None and runs[running_run[0]]["status"] == "running":
                    return self.send(409, json.dumps(
                        {"error": f"Run {running_run[0]} ({runs[running_run[0]]['label']}) is still going"}).encode(),
                        "application/json")
                run_number[0] += 1
                job = {"id": run_number[0], "label": label, "mode": run_mode, "predictions": predictions_name,
                       "file": None, "command": " ".join(command),
                       "status": "running", "returncode": None,
                       "started": time.time(), "finished": None, "log": ""}
                runs[job["id"]] = job
                running_run[0] = job["id"]
                for old in [job_id for job_id in sorted(runs) if runs[job_id]["status"] != "running"][:-MAX_RUNS + 1]:
                    del runs[old]
            # Predict modes write straight to output/history/<timestamp>.json of their own (--output-only);
            # output/annotations.json is only ever touched by promoting a run (POST /api/latest).
            versioned_stem = None
            if predictions_name:
                history_dir = (Path.cwd() / self.options.predictions_dir).parent / "history"
                stem = str(int(time.time()))
                while (history_dir / f"{stem}.json").exists():
                    stem = str(int(stem) + 1)
                versioned_stem = stem
                versioned_path = str((history_dir / f"{stem}.json").resolve())
                command, label, run_mode, predictions_name = self.build_command(body, versioned_path)
                with runs_lock:
                    job["command"] = " ".join(command)
            save_runs(self.options.runs)
            threading.Thread(target=run_worker,
                             args=(job["id"], command, self.options.runs, versioned_stem),
                             daemon=True).start()
            return self.send_json({"id": job["id"], "label": label})
        self.send(404, b"Not found", "text/plain")

    def upload(self, length):
        """Save one uploaded photo to the uploads folder for the final run.

        POST /api/upload?camera=<id>&name=<file> with the JPEG as the raw body. Uploads live in
        input/uploads/<camera>/, apart from the training photos, so they can never be annotated
        or trained on. The camera must be in the planogram, the name a plain .jpg filename, the
        bytes a decodable JPEG under MAX_UPLOAD, and the file must not exist yet.
        """
        query = parse_qs(urlsplit(self.path).query)
        camera = (query.get("camera") or [None])[0]
        name = (query.get("name") or [None])[0]
        if camera not in self.planogram:
            return self.send(400, json.dumps({"error": f"Unknown camera {camera!r}"}).encode(), "application/json")
        if not name or not UPLOAD_NAME.match(name):
            return self.send(400, json.dumps({"error": f"Name must be a plain .jpg filename, not {name!r}"}).encode(),
                             "application/json")
        if length > MAX_UPLOAD:
            # Drain nothing: Content-Length exceeds the cap, so refuse before reading the body.
            return self.send(413, json.dumps({"error": f"File exceeds {MAX_UPLOAD // 1024 // 1024} MB"}).encode(),
                             "application/json")
        data = self.rfile.read(length)
        if data[:2] != b"\xff\xd8" or cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            return self.send(400, json.dumps({"error": f"{name} is not a decodable JPEG"}).encode(), "application/json")
        dest = Path(self.options.uploads) / camera / name
        if dest.exists():
            return self.send(409, json.dumps({"error": f"{camera}/{name} already exists; not overwritten"}).encode(),
                             "application/json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".jpg.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, dest)
        return self.send_json({"saved": True, "file": name})

    def clean_solutions(self, data):
        """Validate a solution file ({camera: {photo file: {area id: label}}}); raises ValueError."""
        if not isinstance(data, dict):
            raise ValueError("Solution must be {camera: {photo: {area: label}}}")
        cleaned = {}
        for camera, files in data.items():
            if camera not in self.planogram:
                raise ValueError(f"Unknown camera {camera!r}")
            if not isinstance(files, dict):
                raise ValueError(f"Camera {camera!r} must map photos to areas")
            cleaned[camera] = {}
            for file, areas in files.items():
                if not isinstance(areas, dict):
                    raise ValueError(f"Photo {file!r} must map areas to labels")
                cleaned[camera][file] = {}
                for area_id, label in areas.items():
                    if not isinstance(label, dict):
                        raise ValueError(f"Label for {area_id!r} must be an object")
                    compliant = label.get("planogram_compliant")
                    level = label.get("stock_level")
                    value = label.get("stock_value")
                    if not isinstance(compliant, bool):
                        raise ValueError(f"Label for {area_id!r} needs planogram_compliant true/false")
                    if level not in STOCK_VALUES:
                        raise ValueError(f"Label for {area_id!r} needs stock_level empty/half/full")
                    if value != STOCK_VALUES[level]:
                        raise ValueError(f"Label for {area_id!r} needs stock_value {STOCK_VALUES[level]} for {level}")
                    cleaned[camera][file][area_id] = {
                        "planogram_compliant": compliant, "stock_level": level, "stock_value": value}
        return cleaned

    def build_command(self, params, output=None):
        """A `python -m shelf.predict` command from the run form's values; raises ValueError when invalid.

        Paths stay server-configured: labels come from --output (the file the annotate page saves).
        With an output path (one per run) predict writes only there (--output-only) and never touches
        --predictions (the pages' "Latest run"); without one it behaves like the CLI default.
        """
        if not isinstance(params, dict):
            raise ValueError("Run options must be a JSON object")
        mode = params.get("mode", "predict")
        if mode not in ("predict", "out-of-fold", "evaluate"):
            raise ValueError(f"Unknown mode {mode!r}; want predict, out-of-fold or evaluate")
        models = params.get("models") or [DEFAULT_MODEL]
        unknown = [name for name in models if name not in MODELS]
        if not models:
            raise ValueError("Pick at least one model")
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}; want {', '.join(sorted(MODELS))}")
        device = params.get("device", "auto")
        if device not in DEVICES:
            raise ValueError(f"Unknown device {device!r}; want {', '.join(DEVICES)}")
        photos = params.get("photos", "training")
        if photos not in ("training", "uploaded"):
            raise ValueError(f"Unknown photos {photos!r}; want training or uploaded")
        cameras_dir = self.options.cameras if photos == "training" else Path(self.options.uploads)
        if photos == "uploaded" and not any(Path(cameras_dir).glob("*/*.jpg")):
            raise ValueError("No uploaded photos yet — upload some first")
        try:
            jobs = int(params.get("jobs", -1))
        except (TypeError, ValueError):
            raise ValueError(f"Jobs must be a whole number, not {params.get('jobs')!r}")
        out = str(Path(output).resolve()) if output is not None else str(Path(self.options.predictions).resolve())
        command = [sys.executable, "-m", "shelf.predict", "--model", *models,
                   "--planogram", str(Path(self.options.planogram).resolve()),
                   "--cameras", str(Path(cameras_dir).resolve()),
                   "--train-cameras", str(Path(self.options.cameras).resolve()),
                   "--annotations", str(Path(self.options.output).resolve()),
                   "--area-biases", str(Path(self.options.area_biases).resolve()),
                   "--output", out,
                   "--jobs", str(jobs), "--device", device]
        if output is not None:
            command.append("--output-only")
        if mode == "evaluate":
            try:
                rounds = int(params.get("rounds", 20))
            except (TypeError, ValueError):
                raise ValueError(f"Rounds must be a whole number, not {params.get('rounds')!r}")
            if rounds < 1:
                raise ValueError("Rounds must be at least 1")
            biases = params.get("biases", [0.0])
            if isinstance(biases, str):
                biases = biases.replace(",", " ").split()
            try:
                biases = [float(bias) for bias in biases]
            except (TypeError, ValueError):
                raise ValueError(f"Biases must be numbers, not {params.get('biases')!r}")
            if not biases:
                raise ValueError("Give at least one bias to compare")
            split = params.get("split", "photos")
            if split not in ("photos", "cameras"):
                raise ValueError(f"Unknown split {split!r}; want photos or cameras")
            command += ["--evaluate", str(rounds), "--split", split,
                        "--biases", *(str(bias) for bias in biases)]
            return command, f"evaluate {', '.join(models)} ({rounds} rounds)", mode, None
        scope = " on uploads" if photos == "uploaded" else ""
        try:
            lower_bias = float(params.get("lower_bias", 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"Lower bias must be a number, not {params.get('lower_bias')!r}")
        command += ["--lower-bias", str(lower_bias)]
        if mode == "out-of-fold":
            try:
                folds = int(params.get("folds", 10))
            except (TypeError, ValueError):
                raise ValueError(f"Folds must be a whole number, not {params.get('folds')!r}")
            if folds < 2:
                raise ValueError("Folds must be at least 2")
            command += ["--out-of-fold", str(folds)]
            return command, f"out-of-fold {models[0]} ({folds} folds, bias {lower_bias}){scope}", mode, f"{models[0]}.out-of-fold"
        return command, f"predict {models[0]} (bias {lower_bias}){scope}", mode, models[0]

    def prediction_files(self):
        """The predictions to choose from: the latest run, each model's saved predictions,
        and every kept historical output (output/history/<timestamp>.json, newest last)."""
        files = {"latest": self.options.predictions}
        files.update({path.stem: path for path in sorted(self.options.predictions_dir.glob("*.json"))})
        history_dir = (Path.cwd() / self.options.predictions_dir).parent / "history"
        files.update({path.stem: path for path in sorted(history_dir.glob("*.json")) if path.stem not in files})
        return files

    def state(self, chosen="latest"):
        files = self.prediction_files()
        chosen = chosen if chosen in files else "latest"
        output, predictions = self.options.output, files[chosen]
        with runs_lock:
            history = [{"id": runs[job_id]["id"], "label": runs[job_id]["label"],
                        "status": runs[job_id]["status"], "file": runs[job_id].get("file"),
                        "finished": runs[job_id]["finished"]}
                       for job_id in sorted(runs, reverse=True)]
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
            "uploads": {
                camera: sorted(path.name for path in (Path(self.options.uploads) / camera).glob("*.jpg"))
                if (Path(self.options.uploads) / camera).is_dir() else []
                for camera in sorted(self.planogram)
            },
            "annotations": json.loads(output.read_text()) if output.exists() else {},
            "biases": json.loads(self.options.area_biases.read_text()) if self.options.area_biases.exists() else {},
            "solutions": json.loads(solution.read_text()) if (solution := Path(self.options.uploads) / "annotations.json").exists() else {},
            "predictions": json.loads(predictions.read_text()) if predictions.exists() else {},
            "predictionsFile": str(predictions),
            "predictionsName": chosen,
            "predictionOptions": list(files),
            "history": history,
            "models": sorted(MODELS),
            "defaultModel": DEFAULT_MODEL,
            "done": sorted(self.read_done()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--uploads", type=Path, default=Path("input/uploads"),
                        help="uploaded validation photos, apart from the training photos (see POST /api/upload)")
    parser.add_argument("--output", type=Path, default=Path("input/annotations.json"))
    parser.add_argument("--area-biases", type=Path, default=Path("input/area-biases.json"),
                        help="per-area lower biases tuned in the annotate view")
    parser.add_argument("--predictions-dir", type=Path, default=Path("output/predictions"),
                        help="each model's predictions, selectable on the pages (from python -m shelf.predict --model)")
    parser.add_argument("--predictions", type=Path, default=Path("output/annotations.json"),
                        help="predictions to compare with (from python -m shelf.predict; use --out-of-fold for honest ones)")
    parser.add_argument("--done", type=Path, default=Path("output/staff-done.json"), help="tasks staff marked done")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    options.runs = str((Path.cwd() / options.predictions).parent / "runs.json")
    load_runs(options.runs)

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
