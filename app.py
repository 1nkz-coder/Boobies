"""
BreastScan Twin — web app

Local:       python app.py            then open  http://127.0.0.1:5000
Production:  gunicorn app:app -c gunicorn.conf.py   (what Render runs, see render.yaml)
"""
from __future__ import annotations

import os
import threading
import warnings

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from core import birads, scan  # noqa: E402
from core.deep import KINDS, MODELS_DIR, DeepModels  # noqa: E402
from core.tabular import TwinEngine  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "60")) * 1024 * 1024  # DICOM can be large
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 300  # let browsers cache CSS/JS/images for 5 minutes

print("Loading Digital Twin model ...")
twin = TwinEngine.load()
scan_model = scan.ScanModel()      # optional generic model (train_scan_model.py)
deep = DeepModels()                # Kaggle-trained mammogram / histopathology models
for k, v in deep.info().items():
    print(f"  {k:15s}: {'ready' if v['ready'] else v.get('error', 'not installed')}")
print("Ready.")


# Image analysis is memory-hungry. On a small instance, scans queue up and run one at a time
# (other pages and the Digital Twin stay responsive). Raise MAX_CONCURRENT_SCANS on bigger machines.
_scan_gate = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_SCANS", "2")))
_MAX_WAITING = int(os.environ.get("MAX_WAITING_SCANS", "4"))  # beyond this the browser is told to retry shortly
_waiting = 0
_waiting_lock = threading.Lock()


def _busy():
    resp = jsonify({"error": "The server is busy with other scans. Retrying…", "busy": True, "retry_after": 3})
    resp.status_code = 503
    resp.headers["Retry-After"] = "3"
    return resp


class _NoModel:
    """Stand-in so the classic image analysis runs without the generic classifier."""
    ready = False
    def info(self):
        return {"ready": False}


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    """Health check used by Render: 200 when the app and its models are loaded."""
    info = deep.info()
    return jsonify({"status": "ok", "digital_twin": True,
                    **{k: v["ready"] for k, v in info.items()}})


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")  # results are never cached
    return resp


@app.get("/api/schema")
def api_schema():
    return jsonify(twin.schema())


@app.get("/api/metrics")
def api_metrics():
    return jsonify({"tabular": twin.metrics, "scan": scan_model.info(), "deep": deep.info(),
                    "population": twin.population_map()})


@app.get("/api/birads")
def api_birads():
    return jsonify({"categories": birads.CATEGORIES, "management": birads.MANAGEMENT, "urgency": birads.URGENCY})


@app.get("/api/models")
def api_models():
    return jsonify({"scan": scan_model.info(), "deep": deep.info()})


@app.get("/models/<kind>/<name>")
def model_file(kind, name):
    # only the training charts produced by the Kaggle notebook are served
    if kind not in KINDS or name not in ("training_report.png", "confusion_matrix.png", "preprocessing_examples.jpg"):
        abort(404)
    return send_from_directory(os.path.join(MODELS_DIR, kind), name)


@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    try:
        r = twin.predict(body.get("values", []))
        r["assessment"] = birads.assess_cytology(r["probability"])
        return jsonify(r)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/case/<int:i>")
def api_case(i):
    try:
        return jsonify(twin.case(i))
    except IndexError as e:
        return jsonify({"error": str(e)}), 404


@app.post("/api/scan")
def api_scan():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded."}), 400
    polarity = request.form.get("polarity", "auto")
    if polarity not in ("auto", "bright", "dark"):
        polarity = "auto"
    kind = request.form.get("kind", "generic")
    # Large uploads wait in a temporary file on disk; they are only read into memory when it is their turn.
    global _waiting
    with _waiting_lock:
        if _waiting >= _MAX_WAITING:
            return _busy()
        _waiting += 1
    try:
        _scan_gate.acquire()
    finally:
        with _waiting_lock:
            _waiting -= 1
    try:
        data = f.read()
        return _run_scan(kind, data, f.filename, polarity)
    finally:
        _scan_gate.release()


def _run_scan(kind, data, filename, polarity):
    try:
        if kind in KINDS:
            result = deep.analyse(kind, data, filename)
            if kind == "mammogram":  # add the classic measurements of the brightest lesion-like region
                try:
                    classic = scan.analyse(data, filename, _NoModel(), "bright")
                    result["classic"] = {k: classic[k] for k in ("shape", "histogram", "roi", "polarity")}
                    result["images"]["enhanced"] = classic["images"]["enhanced"]
                    result["images"]["overlay"] = classic["images"]["overlay"]
                    result["quality"].update(classic["quality"])
                except Exception:
                    pass
            return jsonify(result)
        result = scan.analyse(data, filename, scan_model, polarity)
        result["kind"] = "generic"
        if result.get("classified"):
            result["assessment"] = birads.assess_mammogram(result["probability"], 0.5)
            result["assessment"]["note"] = "Suggested from a model you trained with train_scan_model.py; a radiologist must assign the final category."
        else:
            result["assessment"] = birads.assess_generic()
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # keep the UI alive on unexpected images
        app.logger.exception("scan failed")
        return jsonify({"error": f"Could not analyse this image ({type(e).__name__})."}), 500


@app.post("/api/scan/<sid>/explain")
def api_explain(sid):
    try:
        with _scan_gate:
            return jsonify(scan.explain(sid, scan_model))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File is larger than {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  BreastScan Twin running at  http://127.0.0.1:{port}\n")
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=False)
