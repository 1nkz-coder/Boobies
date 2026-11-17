"""
Deep-learning scan models trained on Kaggle (kaggle/breastscan_kaggle_train.ipynb).

    models/mammogram/model.onnx       + meta.json   CBIS-DDSM, malignant vs benign finding
    models/histopathology/model.onnx  + meta.json   IDC vs non-IDC tissue patches

The models are run with onnxruntime (no PyTorch needed). Each returns class
probabilities and a class-activation map (CAM) showing where it looked.
The preprocessing functions below are copied verbatim from the training
notebook, so the app sees images exactly the way the model was trained on them.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading

import cv2
import numpy as np
from PIL import Image

from . import birads

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
KINDS = ("mammogram", "histopathology")

# Deployment knobs (environment variables). Defaults suit a laptop; render.yaml lowers them
# so the app fits a small cloud instance (512 MB RAM).
HISTO_BATCH = int(os.environ.get("HISTO_BATCH", "32"))            # patches per inference batch
MAX_HISTO_TILES = int(os.environ.get("MAX_HISTO_TILES", "3000"))  # above this, an even sample of tissue tiles is analysed
MAX_MEGAPIXELS = float(os.environ.get("MAX_IMAGE_MEGAPIXELS", "60"))  # larger images are refused (memory guard)
HISTO_MAX_MEGAPIXELS = float(os.environ.get("HISTO_MAX_MEGAPIXELS", "60"))  # histology is RGB at full scale: own limit
MAMMO_WORK_SIDE = 2048  # mammograms are reduced to this before analysis (the model input is only 640x384)
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0"))             # 0 = onnxruntime default (all cores)


# ----------------------------------------------------------------------------
# Preprocessing: identical to the training notebook
# ----------------------------------------------------------------------------
def mammo_preprocess(gray, out_h, out_w):
    """Mammogram: crop to the breast, put the breast on the left, CLAHE, letterbox to out_h x out_w. Returns uint8 HxW."""
    g = gray if gray.dtype == np.uint8 else cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = g.shape
    s = 512.0 / max(h, w)
    small = cv2.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    _, m = cv2.threshold(cv2.GaussianBlur(small, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh = stats[k, 0], stats[k, 1], stats[k, 2], stats[k, 3]
        if bw * bh > 0.05 * small.size:
            x0, y0 = int(x / s), int(y / s)
            x1, y1 = int((x + bw) / s), int((y + bh) / s)
            g = g[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    # breast facing left (chest wall on the left edge)
    half = g.shape[1] // 2
    if g[:, half:].mean() > g[:, :half].mean():
        g = g[:, ::-1]
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.ascontiguousarray(g))
    # letterbox
    sc = min(out_h / g.shape[0], out_w / g.shape[1])
    nh, nw = max(1, int(round(g.shape[0] * sc))), max(1, int(round(g.shape[1] * sc)))
    r = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w), np.uint8)
    canvas[:nh, :nw] = r
    return canvas


def histo_preprocess(rgb, size=50):
    """Histology patch: RGB uint8 resized to size x size (dataset patches are 50x50 at 40x)."""
    if rgb.shape[0] != size or rgb.shape[1] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class ImageTooLarge(ValueError):
    pass


def _check_size(w: int, h: int, limit_mp: float | None = None):
    limit = min(MAX_MEGAPIXELS, limit_mp) if limit_mp else MAX_MEGAPIXELS
    if w * h > limit * 1e6:
        raise ImageTooLarge(f"This image is {w}×{h} px ({w * h / 1e6:.0f} megapixels); the limit here is "
                            f"{limit:.0f} MP. Crop the region of interest (or export a smaller view) and try again.")


def read_dicom(data: bytes, limit_mp: float | None = None):
    """(header dataset, pixel array) for a DICOM file. Uses pydicom 3's buffer decoder, which does not
    keep a second copy of the pixel bytes; falls back to the classic API on older pydicom."""
    try:
        import pydicom
    except ImportError as e:
        raise ValueError("DICOM support needs pydicom:  pip install pydicom") from e
    ds = pydicom.dcmread(io.BytesIO(data), force=True, stop_before_pixels=True)
    _check_size(int(getattr(ds, "Columns", 0) or 0), int(getattr(ds, "Rows", 0) or 0), limit_mp)
    try:
        from pydicom.pixels import pixel_array
        arr = pixel_array(io.BytesIO(data))
    except Exception:
        arr = pydicom.dcmread(io.BytesIO(data), force=True).pixel_array
    return ds, arr


def decode_image(data: bytes, filename: str = "", color: bool = False, limit_mp: float | None = None,
                 max_side: int | None = None) -> np.ndarray:
    """uint8 grey (HxW) or RGB (HxWx3). Supports PNG/JPG/TIFF/BMP and DICOM.
    Memory-conscious: the size is checked from the header before decoding, the image is decoded once
    (OpenCV), and `max_side` shrinks it straight away when full resolution isn't needed."""
    name = filename.lower()
    if name.endswith((".dcm", ".dicom")) or data[128:132] == b"DICM":
        ds, a = read_dicom(data, limit_mp)
        if a.ndim == 3 and a.shape[-1] not in (3, 4):
            a = a[a.shape[0] // 2]
        if a.ndim == 3:
            a = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)
        if a.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
            a = a.astype(np.float32)
        if max_side and max(a.shape) > max_side:  # shrink in the native bit depth first: much less memory
            sc = max_side / max(a.shape)
            a = cv2.resize(np.ascontiguousarray(a), (int(a.shape[1] * sc), int(a.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = 255 - a
        return np.dstack([a] * 3) if color else a
    try:
        with Image.open(io.BytesIO(data)) as probe:  # reads the header only
            _check_size(*probe.size, limit_mp)
    except ImageTooLarge:
        raise
    except Exception as e:
        raise ValueError("Could not read this file as an image. Use PNG, JPG, TIFF or DICOM.") from e
    buf = np.frombuffer(data, np.uint8)
    a = cv2.imdecode(buf, cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED)
    if a is None:  # formats OpenCV can't read: fall back to Pillow
        img = Image.open(io.BytesIO(data))
        img.seek(0)
        if img.mode in ("I;16", "I;16B", "I", "F") and not color:
            a = np.asarray(img, dtype=np.float32)
        else:
            a = np.asarray(img.convert("RGB" if color else "L"))
            if color:
                a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    if color:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB, dst=a) if a.flags.writeable else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    else:
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY if a.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        if a.dtype != np.uint8:  # 16-bit / float images: stretch to 0-255
            a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if max_side and max(a.shape[:2]) > max_side:
        sc = max_side / max(a.shape[:2])
        a = cv2.resize(a, (int(a.shape[1] * sc), int(a.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return a


def png_b64(img: np.ndarray, max_side: int = 900) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colormap(v: np.ndarray) -> np.ndarray:
    """Perceptual dark-blue → teal → amber → cream ramp (colour-blind friendly). v in [0,1] -> RGB float."""
    stops = np.array([[0.05, 0.08, 0.20], [0.05, 0.45, 0.60], [0.95, 0.62, 0.20], [1.0, 0.95, 0.85]], np.float32)
    x = np.clip(np.asarray(v, np.float32), 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def overlay(base: np.ndarray, heat: np.ndarray, alpha_max: float = 0.65) -> np.ndarray:
    """Blend a [0,1] heatmap onto a grey/RGB uint8 image; low values stay transparent."""
    b = base.astype(np.float32) / 255.0
    if b.ndim == 2:
        b = np.dstack([b] * 3)
    h = np.clip(heat, 0, 1)
    a = (alpha_max * np.clip((h - 0.15) / 0.85, 0, 1))[..., None]
    out = b * (1 - a) + colormap(h) * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def heat_rgba(heat: np.ndarray, max_side: int = 900) -> str:
    """Heatmap as a transparent PNG layer (alpha grows with the value), so the browser can change its opacity."""
    h = np.clip(heat, 0, 1).astype(np.float32)
    rgb = (colormap(h) * 255).astype(np.uint8)
    a = (255 * np.clip((h - 0.12) / 0.88, 0, 1) ** 0.8).astype(np.uint8)
    return png_b64(np.dstack([rgb, a]), max_side)


def _make_trim():
    """On Linux, hand memory freed after an inference back to the OS (glibc otherwise keeps it),
    so the process stays small on a 512 MB instance."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        return lambda: libc.malloc_trim(0)
    except Exception:  # Windows / macOS: nothing to do
        return lambda: None


_release_memory = _make_trim()


def risk_of(p: float, thr: float) -> str:
    if p >= max(0.85, thr):
        return "High"
    if p >= thr:
        return "Elevated"
    if p >= thr * 0.5:
        return "Uncertain"
    return "Low"


# ----------------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------------
class DeepModel:
    def __init__(self, kind: str):
        self.kind = kind
        self.folder = os.path.join(MODELS_DIR, kind)
        self.sess = None
        self.meta = None
        self.mtime = None
        self.error = None
        self.lock = threading.Lock()

    def _paths(self):
        return os.path.join(self.folder, "model.onnx"), os.path.join(self.folder, "meta.json")

    def reload(self):
        onnx_path, meta_path = self._paths()
        if not (os.path.exists(onnx_path) and os.path.exists(meta_path)):
            self.sess, self.meta, self.mtime, self.error = None, None, None, None
            return
        mt = max(os.path.getmtime(onnx_path), os.path.getmtime(meta_path))
        if mt == self.mtime:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self.sess, self.error = None, "onnxruntime is not installed. Run: pip install onnxruntime"
            return
        try:
            with open(meta_path) as f:
                self.meta = json.load(f)
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            # The memory arena keeps the peak allocation of the biggest batch forever; turning it off
            # keeps the process small enough for a 512 MB instance at a small speed cost.
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            if ORT_THREADS > 0:
                opts.intra_op_num_threads = ORT_THREADS
                opts.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self.mtime, self.error = mt, None
        except Exception as e:  # corrupt file etc.
            self.sess, self.error = None, f"Could not load {self.kind} model: {e}"

    @property
    def ready(self) -> bool:
        with self.lock:
            self.reload()
        return self.sess is not None

    def info(self) -> dict:
        ready = self.ready
        out = {"kind": self.kind, "ready": ready}
        if self.error:
            out["error"] = self.error
        if ready:
            m = self.meta
            mt = m.get("metrics", {})
            out.update({
                "title": m.get("title", self.kind), "classes": m.get("classes"), "positive": m.get("positive"),
                "threshold": m.get("threshold", 0.5), "dataset": m.get("dataset"), "architecture": m.get("architecture"),
                "trained_at": m.get("trained_at"), "n": m.get("n"), "caveat": m.get("caveat"),
                "metrics": {k: mt.get(k) for k in ("auc", "accuracy", "sensitivity", "specificity", "precision",
                                                     "f1", "confusion", "n_test", "roc")},
                "has_report": os.path.exists(os.path.join(self.folder, "training_report.png")),
            })
        return out

    def run(self, x: np.ndarray):
        """x: (B,3,H,W) float32 in [0,1] -> probs (B,K), cam (B,K,h,w)."""
        with self.lock:
            probs, cam = self.sess.run(None, {"image": x.astype(np.float32)})
        _release_memory()
        return probs, cam

    @property
    def pos_index(self) -> int:
        return self.meta["classes"].index(self.meta["positive"])


# ----------------------------------------------------------------------------
# Mammogram
# ----------------------------------------------------------------------------
def analyse_mammogram(model: DeepModel, data: bytes, filename: str) -> dict:
    meta = model.meta
    H, W = meta["input"]["height"], meta["input"]["width"]
    gray = decode_image(data, filename, color=False, max_side=MAMMO_WORK_SIDE)
    prep = mammo_preprocess(gray, H, W)
    x = np.repeat(prep[None, None].astype(np.float32) / 255.0, 3, axis=1)
    # test-time flip, as in the notebook's evaluation; run one image at a time to halve peak memory
    p1, c1 = model.run(np.ascontiguousarray(x))
    p2, c2 = model.run(np.ascontiguousarray(x[..., ::-1]))
    probs, cam = np.concatenate([p1, p2]), np.concatenate([c1, c2])
    k = model.pos_index
    p = float(probs[:, k].mean())
    c = (cam[0, k] + cam[1, k][:, ::-1]) / 2
    c = np.maximum(c, 0)
    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
    # fade the map out towards the breast boundary: activation maps often light up on the skin line /
    # image edge, which is not a lesion
    otsu, _ = cv2.threshold(prep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    breast = (prep > max(15, 0.5 * otsu)).astype(np.uint8)
    breast = cv2.morphologyEx(breast, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dist = cv2.distanceTransform(breast, cv2.DIST_L2, 5)
    c = c * np.clip(dist / (0.04 * max(H, W)), 0, 1)
    c = c / (c.max() + 1e-6)
    thr = float(meta.get("threshold", 0.5))
    breast_fraction = float(breast.mean())
    image_ok = breast_fraction > 0.12 and prep.std() > 8
    focus = None
    n_cc, lab_cc = cv2.connectedComponents((c > 0.6).astype(np.uint8))
    peak = np.unravel_index(int(np.argmax(c)), c.shape)
    ys, xs = np.where(lab_cc == lab_cc[peak]) if n_cc > 1 and lab_cc[peak] > 0 else (np.array([]), np.array([]))
    if len(ys):
        focus = {"x": float(xs.min() / W), "y": float(ys.min() / H),
                 "w": float((xs.max() - xs.min() + 1) / W), "h": float((ys.max() - ys.min() + 1) / H)}
    return {
        "kind": "mammogram", "classified": True, "probability": p, "threshold": thr,
        "label": ("Suspicious for malignancy" if p >= thr else "Likely benign finding") if image_ok else "Image not assessable",
        "risk": risk_of(p, thr),
        "class_probabilities": {cls: float(probs[:, i].mean()) for i, cls in enumerate(meta["classes"])},
        "images": {"input": png_b64(prep), "heatmap": png_b64(overlay(prep, c)), "heat": heat_rgba(c)},
        "focus": focus, "model": model.info(),
        "quality": {"width": int(gray.shape[1]), "height": int(gray.shape[0]), "breast_fraction": breast_fraction,
                    "ok": bool(image_ok)},
        "assessment": birads.assess_mammogram(p, thr, image_ok, None if image_ok else
                                              "Too little breast tissue was found in the image (is it a mammogram?)."),
    }


# ----------------------------------------------------------------------------
# Histopathology
# ----------------------------------------------------------------------------
def _tissue(patch: np.ndarray) -> bool:
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return patch.mean() < 225 and hsv[..., 1].mean() > 18


def analyse_histology(model: DeepModel, data: bytes, filename: str) -> dict:
    meta = model.meta
    P = int(meta["preprocess"].get("patch_px", 50))
    S = int(meta["input"]["height"])
    rgb = decode_image(data, filename, color=True, limit_mp=HISTO_MAX_MEGAPIXELS)
    h, w = rgb.shape[:2]
    thr = float(meta.get("threshold", 0.5))
    k = model.pos_index

    def infer(patches):
        out = []
        for i in range(0, len(patches), HISTO_BATCH):  # build each batch on demand to keep memory flat
            xb = np.stack([cv2.resize(histo_preprocess(pt, P), (S, S), interpolation=cv2.INTER_LINEAR)
                           for pt in patches[i:i + HISTO_BATCH]])
            xb = xb.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            pr, _ = model.run(np.ascontiguousarray(np.concatenate([xb, xb[..., ::-1]], 0)))  # flip TTA
            n = len(xb)
            out.append((pr[:n, k] + pr[n:, k]) / 2)
        return np.concatenate(out)

    # one patch (e.g. an image straight from the dataset)
    if max(h, w) <= 2 * P:
        p = float(infer([rgb])[0])
        heat = np.full((h, w), p, np.float32)
        z = max(1, 240 // max(h, w))  # enlarge tiny patches for display
        view = cv2.resize(rgb, (w * z, h * z), interpolation=cv2.INTER_NEAREST)
        return {
            "kind": "histopathology", "classified": True, "probability": p, "threshold": thr,
            "label": "IDC detected" if p >= thr else "No IDC detected", "risk": risk_of(p, thr),
            "mode": "single patch", "patches": {"total": 1, "tissue": 1, "positive": int(p >= thr), "fraction": float(p >= thr), "patch_px": P},
            "images": {"input": png_b64(view), "heatmap": png_b64(overlay(view, cv2.resize(heat, view.shape[1::-1]))),
                       "heat": heat_rgba(cv2.resize(heat, view.shape[1::-1]))},
            "model": model.info(), "quality": {"width": int(w), "height": int(h)},
            "assessment": birads.assess_histology(p >= thr, p, float(p >= thr)),
        }

    # Large image: tile into P x P patches at NATIVE scale. The model was trained at one magnification,
    # so the image is never shrunk; if there are more tissue tiles than MAX_HISTO_TILES, an evenly spaced
    # sample of them is analysed instead (and the result says so).
    gy, gx = h // P, w // P
    coords, patches = [], []
    for i in range(gy):
        for j in range(gx):
            pt = rgb[i * P:(i + 1) * P, j * P:(j + 1) * P]
            if _tissue(pt):
                coords.append((i, j)); patches.append(pt)
    n_tissue_all = len(patches)
    sampled = n_tissue_all > MAX_HISTO_TILES
    if sampled:
        keep = np.linspace(0, n_tissue_all - 1, MAX_HISTO_TILES).round().astype(int)
        coords = [coords[t] for t in keep]
        patches = [patches[t] for t in keep]
    grid = np.full((max(gy, 1), max(gx, 1)), np.nan, np.float32)
    if patches:
        probs = infer(patches)
        for (i, j), pr in zip(coords, probs):
            grid[i, j] = pr
    else:
        probs = np.array([])
    n_t = len(probs)
    if n_t == 0:
        return {
            "kind": "histopathology", "classified": False, "label": "No tissue found",
            "mode": "tiled", "patches": {"total": int(gy * gx), "tissue": 0, "positive": 0, "fraction": 0.0, "patch_px": P},
            "images": {"input": png_b64(rgb)}, "model": model.info(), "quality": {"width": int(w), "height": int(h)},
            "assessment": birads.assess_histology_inadequate(),
        }
    n_pos = int((probs >= thr).sum()) if n_t else 0
    frac = n_pos / n_t if n_t else 0.0
    # slide-level score: mean of the top 5% patch probabilities (robust to single noisy patches)
    top = np.sort(probs)[-max(1, n_t // 20):] if n_t else np.array([0.0])
    score = float(top.mean())
    detected = n_t > 0 and n_pos >= max(2, int(0.02 * n_t)) and score >= thr
    # Draw the heatmap at display size (<= 900 px), not at full resolution: same picture, far less memory.
    ds = min(1.0, 900 / max(h, w))
    disp = cv2.resize(rgb, (max(1, int(w * ds)), max(1, int(h * ds))), interpolation=cv2.INTER_AREA) if ds < 1 else rgb
    dh, dw = disp.shape[:2]
    gh, gw = min(dh, int(round(gy * P * ds))), min(dw, int(round(gx * P * ds)))
    heat = cv2.resize(np.nan_to_num(grid, nan=0.0), (max(1, gw), max(1, gh)), interpolation=cv2.INTER_CUBIC)
    full = np.zeros((dh, dw), np.float32)
    full[:gh, :gw] = np.clip(heat, 0, 1)
    return {
        "kind": "histopathology", "classified": True, "probability": score, "threshold": thr,
        "label": "IDC detected" if detected else "No IDC detected",
        "risk": risk_of(score, thr) if detected else ("Uncertain" if n_pos else "Low"),
        "mode": "tiled",
        "patches": {"total": int(gy * gx), "tissue": n_t, "positive": n_pos, "fraction": frac, "patch_px": P,
                    "sampled": bool(sampled), "tissue_all": int(n_tissue_all)},
        "images": {"input": png_b64(disp), "heatmap": png_b64(overlay(disp, full, 0.55)), "heat": heat_rgba(full)},
        "model": model.info(), "quality": {"width": int(w), "height": int(h)},
        "assessment": birads.assess_histology(bool(detected), score, frac),
    }


class DeepModels:
    def __init__(self):
        self.models = {k: DeepModel(k) for k in KINDS}

    def info(self) -> dict:
        return {k: m.info() for k, m in self.models.items()}

    def analyse(self, kind: str, data: bytes, filename: str) -> dict:
        m = self.models[kind]
        if not m.ready:
            raise ValueError(m.error or f"No {kind} model found. Put model.onnx and meta.json in models/{kind}/.")
        fn = analyse_mammogram if kind == "mammogram" else analyse_histology
        return fn(m, data, filename)
