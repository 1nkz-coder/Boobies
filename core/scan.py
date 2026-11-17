"""
Scan Lab — image analysis for breast scans (MRI slices, mammograms, ultrasound).

Pipeline
  1. Load   : PNG / JPG / BMP / TIFF, or DICOM (.dcm) when pydicom is installed.
  2. Enhance: grey-scale, percentile windowing, CLAHE, light denoising.
  3. Segment: find the most lesion-like region (bright or dark blob) that does not
              touch the image border.
  4. Measure: shape features that mirror the WDBC definitions (radius, perimeter,
              area, compactness, concavity, symmetry, fractal dimension …) plus
              texture (GLCM, LBP) and intensity statistics.
  5. Classify: ONLY if a scan model trained on labelled scans exists
              (models/scan_model.joblib, created by train_scan_model.py).
              Without it the app reports the measurements but makes no diagnosis.
  6. Explain : occlusion-sensitivity heatmap (with a model) or a texture
              irregularity map (without one).
"""
from __future__ import annotations

import base64
import io
import os

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, feature, filters, measure, morphology, transform

from .deep import ImageTooLarge, _check_size, read_dicom

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_MAX_SIDE = 2048
SCAN_MODEL_PATH = os.path.join(BASE_DIR, "models", "scan_model.joblib")
WORK_SIZE = 384      # analysis resolution (longest side)
FEAT_SIZE = 128      # resolution for global texture descriptors
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}


# --------------------------------------------------------------------------- I/O
def load_image(data: bytes, filename: str = "") -> np.ndarray:
    """Return a float32 grey image in [0, 1]."""
    name = filename.lower()
    arr = None
    if name.endswith((".dcm", ".dicom")) or data[128:132] == b"DICM":
        ds, raw = read_dicom(data)
        if raw.ndim == 3 and raw.shape[-1] not in (3, 4):  # multi-slice volume: middle slice
            raw = raw[raw.shape[0] // 2]
        if raw.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
            raw = raw.astype(np.float32)
        if raw.ndim == 2 and max(raw.shape) > WORK_MAX_SIDE:  # shrink first (native bit depth): analysis runs at 384 px
            k = WORK_MAX_SIDE / max(raw.shape)
            raw = cv2.resize(np.ascontiguousarray(raw), (int(raw.shape[1] * k), int(raw.shape[0] * k)), interpolation=cv2.INTER_AREA)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = raw.astype(np.float32) * slope + inter
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.seek(0)
            _check_size(*img.size)
            if img.mode in ("RGB", "L", "CMYK", "YCbCr"):
                img.draft("L", (WORK_MAX_SIDE, WORK_MAX_SIDE))  # JPEG: decode at reduced size directly
            img = img.convert("L") if img.mode not in ("I;16", "I;16B", "I", "F") else img
            img.thumbnail((WORK_MAX_SIDE, WORK_MAX_SIDE))  # the analysis runs at 384 px anyway
            arr = np.asarray(img, dtype=np.float32)
        except ImageTooLarge:
            raise
        except Exception as e:
            raise ValueError("Could not read this file as an image. Use PNG, JPG, TIFF or DICOM.") from e
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi - lo < 1e-6:
        raise ValueError("The image is blank (no contrast).")
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def _resize_longest(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    s = size / max(h, w)
    return transform.resize(img, (max(16, round(h * s)), max(16, round(w * s))), anti_aliasing=True).astype(np.float32)


def enhance(img: np.ndarray) -> np.ndarray:
    img = _resize_longest(img, WORK_SIZE)
    img = exposure.equalize_adapthist(img, clip_limit=0.015).astype(np.float32)
    return filters.gaussian(img, sigma=1.0).astype(np.float32)


# ---------------------------------------------------------------- segmentation
def _remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def _tissue_mask(img: np.ndarray) -> np.ndarray:
    t = filters.threshold_otsu(img)
    m = img > t * 0.5
    m = _remove_small(m, max(64, img.size // 200))
    m = ndi.binary_fill_holes(m)
    return m if m.mean() > 0.05 else np.ones_like(m, bool)


def segment_lesion(img: np.ndarray, polarity: str = "auto"):
    """Return (mask, polarity_used, candidates_count)."""
    tissue = _tissue_mask(img)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    centre_w = 1 - 0.6 * np.hypot((yy - h / 2) / h, (xx - w / 2) / w)

    def candidates(pol):
        try:
            th = filters.threshold_multiotsu(img[tissue] if tissue.any() else img, classes=3)
        except ValueError:
            th = [filters.threshold_otsu(img)] * 2
        m = (img > th[-1]) if pol == "bright" else (img < th[0])
        m &= tissue
        m = ndi.binary_opening(m, structure=morphology.disk(2))
        m = ndi.binary_fill_holes(m)
        lab = measure.label(m)
        out = []
        for r in measure.regionprops(lab):
            if r.area < img.size * 0.002 or r.area > img.size * 0.35:
                continue
            minr, minc, maxr, maxc = r.bbox
            if minr == 0 or minc == 0 or maxr == h or maxc == w:
                continue
            r0, c0 = max(0, minr - 6), max(0, minc - 6)
            sub = lab[r0:maxr + 6, c0:maxc + 6] == r.label
            ring = ndi.binary_dilation(sub, iterations=5) & ~sub & tissue[r0:maxr + 6, c0:maxc + 6]
            patch = img[r0:maxr + 6, c0:maxc + 6]
            contrast = abs(float(patch[sub].mean() - (patch[ring].mean() if ring.any() else img.mean())))
            region = lab == r.label
            score = (contrast * r.area ** 0.25 * (0.4 + r.solidity)
                     * float(centre_w[int(r.centroid[0]), int(r.centroid[1])])
                     * (1.15 if pol == "bright" else 1.0))
            out.append((score, region))
            continue
            out.append((score, lab == r.label))
        return out

    pols = ["bright", "dark"] if polarity == "auto" else [polarity]
    best, best_pol, n = None, pols[0], 0
    for pol in pols:
        c = candidates(pol)
        n += len(c)
        if c:
            top = max(c, key=lambda t: t[0])
            if best is None or top[0] > best[0]:
                best, best_pol = top, pol
    if best is None:  # fallback: central disc
        r = min(h, w) // 6
        mask = np.hypot(yy - h / 2, xx - w / 2) < r
        return mask, best_pol, 0
    return best[1], best_pol, n


# ------------------------------------------------------------------- features
def _fractal_dimension(contour_mask: np.ndarray) -> float:
    edge = contour_mask ^ ndi.binary_erosion(contour_mask)
    if edge.sum() < 10:
        return 1.0
    sizes, counts = [], []
    s = 2
    while s < min(edge.shape) // 2:
        h, w = (edge.shape[0] // s) * s, (edge.shape[1] // s) * s
        blocks = edge[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        sizes.append(s)
        counts.append(max(1, blocks.sum()))
        s *= 2
    if len(sizes) < 2:
        return 1.0
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def shape_features(img: np.ndarray, mask: np.ndarray) -> dict:
    """WDBC-style morphometrics of the segmented region (units: pixels at 384px)."""
    props = measure.regionprops(mask.astype(np.uint8), intensity_image=img)[0]
    contours = measure.find_contours(mask.astype(float), 0.5)
    c = max(contours, key=len)
    cy, cx = props.centroid
    radii = np.hypot(c[:, 0] - cy, c[:, 1] - cx)
    perimeter = float(props.perimeter) or 1.0
    area = float(props.area)
    hull_area = float(getattr(props, 'area_convex', None) or props.convex_area) or area
    # concave points: sign changes of curvature along a smoothed contour
    k = max(3, len(radii) // 40)
    sm = np.convolve(np.r_[radii[-k:], radii, radii[:k]], np.ones(k) / k, mode="same")[k:-k]
    d2 = np.diff(np.sign(np.diff(sm)))
    concave_pts = int((d2 > 0).sum())
    # symmetry: compare the two halves across the major axis
    rot = transform.rotate(mask.astype(float), -np.degrees(props.orientation) + 90, center=(cx, cy), order=0) > 0.5
    flipped = rot[::-1, :]
    sym = float((rot & flipped).sum() / max(1, (rot | flipped).sum()))
    inside = img[mask]
    ring = ndi.binary_dilation(mask, structure=morphology.disk(6)) & ~mask
    surround = img[ring] if ring.any() else img.ravel()
    return {
        "radius": float(radii.mean()),
        "texture": float(inside.std() * 100),
        "perimeter": perimeter,
        "area": area,
        "smoothness": float(np.abs(np.diff(radii)).mean() / max(radii.mean(), 1e-6)),
        "compactness": float(perimeter ** 2 / (4 * np.pi * area)),  # 1.0 = perfect circle
        "concavity": float(1 - area / hull_area),
        "concave_points": concave_pts,
        "symmetry": sym,
        "fractal_dimension": _fractal_dimension(mask),
        "eccentricity": float(props.eccentricity),
        "solidity": float(props.solidity),
        "margin_irregularity": float(radii.std() / max(radii.mean(), 1e-6)),
        "contrast_to_surround": float(inside.mean() - surround.mean()),
        "size_fraction": float(area / mask.size),
    }


def texture_features(img: np.ndarray) -> np.ndarray:
    g = (_resize_longest(img, FEAT_SIZE) * 255).astype(np.uint8)
    q = (g // 32).astype(np.uint8)
    glcm = feature.graycomatrix(q, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=8, symmetric=True, normed=True)
    gl = [feature.graycoprops(glcm, p).mean(axis=1) for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")]
    lbp = feature.local_binary_pattern(g, 8, 1, "uniform")
    lbp_h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    hist, _ = np.histogram(g, bins=16, range=(0, 255), density=True)
    sq = transform.resize(g / 255.0, (FEAT_SIZE, FEAT_SIZE), anti_aliasing=True)
    hog = feature.hog(sq, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)
    stats = [g.mean() / 255, g.std() / 255, float(((g - g.mean()) ** 3).mean() / (g.std() ** 3 + 1e-6)),
             float(((g - g.mean()) ** 4).mean() / (g.std() ** 4 + 1e-6))]
    return np.concatenate([np.ravel(gl), lbp_h, hist * 16, hog, stats]).astype(np.float32)


SHAPE_KEYS = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity",
              "concave_points", "symmetry", "fractal_dimension", "eccentricity", "solidity",
              "margin_irregularity", "contrast_to_surround", "size_fraction"]


def feature_vector(enhanced: np.ndarray, polarity: str = "auto"):
    mask, pol, _ = segment_lesion(enhanced, polarity)
    shp = shape_features(enhanced, mask)
    ys, xs = np.where(mask)
    pad = 16
    roi = enhanced[max(0, ys.min() - pad): ys.max() + pad, max(0, xs.min() - pad): xs.max() + pad]
    vec = np.concatenate([
        np.array([shp[k] for k in SHAPE_KEYS], dtype=np.float32),
        texture_features(enhanced),
        texture_features(roi) if min(roi.shape) >= 16 else np.zeros_like(texture_features(enhanced)),
    ])
    return np.nan_to_num(vec), mask, pol, shp


# ------------------------------------------------------------------ rendering
def _png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _colormap(v: np.ndarray) -> np.ndarray:
    """Perceptual blue→teal→amber→white ramp (colour-blind friendly)."""
    stops = np.array([[0.05, 0.08, 0.20], [0.05, 0.45, 0.60], [0.95, 0.62, 0.20], [1.0, 0.95, 0.85]])
    x = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def render(enhanced: np.ndarray, mask: np.ndarray, heat: np.ndarray | None):
    grey = np.dstack([enhanced] * 3)
    edge = mask ^ ndi.binary_erosion(mask, structure=morphology.disk(2))
    overlay = grey.copy()
    overlay[mask] = overlay[mask] * 0.65 + np.array([0.98, 0.62, 0.2]) * 0.35
    overlay[edge] = [1.0, 0.75, 0.3]
    out = {"enhanced": _png_b64(grey), "overlay": _png_b64(overlay)}
    if heat is not None:
        h = (heat - heat.min()) / (np.ptp(heat) + 1e-6)
        out["heatmap"] = _png_b64(grey * 0.45 + _colormap(h) * 0.55)
    return out


def irregularity_map(img: np.ndarray) -> np.ndarray:
    """Local texture irregularity (local std × gradient magnitude), no model needed."""
    m = ndi.uniform_filter(img, 9)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(img ** 2, 9) - m ** 2, 0))
    grad = filters.sobel(img)
    return filters.gaussian(sd * (grad + grad.mean()), sigma=6)


# ---------------------------------------------------------------------- model
class ScanModel:
    def __init__(self):
        self.bundle = None
        self.mtime = None
        self.reload()

    def reload(self):
        if os.path.exists(SCAN_MODEL_PATH):
            mt = os.path.getmtime(SCAN_MODEL_PATH)
            if mt != self.mtime:
                try:
                    self.bundle = joblib.load(SCAN_MODEL_PATH)
                    self.mtime = mt
                except Exception:
                    self.bundle = None
        else:
            self.bundle, self.mtime = None, None

    @property
    def ready(self) -> bool:
        self.reload()
        return self.bundle is not None

    def info(self) -> dict:
        if not self.ready:
            return {"ready": False}
        b = self.bundle
        return {"ready": True, "classes": b["classes"], "positive": b["positive"],
                "metrics": b.get("metrics", {}), "n_images": b.get("n_images"),
                "trained_at": b.get("trained_at"), "modality": b.get("modality", "unspecified")}

    def proba(self, vecs: np.ndarray) -> np.ndarray:
        """Probability of the positive (cancerous) class."""
        b = self.bundle
        p = b["pipeline"].predict_proba(vecs)
        return p[:, b["classes"].index(b["positive"])]

    def class_probas(self, vec: np.ndarray) -> dict:
        b = self.bundle
        p = b["pipeline"].predict_proba(vec[None])[0]
        return {c: float(v) for c, v in zip(b["classes"], p)}


def occlusion_map(enhanced, model: ScanModel, polarity, base_p, grid=6):
    h, w = enhanced.shape
    heat = np.zeros((grid, grid), np.float32)
    fill = float(np.median(enhanced))
    vecs = []
    for i in range(grid):
        for j in range(grid):
            im = enhanced.copy()
            im[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid] = fill
            vecs.append(feature_vector(im, polarity)[0])
    p = model.proba(np.stack(vecs)).reshape(grid, grid)
    heat = np.maximum(base_p - p, 0)  # regions whose removal lowers the cancer probability
    return filters.gaussian(transform.resize(heat, (h, w), order=1), sigma=8)


_CACHE: "dict[str, tuple]" = {}


def analyse(data: bytes, filename: str, model: ScanModel, polarity: str = "auto") -> dict:
    """Fast pass: enhancement, segmentation, measurements, classification.
    The (slower) explanation heatmap is computed separately by explain()."""
    import uuid
    raw = load_image(data, filename)
    enhanced = enhance(raw)
    vec, mask, pol, shp = feature_vector(enhanced, polarity)
    quality = {
        "width": int(raw.shape[1]), "height": int(raw.shape[0]),
        "contrast": float(enhanced.std()),
        "sharpness": float(filters.laplace(enhanced).var() * 1e4),
    }
    result = {"shape": shp, "polarity": pol, "quality": quality, "model": model.info(), "classified": False}
    p = None
    if model.ready:
        probs = model.class_probas(vec)
        p = probs[model.bundle["positive"]]
        result.update({
            "classified": True, "probability": p, "class_probabilities": probs,
            "label": "Suspicious for malignancy" if p >= 0.5 else "No malignancy detected",
            "risk": "High" if p >= 0.85 else "Elevated" if p >= 0.5 else "Uncertain" if p > 0.15 else "Low",
        })
    heat = irregularity_map(enhanced)
    result["heatmap_kind"] = "irregularity"
    result["images"] = render(enhanced, mask, heat)
    ys, xs = np.where(mask)
    result["roi"] = {"x": float(xs.min() / mask.shape[1]), "y": float(ys.min() / mask.shape[0]),
                     "w": float((xs.max() - xs.min() + 1) / mask.shape[1]), "h": float((ys.max() - ys.min() + 1) / mask.shape[0])}
    hist, _ = np.histogram(enhanced, bins=48, range=(0, 1))
    result["histogram"] = (hist / max(1, hist.max())).round(4).tolist()
    sid = uuid.uuid4().hex[:12]
    _CACHE[sid] = (enhanced, mask, pol, p)
    while len(_CACHE) > 20:
        _CACHE.pop(next(iter(_CACHE)))
    result["id"] = sid
    return result


def explain(sid: str, model: ScanModel) -> dict:
    """Occlusion-sensitivity heatmap: which parts of the scan drive the cancer score."""
    if sid not in _CACHE:
        raise KeyError("Scan expired; please analyse it again.")
    enhanced, mask, pol, p = _CACHE[sid]
    if not model.ready or p is None:
        raise ValueError("Explanations need a trained scan model.")
    heat = occlusion_map(enhanced, model, pol, p)
    return {"heatmap": render(enhanced, mask, heat)["heatmap"], "heatmap_kind": "occlusion"}
