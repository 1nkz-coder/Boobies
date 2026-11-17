# %% [markdown]
# # BreastScan Twin: train all models on Kaggle
#
# This notebook trains three models and packs them into **one zip** for the BreastScan Twin app:
#
# | Part | Dataset | Model | Output |
# |---|---|---|---|
# | A | `uciml/breast-cancer-wisconsin-data` | Best of LR / SVM / RF / GB / KNN (scikit-learn) | `tabular_model.joblib` |
# | B | `paultimothymooney/breast-histopathology-images` | EfficientNet-B0, IDC vs non-IDC tissue patches | `histopathology/model.onnx` |
# | C | `awsaf49/cbis-ddsm-breast-cancer-image-dataset` | EfficientNet-B0, malignant vs benign mammograms | `mammogram/model.onnx` |
#
# **Before you run:**
# 1. Right-hand panel → **Session options → Accelerator → GPU T4 x2** (or P100).
# 2. Right-hand panel → **Internet → On** (needed once to download the ImageNet starting weights).
# 3. Add the three datasets with **+ Add Input** (you already have).
# 4. **Run All** in a fresh session (Run → Restart & clear outputs first). It takes roughly 1–1.5 hours on a T4.
#    Please don't edit the code cells; every setting lives in the CONFIG cell.
#    If one part fails, the others still run and the zip is still made. Each model is saved the moment
#    its training ends, so re-running after a fix reuses it instead of training again.
# 5. When it finishes, download **`/kaggle/working/breastscan_models.zip`** from the Output panel
#    and unzip it into the app's `models/` folder.
#
# Everything you might want to change is in the **CONFIG** cell.

# %%
# ============================== CONFIG ==============================
CONFIG = {
    # Dataset locations. If a path doesn't exist, the notebook searches /kaggle/input for it.
    "WDBC_DIR":  "/kaggle/input/datasets/uciml/breast-cancer-wisconsin-data",
    "HISTO_DIR": "/kaggle/input/datasets/paultimothymooney/breast-histopathology-images",
    "CBIS_DIR":  "/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset",
    "OUT_DIR":   "/kaggle/working/breastscan_models",

    # Which parts to run (set False to skip one)
    "RUN_TABULAR": True,
    "RUN_HISTO":   True,
    "RUN_MAMMO":   True,

    # Part B: histopathology (IDC) patches
    "HISTO_MAX_PATCHES": 90_000,   # cap on patches used (dataset has ~277k); more = slower, slightly better
    "HISTO_IMG": 96,               # patches are 50x50 px; upsampled to this size for the network
    "HISTO_EPOCHS": 8,
    "HISTO_BATCH": 256,
    "HISTO_LR": 1e-3,

    # Part C: CBIS-DDSM full mammograms
    "MAMMO_H": 640, "MAMMO_W": 384,  # input size (tall, like a mammogram)
    "MAMMO_EPOCHS": 20,
    "MAMMO_BATCH": 16,
    "MAMMO_LR": 3e-4,

    # Shared
    "PATIENCE": 5,           # early stopping on validation AUC
    "SEED": 7,
    "NUM_WORKERS": 4,
    "ALLOW_NO_PRETRAINED": False,  # True = train from scratch if ImageNet weights can't download (much worse)
    "REUSE_CHECKPOINTS": True,     # if a model_state.pt from an earlier run in this session exists, skip training and reuse it
}
print(CONFIG)

# %%
# ============================== SETUP ==============================
import os, re, gc, json, time, glob, random, zipfile, warnings, datetime as dt
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(CONFIG["SEED"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
OUT = CONFIG["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)
print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | device: {DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
if DEVICE.type != "cuda":
    print("⚠️  No GPU detected. Turn on a GPU accelerator in Session options, otherwise parts B and C will be very slow.")

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k): return x

def now(): return dt.datetime.now().isoformat(timespec="minutes")

# results of each part (stay None if a part is skipped or fails, so the packaging cell always works)
TAB_METRICS = HISTO_METRICS = MAMMO_METRICS = None

def run_part(name, fn):
    """Run one part; if it fails, print the error and carry on with the next part."""
    import traceback
    try:
        return fn()
    except Exception:
        print(f"\n❌ {name} failed. The other parts will still run. Error:\n")
        traceback.print_exc()
        return None

# %%
# ============================== FIND DATASETS ==============================
def find_dir(preferred, marker, max_depth=6):
    """Return `preferred` if it contains `marker`, else search /kaggle/input for a folder that does."""
    def has(d):
        return os.path.exists(os.path.join(d, marker))
    if preferred and os.path.isdir(preferred):
        if has(preferred):
            return preferred
        for root, dirs, files in os.walk(preferred):
            if root[len(preferred):].count(os.sep) > max_depth: dirs[:] = []; continue
            if marker in files or marker in dirs:
                return root
    for base in ["/kaggle/input"]:
        if not os.path.isdir(base): continue
        for root, dirs, files in os.walk(base):
            if root[len(base):].count(os.sep) > max_depth: dirs[:] = []; continue
            if marker in files or marker in dirs:
                return root
            # don't descend into big image folders while searching
            if len(files) > 200: dirs[:] = []
    return None

WDBC_DIR = find_dir(CONFIG["WDBC_DIR"], "data.csv") if CONFIG["RUN_TABULAR"] else None
CBIS_CSV = find_dir(CONFIG["CBIS_DIR"], "dicom_info.csv") if CONFIG["RUN_MAMMO"] else None
HISTO_DIR = CONFIG["HISTO_DIR"] if os.path.isdir(CONFIG["HISTO_DIR"]) else find_dir("/kaggle/input", "IDC_regular_ps50_idx5")
print("WDBC     :", WDBC_DIR)
print("CBIS csv :", CBIS_CSV)
print("Histology:", HISTO_DIR)

# %% [markdown]
# ## Part A: Wisconsin (WDBC) tabular model for the Digital Twin
# Compares several algorithms with 10-fold cross-validation on the training split (70/30, `random_state=7`,
# exactly as in the original notebook), picks the one with the best CV balanced accuracy and evaluates it once on the test split.

# %%
def run_tabular():
    import sklearn, joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    df = pd.read_csv(os.path.join(WDBC_DIR, "data.csv"))
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("id",) or c.startswith("Unnamed")])
    y = (df.pop("diagnosis").str.upper().str[0] == "M").astype(int).values  # M=1, B=0
    feature_names = list(df.columns)
    X = df.values.astype(float)
    assert X.shape[1] == 30, f"Expected 30 features, got {X.shape[1]}: {feature_names}"
    print(f"WDBC: {X.shape[0]} cases | malignant={y.sum()} benign={(1-y).sum()}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7)
    cv = StratifiedKFold(10, shuffle=True, random_state=7)

    def svc(C): return CalibratedClassifierCV(SVC(kernel="rbf", C=C, gamma="scale"), method="sigmoid", cv=5)
    algos = {
        "LR":   LogisticRegression(max_iter=5000),
        "LDA":  LinearDiscriminantAnalysis(),
        "KNN":  KNeighborsClassifier(n_neighbors=9),
        "CART": DecisionTreeClassifier(random_state=7),
        "NB":   GaussianNB(),
        "SVM":  svc(10),
        "RF":   RandomForestClassifier(n_estimators=500, random_state=7, n_jobs=-1),
        "GB":   GradientBoostingClassifier(random_state=7),
    }
    comparison, best = [], None
    for name, clf in algos.items():
        raw = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1).mean()
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        sc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        bal = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="balanced_accuracy", n_jobs=-1).mean()
        comparison.append({"name": name, "raw": round(float(raw), 4), "scaled": round(float(sc.mean()), 4),
                           "cv_auc": round(float(auc), 4), "cv_balanced_acc": round(float(bal), 4)})
        print(f"  {name:5s} raw acc {raw:.4f} | scaled acc {sc.mean():.4f} ± {sc.std():.4f} | balanced {bal:.4f} | AUC {auc:.4f}")
        # pick by balanced accuracy (treats missed cancers and false alarms equally), AUC breaks ties
        if best is None or (bal, auc) > (best[1], best[4]):
            best = (name, bal, sc.mean(), sc.std(), auc)

    # small tuning pass for the winner if it is the SVM or LR
    name = best[0]
    clf = algos[name]
    if name == "SVM":
        scores = {C: cross_val_score(Pipeline([("scaler", StandardScaler()), ("classifier", svc(C))]),
                                     X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean() for C in (0.5, 1, 3, 10, 30, 100)}
        C = max(scores, key=scores.get); clf = svc(C); name = f"SVM (RBF, C={C})"
    elif name == "LR":
        scores = {C: cross_val_score(Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(C=C, max_iter=5000))]),
                                     X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean() for C in (0.01, 0.1, 0.3, 1, 3, 10)}
        C = max(scores, key=scores.get); clf = LogisticRegression(C=C, max_iter=5000); name = f"Logistic Regression (C={C})"

    model = Pipeline([("scaler", StandardScaler()), ("classifier", clf)]).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    metrics = {
        "model": f"StandardScaler → {name}",
        "dataset": "Wisconsin Diagnostic Breast Cancer (569 cases, 30 features) — Kaggle uciml/breast-cancer-wisconsin-data",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, pred)), "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)), "f1": float(f1_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, proba)), "train_accuracy": float(model.score(X_tr, y_tr)),
        "cv_mean": float(best[2]), "cv_std": float(best[3]),
        "confusion": confusion_matrix(y_te, pred).tolist(),
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        "notebook_comparison": comparison,
        "feature_names": feature_names,
        "sklearn_version": sklearn.__version__, "trained_at": now(), "source": "kaggle",
    }
    joblib.dump(model, os.path.join(OUT, "tabular_model.joblib"))
    with open(os.path.join(OUT, "tabular_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Tabular model: {metrics['model']}\n   test accuracy {metrics['accuracy']:.4f} | AUC {metrics['auc']:.4f} | "
          f"sensitivity {metrics['recall']:.4f}\n   confusion [[TN, FP], [FN, TP]] = {metrics['confusion']}")
    return metrics

if CONFIG["RUN_TABULAR"] and WDBC_DIR:
    TAB_METRICS = run_part("Part A (tabular)", run_tabular)

# %% [markdown]
# ## Shared image tools
# * **Preprocessing** functions here are copied verbatim into the app, so training and prediction see identical images.
# * **Model**: ImageNet-pretrained EfficientNet-B0, fine-tuned end to end with mixed precision, class-balanced loss,
#   cosine learning rate and early stopping on validation AUC.
# * **Export**: ONNX with two outputs, `probs` and a class activation map `cam` (shows *where* the model looked).
#   The app runs it with `onnxruntime`, so PyTorch is not needed on your laptop.

# %%
# ---------- preprocessing (identical in the app) ----------
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

# ---------- model ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_effnet(n_classes=2):
    try:
        w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        m = torchvision.models.efficientnet_b0(weights=w)
        print("Loaded ImageNet-pretrained EfficientNet-B0")
    except Exception as e:
        if not CONFIG["ALLOW_NO_PRETRAINED"]:
            raise RuntimeError("Could not download ImageNet weights. Turn Internet ON in the notebook settings "
                               "(or set ALLOW_NO_PRETRAINED=True to train from scratch, which is much worse).") from e
        print("⚠️  Training from scratch (no pretrained weights):", e)
        m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m

class CAMExport(nn.Module):
    """Wraps the classifier for export. Input: (B,3,H,W) float32 in [0,1].
    Outputs: probs (B,K) and class-activation maps cam (B,K,h,w)."""
    def __init__(self, m):
        super().__init__()
        self.features, self.pool, self.fc = m.features, m.avgpool, m.classifier[1]
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
    def forward(self, x):
        f = self.features((x - self.mean) / self.std)
        logits = self.fc(torch.flatten(self.pool(f), 1))
        cam = torch.einsum("bchw,kc->bkhw", f, self.fc.weight)
        return torch.softmax(logits, dim=1), cam

def normalize_batch(x):  # x float in [0,1], (B,3,H,W)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

# ---------- training ----------
def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=USE_AMP)

def predict_probs(model, loader, gpu_tf=None, tta=True):
    model.eval(); out, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True).float() / 255.0
            if gpu_tf: xb = gpu_tf(xb, train=False)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                p = torch.softmax(model(normalize_batch(xb)).float(), 1)
                if tta:
                    p = (p + torch.softmax(model(normalize_batch(torch.flip(xb, dims=[3]))).float(), 1)) / 2
            out.append(p[:, 1].float().cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(out)

def train_model(name, model, train_dl, val_dl, epochs, lr, class_weights, gpu_tf=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
    scaler = make_scaler()
    best_auc, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for xb, yb in tqdm(train_dl, desc=f"{name} epoch {ep}/{epochs}", leave=False):
            xb = xb.to(DEVICE, non_blocking=True).float() / 255.0
            yb = yb.to(DEVICE, non_blocking=True)
            if gpu_tf: xb = gpu_tf(xb, train=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = crit(model(normalize_batch(xb)), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(yb); n += len(yb)
        yv, pv = predict_probs(model, val_dl, gpu_tf, tta=False)
        auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else float("nan")
        acc = accuracy_score(yv, pv >= 0.5)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_auc": auc, "val_acc": acc})
        print(f"{name} | epoch {ep:2d} | loss {tot / max(n, 1):.4f} | val AUC {auc:.4f} | val acc {acc:.4f} | {time.time() - t0:.0f}s")
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CONFIG["PATIENCE"]:
                print(f"Early stopping (no val AUC improvement for {CONFIG['PATIENCE']} epochs)."); break
    model.load_state_dict(best_state)
    return model, hist, best_auc

def evaluate(name, model, val_dl, test_dl, gpu_tf=None):
    yv, pv = predict_probs(model, val_dl, gpu_tf)
    fpr, tpr, thr = roc_curve(yv, pv)
    j = int(np.argmax(tpr - fpr)); threshold = float(np.clip(thr[j], 0.05, 0.95))  # Youden's J on validation
    yt, pt = predict_probs(model, test_dl, gpu_tf)
    res = {}
    for tag, t in (("at_0.5", 0.5), ("at_threshold", threshold)):
        pred = (pt >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        res[tag] = {"threshold": float(t), "accuracy": float(accuracy_score(yt, pred)),
                    "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
                    "precision": float(tp / max(tp + fp, 1)), "f1": float(f1_score(yt, pred, zero_division=0)),
                    "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    tf, tt, _ = roc_curve(yt, pt)
    keep = np.unique(np.linspace(0, len(tf) - 1, min(200, len(tf))).astype(int))
    metrics = {"auc": float(roc_auc_score(yt, pt)), "threshold": threshold, **res["at_threshold"],
               "at_0.5": res["at_0.5"], "n_test": int(len(yt)),
               "roc": {"fpr": [round(float(v), 4) for v in tf[keep]], "tpr": [round(float(v), 4) for v in tt[keep]]}}
    print(f"\n✅ {name} TEST: AUC {metrics['auc']:.4f} | threshold {threshold:.3f} | acc {metrics['accuracy']:.4f} | "
          f"sensitivity {metrics['sensitivity']:.4f} | specificity {metrics['specificity']:.4f}")
    print(f"   at 0.5: acc {res['at_0.5']['accuracy']:.4f} | sens {res['at_0.5']['sensitivity']:.4f} | spec {res['at_0.5']['specificity']:.4f}")
    return metrics, (yt, pt)

def save_plots(folder, title, hist, metrics, yt_pt):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    e = [h["epoch"] for h in hist]
    ax[0].plot(e, [h["train_loss"] for h in hist], "o-"); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(e, [h["val_auc"] for h in hist], "o-", label="val AUC")
    ax[1].plot(e, [h["val_acc"] for h in hist], "s--", label="val acc"); ax[1].legend(); ax[1].set_title("Validation"); ax[1].set_xlabel("epoch")
    ax[2].plot(metrics["roc"]["fpr"], metrics["roc"]["tpr"], lw=2, label=f"AUC = {metrics['auc']:.3f}")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1); ax[2].set_title("Test ROC"); ax[2].set_xlabel("FPR"); ax[2].set_ylabel("TPR"); ax[2].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(folder, "training_report.png"), dpi=120); plt.close(fig)
    cm = np.array(metrics["confusion"])
    fig, a = plt.subplots(figsize=(4.5, 4))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_title(f"Confusion (threshold {metrics['threshold']:.2f})"); fig.tight_layout()
    fig.savefig(os.path.join(folder, "confusion_matrix.png"), dpi=120); plt.close(fig)

def export_model(folder, model, in_h, in_w, meta):
    os.makedirs(folder, exist_ok=True)
    model = model.eval().float().cpu()
    if not os.path.exists(os.path.join(folder, "model_state.pt")):
        torch.save(model.state_dict(), os.path.join(folder, "model_state.pt"))
    wrapper = CAMExport(model).eval()
    dummy = torch.rand(1, 3, in_h, in_w)
    onnx_path = os.path.join(folder, "model.onnx")
    kw = dict(input_names=["image"], output_names=["probs", "cam"],
              dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}, "cam": {0: "batch"}}, opset_version=17)
    try:
        try:
            torch.onnx.export(wrapper, dummy, onnx_path, dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(wrapper, dummy, onnx_path, **kw)
        # merge external weight data (if any) into one file
        try:
            import onnx
            m = onnx.load(onnx_path, load_external_data=True); onnx.save(m, onnx_path, save_as_external_data=False)
            for f in glob.glob(onnx_path + ".data"): os.remove(f)
        except Exception:
            pass
        meta["onnx"] = True
        print("Exported", onnx_path, f"({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            x = torch.rand(2, 3, in_h, in_w)
            p_ort = sess.run(None, {"image": x.numpy()})[0]
            with torch.no_grad():
                p_t = wrapper(x)[0].numpy()
            print(f"ONNX check: max |onnxruntime - torch| = {np.abs(p_ort - p_t).max():.2e}")
        except ImportError:
            print("(onnxruntime not installed here; skipping the ONNX check. The app will still load the model.)")
    except Exception as e:
        meta["onnx"] = False
        print("⚠️  ONNX export failed, keeping model_state.pt only:", repr(e))
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def balanced_weights(y):
    c = np.bincount(y, minlength=2).astype(float)
    return (c.sum() / (2 * np.maximum(c, 1))).tolist()

def train_or_reuse(name, folder, train_dl, val_dl, epochs, lr, class_weights, gpu_tf=None):
    """Train, then IMMEDIATELY save the weights (so a later error can't lose them).
    If REUSE_CHECKPOINTS is on and weights from an earlier run exist, load them instead of training."""
    os.makedirs(folder, exist_ok=True)
    ckpt = os.path.join(folder, "model_state.pt")
    hist_path = os.path.join(folder, "history.json")
    model = build_effnet(2)
    if CONFIG["REUSE_CHECKPOINTS"] and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
        print(f"♻️  Reusing trained weights from {ckpt} (set REUSE_CHECKPOINTS=False to retrain)")
        return model.to(DEVICE), hist
    model, hist, _ = train_model(name, model, train_dl, val_dl, epochs, lr, class_weights, gpu_tf)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
    with open(hist_path, "w") as f:
        json.dump(hist, f)
    print(f"💾 Saved trained weights to {ckpt}")
    return model, hist

# %% [markdown]
# ## Part B: Histopathology. Invasive ductal carcinoma (IDC) in tissue patches
# * 50×50 px patches from 162 whole-slide images (40×). Label comes from the file name (`..._class1.png` = IDC).
# * The split is **by patient** (70/15/15), so no patient appears in both training and test (avoids leakage).

# %%
def _read_patch(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None:
        return np.zeros((50, 50, 3), np.uint8)
    return histo_preprocess(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 50)

class ArrayDS(Dataset):
    """Images already in memory as uint8 (N,H,W) or (N,H,W,3); light CPU augmentation."""
    def __init__(self, X, y, train=False, kind="histo"):
        self.X, self.y, self.train, self.kind = X, y, train, kind
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        im = self.X[i]
        if self.train:
            if random.random() < 0.5: im = im[:, ::-1]
            if self.kind == "histo":
                if random.random() < 0.5: im = im[::-1, :]
                im = np.rot90(im, random.randint(0, 3))
            else:  # mammogram: mild zoom/shift + brightness/contrast
                if random.random() < 0.7:
                    h, w = im.shape[:2]
                    sc = random.uniform(0.9, 1.1); ang = random.uniform(-8, 8)
                    M = cv2.getRotationMatrix2D((w * 0.3, h / 2), ang, sc)
                    M[:, 2] += (random.uniform(-0.04, 0.04) * w, random.uniform(-0.04, 0.04) * h)
                    im = cv2.warpAffine(np.ascontiguousarray(im), M, (w, h), borderValue=0)
                if random.random() < 0.7:
                    a = random.uniform(0.85, 1.15); b = random.uniform(-15, 15)
                    im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        im = np.ascontiguousarray(im)
        t = torch.from_numpy(im)
        t = t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0).repeat(3, 1, 1)
        return t, int(self.y[i])

def run_histo():
    t0 = time.time()
    # The dataset stores every patch twice (top-level patient folders AND IDC_regular_ps50_idx5/).
    # Walking only one copy halves the indexing time.
    walk_root = os.path.join(HISTO_DIR, "IDC_regular_ps50_idx5")
    if not os.path.isdir(walk_root):
        walk_root = HISTO_DIR
    print(f"Indexing patches under {walk_root} (several minutes on Kaggle's disk)...")
    seen, rows = set(), []
    for root, _, files in os.walk(walk_root):
        for f in files:
            if not f.endswith(".png") or f in seen:  # dataset contains a duplicate copy of every patch
                continue
            mm = re.search(r"class([01])", f)
            if not mm: continue
            seen.add(f)
            rows.append((os.path.join(root, f), f.split("_")[0], int(mm.group(1))))
    df = pd.DataFrame(rows, columns=["path", "patient", "label"])
    print(f"Found {len(df):,} unique patches from {df.patient.nunique()} patients | IDC+ {df.label.mean():.1%} | {time.time() - t0:.0f}s")
    if df.patient.nunique() < 10:
        raise RuntimeError("Too few patients found. Check HISTO_DIR.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=CONFIG["SEED"])
    tr_idx, rest = next(gss.split(df, groups=df.patient))
    rest_df = df.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=CONFIG["SEED"])
    va_i, te_i = next(gss2.split(rest_df, groups=rest_df.patient))
    parts = {"train": df.iloc[tr_idx], "val": rest_df.iloc[va_i], "test": rest_df.iloc[te_i]}

    cap = CONFIG["HISTO_MAX_PATCHES"]
    share = {"train": 0.70, "val": 0.15, "test": 0.15}
    for k in parts:
        n = int(cap * share[k])
        if len(parts[k]) > n:
            parts[k] = parts[k].sample(n, random_state=CONFIG["SEED"])
        print(f"  {k:5s}: {len(parts[k]):6,} patches | {parts[k].patient.nunique():3d} patients | IDC+ {parts[k].label.mean():.1%}")

    arrays = {}
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for k, d in parts.items():
            imgs = list(tqdm(pool.imap(_read_patch, d.path.tolist(), chunksize=256), total=len(d), desc=f"loading {k}"))
            arrays[k] = (np.stack(imgs), d.label.values.astype(np.int64))
    print(f"Loaded patches into memory in {time.time() - t0:.0f}s")

    size = CONFIG["HISTO_IMG"]
    def gpu_tf(x, train):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        if train:  # stain/colour jitter on GPU
            b = x.shape[0]
            x = x * (1 + 0.15 * (torch.rand(b, 3, 1, 1, device=x.device) - 0.5)) + 0.08 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
            x = x.clamp(0, 1)
        return x

    kw = dict(num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    train_dl = DataLoader(ArrayDS(*arrays["train"], train=True), batch_size=CONFIG["HISTO_BATCH"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(ArrayDS(*arrays["val"]), batch_size=CONFIG["HISTO_BATCH"] * 2, **kw)
    test_dl = DataLoader(ArrayDS(*arrays["test"]), batch_size=CONFIG["HISTO_BATCH"] * 2, **kw)

    folder = os.path.join(OUT, "histopathology")
    model, hist = train_or_reuse("Histopathology", folder, train_dl, val_dl, CONFIG["HISTO_EPOCHS"],
                                 CONFIG["HISTO_LR"], balanced_weights(arrays["train"][1]), gpu_tf)
    metrics, yt_pt = evaluate("Histopathology", model, val_dl, test_dl, gpu_tf)
    meta = {
        "modality": "histopathology", "title": "Histopathology: invasive ductal carcinoma (IDC)",
        "classes": ["non-IDC", "IDC"], "positive": "IDC",
        "input": {"height": size, "width": size, "channels": "RGB", "range": "0-1"},
        "preprocess": {"type": "histo_patch", "patch_px": 50, "resize_to": size,
                       "note": "Patches are 50x50 px at 40x magnification. Larger images are tiled into 50x50 patches."},
        "threshold": metrics["threshold"], "metrics": metrics, "history": hist,
        "dataset": "Kaggle paultimothymooney/breast-histopathology-images (IDC, 162 slides)",
        "split": "patient-level 70/15/15", "n": {k: int(len(v[1])) for k, v in arrays.items()},
        "architecture": "EfficientNet-B0 (ImageNet pretrained)", "trained_at": now(),
    }
    save_plots(folder, "Histopathology (IDC) EfficientNet-B0", hist, metrics, yt_pt)
    export_model(folder, model, size, size, meta)
    del arrays, train_dl, val_dl, test_dl; gc.collect(); torch.cuda.empty_cache()
    return metrics

if CONFIG["RUN_HISTO"] and HISTO_DIR:
    HISTO_METRICS = run_part("Part B (histopathology)", run_histo)

# %% [markdown]
# ## Part C: CBIS-DDSM mammograms. Malignant vs benign
# * Uses the **full mammogram** images (what a user would upload), with the official train/test split.
#   Validation is 15% of the training patients.
# * An image is labelled malignant if **any** lesion on it is malignant (`BENIGN_WITHOUT_CALLBACK` counts as benign).
# * Note: every CBIS-DDSM image contains a lesion, so this model separates *malignant from benign findings*;
#   it was not trained on normal (lesion-free) mammograms.

# %%
def _norm_cols(d):
    d.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in d.columns]
    return d

def build_cbis_table():
    csv_dir = CBIS_CSV
    root = os.path.dirname(csv_dir.rstrip("/"))
    jpeg_dir = os.path.join(root, "jpeg")
    if not os.path.isdir(jpeg_dir):
        jpeg_dir = find_dir(root, "jpeg") and os.path.join(find_dir(root, "jpeg"), "jpeg")
    print("CBIS jpeg dir:", jpeg_dir)

    info = _norm_cols(pd.read_csv(os.path.join(csv_dir, "dicom_info.csv")))
    # image_path looks like "CBIS-DDSM/jpeg/<series folder>/1-249.jpg"
    parts_ = info["image_path"].astype(str).str.strip().str.split("/")
    info["series"] = parts_.str[-2]
    info["file"] = parts_.str[-1]
    info["desc"] = info["seriesdescription"].astype(str).str.lower()
    full_by_series = {}
    for s, f, d in zip(info.series, info.file, info.desc):
        if "full mammogram" in d:
            p = os.path.join(jpeg_dir, s, f)
            full_by_series[s] = p

    cases = []
    for name in ["mass_case_description_train_set.csv", "calc_case_description_train_set.csv",
                 "mass_case_description_test_set.csv", "calc_case_description_test_set.csv"]:
        p = os.path.join(csv_dir, name)
        if not os.path.exists(p):
            print("  missing", name); continue
        d = _norm_cols(pd.read_csv(p))
        d["split"] = "test" if "test" in name else "train"
        d["lesion"] = "mass" if name.startswith("mass") else "calcification"
        cases.append(d)
    cases = pd.concat(cases, ignore_index=True)
    cases["series"] = cases["image_file_path"].astype(str).str.strip().str.split("/").str[2].str.strip()
    cases["malignant"] = (cases["pathology"].astype(str).str.upper() == "MALIGNANT").astype(int)
    cases["path"] = cases["series"].map(full_by_series)

    # fall back: if dicom_info mapping misses, take the single jpg in the series folder
    miss = cases["path"].isna() | ~cases["path"].fillna("").map(os.path.exists)
    if miss.any():
        def first_jpg(s):
            d = os.path.join(jpeg_dir, s)
            if os.path.isdir(d):
                fs = sorted(x for x in os.listdir(d) if x.lower().endswith((".jpg", ".jpeg", ".png")))
                return os.path.join(d, fs[0]) if len(fs) == 1 else None
            return None
        cases.loc[miss, "path"] = cases.loc[miss, "series"].map(first_jpg)
    found = cases["path"].notna() & cases["path"].fillna("").map(os.path.exists)
    print(f"Matched {found.sum():,}/{len(cases):,} lesion rows to full-mammogram JPEGs")
    cases = cases[found]

    img = (cases.groupby("path")
           .agg(malignant=("malignant", "max"), split=("split", "first"), patient=("patient_id", "first"),
                lesion=("lesion", lambda s: "+".join(sorted(set(s)))))
           .reset_index())
    print(f"Images: {len(img):,} | malignant {img.malignant.mean():.1%} | train {sum(img.split == 'train')} / test {sum(img.split == 'test')}")
    return img

def _load_mammo(args):
    p, h, w = args
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return np.zeros((h, w), np.uint8)
    return mammo_preprocess(g, h, w)

def run_mammo():
    t0 = time.time()
    img = build_cbis_table()
    if len(img) < 100:
        raise RuntimeError("Too few mammograms matched. Check CBIS_DIR.")
    train_all = img[img.split == "train"].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=CONFIG["SEED"])
    tr_i, va_i = next(gss.split(train_all, groups=train_all.patient))
    parts = {"train": train_all.iloc[tr_i], "val": train_all.iloc[va_i], "test": img[img.split == "test"]}
    H, W = CONFIG["MAMMO_H"], CONFIG["MAMMO_W"]
    arrays = {}
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for k, d in parts.items():
            ims = list(tqdm(pool.imap(_load_mammo, [(p, H, W) for p in d.path], chunksize=4), total=len(d), desc=f"preprocessing {k}"))
            arrays[k] = (np.stack(ims), d.malignant.values.astype(np.int64))
            print(f"  {k:5s}: {len(d):5,} images | {d.patient.nunique():4d} patients | malignant {d.malignant.mean():.1%}")
    print(f"Preprocessed mammograms in {time.time() - t0:.0f}s")
    # save a few examples so you can check the preprocessing
    ex = np.concatenate([arrays["train"][0][i] for i in range(min(6, len(arrays["train"][0])))], axis=1)
    folder = os.path.join(OUT, "mammogram"); os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "preprocessing_examples.jpg"), ex)

    kw = dict(num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    train_dl = DataLoader(ArrayDS(*arrays["train"], train=True, kind="mammo"), batch_size=CONFIG["MAMMO_BATCH"], shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(ArrayDS(*arrays["val"], kind="mammo"), batch_size=CONFIG["MAMMO_BATCH"] * 2, **kw)
    test_dl = DataLoader(ArrayDS(*arrays["test"], kind="mammo"), batch_size=CONFIG["MAMMO_BATCH"] * 2, **kw)

    model, hist = train_or_reuse("Mammogram", folder, train_dl, val_dl, CONFIG["MAMMO_EPOCHS"],
                                 CONFIG["MAMMO_LR"], balanced_weights(arrays["train"][1]))
    metrics, yt_pt = evaluate("Mammogram", model, val_dl, test_dl)
    meta = {
        "modality": "mammogram", "title": "Mammogram: malignant vs benign finding",
        "classes": ["benign", "malignant"], "positive": "malignant",
        "input": {"height": H, "width": W, "channels": "gray→3", "range": "0-1"},
        "preprocess": {"type": "mammo", "height": H, "width": W,
                       "steps": ["breast crop (Otsu, largest component)", "breast facing left", "CLAHE 2.0/8x8", "letterbox"]},
        "threshold": metrics["threshold"], "metrics": metrics, "history": hist,
        "dataset": "Kaggle awsaf49/cbis-ddsm-breast-cancer-image-dataset (CBIS-DDSM, full mammograms)",
        "split": "official CBIS-DDSM train/test; validation = 15% of training patients",
        "n": {k: int(len(v[1])) for k, v in arrays.items()},
        "caveat": "Trained only on mammograms that contain a lesion; it ranks malignant vs benign findings, it was not trained to detect normal studies.",
        "architecture": "EfficientNet-B0 (ImageNet pretrained)", "trained_at": now(),
    }
    save_plots(folder, "CBIS-DDSM mammograms EfficientNet-B0", hist, metrics, yt_pt)
    export_model(folder, model, H, W, meta)
    del arrays, train_dl, val_dl, test_dl; gc.collect(); torch.cuda.empty_cache()
    return metrics

if CONFIG["RUN_MAMMO"] and CBIS_CSV:
    MAMMO_METRICS = run_part("Part C (mammogram)", run_mammo)

# %% [markdown]
# ## Package everything for the app

# %%
def pick(m, keys):
    return {k: m.get(k) for k in keys} if isinstance(m, dict) else None

TAB_METRICS, HISTO_METRICS, MAMMO_METRICS = (globals().get(n) for n in ("TAB_METRICS", "HISTO_METRICS", "MAMMO_METRICS"))
summary = {"created": now(),
           "tabular": pick(TAB_METRICS, ("model", "accuracy", "auc", "recall")),
           "histopathology": pick(HISTO_METRICS, ("auc", "accuracy", "sensitivity", "specificity", "threshold")),
           "mammogram": pick(MAMMO_METRICS, ("auc", "accuracy", "sensitivity", "specificity", "threshold"))}
missing = [k for k in ("histopathology", "mammogram") if CONFIG[{"histopathology": "RUN_HISTO", "mammogram": "RUN_MAMMO"}[k]]
           and not os.path.exists(os.path.join(OUT, k, "model.onnx"))]
if missing:
    print("⚠️  No model.onnx for:", ", ".join(missing), "(scroll up for the error message of that part)")
with open(os.path.join(OUT, "SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write("BreastScan Twin models trained on Kaggle\n\n"
            "Unzip into the app's  models/  folder so you get:\n"
            "  models/tabular_model.joblib, models/tabular_metrics.json\n"
            "  models/histopathology/model.onnx + meta.json\n"
            "  models/mammogram/model.onnx + meta.json\n\n" + json.dumps(summary, indent=2))

zip_path = "/kaggle/working/breastscan_models.zip" if OUT.startswith("/kaggle") else OUT.rstrip("/") + ".zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in files:
            if fn == "model_state.pt":  # PyTorch backup; not needed by the app (keeps the zip small)
                continue
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, OUT))
print(json.dumps(summary, indent=2))
print(f"\n📦 Download this file from the Output panel: {zip_path}  ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
