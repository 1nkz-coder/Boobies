"""
Train the Scan Lab classifier on YOUR labelled scans.

Folder layout (one sub-folder per class, any image format incl. DICOM):

    data/scans/
        benign/      img001.png  img002.jpg ...
        malignant/   img101.png  ...
        normal/      (optional third class)

Run:
    python train_scan_model.py
    python train_scan_model.py --data "D:/datasets/breast_mri" --modality MRI

Files whose name contains "_mask" (segmentation masks, e.g. in the BUSI
dataset) are skipped automatically.

The model: hand-crafted radiomics-style features (lesion shape, GLCM / LBP
texture, HOG, intensity) → StandardScaler → SVM / Random Forest, picked by
5-fold cross-validation, in the same spirit as the notebook.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from core.scan import IMAGE_EXTS, SCAN_MODEL_PATH, enhance, feature_vector, load_image

POSITIVE_NAMES = ("malignant", "cancer", "cancerous", "tumor", "tumour", "positive", "m", "1")


def find_images(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))
    items = []
    for c in classes:
        for dirpath, _, files in os.walk(os.path.join(root, c)):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS and "_mask" not in f.lower():
                    items.append((os.path.join(dirpath, f), c))
    return classes, items


def extract(path, polarity, augment):
    with open(path, "rb") as fh:
        data = fh.read()
    img = enhance(load_image(data, path))
    vecs = [feature_vector(img, polarity)[0]]
    if augment:
        vecs.append(feature_vector(img[:, ::-1].copy(), polarity)[0])
    return vecs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("data", "scans"), help="folder with one sub-folder per class")
    ap.add_argument("--positive", default=None, help="name of the cancerous class folder (auto-detected)")
    ap.add_argument("--polarity", default="auto", choices=["auto", "bright", "dark"],
                    help="lesions look bright (MRI contrast, mammography) or dark (ultrasound)")
    ap.add_argument("--modality", default="unspecified", help="e.g. MRI, mammogram, ultrasound (shown in the app)")
    ap.add_argument("--no-augment", action="store_true", help="disable horizontal-flip augmentation")
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        sys.exit(f"Data folder not found: {args.data}")
    classes, items = find_images(args.data)
    counts = {c: sum(1 for _, k in items if k == c) for c in classes}
    classes = [c for c in classes if counts[c] > 0]
    print("Found:", ", ".join(f"{c}={counts[c]}" for c in classes))
    if len(classes) < 2 or min(counts[c] for c in classes) < 5:
        sys.exit("Need at least two class folders with 5+ images each (e.g. benign/ and malignant/).")

    positive = args.positive or next((c for c in classes if c.lower() in POSITIVE_NAMES), None)
    if positive not in classes:
        sys.exit(f"Could not tell which folder is cancerous. Pass --positive <folder>. Folders: {classes}")

    paths = np.array([p for p, _ in items])
    labels = np.array([c for _, c in items])
    tr_idx, te_idx = train_test_split(np.arange(len(items)), test_size=0.2, stratify=labels, random_state=7)

    t0 = time.time()
    feats, bad = {}, []
    train_set = set(tr_idx.tolist())
    for n, i in enumerate(range(len(items)), 1):
        try:
            feats[i] = extract(paths[i], args.polarity, augment=(i in train_set) and not args.no_augment)
        except Exception as e:  # unreadable / blank file
            bad.append((paths[i], str(e)))
        if n % 25 == 0 or n == len(items):
            print(f"  features {n}/{len(items)}  ({time.time() - t0:.0f}s)", end="\r")
    print()
    for p, e in bad[:10]:
        print("  skipped:", p, "-", e)

    def stack(idx):
        X, y = [], []
        for i in idx:
            for v in feats.get(i, []):
                X.append(v); y.append(labels[i])
        return np.array(X), np.array(y)

    X_tr, y_tr = stack(tr_idx)
    # test images: only the un-augmented vector
    X_te = np.array([feats[i][0] for i in te_idx if i in feats])
    y_te = np.array([labels[i] for i in te_idx if i in feats])

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, class_weight="balanced", random_state=7))])
    grid = [
        {"clf": [SVC(probability=True, class_weight="balanced", random_state=7)],
         "clf__C": [0.5, 1, 4, 16], "clf__gamma": ["scale", 0.01, 0.001]},
        {"clf": [RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=7, n_jobs=-1)],
         "clf__max_depth": [None, 12]},
    ]
    cv = StratifiedKFold(5, shuffle=True, random_state=7)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    model = gs.best_estimator_
    print("Best:", gs.best_params_["clf"].__class__.__name__, f"CV balanced acc {gs.best_score_:.3f}")

    model_classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(X_te)[:, model_classes.index(positive)]
    pred = model.predict(X_te)
    y_bin = (y_te == positive).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "auc": float(roc_auc_score(y_bin, proba)) if len(set(y_bin)) == 2 else None,
        "cv_balanced_accuracy": float(gs.best_score_),
        "confusion": confusion_matrix(y_te, pred, labels=model_classes).tolist(),
        "n_test": int(len(y_te)),
        "algorithm": gs.best_params_["clf"].__class__.__name__,
    }
    print(classification_report(y_te, pred))
    print("Confusion (rows=true, cols=pred, order", model_classes, ")\n", np.array(metrics["confusion"]))

    # refit on all data for the deployed model
    X_all, y_all = stack(sorted(feats))
    model.fit(X_all, y_all)
    os.makedirs(os.path.dirname(SCAN_MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": model, "classes": [str(c) for c in model.classes_], "positive": positive,
                 "metrics": metrics, "n_images": len(feats), "polarity": args.polarity,
                 "modality": args.modality, "trained_at": dt.datetime.now().isoformat(timespec="minutes")},
                SCAN_MODEL_PATH)
    print(f"Saved -> {SCAN_MODEL_PATH}\nRestart is not needed: the app picks up the new model automatically.")


if __name__ == "__main__":
    main()
