"""
Patient Digital Twin — tabular model.

This reproduces the final model from the notebook "Breast cancer.ipynb":

    Pipeline([('scaler', StandardScaler()), ('classifier', SVC(kernel='rbf', C=100))])

trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset, 30 features,
70/30 split with random_state=7 and labels encoded B=0, M=1 (LabelEncoder).

The WDBC data ships with scikit-learn (load_breast_cancer), so no CSV is needed.
The only change from the notebook is a probability calibration step, so the app can show a
malignancy probability instead of a bare 0/1.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tabular_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "tabular_metrics.json")

# Column names exactly as they appear in the project's CSV / notebook.
BASES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave points", "symmetry", "fractal_dimension"]
FEATURES = [f"{b}_mean" for b in BASES] + [f"{b}_se" for b in BASES] + [f"{b}_worst" for b in BASES]

DESCRIPTIONS = {
    "radius": "Mean distance from the centre to points on the cell-nucleus perimeter.",
    "texture": "Standard deviation of grey-scale values inside the nucleus.",
    "perimeter": "Length of the nucleus boundary.",
    "area": "Area enclosed by the nucleus boundary.",
    "smoothness": "Local variation in radius lengths.",
    "compactness": "perimeter² / area − 1.0",
    "concavity": "Severity of concave portions of the contour.",
    "concave points": "Number of concave portions of the contour.",
    "symmetry": "Symmetry of the nucleus shape.",
    "fractal_dimension": "'Coastline approximation' − 1 (boundary roughness).",
}
GROUP_LABEL = {"mean": "Mean", "se": "Standard error", "worst": "Worst (largest)"}


def load_data():
    d = load_breast_cancer()
    X = d.data.astype(float)
    y = 1 - d.target  # sklearn: 0=malignant,1=benign  ->  project: B=0, M=1
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        # RBF-SVM chosen by the Kaggle notebook (best 10-fold CV balanced accuracy of 8 algorithms);
        # Platt (sigmoid) calibration turns its score into a probability.
        ("classifier", CalibratedClassifierCV(SVC(kernel="rbf", C=3, gamma="scale"), method="sigmoid", cv=5)),
    ])


def train_and_save(verbose: bool = True) -> dict:
    previous = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                previous = json.load(f)
        except Exception:
            previous = {}
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=7)
    model = build_pipeline().fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    fpr, tpr, _ = roc_curve(y_te, proba)
    cv = cross_val_score(build_pipeline(), X_tr, y_tr, cv=10)
    metrics = {
        "model": "StandardScaler → SVM (RBF, C=3)",
        "dataset": "Wisconsin Diagnostic Breast Cancer (569 cases, 30 features)",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "auc": float(roc_auc_score(y_te, proba)),
        "train_accuracy": float(model.score(X_tr, y_tr)),
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "confusion": confusion_matrix(y_te, pred).tolist(),  # [[TN, FP],[FN, TP]]
        "roc": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        # Spot-check results recorded in the notebook (10-fold CV accuracy, training data)
        "notebook_comparison": [
            {"name": "LR", "raw": 0.942, "scaled": 0.975},
            {"name": "LDA", "raw": 0.955, "scaled": 0.955},
            {"name": "KNN", "raw": 0.937, "scaled": 0.957},
            {"name": "CART", "raw": 0.930, "scaled": 0.919},
            {"name": "NB", "raw": 0.937, "scaled": 0.937},
            {"name": "SVM", "raw": 0.902, "scaled": 0.967},
        ],
    }
    import sklearn
    metrics["sklearn_version"] = sklearn.__version__
    # keep the 8-algorithm comparison measured on Kaggle, if we have it
    if any("cv_auc" in a for a in previous.get("notebook_comparison", [])):
        metrics["notebook_comparison"] = previous["notebook_comparison"]
    # Final model is trained on the same training split as the notebook.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"Saved model   -> {MODEL_PATH}")
        print(f"Test accuracy  {metrics['accuracy']:.4f}   AUC {metrics['auc']:.4f}")
        print(f"Confusion      {metrics['confusion']}")
    return metrics


@dataclass
class TwinEngine:
    model: Pipeline
    metrics: dict
    X: np.ndarray
    y: np.ndarray

    @classmethod
    def load(cls) -> "TwinEngine":
        if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
            train_and_save()
        import sklearn
        import warnings
        try:
            with open(METRICS_PATH) as f:
                saved_version = json.load(f).get("sklearn_version")
            if saved_version and saved_version != sklearn.__version__:
                raise RuntimeError(f"model saved with scikit-learn {saved_version}")
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a version-mismatch warning means: rebuild
                model = joblib.load(MODEL_PATH)
            model.predict_proba(load_data()[0][:1])
        except Exception as e:  # different scikit-learn version: rebuild the same recipe locally (~1 s)
            print(f"  Rebuilding the Digital Twin model locally ({e})")
            train_and_save(verbose=False)
            model = joblib.load(MODEL_PATH)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        X, y = load_data()
        eng = cls(model, metrics, X, y)
        eng._prepare()
        return eng

    # ---------- precomputed helpers ----------
    def _prepare(self):
        scaler = self.model.named_steps["scaler"]
        self.Xs = scaler.transform(self.X)
        self.pca = PCA(n_components=2, random_state=7).fit(self.Xs)
        self.P = self.pca.transform(self.Xs)
        self.benign_mean = self.X[self.y == 0].mean(axis=0)
        self.malig_mean = self.X[self.y == 1].mean(axis=0)
        self.proba_all = self.model.predict_proba(self.X)[:, 1]

    def schema(self) -> dict:
        feats = []
        for i, name in enumerate(FEATURES):
            base, grp = name.rsplit("_", 1)
            col = self.X[:, i]
            feats.append({
                "key": name, "index": i, "base": base, "group": grp,
                "label": base.replace("_", " ").title(), "group_label": GROUP_LABEL[grp],
                "description": DESCRIPTIONS[base],
                "min": float(col.min()), "max": float(col.max()),
                "mean": float(col.mean()), "std": float(col.std()), "median": float(np.median(col)),
                "benign_mean": float(self.benign_mean[i]), "malignant_mean": float(self.malig_mean[i]),
                "step": float(_nice_step(col.max() - col.min())),
            })
        presets = self._presets()
        return {"features": feats, "presets": presets}

    def _presets(self):
        def row(i, label, note):
            return {"label": label, "note": note, "diagnosis": "Malignant" if self.y[i] else "Benign",
                    "values": [float(v) for v in self.X[i]]}
        order = np.argsort(self.proba_all)
        b_typ = int(order[len(order) // 6])
        m_typ = int(order[-len(order) // 6])
        border = int(np.argmin(np.abs(self.proba_all - 0.5)))
        return [
            {"label": "Population average", "note": "Mean of all 569 cases", "diagnosis": "—",
             "values": [float(v) for v in self.X.mean(axis=0)]},
            row(b_typ, "Typical benign case", f"Case #{b_typ} from the dataset"),
            row(m_typ, "Typical malignant case", f"Case #{m_typ} from the dataset"),
            row(border, "Borderline case", f"Case #{border}: the model is least certain here"),
        ]

    # ---------- inference ----------
    def predict(self, values) -> dict:
        x = np.asarray(values, dtype=float).reshape(1, -1)
        if x.shape[1] != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} values, got {x.shape[1]}")
        p = float(self.model.predict_proba(x)[0, 1])
        pc = min(max(p, 1e-6), 1 - 1e-6)
        margin = float(np.log(pc / (1 - pc)))

        # Contribution: how much does the probability fall if this one feature is
        # moved to the benign average? (positive = pushes towards malignant)
        X_rep = np.repeat(x, len(FEATURES), axis=0)
        X_rep[np.arange(len(FEATURES)), np.arange(len(FEATURES))] = self.benign_mean
        p_rep = self.model.predict_proba(X_rep)[:, 1]
        contrib = p - p_rep
        order = np.argsort(-np.abs(contrib))[:10]
        contributions = [{"key": FEATURES[i], "value": float(x[0, i]), "delta": float(contrib[i]),
                          "benign_mean": float(self.benign_mean[i]),
                          "malignant_mean": float(self.malig_mean[i])} for i in order]

        xs = self.model.named_steps["scaler"].transform(x)
        d = np.linalg.norm(self.Xs - xs, axis=1)
        nn = np.argsort(d)[:6]
        neighbours = [{"case": int(i), "diagnosis": "Malignant" if self.y[i] else "Benign",
                       "distance": float(d[i]), "probability": float(self.proba_all[i])} for i in nn]
        pos = self.pca.transform(xs)[0]

        # z-scores vs the whole population, used for the "body map" radar
        z = ((x[0] - self.X.mean(axis=0)) / self.X.std(axis=0)).tolist()

        if p >= 0.5:
            label, risk = "Malignant", ("High" if p >= 0.85 else "Elevated")
        else:
            label, risk = "Benign", ("Low" if p <= 0.15 else "Uncertain")
        return {"probability": p, "label": label, "risk": risk, "margin": margin,
                "contributions": contributions, "neighbours": neighbours,
                "pca": [float(pos[0]), float(pos[1])], "z": z}

    def case(self, i: int) -> dict:
        if not 0 <= i < len(self.y):
            raise IndexError("No such case")
        return {"case": i, "diagnosis": "Malignant" if self.y[i] else "Benign",
                "values": [float(v) for v in self.X[i]]}

    def population_map(self) -> dict:
        return {"points": [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(self.P, self.y)],
                "explained": [float(v) for v in self.pca.explained_variance_ratio_]}


def _nice_step(span: float) -> float:
    raw = span / 400.0
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 0.001
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag
