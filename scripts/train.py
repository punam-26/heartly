"""
Heartly - data cleaning + model training.

Run from the project root:   python scripts/train.py

Outputs
  data/heart_clean.csv   cleaned dataset
  model/model.json       scaler + KNN training set (read by the web app, no sklearn needed)
  model/stats.json       numbers used by the dashboard / insights pages
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
K = 15
SEED = 42

# ------------------------------------------------------------------ 1. LOAD
raw = pd.read_csv(ROOT / "data" / "heart_raw.csv")
print(f"Raw rows: {len(raw)}")

# ----------------------------------------------------------------- 2. CLEAN
df = raw.copy()
df = df.dropna()                                   # no NaNs here, but be safe
n_dups = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)   # 723 exact duplicates
print(f"Exact duplicates removed: {n_dups} -> {len(df)} unique rows")

# sanity ranges: anything outside is a data-entry error
limits = {"age": (1, 120), "trestbps": (60, 260), "chol": (80, 700),
          "thalach": (50, 230), "oldpeak": (0, 10)}
for col, (lo, hi) in limits.items():
    bad = ~df[col].between(lo, hi)
    assert not bad.any(), f"{col} has out-of-range values"

# IMPORTANT: in this file target==1 means NO disease (label is flipped vs. the
# clinical meaning). Evidence is printed below. We create an explicit column.
grp = df.groupby("target")[["thalach", "oldpeak", "ca", "exang"]].mean().round(2)
print("\nFeature means by target (disease shows lower thalach, higher oldpeak/ca/exang):")
print(grp)
df["disease"] = (df["target"] == 0).astype(int)

df.to_csv(ROOT / "data" / "heart_clean.csv", index=False)

X = df[FEATURES]
y = df["disease"]

# ----------------------------------------------------------- 3. COMPARE MODELS
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
candidates = {
    f"KNN (k={K})": make_pipeline(StandardScaler(), KNeighborsClassifier(K)),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Random forest": RandomForestClassifier(300, random_state=1),
}
comparison = []
for name, model in candidates.items():
    s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    comparison.append({"model": name, "cv_accuracy": round(float(s.mean()), 4),
                       "cv_std": round(float(s.std()), 4)})
    print(f"{name:22s} CV accuracy {s.mean():.3f} +/- {s.std():.3f}")

# ------------------------------------------------- 4. HOLD-OUT CHECK (KNN)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
knn_eval = make_pipeline(StandardScaler(), KNeighborsClassifier(K)).fit(Xtr, ytr)
pred = knn_eval.predict(Xte)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
holdout = {
    "n_test": int(len(yte)),
    "accuracy": round(float((tp + tn) / len(yte)), 4),
    "sensitivity": round(float(tp / (tp + fn)), 4),   # disease correctly caught
    "specificity": round(float(tn / (tn + fp)), 4),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
print("\nHold-out:", holdout)

# ------------------------------------------- 5. FINAL MODEL (fit on all rows)
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
final = KNeighborsClassifier(K).fit(Xs, y)

model_json = {
    "type": "knn",
    "k": K,
    "features": FEATURES,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "X": np.round(Xs, 6).tolist(),
    "y": y.tolist(),            # 1 = disease, 0 = no disease
    "bands": {"low": 0.34, "high": 0.67},
}
(ROOT / "model" / "model.json").write_text(json.dumps(model_json))

# ----------------------------------------------------------- 6. DASHBOARD STATS
bands = list(range(25, 80, 5))
age_counts = []
for lo in bands:
    age_counts.append({"label": f"{lo}-{lo+4}",
                       "count": int(df["age"].between(lo, lo + 4).sum())})

def rate_by(col, labels):
    out = []
    for code, g in df.groupby(col):
        out.append({"code": int(code), "label": labels.get(int(code), str(code)),
                    "n": int(len(g)), "disease_rate": round(float(g["disease"].mean()), 3)})
    return out

stats = {
    "raw_rows": int(len(raw)),
    "clean_rows": int(len(df)),
    "duplicates_removed": n_dups,
    "n_features": len(FEATURES),
    "disease": int(y.sum()),
    "no_disease": int((1 - y).sum()),
    "disease_share": round(float(y.mean()), 4),
    "age_bands": age_counts,
    "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in FEATURES},
    "by_cp": rate_by("cp", {0: "Code 0", 1: "Code 1", 2: "Code 2", 3: "Code 3"}),
    "by_sex": rate_by("sex", {0: "Female", 1: "Male"}),
    "by_exang": rate_by("exang", {0: "No", 1: "Yes"}),
    "by_ca": rate_by("ca", {}),
    "by_thal": rate_by("thal", {}),
    "by_slope": rate_by("slope", {}),
    "feature_means": {
        c: {"disease": round(float(df[df.disease == 1][c].mean()), 2),
            "no_disease": round(float(df[df.disease == 0][c].mean()), 2)}
        for c in ["age", "trestbps", "chol", "thalach", "oldpeak"]
    },
    "comparison": comparison,
    "holdout": holdout,
}
(ROOT / "model" / "stats.json").write_text(json.dumps(stats, indent=1))
print("\nSaved data/heart_clean.csv, model/model.json, model/stats.json")
