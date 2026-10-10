"""
Heartly - Flask backend (runs locally and on Vercel).

Only dependency: Flask. The KNN model is stored as plain JSON by scripts/train.py,
so there is no scikit-learn / numpy / pickle to break on deployment.
"""
import csv
import json
import math
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__,
            template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"),
            static_url_path="/static")

MODEL = json.loads((ROOT / "model" / "model.json").read_text())
STATS = json.loads((ROOT / "model" / "stats.json").read_text())
with open(ROOT / "data" / "heart_clean.csv", newline="") as f:
    ROWS = [{k: (float(v) if "." in v else int(v)) for k, v in r.items()}
            for r in csv.DictReader(f)]

_EXAMPLES = None
FEATURES = MODEL["features"]
K = MODEL["k"]

# Labels are the usual Cleveland names, re-ordered to match how the codes are
# distributed in THIS file (checked against the original Cleveland frequencies).
FIELDS = [
    {"name": "age", "label": "Age", "unit": "years", "kind": "number", "group": "Resting measurements"},
    {"name": "sex", "label": "Sex", "kind": "select", "group": "Resting measurements",
     "options": {0: "Female", 1: "Male"}},
    {"name": "trestbps", "label": "Resting blood pressure", "unit": "mm Hg", "kind": "number",
     "group": "Resting measurements"},
    {"name": "chol", "label": "Serum cholesterol", "unit": "mg/dl", "kind": "number",
     "group": "Resting measurements"},
    {"name": "fbs", "label": "Fasting blood sugar", "kind": "select", "group": "Resting measurements",
     "options": {0: "120 mg/dl or lower", 1: "Above 120 mg/dl"}},
    {"name": "restecg", "label": "Resting ECG result", "kind": "select", "group": "Resting measurements",
     "options": {0: "Normal", 1: "Left ventricular hypertrophy", 2: "ST-T wave abnormality"}},
    {"name": "cp", "label": "Chest pain type", "kind": "select", "group": "Exercise test",
     "options": {0: "Asymptomatic", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Typical angina"}},
    {"name": "thalach", "label": "Maximum heart rate reached", "unit": "bpm", "kind": "number",
     "group": "Exercise test"},
    {"name": "exang", "label": "Exercise-induced angina", "kind": "select", "group": "Exercise test",
     "options": {0: "No", 1: "Yes"}},
    {"name": "oldpeak", "label": "ST depression after exercise", "unit": "mm", "kind": "number",
     "step": 0.1, "group": "Exercise test"},
    {"name": "slope", "label": "Slope of the peak-exercise ST segment", "kind": "select",
     "group": "Exercise test", "options": {0: "Downsloping", 1: "Flat", 2: "Upsloping"}},
    {"name": "ca", "label": "Major vessels coloured by fluoroscopy", "kind": "select",
     "group": "Imaging", "options": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}},
    {"name": "thal", "label": "Thalassemia scan result", "kind": "select", "group": "Imaging",
     "options": {0: "0 (not recorded)", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}},
]
for fld in FIELDS:
    if fld["kind"] == "number":
        lo, hi = STATS["ranges"][fld["name"]]
        fld["min"], fld["max"] = lo, hi
        fld.setdefault("step", 1)


# ------------------------------------------------------------------ model
def predict(values):
    """values: dict feature -> float. Returns dict with risk probability."""
    z = [(values[f] - m) / s for f, m, s in zip(FEATURES, MODEL["mean"], MODEL["scale"])]
    dists = []
    for i, row in enumerate(MODEL["X"]):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(z, row)))
        dists.append((d, i))
    dists.sort()
    nearest = dists[:K]
    disease = sum(MODEL["y"][i] for _, i in nearest)
    p = disease / K
    b = MODEL["bands"]
    status = "Low" if p < b["low"] else ("High" if p >= b["high"] else "Moderate")
    return {"risk_probability": round(p, 4), "risk_status": status,
            "neighbors_with_disease": disease, "neighbors_used": K}


def validate(payload):
    """Return (values, errors). Every field is required and must be in range."""
    values, errors = {}, {}
    if not isinstance(payload, dict):
        return values, {"_": "Send a JSON object with the 13 fields."}
    for fld in FIELDS:
        n = fld["name"]
        raw = payload.get(n)
        if raw is None or raw == "":
            errors[n] = "Required."
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[n] = "Must be a number."
            continue
        if not math.isfinite(v):
            errors[n] = "Must be a finite number."
        elif fld["kind"] == "select" and (int(v) != v or int(v) not in fld["options"]):
            errors[n] = "Not one of the allowed codes."
        elif fld["kind"] == "number" and not (fld["min"] <= v <= fld["max"]):
            errors[n] = f"Outside the range seen in the dataset ({fld['min']:g}-{fld['max']:g})."
        else:
            values[n] = v
    return values, errors


# ----------------------------------------------------------------- routes
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model="knn", k=K, training_rows=len(MODEL["y"]))


@app.get("/api/meta")
def meta():
    return jsonify(fields=FIELDS, stats=STATS)


@app.get("/api/examples")
def examples():
    """Two real rows from the dataset that the model scores clearly high / low."""
    global _EXAMPLES
    if _EXAMPLES:
        return jsonify(_EXAMPLES)
    pick = lambda r: {f: r[f] for f in FEATURES}
    scored = [(predict({f: float(r[f]) for f in FEATURES})["risk_probability"], i)
              for i, r in enumerate(ROWS) if r["disease"] == (1)]
    hi = ROWS[max(scored)[1]]
    scored0 = [(predict({f: float(r[f]) for f in FEATURES})["risk_probability"], i)
               for i, r in enumerate(ROWS) if r["disease"] == 0]
    lo = ROWS[min(scored0)[1]]
    _EXAMPLES = {"high": pick(hi), "low": pick(lo)}
    return jsonify(_EXAMPLES)


@app.post("/api/predict")
def api_predict():
    values, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify(error="Some inputs are missing or out of range.", fields=errors), 400
    return jsonify(predict(values))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
