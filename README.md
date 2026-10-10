# Heartly – Heart Disease Risk Prediction (Flask + KNN, deployed on Vercel)

Educational machine-learning case study. Not a medical device and not a diagnosis.

## What's inside
```
heartly/
├── api/index.py          Flask backend (serves the app and the prediction endpoint)   <- Vercel entry point
├── templates/index.html  Front end (3 tabs: Home, Dashboard, Make a prediction)
├── static/               style.css, app.js
├── model/                model.json (trained KNN + scaler), stats.json (dashboard numbers)
├── data/                 heart_raw.csv (your file), heart_clean.csv (after cleaning)
├── scripts/train.py      cleaning + training pipeline
├── requirements.txt      runtime deps for Vercel (Flask only)
├── requirements-train.txt  deps for re-training locally
└── vercel.json           Vercel routing / file inclusion
```

## Data cleaning (what scripts/train.py does)
1. 1025 rows -> **302** after removing 723 exact duplicates (duplicates leak test data into training).
2. Checked for missing values / impossible numbers (none).
3. `target` in this file is **flipped**: 1 = no disease. A clean `disease` column (= 1 - target) is added.
4. Codes kept exactly as supplied.

## Run locally
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python api/index.py             # open http://127.0.0.1:5000
```
Re-train (optional): `pip install -r requirements-train.txt` then `python scripts/train.py`

## Endpoints used by the app
POST /api/predict (13 fields -> risk status), GET /api/meta, GET /api/examples, GET /api/health
