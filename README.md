# Ledger — Credit Card Attrition Risk Console

A Flask application that trains and compares three classifiers (Random Forest,
Gradient Boosting, Logistic Regression) on the [BankChurners credit-card
customer dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers),
exposes the results and a live predictor through a JSON API, and serves a
dashboard ("Ledger") on top for exploring churn patterns and assessing
individual customers.

This version is set up to deploy on [Render](https://render.com) as well as
run locally. Every route and the full dashboard were verified end-to-end
against the real `BankChurners.csv`.

## Verified results (on the real `BankChurners.csv`, 10,127 customers)

| Model | Accuracy | ROC AUC | 5-fold CV AUC |
|---|---|---|---|
| Gradient Boosting | 96.40% | 0.9879 | 0.9400 |
| Random Forest | 96.25% | 0.9873 | 0.9238 |
| Logistic Regression | 89.93% | 0.9171 | 0.9163 |

Dataset: 8,500 existing customers, 1,627 attrited (16.07% churn rate).

## Deploying to Render

1. Push this folder to a GitHub (or GitLab) repository.
2. In the Render dashboard, choose **New → Blueprint** and point it at the
   repo — `render.yaml` in this folder is picked up automatically and
   configures everything (build command, start command, health check).
   - No Blueprint? Choose **New → Web Service** instead and set:
     - **Build command:** `pip install -r requirements.txt`
     - **Start command:** `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
     - **Health check path:** `/healthz`
3. Deploy. The first boot trains three models against the bundled CSV
   (roughly 20–40 seconds) before the health check passes — this is
   expected, not a hang.

**Why `--workers 1`:** models train once at import time, in-memory, when the
app boots (see `load_and_prepare()` in `app.py`). Every gunicorn worker
imports the app separately, so more workers means the training work (and
memory for three fitted models) is multiplied per worker with no shared
cache. One worker with several threads comfortably serves this app's
request volume; if load ever requires more workers, moving training to a
one-time step that pickles the fitted models to disk (and has each worker
load rather than retrain them) is the next step — see "Known limitations."

`app.py` reads `$PORT` (Render sets this) and binds `0.0.0.0`, and falls
back to port 5000 for local runs where `$PORT` isn't set.

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Model training runs once at startup (roughly 20 seconds — three models plus
5-fold cross-validation each) before the server starts. Then visit:

```
http://127.0.0.1:5000/
```

Windows users without a terminal workflow can instead double-click
`start.bat` — see `HOW_TO_RUN.txt`.

## What the dashboard actually shows

`templates/index.html` + `static/css/style.css` + `static/js/app.js` — no
build step (Chart.js loads from a CDN in the browser), calls the real API
routes below directly:

- **Book summary** — live totals from `/api/overview`
- **Model comparison** — the three models' held-out accuracy/AUC/CV-AUC from
  `/api/models`, plus a feature-importance chart (switchable per model) from
  `/api/feature_importance`, and ROC curves for all three models
- **Demographics** — age/income/card-category breakdowns by churn status,
  from `/api/demographics`
- **Transaction behavior** — a transaction-count-vs-amount scatter and an
  inactivity-months distribution, from `/api/behavioral`
- **Churn rate by segment** — income, card tier, and education level, from
  `/api/churn_by_segment`
- **Assess a customer** — a form covering every feature the model actually
  uses, posting to `/api/predict` and rendering the result as a risk stamp

## API reference

```
GET  /healthz                  -> { status: "ok" }  (used by Render's health check)
GET  /api/overview             -> { total_customers, attrited, existing, churn_rate, avg_age, avg_credit, avg_trans_amt }
GET  /api/demographics          -> breakdowns by gender/education/income/card/marital status, plus age bins
GET  /api/behavioral           -> mean/median behavioral metrics, a sampled scatter, and inactivity distribution
GET  /api/models               -> per-model accuracy, AUC, CV AUC, confusion matrix, ROC/PR curve points, feature importance
GET  /api/feature_importance?model=Random%20Forest  -> sorted feature importances for the named model
GET  /api/churn_by_segment     -> churn rate % grouped by income/card/education/months-inactive
POST /api/predict              -> single-customer prediction (see below)
```

**`POST /api/predict` request** — every key is required and must match one
of the model's real training features:

```json
{
  "Customer_Age": 45, "Gender": "M", "Dependent_count": 2,
  "Education_Level": "Graduate", "Marital_Status": "Married",
  "Income_Category": "$60K - $80K", "Card_Category": "Blue",
  "Months_on_book": 36, "Total_Relationship_Count": 3,
  "Months_Inactive_12_mon": 2, "Contacts_Count_12_mon": 2,
  "Credit_Limit": 8500, "Total_Revolving_Bal": 1100,
  "Avg_Open_To_Buy": 7400, "Total_Amt_Chng_Q4_Q1": 0.75,
  "Total_Trans_Amt": 4200, "Total_Trans_Ct": 65,
  "Total_Ct_Chng_Q4_Q1": 0.7, "Avg_Utilization_Ratio": 0.13
}
```

**Response** (verified real output for the example above):

```json
{
  "prediction": "Existing Customer",
  "churn_probability": 0.0,
  "retain_probability": 100.0,
  "risk_level": "Low"
}
```

## Project structure

```
ledger/
├── README.md
├── requirements.txt
├── Procfile                 # process command for Render/Heroku-style platforms
├── render.yaml               # one-click Render Blueprint config
├── .gitignore
├── app.py                    # Flask app: trains models at startup, serves API + dashboard
├── BankChurners.csv          # real dataset, 10,127 rows
├── start.bat                 # Windows local-run launcher (optional, not used on Render)
├── HOW_TO_RUN.txt            # instructions for start.bat
├── templates/
│   └── index.html            # dashboard markup
└── static/
    ├── css/style.css          # dashboard styling
    └── js/app.js               # fetches the API, renders charts + predictor
```

## Known limitations

- **No model persistence** — models retrain from scratch every time the
  process starts (roughly 20–40 seconds, not a runtime cost per request, but
  there's no saved `.pkl`/checkpoint to load instead). On Render's free
  tier, a spun-down service means the next request pays that cost again.
- **Flask's dev server is never used in production** — `app.py` only calls
  `app.run()` under `if __name__ == '__main__'`, i.e. for local runs;
  Render (per `Procfile`/`render.yaml`) always serves through gunicorn.
- **Single gunicorn worker** — see "Why `--workers 1`" above.
- **No automated tests.**
- **Class imbalance** (83.9% existing / 16.1% attrited) — plain accuracy is
  a bit optimistic for this reason; AUC and the CV score are more informative
  than accuracy alone, which is why both are surfaced in the dashboard rather
  than accuracy in isolation.

## License

MIT License.

## Author

**Oba Oriekwo**
