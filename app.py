import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Global state ────────────────────────────────────────────────────────────
df_global = None
model_results = {}
encoders = {}
feature_names = []
predict_model = None   # Random Forest fit once at startup, reused by /api/predict

FEATURE_COLS = [
    'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
    'Marital_Status', 'Income_Category', 'Card_Category',
    'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]
TARGET_COL = 'Attrition_Flag'
CAT_COLS = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

# Absolute path to the CSV so it's found regardless of the process's working
# directory (Render, gunicorn, etc. may not cwd into this folder).
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BankChurners.csv')


def generate_synthetic_data(n=10127):
    """Generate realistic BankChurner-like synthetic data (fallback only)."""
    np.random.seed(42)

    attrition = np.random.choice(['Existing Customer', 'Attrited Customer'],
                                 size=n, p=[0.839, 0.161])
    attrited = attrition == 'Attrited Customer'

    n_att = int(attrited.sum())
    n_ext = n - n_att

    months_inactive = np.empty(n, dtype=int)
    months_inactive[attrited]  = np.random.choice([2,3,4,5,6], n_att, p=[0.15,0.3,0.3,0.15,0.1])
    months_inactive[~attrited] = np.random.choice([0,1,2,3,4,5,6], n_ext, p=[0.05,0.25,0.35,0.2,0.1,0.04,0.01])

    contacts = np.empty(n, dtype=int)
    contacts[attrited]  = np.random.choice([3,4,5,6], n_att, p=[0.3,0.3,0.25,0.15])
    contacts[~attrited] = np.random.choice([0,1,2,3,4,5,6], n_ext, p=[0.05,0.2,0.3,0.25,0.12,0.06,0.02])

    credit = np.empty(n)
    credit[attrited]  = np.clip(np.random.normal(8000, 5000, n_att), 1438, 34516)
    credit[~attrited] = np.clip(np.random.normal(9000, 6000, n_ext), 1438, 34516)

    revolving = np.empty(n)
    revolving[attrited]  = np.random.choice([0]+list(range(100,2600,100)), n_att)
    revolving[~attrited] = np.clip(np.random.normal(1162, 815, n_ext), 0, 2517)

    trans_amt = np.empty(n)
    trans_amt[attrited]  = np.clip(np.random.normal(3095, 1400, n_att), 510, 10000)
    trans_amt[~attrited] = np.clip(np.random.normal(4762, 3559, n_ext), 510, 18484)

    trans_ct = np.empty(n, dtype=int)
    trans_ct[attrited]  = np.clip(np.random.normal(44, 15, n_att), 10, 80).astype(int)
    trans_ct[~attrited] = np.clip(np.random.normal(68, 23, n_ext), 10, 139).astype(int)

    util = np.empty(n)
    util[attrited]  = np.clip(np.random.beta(1, 5, n_att), 0, 1)
    util[~attrited] = np.clip(np.random.beta(2, 5, n_ext), 0, 1)

    data = {
        'CLIENTNUM': np.arange(700000000, 700000000 + n),
        'Attrition_Flag': attrition,
        'Customer_Age': np.clip(np.random.normal(46, 8, n), 26, 73).astype(int),
        'Gender': np.random.choice(['M', 'F'], n, p=[0.47, 0.53]),
        'Dependent_count': np.random.choice([0,1,2,3,4,5], n, p=[0.17,0.22,0.27,0.2,0.1,0.04]),
        'Education_Level': np.random.choice(
            ['High School','Graduate','Uneducated','Unknown','College','Post-Graduate','Doctorate'],
            n, p=[0.28,0.30,0.15,0.15,0.05,0.04,0.03]),
        'Marital_Status': np.random.choice(['Married','Single','Unknown','Divorced'], n, p=[0.46,0.39,0.07,0.08]),
        'Income_Category': np.random.choice(
            ['Less than $40K','$40K - $60K','$60K - $80K','$80K - $120K','$120K +','Unknown'],
            n, p=[0.35,0.18,0.15,0.15,0.07,0.10]),
        'Card_Category': np.random.choice(['Blue','Silver','Gold','Platinum'], n, p=[0.93,0.05,0.016,0.004]),
        'Months_on_book': np.clip(np.random.normal(36, 8, n), 13, 56).astype(int),
        'Total_Relationship_Count': np.random.choice([1,2,3,4,5,6], n, p=[0.08,0.12,0.18,0.22,0.22,0.18]),
        'Months_Inactive_12_mon': months_inactive,
        'Contacts_Count_12_mon': contacts,
        'Credit_Limit': credit,
        'Total_Revolving_Bal': revolving,
        'Avg_Open_To_Buy': np.clip(np.random.normal(7469, 9090, n), 3, 34516),
        'Total_Amt_Chng_Q4_Q1': np.clip(np.random.normal(0.76, 0.22, n), 0, 3.4),
        'Total_Trans_Amt': trans_amt,
        'Total_Trans_Ct': trans_ct,
        'Total_Ct_Chng_Q4_Q1': np.clip(np.random.normal(0.71, 0.24, n), 0, 3.71),
        'Avg_Utilization_Ratio': util,
    }
    return pd.DataFrame(data)


def load_and_prepare():
    global df_global, model_results, encoders, feature_names, predict_model

    try:
        df = pd.read_csv(CSV_PATH)
        nb_cols = [c for c in df.columns if 'Naive_Bayes' in c]
        df.drop(columns=nb_cols, inplace=True, errors='ignore')
    except FileNotFoundError:
        print("WARNING: 'BankChurners.csv' not found next to app.py — "
              "training on SYNTHETIC generated data instead of real data. "
              "Metrics reported by this run do NOT reflect the real dataset.")
        df = generate_synthetic_data()

    df_global = df.copy()

    le = LabelEncoder()
    df_enc = df.copy()
    df_enc[TARGET_COL] = le.fit_transform(df_enc[TARGET_COL])   # Attrited=0, Existing=1
    encoders['target'] = le

    for col in CAT_COLS:
        enc = LabelEncoder()
        df_enc[col] = enc.fit_transform(df_enc[col].astype(str))
        encoders[col] = enc

    available_features = [c for c in FEATURE_COLS if c in df_enc.columns]
    feature_names = available_features

    X = df_enc[available_features]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr, rc, _ = precision_recall_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        cv = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()

        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            fi = model.named_steps['clf'].coef_[0]
        else:
            fi = np.zeros(len(available_features))

        results[name] = {
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'cv_auc': round(cv, 4),
            'confusion_matrix': cm.tolist(),
            'roc': {'fpr': fpr.tolist()[::5], 'tpr': tpr.tolist()[::5]},
            'pr_curve': {'precision': pr.tolist()[::5], 'recall': rc.tolist()[::5]},
            'feature_importance': dict(zip(available_features, np.abs(fi).tolist())),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    model_results = results

    predict_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    predict_model.fit(X, y)

    return df, results


# Train once when the module is imported (works for both `python app.py`
# locally and `gunicorn app:app` on Render — gunicorn imports this module
# once per worker, so run with a single worker unless training cost is
# acceptable per-worker; see render.yaml / Procfile).
print("Training models…")
load_and_prepare()
print("Done.")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    """Lightweight health check for Render."""
    return jsonify({'status': 'ok'})


@app.route('/api/overview')
def api_overview():
    df = df_global
    total = len(df)
    attrited = int((df[TARGET_COL] == 'Attrited Customer').sum())
    existing = total - attrited
    churn_rate = round(attrited / total * 100, 2)
    avg_age = round(df['Customer_Age'].mean(), 1)
    avg_credit = round(df['Credit_Limit'].mean(), 0)
    avg_trans = round(df['Total_Trans_Amt'].mean(), 0)

    return jsonify({
        'total_customers': total,
        'attrited': attrited,
        'existing': existing,
        'churn_rate': churn_rate,
        'avg_age': avg_age,
        'avg_credit': avg_credit,
        'avg_trans_amt': avg_trans
    })


@app.route('/api/demographics')
def api_demographics():
    df = df_global
    attrited = df[df[TARGET_COL] == 'Attrited Customer']
    existing = df[df[TARGET_COL] == 'Existing Customer']

    def vc(series):
        return series.value_counts().to_dict()

    return jsonify({
        'gender': {'attrited': vc(attrited['Gender']), 'existing': vc(existing['Gender'])},
        'education': {'attrited': vc(attrited['Education_Level']), 'existing': vc(existing['Education_Level'])},
        'income': {'attrited': vc(attrited['Income_Category']), 'existing': vc(existing['Income_Category'])},
        'card': {'attrited': vc(attrited['Card_Category']), 'existing': vc(existing['Card_Category'])},
        'marital': {'attrited': vc(attrited['Marital_Status']), 'existing': vc(existing['Marital_Status'])},
        'age_bins': {
            'attrited_vals': pd.cut(attrited['Customer_Age'], bins=[20,30,40,50,60,70,80]).value_counts().sort_index().values.tolist(),
            'existing_vals': pd.cut(existing['Customer_Age'], bins=[20,30,40,50,60,70,80]).value_counts().sort_index().values.tolist(),
            'labels': ['20-30','30-40','40-50','50-60','60-70','70-80']
        }
    })


@app.route('/api/behavioral')
def api_behavioral():
    df = df_global
    att = df[df[TARGET_COL] == 'Attrited Customer']
    ext = df[df[TARGET_COL] == 'Existing Customer']

    metrics = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal',
               'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Avg_Utilization_Ratio']

    result = {}
    for m in metrics:
        if m in df.columns:
            result[m] = {
                'attrited_mean': round(float(att[m].mean()), 2),
                'existing_mean': round(float(ext[m].mean()), 2),
                'attrited_median': round(float(att[m].median()), 2),
                'existing_median': round(float(ext[m].median()), 2),
            }

    sample = df.sample(min(500, len(df)), random_state=1)
    result['scatter'] = {
        'x': sample['Total_Trans_Ct'].tolist(),
        'y': sample['Total_Trans_Amt'].tolist(),
        'label': sample[TARGET_COL].tolist()
    }

    result['inactivity_dist'] = {
        'attrited': att['Months_Inactive_12_mon'].value_counts().sort_index().to_dict(),
        'existing': ext['Months_Inactive_12_mon'].value_counts().sort_index().to_dict()
    }

    return jsonify(result)


@app.route('/api/models')
def api_models():
    return jsonify(model_results)


@app.route('/api/feature_importance')
def api_feature_importance():
    model_name = request.args.get('model', 'Random Forest')
    if model_name not in model_results:
        return jsonify({'error': 'Model not found'}), 404

    fi = model_results[model_name]['feature_importance']
    sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    return jsonify(sorted_fi)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Single customer churn prediction."""
    data = request.json or {}

    row = {}
    for col in feature_names:
        val = data.get(col, 0)
        if col in CAT_COLS:
            enc = encoders.get(col)
            try:
                val = enc.transform([str(val)])[0]
            except Exception:
                val = 0
        row[col] = [float(val)]

    X_pred = pd.DataFrame(row)[feature_names]

    le = encoders['target']
    prob = predict_model.predict_proba(X_pred)[0]
    pred = predict_model.predict(X_pred)[0]

    label_map = {v: k for k, v in zip(le.classes_, le.transform(le.classes_))}
    prediction_label = label_map[pred]

    return jsonify({
        'prediction': prediction_label,
        'churn_probability': round(float(prob[0]) * 100, 1),
        'retain_probability': round(float(prob[1]) * 100, 1),
        'risk_level': 'High' if prob[0] > 0.6 else 'Medium' if prob[0] > 0.3 else 'Low'
    })


@app.route('/api/churn_by_segment')
def api_churn_by_segment():
    df = df_global
    result = {}

    for seg in ['Income_Category', 'Card_Category', 'Education_Level']:
        if seg not in df.columns:
            continue
        grp = df.groupby(seg)[TARGET_COL].apply(
            lambda x: round((x == 'Attrited Customer').mean() * 100, 1)
        ).to_dict()
        result[seg] = grp

    df['inactive_bin'] = df['Months_Inactive_12_mon'].astype(str) + ' months'
    result['months_inactive'] = df.groupby('inactive_bin')[TARGET_COL].apply(
        lambda x: round((x == 'Attrited Customer').mean() * 100, 1)
    ).to_dict()

    return jsonify(result)


if __name__ == '__main__':
    # Render (and most PaaS hosts) inject the port to bind to via $PORT and
    # expect the app to listen on 0.0.0.0, not 127.0.0.1. Falls back to 5000
    # for local runs where $PORT isn't set.
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
