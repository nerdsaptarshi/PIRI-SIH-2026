"""
PIRI machine-learning pipeline.

Four Random Forest models answer four related questions:
1. probability of cost overrun
2. probability of schedule delay
3. expected cost overrun percentage
4. expected schedule delay in months

The project is a prototype. Training data may be synthetic until authorized
PAIMANA/OCMS integration is available.
"""

from pathlib import Path
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = [
    "sector",
    "ministry",
    "original_cost_cr",
    "revised_cost_cr",
    "expenditure_cr",
    "physical_progress_pct",
    "schedule_progress_pct",
    "planned_duration_months",
    "elapsed_months",
    "milestones_due",
    "milestones_delayed",
    "monthly_expenditure_growth_pct",
    "cost_growth_pct",
    "agency_delay_count",
    "contract_variation_count",
    "clearance_pending",
]

CAT = ["sector", "ministry"]
NUM = [x for x in FEATURES if x not in CAT]

MODEL_VERSION = "piri-rf-v1"


def _normalise_frame(data):
    """Return a DataFrame with exactly the columns expected by the models."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    for col in FEATURES:
        if col not in frame.columns:
            frame[col] = 0

    frame = frame[FEATURES].copy()

    for col in CAT:
        frame[col] = frame[col].fillna("Unknown").astype(str)

    for col in NUM:
        if col == "clearance_pending":
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    frame["clearance_pending"] = (
        frame["clearance_pending"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
        if frame["clearance_pending"].dtype == object
        else frame["clearance_pending"].fillna(0).astype(int)
    )

    return frame


def make_preprocessor():
    # sparse_output=False is required by the deployed scikit-learn version.
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT,
            ),
            ("num", StandardScaler(), NUM),
        ],
        remainder="drop",
    )


def _classifier(seed):
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _regressor(seed):
    return RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )


def build_models():
    """Create the four untrained model pipelines."""
    return {
        "cost": Pipeline([
            ("prep", make_preprocessor()),
            ("model", _classifier(42)),
        ]),
        "delay": Pipeline([
            ("prep", make_preprocessor()),
            ("model", _classifier(43)),
        ]),
        "cost_pct": Pipeline([
            ("prep", make_preprocessor()),
            ("model", _regressor(44)),
        ]),
        "delay_months": Pipeline([
            ("prep", make_preprocessor()),
            ("model", _regressor(45)),
        ]),
    }


def _safe_auc(y_true, probability):
    try:
        return float(roc_auc_score(y_true, probability))
    except ValueError:
        return None


def train_models(
    df,
    model_dir="models",
    cost_target="cost_overrun",
    delay_target="schedule_delay",
    cost_pct_target="cost_overrun_pct",
    delay_month_target="delay_months",
):
    """
    Train the four models and save them.

    Expected target columns:
      cost_overrun        -> 0/1
      schedule_delay      -> 0/1
      cost_overrun_pct    -> numeric
      delay_months        -> numeric

    If your seed script uses different target names, pass them explicitly.
    A time-aware split should be used for real historical PAIMANA/OCMS data.
    """
    df = df.copy()
    frame = _normalise_frame(df)

    targets = [
        cost_target,
        delay_target,
        cost_pct_target,
        delay_month_target,
    ]
    missing = [x for x in targets if x not in df.columns]
    if missing:
        raise ValueError(f"Missing training target columns: {missing}")

    n = len(frame)
    if n < 10:
        raise ValueError("At least 10 records are recommended for prototype training.")

    split = max(1, int(n * 0.8))
    if split >= n:
        split = n - 1

    x_train, x_test = frame.iloc[:split], frame.iloc[split:]

    y_cost_train = pd.to_numeric(df[cost_target], errors="coerce").fillna(0).astype(int).iloc[:split]
    y_cost_test = pd.to_numeric(df[cost_target], errors="coerce").fillna(0).astype(int).iloc[split:]

    y_delay_train = pd.to_numeric(df[delay_target], errors="coerce").fillna(0).astype(int).iloc[:split]
    y_delay_test = pd.to_numeric(df[delay_target], errors="coerce").fillna(0).astype(int).iloc[split:]

    y_cost_pct_train = pd.to_numeric(df[cost_pct_target], errors="coerce").fillna(0.0).iloc[:split]
    y_cost_pct_test = pd.to_numeric(df[cost_pct_target], errors="coerce").fillna(0.0).iloc[split:]

    y_delay_m_train = pd.to_numeric(df[delay_month_target], errors="coerce").fillna(0.0).iloc[:split]
    y_delay_m_test = pd.to_numeric(df[delay_month_target], errors="coerce").fillna(0.0).iloc[split:]

    models = build_models()
    metrics = {}

    models["cost"].fit(x_train, y_cost_train)
    cost_prob = models["cost"].predict_proba(x_test)[:, 1]
    metrics["cost_auc"] = _safe_auc(y_cost_test, cost_prob)
    metrics["cost_accuracy"] = float(
        accuracy_score(y_cost_test, (cost_prob >= 0.5).astype(int))
    )

    models["delay"].fit(x_train, y_delay_train)
    delay_prob = models["delay"].predict_proba(x_test)[:, 1]
    metrics["delay_auc"] = _safe_auc(y_delay_test, delay_prob)
    metrics["delay_accuracy"] = float(
        accuracy_score(y_delay_test, (delay_prob >= 0.5).astype(int))
    )

    models["cost_pct"].fit(x_train, y_cost_pct_train)
    cost_pct_pred = models["cost_pct"].predict(x_test)
    metrics["cost_mae_pct"] = float(mean_absolute_error(y_cost_pct_test, cost_pct_pred))
    metrics["cost_rmse_pct"] = float(np.sqrt(mean_squared_error(y_cost_pct_test, cost_pct_pred)))

    models["delay_months"].fit(x_train, y_delay_m_train)
    delay_pred = models["delay_months"].predict(x_test)
    metrics["delay_mae_months"] = float(mean_absolute_error(y_delay_m_test, delay_pred))
    metrics["delay_rmse_months"] = float(np.sqrt(mean_squared_error(y_delay_m_test, delay_pred)))

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")

    (model_dir / "version.txt").write_text(MODEL_VERSION)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return models, metrics


def load_models(model_dir="models"):
    """Load all four trained models from disk."""
    model_dir = Path(model_dir)

    files = {
        "cost": model_dir / "cost.joblib",
        "delay": model_dir / "delay.joblib",
        "cost_pct": model_dir / "cost_pct.joblib",
        "delay_months": model_dir / "delay_months.joblib",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Trained PIRI model files are missing. Run scripts/seed_and_train.py first. "
            f"Missing: {missing}"
        )

    return {name: joblib.load(path) for name, path in files.items()}


def _risk_level(score):
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def predict(models, row):
    """
    Return:
      cost probability,
      delay probability,
      predicted cost overrun %,
      predicted delay months,
      PIRI composite score,
      risk level.
    """
    frame = _normalise_frame(row)

    pc = float(models["cost"].predict_proba(frame[FEATURES])[0, 1])
    pdly = float(models["delay"].predict_proba(frame[FEATURES])[0, 1])
    cost_pct = float(models["cost_pct"].predict(frame[FEATURES])[0])
    delay_months = float(models["delay_months"].predict(frame[FEATURES])[0])

    # PIRI is a composite prioritisation score, NOT a probability.
    score = round(
        100
        * (
            0.45 * pc
            + 0.40 * pdly
            + 0.15 * min(max(cost_pct / 30.0, 0.0), 1.0)
        ),
        1,
    )

    return (
        round(pc, 4),
        round(pdly, 4),
        round(cost_pct, 2),
        round(delay_months, 2),
        score,
        _risk_level(score),
    )


def _get_feature_names(model):
    prep = model.named_steps["prep"]
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        return FEATURES


def _shap_values_for_positive_class(explainer, transformed):
    """
    Normalize SHAP's different return shapes across SHAP versions.
    """
    sv = explainer.shap_values(transformed)

    if isinstance(sv, list):
        arr = np.asarray(sv[1] if len(sv) > 1 else sv[0])
    else:
        arr = np.asarray(sv)

    if arr.ndim == 3:
        # Usually: samples x features x classes
        arr = arr[0, :, 1]
    elif arr.ndim == 2:
        arr = arr[0]
    else:
        arr = arr.reshape(-1)

    return arr


def _explain_one(model, frame, label):
    try:
        prep = model.named_steps["prep"]
        estimator = model.named_steps["model"]
        transformed = prep.transform(frame[FEATURES])

        feature_names = _get_feature_names(model)

        explainer = shap.TreeExplainer(estimator)
        values = _shap_values_for_positive_class(explainer, transformed)

        count = min(len(feature_names), len(values))
        pairs = [
            {
                "model": label,
                "feature": feature_names[i].replace("cat__", "").replace("num__", ""),
                "impact": round(float(values[i]), 4),
            }
            for i in range(count)
        ]

        pairs.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return pairs[:6]

    except Exception:
        # SHAP is explanatory, not essential to the prediction endpoint.
        # If a library-version edge case occurs, return an empty explanation.
        return []


def explain_project(models, row):
    """Return the strongest SHAP drivers for cost and schedule risk."""
    frame = _normalise_frame(row)

    drivers = []
    drivers.extend(_explain_one(models["cost"], frame, "Cost-overrun"))
    drivers.extend(_explain_one(models["delay"], frame, "Schedule-delay"))

    return drivers
