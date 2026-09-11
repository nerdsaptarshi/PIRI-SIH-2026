from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from .config import MODEL_DIR

FEATURES = [
    "sector","ministry","original_cost_cr","revised_cost_cr","expenditure_cr",
    "physical_progress_pct","schedule_progress_pct","planned_duration_months",
    "elapsed_months","milestones_due","milestones_delayed",
    "monthly_expenditure_growth_pct","cost_growth_pct","agency_delay_count",
    "contract_variation_count","clearance_pending"
]
CAT = ["sector","ministry"]
NUM = [x for x in FEATURES if x not in CAT]

def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM)
    ])

def train_models(df: pd.DataFrame):
    X=df[FEATURES]
    y_cost=df["actual_cost_overrun"]
    y_delay=df["actual_time_overrun"]
    y_cost_pct=df["actual_cost_overrun_pct"]
    y_delay_months=df["actual_delay_months"]

    # A real train/evaluate pipeline. For a production deployment use a time-based split.
    split=max(int(len(df)*0.8), 1)
    Xtr,Xte=X.iloc[:split],X.iloc[split:]
    yc_tr,yc_te=y_cost.iloc[:split],y_cost.iloc[split:]
    yd_tr,yd_te=y_delay.iloc[:split],y_delay.iloc[split:]
    ycp_tr,ycp_te=y_cost_pct.iloc[:split],y_cost_pct.iloc[split:]
    ydm_tr,ydm_te=y_delay_months.iloc[:split],y_delay_months.iloc[split:]

    cost=Pipeline([("prep",make_preprocessor()),("model",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42))])
    delay=Pipeline([("prep",make_preprocessor()),("model",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=43))])
    cost_pct=Pipeline([("prep",make_preprocessor()),("model",RandomForestRegressor(n_estimators=300,random_state=44))])
    delay_m=Pipeline([("prep",make_preprocessor()),("model",RandomForestRegressor(n_estimators=300,random_state=45))])

    cost.fit(Xtr,yc_tr); delay.fit(Xtr,yd_tr); cost_pct.fit(Xtr,ycp_tr); delay_m.fit(Xtr,ydm_tr)

    metrics={}
    if len(Xte):
        metrics["cost_auc"]=float(roc_auc_score(yc_te,cost.predict_proba(Xte)[:,1])) if len(set(yc_te))>1 else None
        metrics["delay_auc"]=float(roc_auc_score(yd_te,delay.predict_proba(Xte)[:,1])) if len(set(yd_te))>1 else None
        metrics["cost_accuracy"]=float(accuracy_score(yc_te,cost.predict(Xte)))
        metrics["delay_accuracy"]=float(accuracy_score(yd_te,delay.predict(Xte)))
        metrics["cost_mae_pct"]=float(mean_absolute_error(ycp_te,cost_pct.predict(Xte)))
        metrics["delay_mae_months"]=float(mean_absolute_error(ydm_te,delay_m.predict(Xte)))
    else:
        metrics={"cost_auc":None,"delay_auc":None,"cost_accuracy":None,"delay_accuracy":None,"cost_mae_pct":None,"delay_mae_months":None}

    version="piri-rf-v1"
    for name,obj in {"cost":cost,"delay":delay,"cost_pct":cost_pct,"delay_months":delay_m}.items():
        joblib.dump(obj,MODEL_DIR/f"{name}.joblib")
    (MODEL_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (MODEL_DIR/"version.txt").write_text(version)
    return metrics

def load_models():
    files=["cost","delay","cost_pct","delay_months"]
    return {x:joblib.load(MODEL_DIR/f"{x}.joblib") for x in files}

def explain_project(models, row: pd.DataFrame):
    # SHAP on the fitted RandomForest after preprocessing.
    import shap
    drivers=[]
    for key,label in [("cost","Cost-overrun"),("delay","Schedule-delay")]:
        pipe=models[key]
        Xt=pipe.named_steps["prep"].transform(row[FEATURES])
        model=pipe.named_steps["model"]
        explainer=shap.TreeExplainer(model)
        sv=explainer.shap_values(Xt)
        vals=sv[1][0] if isinstance(sv,list) else sv[0]
        names=list(pipe.named_steps["prep"].get_feature_names_out())
        order=np.argsort(np.abs(vals))[::-1][:6]
        drivers += [{"model":label,"feature":names[i].replace("num__","").replace("cat__",""),"impact":round(float(vals[i]),4)} for i in order]
    return drivers

def predict(models,row:pd.DataFrame):
    pc=float(models["cost"].predict_proba(row[FEATURES])[:,1][0])
    pdly=float(models["delay"].predict_proba(row[FEATURES])[:,1][0])
    cost_pct=float(models["cost_pct"].predict(row[FEATURES])[0])
    delay_m=float(max(0,models["delay_months"].predict(row[FEATURES])[0]))
    score=round(100*(0.45*pc+0.40*pdly+0.15*min(max(cost_pct/30,0),1)),1)
    level="High" if score>=70 else "Medium" if score>=45 else "Low"
    return pc,pdly,cost_pct,delay_m,score,level
