from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date
import pandas as pd, json, os
from .db import get_db, Project, Prediction, Outcome
from .schemas import ProjectOut, PredictionOut, AlertOut, ChatRequest, ChatResponse
from .model_pipeline import load_models, predict, explain_project, FEATURES
from .config import MODEL_DIR

app=FastAPI(title="PIRI AI Infrastructure Monitoring API",version="1.0.0")
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

MODELS=None
MODEL_VERSION="untrained"

@app.on_event("startup")
def startup():
    global MODELS,MODEL_VERSION
    try:
        MODELS=load_models()
        MODEL_VERSION=(MODEL_DIR/"version.txt").read_text().strip()
    except Exception:
        MODELS=None

def project_row(p):
    return pd.DataFrame([{k:getattr(p,k) for k in FEATURES}])

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":MODELS is not None,"model_version":MODEL_VERSION}

@app.get("/api/projects",response_model=list[ProjectOut])
def projects(db:Session=Depends(get_db), q:str="", sector:str="", risk:str=""):
    ps=db.query(Project).filter(Project.status=="Ongoing").all()
    # risk filter is applied after prediction to keep API simple and model-driven.
    if q: ps=[p for p in ps if q.lower() in (p.name+" "+p.project_code+" "+p.ministry).lower()]
    if sector: ps=[p for p in ps if p.sector==sector]
    if risk and MODELS:
        ps=[p for p in ps if predict(MODELS,project_row(p))[-1]==risk]
    return ps

@app.get("/api/projects/{code}",response_model=ProjectOut)
def project(code:str,db:Session=Depends(get_db)):
    p=db.query(Project).filter(Project.project_code==code).first()
    if not p: raise HTTPException(404,"Project not found")
    return p

@app.post("/api/projects/{code}/predict",response_model=PredictionOut)
def project_prediction(code:str,db:Session=Depends(get_db)):
    if not MODELS: raise HTTPException(503,"Models not trained. Run scripts/seed_and_train.py first.")
    p=db.query(Project).filter(Project.project_code==code).first()
    if not p: raise HTTPException(404,"Project not found")
    row=project_row(p)
    pc,pdly,cost_pct,delay_m,score,level=predict(MODELS,row)
    drivers=explain_project(MODELS,row)
    rec={"project_code":code,"cost_overrun_probability":round(pc,4),"delay_probability":round(pdly,4),
         "risk_score":score,"risk_level":level,"top_drivers":drivers,"model_version":MODEL_VERSION}
    db.add(Prediction(project_code=code,cost_overrun_probability=pc,delay_probability=pdly,risk_score=score,risk_level=level,
                      top_drivers=json.dumps(drivers),model_version=MODEL_VERSION)); db.commit()
    return rec

@app.get("/api/projects/{code}/prediction")
def get_prediction(code:str,db:Session=Depends(get_db)):
    p=db.query(Prediction).filter(Prediction.project_code==code).order_by(Prediction.id.desc()).first()
    if not p:
        return project_prediction(code,db)
    return {"project_code":p.project_code,"cost_overrun_probability":p.cost_overrun_probability,
            "delay_probability":p.delay_probability,"risk_score":p.risk_score,"risk_level":p.risk_level,
            "top_drivers":json.loads(p.top_drivers),"model_version":p.model_version}

@app.get("/api/alerts",response_model=list[AlertOut])
def alerts(db:Session=Depends(get_db)):
    if not MODELS: return []
    out=[]
    for p in db.query(Project).filter(Project.status=="Ongoing").all():
        pc,pdly,cost_pct,delay_m,score,level=predict(MODELS,project_row(p))
        if score>=70:
            signal="Cost escalation + schedule risk" if pc>=pdly else "Schedule delay risk"
            action="Cost-driver review and milestone recovery plan" if pc>=pdly else "Milestone recovery plan and implementation review"
            out.append({"project_code":p.project_code,"project_name":p.name,"severity":"CRITICAL" if score>=85 else "HIGH",
                         "signal":signal,"probability":round(max(pc,pdly),3),"recommended_action":action})
    return sorted(out,key=lambda x:x["probability"],reverse=True)[:25]

@app.get("/api/analytics/summary")
def summary(db:Session=Depends(get_db)):
    ps=db.query(Project).filter(Project.status=="Ongoing").all()
    if not ps:return {}
    preds=[predict(MODELS,project_row(p)) for p in ps] if MODELS else []
    return {"projects":len(ps),"original_cost_cr":round(sum(p.original_cost_cr for p in ps),2),
            "revised_cost_cr":round(sum(p.revised_cost_cr for p in ps),2),
            "expenditure_cr":round(sum(p.expenditure_cr for p in ps),2),
            "high_risk":sum(1 for x in preds if x[-1]=="High"),"medium_risk":sum(1 for x in preds if x[-1]=="Medium"),
            "low_risk":sum(1 for x in preds if x[-1]=="Low")}

@app.get("/api/analytics/sectors")
def sectors(db:Session=Depends(get_db)):
    if not MODELS:return []
    data=[]
    for sector in [x[0] for x in db.query(Project.sector).distinct().all()]:
        ps=db.query(Project).filter(Project.sector==sector,Project.status=="Ongoing").all()
        scores=[predict(MODELS,project_row(p))[4] for p in ps]
        data.append({"sector":sector,"projects":len(ps),"avg_risk":round(sum(scores)/len(scores),1) if scores else 0,
                     "high_risk":sum(1 for s in scores if s>=70)})
    return sorted(data,key=lambda x:x["avg_risk"],reverse=True)

@app.get("/api/models/metrics")
def model_metrics():
    path=MODEL_DIR/"metrics.json"
    return {"model_version":MODEL_VERSION,"metrics":json.loads(path.read_text()) if path.exists() else {}}

@app.post("/api/assistant",response_model=ChatResponse)
def assistant(req:ChatRequest,db:Session=Depends(get_db)):
    # Safe prototype assistant: grounded in DB/model outputs. Replace with an approved local/enterprise LLM + RAG.
    text=req.message.lower()
    evidence=[]
    if req.project_code:
        p=db.query(Project).filter(Project.project_code==req.project_code).first()
        if p:
            pc,pdly,cost_pct,delay_m,score,level=predict(MODELS,project_row(p)) if MODELS else (0,0,0,0,0,"Unknown")
            evidence=[f"{p.project_code}: physical progress {p.physical_progress_pct:.1f}%, schedule progress {p.schedule_progress_pct:.1f}%",
                      f"Model: cost-risk {pc:.1%}, delay-risk {pdly:.1%}, PIRI {score:.1f} ({level})"]
            if "why" in text or "risk" in text:
                return {"answer":f"{p.name} is currently classified as {level} risk. The model estimates {pc:.1%} cost-overrun probability and {pdly:.1%} delay probability. Use the Explainable AI endpoint to inspect feature-level drivers before taking action.","evidence":evidence}
    return {"answer":"I can answer from authorised project records and model outputs. Provide a project code (for example P-1001) for a grounded project-risk explanation. A production deployment should connect this endpoint to an approved LLM/RAG service with access controls.","evidence":evidence}


# Serve the PIRI dashboard from the same Render web service.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
