from datetime import date, timedelta
from pathlib import Path
import sys, random, numpy as np, pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.db import Base, engine, SessionLocal, Project, Outcome
from backend.model_pipeline import train_models


def generate_demo_data(n=120):
    random.seed(42); np.random.seed(42)
    sectors=["Transport","Energy","Water","Communication","Social Infrastructure"]
    ministries={"Transport":"MoRTH","Energy":"Power","Water":"Jal Shakti","Communication":"DoT","Social Infrastructure":"Health"}
    rows=[]
    for i in range(1,n+1):
        sector=random.choice(sectors); ministry=ministries[sector]
        orig=round(random.uniform(300,18000),2)
        cost_growth=max(0,random.gauss(8,7))
        revised=orig*(1+cost_growth/100)
        expenditure=orig*random.uniform(.18,.92)
        phys=random.uniform(20,96)
        sched=max(5,min(100,phys+random.gauss(-3,12)))
        planned=random.uniform(18,84)
        elapsed=min(planned*1.35, planned*random.uniform(.35,1.15))
        due=random.randint(4,20)
        delay=max(0,int(round(random.gauss(3,3)+max(0,(sched-phys)/12))))
        delayed=min(due,delay)
        exp_growth=random.gauss(3,5)
        agency=max(0,int(random.gauss(2,2)))
        contract=max(0,int(random.gauss(2,2)))
        clearance=1 if random.random()<.28 else 0
        risk_signal=(cost_growth*0.05 + delayed/due*0.9 + max(0,(elapsed/planned)-.85)*1.2 + clearance*.25 + contract*.08 + agency*.08 + max(0,(phys-sched)/100)*.5)
        actual_cost=1 if risk_signal+random.random()*.5>.85 else 0
        actual_delay=1 if risk_signal+random.random()*.4>.75 else 0
        actual_cost_pct=max(0, random.gauss(cost_growth*.9 + actual_cost*7, 3))
        actual_delay_m=max(0, random.gauss((elapsed-planned)*.5 + actual_delay*5,2))
        rows.append(dict(project_code=f"P-{1000+i}",name=f"Demo Infrastructure Project {i:03d}",sector=sector,ministry=ministry,
            original_cost_cr=round(orig,2),revised_cost_cr=round(revised,2),expenditure_cr=round(expenditure,2),
            physical_progress_pct=round(phys,2),schedule_progress_pct=round(sched,2),planned_duration_months=round(planned,2),
            elapsed_months=round(elapsed,2),milestones_due=due,milestones_delayed=delayed,
            monthly_expenditure_growth_pct=round(exp_growth,2),cost_growth_pct=round(cost_growth,2),
            agency_delay_count=agency,contract_variation_count=contract,clearance_pending=clearance,
            last_update=date(2026,4,1)+timedelta(days=random.randint(0,30)),status="Ongoing",
            actual_cost_overrun=actual_cost,actual_time_overrun=actual_delay,
            actual_cost_overrun_pct=round(actual_cost_pct,2),actual_delay_months=round(actual_delay_m,2)))
    return pd.DataFrame(rows), rows


def seed_and_train():
    df, rows = generate_demo_data()
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
    db=SessionLocal()
    try:
        for r in rows:
            db.add(Project(**{k:r[k] for k in Project.__table__.columns.keys() if k!="id"}))
            db.add(Outcome(project_code=r["project_code"],actual_cost_overrun=r["actual_cost_overrun"],actual_time_overrun=r["actual_time_overrun"],
                           actual_cost_overrun_pct=r["actual_cost_overrun_pct"],actual_delay_months=r["actual_delay_months"]))
        db.commit()
    finally:
        db.close()
    metrics=train_models(df)
    print("Seeded",len(df),"projects")
    print("Metrics:",metrics)
    return df, rows, metrics


if __name__ == "__main__":
    seed_and_train()
