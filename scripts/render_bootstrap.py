"""Render startup bootstrap for the SIH prototype.

Fresh database: seed synthetic demo records once and train models.
Existing database: rebuild models from stored project/outcome records without
wiping the database. This makes model files disposable on Render's free
filesystem while keeping the demo data in PostgreSQL.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from backend.db import SessionLocal, Project, Outcome  # noqa: E402
from backend.model_pipeline import train_models  # noqa: E402
from scripts.seed_and_train import generate_demo_data  # noqa: E402


def bootstrap():
    db = SessionLocal()
    try:
        project_count = db.query(Project).count()
        if project_count == 0:
            df, rows = generate_demo_data()
            for r in rows:
                db.add(Project(**{k: r[k] for k in Project.__table__.columns.keys() if k != "id"}))
                db.add(Outcome(
                    project_code=r["project_code"],
                    actual_cost_overrun=r["actual_cost_overrun"],
                    actual_time_overrun=r["actual_time_overrun"],
                    actual_cost_overrun_pct=r["actual_cost_overrun_pct"],
                    actual_delay_months=r["actual_delay_months"],
                ))
            db.commit()
            print(f"[PIRI] Seeded {len(df)} synthetic demo projects.")
        else:
            projects = db.query(Project).all()
            outcomes = {o.project_code: o for o in db.query(Outcome).all()}
            records = []
            feature_names = [c.name for c in Project.__table__.columns if c.name != "id"]
            for p in projects:
                o = outcomes.get(p.project_code)
                if not o:
                    continue
                r = {name: getattr(p, name) for name in feature_names}
                r.update({
                    "actual_cost_overrun": o.actual_cost_overrun,
                    "actual_time_overrun": o.actual_time_overrun,
                    "actual_cost_overrun_pct": o.actual_cost_overrun_pct,
                    "actual_delay_months": o.actual_delay_months,
                })
                records.append(r)
            df = pd.DataFrame(records)
            print(f"[PIRI] Rebuilding models from {len(df)} stored projects.")
        if len(df) < 10:
            raise RuntimeError("Not enough labeled project records to train models.")
        metrics = train_models(df)
        print(f"[PIRI] Models trained successfully: {metrics}")
    finally:
        db.close()


if __name__ == "__main__":
    bootstrap()
