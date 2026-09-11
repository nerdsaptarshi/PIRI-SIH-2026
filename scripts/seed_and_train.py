"""
Seed a reproducible synthetic PIRI demo portfolio and train the four models.

Run from the repository root:
    python scripts/seed_and_train.py

IMPORTANT:
This is demonstration data only. Do not represent it as PAIMANA/OCMS data.
Production use requires authorised data access and time-aware validation.
"""

from datetime import datetime, timedelta
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import MODEL_DIR
from backend.db import Base, Project, SessionLocal, engine, init_db
from backend.model_pipeline import FEATURES, train_models


random.seed(42)
np.random.seed(42)

SECTORS = [
    "Roads & Highways",
    "Railways",
    "Power",
    "Water Resources",
    "Urban Development",
    "Petroleum & Natural Gas",
]

MINISTRIES = {
    "Roads & Highways": "Ministry of Road Transport & Highways",
    "Railways": "Ministry of Railways",
    "Power": "Ministry of Power",
    "Water Resources": "Ministry of Jal Shakti",
    "Urban Development": "Ministry of Housing & Urban Affairs",
    "Petroleum & Natural Gas": "Ministry of Petroleum & Natural Gas",
}


def make_projects(n=120):
    rows = []

    for i in range(1, n + 1):
        sector = random.choice(SECTORS)
        original = round(random.uniform(180, 8000), 2)

        physical = round(random.uniform(8, 96), 1)
        planned = random.randint(18, 84)
        elapsed = round(planned * random.uniform(0.35, 1.25), 1)

        # Synthetic operational signals. Higher values intentionally create
        # learnable risk relationships for the prototype.
        schedule_progress = max(
            2,
            min(100, physical + np.random.normal(0, 8) - max(0, elapsed / planned - 0.85) * 25),
        )

        milestones_due = max(1, int(planned / 6))
        delay_pressure = max(0, (elapsed / planned - 0.75) * 8)
        milestones_delayed = min(
            milestones_due,
            max(0, int(np.random.poisson(1 + delay_pressure + (100 - schedule_progress) / 35))),
        )

        agency_delay_count = max(
            0, int(np.random.poisson(0.7 + milestones_delayed * 0.35))
        )
        contract_variation_count = max(
            0, int(np.random.poisson(0.5 + original / 5000))
        )
        clearance_pending = random.random() < (
            0.08 + 0.18 * (milestones_delayed >= 2)
        )

        monthly_growth = round(
            np.random.normal(2.5 + milestones_delayed * 1.3, 3.5), 2
        )

        cost_growth = round(
            max(
                -1,
                np.random.normal(
                    4
                    + milestones_delayed * 2.2
                    + contract_variation_count * 1.6
                    + (8 if clearance_pending else 0),
                    5,
                ),
            ),
            2,
        )

        revised = round(original * (1 + max(cost_growth, 0) / 100), 2)
        expenditure = round(
            min(
                revised * 0.98,
                original * max(0.03, min(0.98, physical / 100 + np.random.normal(0, 0.05))),
            ),
            2,
        )

        # Latent synthetic labels: deliberately correlated with monitoring signals.
        cost_signal = (
            cost_growth
            + contract_variation_count * 2
            + monthly_growth * 0.7
            + (7 if clearance_pending else 0)
            + np.random.normal(0, 3)
        )
        delay_signal = (
            (100 - schedule_progress)
            + milestones_delayed * 9
            + agency_delay_count * 5
            + max(0, elapsed / planned - 1) * 35
            + (8 if clearance_pending else 0)
            + np.random.normal(0, 5)
        )

        cost_overrun = int(cost_signal >= 13)
        schedule_delay = int(delay_signal >= 42)

        cost_overrun_pct = round(
            max(0, cost_growth + np.random.normal(0, 2.5) + cost_overrun * 3.5), 2
        )
        delay_months = round(
            max(
                0,
                (delay_signal - 18) / 9 + np.random.normal(0, 1.2) + schedule_delay * 1.5,
            ),
            2,
        )

        rows.append(
            {
                "project_code": f"P-{1000 + i}",
                "name": f"{sector} Infrastructure Project {i:03d}",
                "sector": sector,
                "ministry": MINISTRIES[sector],
                "original_cost_cr": original,
                "revised_cost_cr": revised,
                "expenditure_cr": expenditure,
                "physical_progress_pct": round(float(physical), 1),
                "schedule_progress_pct": round(float(schedule_progress), 1),
                "planned_duration_months": float(planned),
                "elapsed_months": float(elapsed),
                "milestones_due": milestones_due,
                "milestones_delayed": milestones_delayed,
                "monthly_expenditure_growth_pct": monthly_growth,
                "cost_growth_pct": cost_growth,
                "agency_delay_count": agency_delay_count,
                "contract_variation_count": contract_variation_count,
                "clearance_pending": clearance_pending,
                "last_update": datetime.utcnow() - timedelta(days=random.randint(0, 90)),
                "status": "Ongoing",
                "cost_overrun": cost_overrun,
                "schedule_delay": schedule_delay,
                "cost_overrun_pct": cost_overrun_pct,
                "delay_months": delay_months,
            }
        )

    return pd.DataFrame(rows)


def seed_database(df):
    init_db()
    db = SessionLocal()
    try:
        # Synthetic demo reset: safe for a disposable prototype database.
        db.query(Project).delete()
        db.commit()

        for row in df.to_dict(orient="records"):
            data = {k: row[k] for k in [
                "project_code", "name", "sector", "ministry",
                "original_cost_cr", "revised_cost_cr", "expenditure_cr",
                "physical_progress_pct", "schedule_progress_pct",
                "planned_duration_months", "elapsed_months",
                "milestones_due", "milestones_delayed",
                "monthly_expenditure_growth_pct", "cost_growth_pct",
                "agency_delay_count", "contract_variation_count",
                "clearance_pending", "last_update", "status",
            ]}
            db.add(Project(**data))

        db.commit()
    finally:
        db.close()


def main():
    df = make_projects(120)
    seed_database(df)

    models, metrics = train_models(
        df,
        model_dir=MODEL_DIR,
        cost_target="cost_overrun",
        delay_target="schedule_delay",
        cost_pct_target="cost_overrun_pct",
        delay_month_target="delay_months",
    )

    print(f"Seeded {len(df)} synthetic ongoing projects.")
    print(f"Models saved to: {MODEL_DIR}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
