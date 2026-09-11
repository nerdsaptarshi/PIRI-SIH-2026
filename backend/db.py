import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DB_URL

# Render provides PostgreSQL URLs as postgresql://...
# Explicitly use Psycopg 3, which is installed in requirements.txt.
if DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

connect_args = (
    {"check_same_thread": False}
    if DB_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DB_URL,
    connect_args=connect_args
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    project_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    sector = Column(String(100), nullable=False)
    ministry = Column(String(150), nullable=False)
    original_cost_cr = Column(Float, nullable=False)
    revised_cost_cr = Column(Float, nullable=False)
    expenditure_cr = Column(Float, nullable=False)
    physical_progress_pct = Column(Float, nullable=False)
    schedule_progress_pct = Column(Float, nullable=False)
    planned_duration_months = Column(Float, nullable=False)
    elapsed_months = Column(Float, nullable=False)
    milestones_due = Column(Integer, nullable=False)
    milestones_delayed = Column(Integer, nullable=False)
    monthly_expenditure_growth_pct = Column(Float, nullable=False)
    cost_growth_pct = Column(Float, nullable=False)
    agency_delay_count = Column(Integer, nullable=False)
    contract_variation_count = Column(Integer, nullable=False)
    clearance_pending = Column(Integer, nullable=False)
    last_update = Column(Date, nullable=False)
    status = Column(String(50), nullable=False, default="Ongoing")


class Outcome(Base):
    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True)
    project_code = Column(String(50), index=True, nullable=False)
    actual_cost_overrun = Column(Integer, nullable=False)
    actual_time_overrun = Column(Integer, nullable=False)
    actual_cost_overrun_pct = Column(Float, nullable=False)
    actual_delay_months = Column(Float, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    project_code = Column(String(50), index=True, nullable=False)
    cost_overrun_probability = Column(Float, nullable=False)
    delay_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    top_drivers = Column(Text, nullable=False)
    model_version = Column(String(100), nullable=False)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
