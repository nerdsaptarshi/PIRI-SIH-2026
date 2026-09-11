from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List, Dict

class ProjectOut(BaseModel):
    project_code: str
    name: str
    sector: str
    ministry: str
    original_cost_cr: float
    revised_cost_cr: float
    expenditure_cr: float
    physical_progress_pct: float
    schedule_progress_pct: float
    planned_duration_months: float
    elapsed_months: float
    milestones_due: int
    milestones_delayed: int
    monthly_expenditure_growth_pct: float
    cost_growth_pct: float
    agency_delay_count: int
    contract_variation_count: int
    clearance_pending: int
    last_update: date
    status: str

    class Config:
        from_attributes = True

class PredictionOut(BaseModel):
    project_code: str
    cost_overrun_probability: float
    delay_probability: float
    risk_score: float
    risk_level: str
    top_drivers: List[Dict]
    model_version: str

class AlertOut(BaseModel):
    project_code: str
    project_name: str
    severity: str
    signal: str
    probability: float
    recommended_action: str

class ChatRequest(BaseModel):
    message: str
    project_code: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    evidence: List[str] = Field(default_factory=list)
