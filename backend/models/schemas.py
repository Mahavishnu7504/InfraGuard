from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# -----------------------------------------------------
# 🔹 DETECTION SCHEMAS
# -----------------------------------------------------

class Detection(BaseModel):
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float


# -----------------------------------------------------
# 🔹 PPE PERSON REPORT
# -----------------------------------------------------

class PersonRisk(BaseModel):
    person_id: int
    risk: str
    missing: List[str]
    assigned_ppe: List[str]
    reason: str


class PPEReport(BaseModel):
    image_risk: str
    persons: List[PersonRisk]


# -----------------------------------------------------
# 🔹 DANGER / PROXIMITY
# -----------------------------------------------------

class DangerZone(BaseModel):
    type: str
    machine: str
    distance: int


class Proximity(BaseModel):
    type: str
    machine: str
    distance: int


# -----------------------------------------------------
# 🔹 FINAL RISK RESPONSE
# -----------------------------------------------------

class RiskResponse(BaseModel):
    risk_level: str
    ppe: PPEReport
    proximity: List[Proximity]
    danger_zones: List[DangerZone]


# -----------------------------------------------------
# 🔹 SAFETY API RESPONSE
# -----------------------------------------------------

class SafetyResponse(BaseModel):
    detections: List[Detection]
    risk: RiskResponse


# -----------------------------------------------------
# 🔹 QUALITY MODULE SCHEMAS
# -----------------------------------------------------

class QualityIssue(BaseModel):
    issue_type: str
    confidence: float
    severity: str
    category: str


class QualityReportItem(BaseModel):
    issue_type: str
    category: str
    severity: str
    confidence: float
    observation: str
    risk: str
    corrective_action: str
    preventive_action: str
    best_practice: str
    guideline_reference: str
    internal_score: Optional[int] = None


class QualityAnalysisResponse(BaseModel):
    issues: List[QualityIssue]
    overall_status: str
    report: Optional[List[QualityReportItem]] = None
    summary: Optional[str] = None
    executive_summary: Optional[str] = None
    priority_action: Optional[str] = None
    compliance_score: Optional[int] = None
    severity_counts: Optional[Dict[str, int]] = None
    image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None


# -----------------------------------------------------
# 🔹 DATABASE EVENT SCHEMA
# -----------------------------------------------------

class EventBase(BaseModel):
    event_type: str
    risk_level: str
    camera_id: Optional[int] = None
    image_path: Optional[str] = None


class EventCreate(EventBase):
    pass


class EventResponse(EventBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# -----------------------------------------------------
# 🔹 WEBSOCKET MESSAGE
# -----------------------------------------------------

class AlertMessage(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    risk_level: str
    message: str
    camera_id: Optional[int] = None