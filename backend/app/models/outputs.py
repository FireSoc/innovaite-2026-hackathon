"""Response Pydantic models for all API endpoints."""

from enum import Enum

from pydantic import BaseModel, Field
from app.models.evidence import ExpenseItem, RenameEntry, DamageClaim, MissingEvidence
from app.models.actions import ChecklistItem
from app.models.inputs import Declaration


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"
    URGENT = "urgent"
    MODERATE = "moderate"


class InsightUrgency(str, Enum):
    CRITICAL = "critical"
    ACTION_NEEDED = "action_needed"
    INFORMATIONAL = "informational"


class KeyInsight(BaseModel):
    """A single AI-generated insight surfaced on the results page."""

    title: str = Field(..., description="Short headline, e.g. 'Payroll deadline in 8 days'")
    detail: str = Field(..., description="1-2 sentence explanation")
    urgency: InsightUrgency = Field(
        default=InsightUrgency.INFORMATIONAL,
        description="Urgency tag for color-coding in the UI",
    )


class EligibilityResponse(BaseModel):
    disaster_id: str = Field(..., description="Primary FEMA disaster declaration number")
    declarations: list[Declaration]
    programs: list[str] = Field(
        ...,
        description="Available program flags, e.g. ['Individual and Households', 'Public Assistance']",
    )
    county: str
    state: str


class DeferrableEstimate(BaseModel):
    category: str
    estimated_savings_days: float
    description: str


class RunwayResponse(BaseModel):
    runway_days: float = Field(..., description="Days of cash remaining at current burn")
    daily_burn: float = Field(..., description="Estimated daily cash outflow")
    deferrable_estimates: list[DeferrableEstimate] = Field(
        ..., description="Potential runway gains from forbearance/waivers"
    )


class EvidenceExtractionResponse(BaseModel):
    expense_items: list[ExpenseItem] = []
    rename_map: list[RenameEntry] = []
    damage_claims: list[DamageClaim] = []
    missing_evidence: list[MissingEvidence] = []


class PacketFileEntry(BaseModel):
    path: str = Field(..., description="Path of the file in the ZIP")
    description: str = Field(..., description="Short description for UI display")


class ResultsSummary(BaseModel):
    damage_claim_count: int = Field(..., description="Number of damage claims in the packet")
    expense_count: int = Field(..., description="Number of expense items in the packet")
    letter_count: int = Field(..., description="Number of letter documents (e.g. 3 letter types x 2 formats)")
    runway_days: float = Field(..., description="Estimated runway in days")
    business_name: str = Field(..., description="Business name from the packet")
    disaster_id: str = Field(..., description="FEMA disaster declaration ID")
    one_line_summary: str = Field(
        ...,
        description="Template-based one-line summary for the results page",
    )
    key_insights: list[KeyInsight] = Field(
        default_factory=list,
        description="3-5 AI-generated headline insights for the results page",
    )
    urgency_level: str = Field(
        default="moderate",
        description="Overall urgency classification: critical / urgent / moderate",
    )


class PacketBuildResponse(BaseModel):
    zip_base64: str = Field(..., description="ZIP file content as base64")
    filename: str = Field(..., description="Suggested download filename")
    results_summary: ResultsSummary
    files_included: list[PacketFileEntry]


class PacketResponse(BaseModel):
    download_url: str = Field(
        ..., description="URL to download the generated ZIP packet"
    )
    files_included: list[str] = Field(
        ..., description="List of files in the ZIP"
    )


class PlanResponse(BaseModel):
    checklist: list[ChecklistItem]
