"""Evidence extraction data contracts (strict schema for AI outputs)."""

from pydantic import BaseModel, Field
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    NEEDS_REVIEW = "needs_review"


class ExpenseItem(BaseModel):
    """A single expense extracted from an uploaded document/image."""

    vendor: str = Field(..., description="Vendor or payee name as seen in the document")
    date: str = Field(..., description="Date of the expense (YYYY-MM-DD or as seen)")
    amount: float = Field(..., description="Dollar amount of the expense")
    category: str = Field(
        ...,
        description="Expense category: rent, utilities, payroll, supplies, repairs, insurance, other",
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Confidence level of extraction"
    )
    source_file: str = Field(..., description="Original filename this was extracted from")
    source_text: str = Field(
        ...,
        description="Literal text snippet from the document that this value was derived from",
    )
    document_type: str = Field(
        default="",
        description="Type of document: receipt, utility_bill, lease, payroll, bank_statement, tax, other",
    )


class RenameEntry(BaseModel):
    """Mapping from original uploaded filename to a standardized name."""

    original_filename: str
    recommended_filename: str = Field(
        ...,
        description="Standardized filename, e.g. receipt_vendorname_2024-01-15_150.00.jpg",
    )
    confidence: ConfidenceLevel


class DamageClaim(BaseModel):
    """A single damage claim bullet extracted from evidence."""

    label: str = Field(..., description="Short label, e.g. 'Water damage - kitchen'")
    detail: str = Field(..., description="1-2 sentence description of the damage")
    confidence: ConfidenceLevel
    source_file: str = Field(
        ..., description="Original filename this damage was extracted from"
    )
    source_text: str = Field(
        ..., description="Text or visual description from the source document"
    )


class MissingEvidence(BaseModel):
    """An evidence item that is expected but was not found in uploads."""

    item: str = Field(
        ..., description="What is missing, e.g. 'Lease agreement', 'Insurance policy'"
    )
    reason: str = Field(
        ...,
        description="Why it matters, e.g. 'Required for SBA loan application'",
    )
