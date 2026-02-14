"""Document requirements — single source of truth for evidence types.

Used to build missing_evidence and to inject into the LLM prompt (required fields,
date ranges, forms, actionable outcomes).
"""

from pydantic import BaseModel, Field


class DocumentRequirement(BaseModel):
    """One evidence type we expect for a complete submission."""

    id: str = Field(..., description="Slug, e.g. lease, insurance, payroll")
    name: str = Field(..., description="Display name for UI and missing_evidence")
    required_fields: list[str] = Field(
        default_factory=list,
        description="Fields to extract, e.g. lessor name, monthly rent, period",
    )
    date_range: str = Field(
        default="",
        description="E.g. Last 3 months, Current policy period, Most recent tax year",
    )
    forms: str = Field(
        default="None",
        description="Relevant forms if any, e.g. IRS 1040/1120, SBA Form 5",
    )
    actionable_outcomes: str = Field(
        ...,
        description="How this doc is used: forbearance letter, SBA verification, etc.",
    )


# Keywords to match extracted category or filename to a requirement (for missing_evidence)
REQUIREMENT_MATCH_KEYWORDS: dict[str, list[str]] = {
    "lease": ["rent", "lease"],
    "insurance": ["insurance"],
    "payroll": ["payroll"],
    "utility": ["utilities", "utility"],
    "damage_photos": ["damage"],
    "bank_statements": ["bank", "statement"],
    "tax_returns": ["tax"],
    "business_license": ["license", "registration"],
}

DOCUMENT_REQUIREMENTS: list[DocumentRequirement] = [
    DocumentRequirement(
        id="lease",
        name="Lease agreement or rent statement",
        required_fields=["lessor name", "property address", "monthly rent", "current period"],
        date_range="Current lease term or latest statement",
        forms="None",
        actionable_outcomes="Landlord forbearance letter; SBA loan rent verification; FEMA/SBA documentation.",
    ),
    DocumentRequirement(
        id="insurance",
        name="Insurance policy or declaration page",
        required_fields=["carrier", "policy number", "coverage period", "property address"],
        date_range="Current policy period",
        forms="None",
        actionable_outcomes="Insurance claim; SBA loan verification of coverage.",
    ),
    DocumentRequirement(
        id="payroll",
        name="Payroll records (last 3 months)",
        required_fields=["employer name", "pay period", "gross pay", "employee count or list"],
        date_range="Last 3 months",
        forms="None",
        actionable_outcomes="SBA disaster loan application; proof of payroll expense.",
    ),
    DocumentRequirement(
        id="utility",
        name="Utility bills (recent)",
        required_fields=["provider", "account or address", "amount due", "service period"],
        date_range="Recent (e.g. last 2 months)",
        forms="None",
        actionable_outcomes="Utility waiver request; expense documentation.",
    ),
    DocumentRequirement(
        id="damage_photos",
        name="Damage photographs",
        required_fields=[],
        date_range="Date of photo if visible",
        forms="None",
        actionable_outcomes="Visual evidence for insurance and FEMA claims.",
    ),
    DocumentRequirement(
        id="bank_statements",
        name="Bank statements (last 3 months)",
        required_fields=["institution", "account type", "period", "ending balance"],
        date_range="Last 3 months",
        forms="None",
        actionable_outcomes="SBA loan and financial verification.",
    ),
    DocumentRequirement(
        id="tax_returns",
        name="Tax returns (most recent year)",
        required_fields=["tax year", "form type (e.g. 1040, 1120)", "key totals"],
        date_range="Most recent tax year",
        forms="IRS 1040, 1120, or equivalent",
        actionable_outcomes="SBA disaster loan application.",
    ),
    DocumentRequirement(
        id="business_license",
        name="Business license or registration",
        required_fields=["entity name", "jurisdiction", "validity date"],
        date_range="Current",
        forms="None",
        actionable_outcomes="Proof of business operation for relief applications.",
    ),
]
