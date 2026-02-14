"""Evidence Auto-Builder — the primary AI feature.

OCR-first: Tesseract extracts text from images/PDFs; the LLM structures and
categorizes using that text as the source of truth for amounts and dates.

Hard guardrails:
- Amounts and dates must be anchored in OCR text; otherwise confidence = needs_review.
- LLM normalizes/categorizes but must not invent values.
- Document requirements (forms, date ranges, actionable outcomes) drive prompt and missing_evidence.
"""

import logging
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.document_requirements import DOCUMENT_REQUIREMENTS, REQUIREMENT_MATCH_KEYWORDS
from app.models.evidence import (
    ConfidenceLevel,
    DamageClaim,
    ExpenseItem,
    MissingEvidence,
    RenameEntry,
)
from app.models.outputs import EvidenceExtractionResponse
from app.services.llm_client import complete_json
from app.services import ocr

logger = logging.getLogger(__name__)


class _RawExtractionResult(BaseModel):
    """Internal schema sent to Gemini for structured extraction."""

    expense_items: list[ExpenseItem] = []
    damage_claims: list[DamageClaim] = []


def _build_document_requirements_block() -> str:
    """Format document requirements for the LLM prompt."""
    lines = []
    for req in DOCUMENT_REQUIREMENTS:
        fields = ", ".join(req.required_fields) if req.required_fields else "(visual only)"
        lines.append(
            f"- {req.name}: required_fields={fields}; date_range={req.date_range}; "
            f"forms={req.forms}; outcomes={req.actionable_outcomes}"
        )
    return "\n".join(lines)


def _build_extraction_prompt(
    filenames: list[str],
    file_ocr_texts: dict[str, str],
    business_type: str = "",
    county: str = "",
    state: str = "",
    disaster_id: str = "",
) -> str:
    """Build the multimodal prompt for evidence extraction (OCR-first)."""
    context_parts = []
    if business_type:
        context_parts.append(f"Business type: {business_type}")
    if county and state:
        context_parts.append(f"Location: {county} County, {state}")
    if disaster_id:
        context_parts.append(f"FEMA Disaster ID: {disaster_id}")

    context_str = "\n".join(context_parts) if context_parts else "No additional context provided."

    doc_reqs_block = _build_document_requirements_block()

    ocr_blocks = []
    for fn in filenames:
        text = file_ocr_texts.get(fn, "")
        ocr_blocks.append(f"--- FILE: {fn} ---\n{text or '(no OCR text)'}\n")

    return f"""You are a disaster-relief document analyst. Extract structured expense and damage data using the OCR text below as the source of truth.

DOCUMENT REQUIREMENTS (use for document_type and categorization):
{doc_reqs_block}

CRITICAL RULES:
1. The OCR text below is the SOURCE OF TRUTH for amounts and dates. Do not invent values.
2. For each expense, set source_text to an exact substring of the OCR text for that file. Set source_file to the filename.
3. If the amount or date is not present in the OCR text, set confidence to "needs_review".
4. Set document_type to one of: receipt, utility_bill, lease, payroll, bank_statement, tax, other.
5. Categories for expenses: rent, utilities, payroll, supplies, repairs, insurance, other.
6. For damage claims, set source_file to the filename of the photo. Describe what damage is visible.

CONTEXT:
{context_str}

UPLOADED FILES ({len(filenames)} files):
{chr(10).join(f"- {fn}" for fn in filenames)}

OCR TEXT PER FILE:
{chr(10).join(ocr_blocks)}

Extract all expense items and damage evidence. Return valid JSON matching the schema exactly."""


def _generate_rename_map(
    filenames: list[str],
    expense_items: list[ExpenseItem],
    damage_claims: list[DamageClaim],
) -> list[RenameEntry]:
    """Generate standardized filenames from extraction results."""
    rename_map: list[RenameEntry] = []

    for fn in filenames:
        # Find expense items associated with this file
        file_expenses = [e for e in expense_items if e.source_file == fn]
        file_damages = [d for d in damage_claims if d.source_file == fn]

        ext = Path(fn).suffix.lower() or ".jpg"
        base = Path(fn).stem

        if file_expenses:
            # Use the first expense to generate the name
            e = file_expenses[0]
            vendor_clean = re.sub(r"[^a-zA-Z0-9]", "_", e.vendor.lower())[:20]
            date_clean = e.date.replace("/", "-").replace(" ", "")[:10]
            recommended = f"receipt_{vendor_clean}_{date_clean}_{e.amount:.2f}{ext}"
            confidence = e.confidence
        elif file_damages:
            d = file_damages[0]
            label_clean = re.sub(r"[^a-zA-Z0-9]", "_", d.label.lower())[:30]
            recommended = f"damage_{label_clean}{ext}"
            confidence = d.confidence
        else:
            recommended = f"evidence_{base}{ext}"
            confidence = ConfidenceLevel.NEEDS_REVIEW

        rename_map.append(
            RenameEntry(
                original_filename=fn,
                recommended_filename=recommended,
                confidence=confidence,
            )
        )

    return rename_map


def _normalize_amount_for_ocr(amount: float) -> str:
    """Normalize amount so we can search for it in OCR text (strip $ and commas)."""
    s = f"{amount:.2f}".rstrip("0").rstrip(".")
    return s


def _normalize_date_for_ocr(date: str) -> list[str]:
    """Return possible substrings to look for in OCR (digits, slashes, dashes)."""
    # Keep original and a few variants (digits only, with slashes/dashes)
    out = [date]
    digits = re.sub(r"[^0-9]", "", date)
    if digits:
        out.append(digits)
    if len(digits) >= 6:  # YYYYMM or MMDDYY etc.
        out.append(digits[:4])
        out.append(digits[-4:])
    return out


def _anchor_expense_to_ocr(
    item: ExpenseItem,
    file_ocr_texts: dict[str, str],
) -> ExpenseItem:
    """
    If amount or date is not found in the OCR text for source_file, set confidence to needs_review.
    """
    ocr_text = file_ocr_texts.get(item.source_file, "")
    if not ocr_text:
        return item.model_copy(
            update={
                "confidence": ConfidenceLevel.NEEDS_REVIEW,
                "source_text": "No OCR text for this file; needs review.",
            }
        )

    amount_str = _normalize_amount_for_ocr(item.amount)
    amount_found = amount_str in ocr_text or str(int(item.amount)) in ocr_text
    date_variants = _normalize_date_for_ocr(item.date)
    date_found = any(v and v in ocr_text for v in date_variants)

    if amount_found and date_found:
        return item

    return item.model_copy(
        update={
            "confidence": ConfidenceLevel.NEEDS_REVIEW,
            "source_text": "Amount/date not found in OCR; needs review.",
        }
    )


def _detect_missing_evidence(
    expense_items: list[ExpenseItem],
    damage_claims: list[DamageClaim],
    filenames: list[str],
) -> list[MissingEvidence]:
    """Compare extracted categories against document requirements; return missing items."""
    missing: list[MissingEvidence] = []

    found_categories: set[str] = set()
    for e in expense_items:
        found_categories.add(e.category.lower())
        if e.document_type:
            found_categories.add(e.document_type.lower())
    for _ in damage_claims:
        found_categories.add("damage")

    fn_lower = " ".join(f.lower() for f in filenames)

    for req in DOCUMENT_REQUIREMENTS:
        keywords = REQUIREMENT_MATCH_KEYWORDS.get(req.id, [req.id])
        found = False
        for kw in keywords:
            if kw in found_categories or kw in fn_lower:
                found = True
                break
        if not found:
            missing.append(
                MissingEvidence(item=req.name, reason=req.actionable_outcomes)
            )

    return missing


async def extract_evidence(
    files: list[tuple[str, bytes, str]],
    business_type: str = "",
    county: str = "",
    state: str = "",
    disaster_id: str = "",
    declaration_title: str = "",
) -> EvidenceExtractionResponse:
    """
    Main entry point for evidence extraction.

    Args:
        files: List of (filename, file_bytes, mime_type) tuples.
        business_type: Type of business.
        county: County name.
        state: State code.
        disaster_id: FEMA disaster declaration ID.
        declaration_title: Title of the disaster declaration.

    Returns:
        EvidenceExtractionResponse with extracted data.
    """
    if not files:
        return EvidenceExtractionResponse(
            expense_items=[],
            rename_map=[],
            damage_claims=[],
            missing_evidence=[
                MissingEvidence(item=req.name, reason=req.actionable_outcomes)
                for req in DOCUMENT_REQUIREMENTS
            ],
        )

    filenames = [f[0] for f in files]
    images = [(f[1], f[2]) for f in files]

    # Step 1 — OCR each file
    file_ocr_texts: dict[str, str] = {}
    for filename, file_bytes, mime_type in files:
        file_ocr_texts[filename] = ocr.extract_text(filename, file_bytes, mime_type)

    # Step 2 — Build prompt with OCR text and document requirements
    prompt = _build_extraction_prompt(
        filenames=filenames,
        file_ocr_texts=file_ocr_texts,
        business_type=business_type,
        county=county,
        state=state,
        disaster_id=disaster_id,
    )

    # Step 3 — LLM extraction (text + images so model can still interpret layout/damage)
    try:
        raw_result = await complete_json(
            schema=_RawExtractionResult,
            prompt=prompt,
            images=images,
            max_retries=1,
        )
        expense_items = raw_result.expense_items
        damage_claims = raw_result.damage_claims
    except (ValidationError, Exception) as e:
        logger.error(f"Evidence extraction failed: {e}")
        expense_items = []
        damage_claims = []

    # Step 4 — OCR anchoring: downgrade confidence if amount/date not in OCR
    expense_items = [
        _anchor_expense_to_ocr(item, file_ocr_texts) for item in expense_items
    ]

    # Step 5 — Rename map and missing evidence from document requirements
    rename_map = _generate_rename_map(filenames, expense_items, damage_claims)
    missing_evidence = _detect_missing_evidence(
        expense_items, damage_claims, filenames
    )

    return EvidenceExtractionResponse(
        expense_items=expense_items,
        rename_map=rename_map,
        damage_claims=damage_claims,
        missing_evidence=missing_evidence,
    )
