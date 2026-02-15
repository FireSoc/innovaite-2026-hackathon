"""Submission Packet Builder — generates PDFs and assembles the ZIP bundle.

Produces:
  - CoverSheet.pdf
  - DamageSummary.pdf
  - ExpenseLedger.pdf + ExpenseLedger.csv
  - EvidenceChecklist.pdf
  - Evidence/ folder with standardized filenames
  - Letters/ folder with ready-to-send PDFs
"""

import base64
import csv
import io
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.evidence import DamageClaim, ExpenseItem, MissingEvidence, RenameEntry
from app.models.inputs import Declaration, PacketBuildRequest, UserInfo
from app.services.letters import render_all_letters

logger = logging.getLogger(__name__)

styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=20
)
HEADING_STYLE = ParagraphStyle(
    "CustomHeading", parent=styles["Heading2"], fontSize=14, spaceAfter=10
)
BODY_STYLE = ParagraphStyle(
    "CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14
)


def _build_cover_sheet(
    user_info: UserInfo,
    disaster_id: str,
    declarations: list[Declaration],
    daily_burn: float,
    runway_days: float,
    monthly_rent: float,
    monthly_payroll: float,
    cash_on_hand: float,
    num_employees: int,
    business_type: str,
) -> bytes:
    """Generate CoverSheet.pdf as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    # Title
    story.append(Paragraph("ReliefBridge — Submission Cover Sheet", TITLE_STYLE))
    story.append(Spacer(1, 12))

    # Date
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", BODY_STYLE))
    story.append(Spacer(1, 8))

    # Business info
    story.append(Paragraph("Business Information", HEADING_STYLE))
    info_data = [
        ["Business Name", user_info.business_name or "N/A"],
        ["Owner", user_info.owner_name or "N/A"],
        ["Business Type", business_type],
        ["Address", user_info.address or "N/A"],
        ["Phone", user_info.phone or "N/A"],
        ["Email", user_info.email or "N/A"],
        ["Employees", str(num_employees)],
    ]
    t = Table(info_data, colWidths=[2.5 * inch, 4 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Disaster info
    story.append(Paragraph("Disaster Declaration", HEADING_STYLE))
    story.append(Paragraph(f"<b>FEMA Disaster ID:</b> {disaster_id or 'N/A'}", BODY_STYLE))
    if declarations:
        d = declarations[0]
        story.append(Paragraph(f"<b>Declaration:</b> {d.declaration_title}", BODY_STYLE))
        story.append(Paragraph(f"<b>Incident Type:</b> {d.incident_type}", BODY_STYLE))
        story.append(Paragraph(f"<b>Declaration Date:</b> {d.declaration_date}", BODY_STYLE))
    story.append(Spacer(1, 16))

    # Financial summary
    story.append(Paragraph("Financial Summary", HEADING_STYLE))
    fin_data = [
        ["Monthly Rent", f"${monthly_rent:,.2f}"],
        ["Monthly Payroll", f"${monthly_payroll:,.2f}"],
        ["Cash on Hand", f"${cash_on_hand:,.2f}"],
        ["Daily Burn Rate", f"${daily_burn:,.2f}"],
        ["Estimated Runway", f"{runway_days:.1f} days"],
    ]
    t2 = Table(fin_data, colWidths=[2.5 * inch, 4 * inch])
    t2.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t2)

    doc.build(story)
    return buf.getvalue()


def _build_damage_summary(damage_claims: list[DamageClaim]) -> bytes:
    """Generate DamageSummary.pdf as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    story.append(Paragraph("Damage Summary", TITLE_STYLE))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        BODY_STYLE
    ))
    story.append(Spacer(1, 12))

    if damage_claims:
        for i, claim in enumerate(damage_claims, 1):
            story.append(Paragraph(
                f"<b>{i}. {claim.label}</b> (Confidence: {claim.confidence.value})",
                BODY_STYLE,
            ))
            story.append(Paragraph(f"   {claim.detail}", BODY_STYLE))
            story.append(Paragraph(
                f"   <i>Source file: {claim.source_file}</i>",
                BODY_STYLE,
            ))
            story.append(Paragraph(
                f"   <i>Source: {claim.source_text[:100]}...</i>" if len(claim.source_text) > 100
                else f"   <i>Source: {claim.source_text}</i>",
                BODY_STYLE,
            ))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph(
            "No damage claims have been extracted from uploaded evidence. "
            "Please upload damage photos and documentation for processing.",
            BODY_STYLE,
        ))

    doc.build(story)
    return buf.getvalue()


def _build_expense_ledger_csv(expense_items: list[ExpenseItem]) -> bytes:
    """Generate ExpenseLedger.csv as bytes."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Vendor", "Date", "Amount", "Category", "Confidence", "Source File", "Source Text"])
    for item in expense_items:
        writer.writerow([
            item.vendor,
            item.date,
            f"{item.amount:.2f}",
            item.category,
            item.confidence.value,
            item.source_file,
            item.source_text,
        ])
    return buf.getvalue().encode("utf-8")


def _build_expense_ledger_pdf(expense_items: list[ExpenseItem]) -> bytes:
    """Generate ExpenseLedger.pdf as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    story.append(Paragraph("Expense Ledger", TITLE_STYLE))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 12))

    if expense_items:
        total = sum(e.amount for e in expense_items)
        story.append(Paragraph(f"<b>Total Documented Expenses: ${total:,.2f}</b>", BODY_STYLE))
        story.append(Spacer(1, 8))

        # Table
        header = ["#", "Vendor", "Date", "Amount", "Category", "Confidence"]
        data = [header]
        for i, item in enumerate(expense_items, 1):
            data.append([
                str(i),
                item.vendor[:25],
                item.date,
                f"${item.amount:,.2f}",
                item.category,
                item.confidence.value,
            ])

        t = Table(data, colWidths=[0.4 * inch, 2 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ]))
        story.append(t)
    else:
        story.append(Paragraph(
            "No expense items extracted. Upload receipts and invoices for processing.",
            BODY_STYLE,
        ))

    doc.build(story)
    return buf.getvalue()


def _build_evidence_checklist(
    rename_map: list[RenameEntry],
    missing_evidence: list[MissingEvidence],
) -> bytes:
    """Generate EvidenceChecklist.pdf as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    story.append(Paragraph("Evidence Checklist", TITLE_STYLE))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 12))

    # Included evidence
    story.append(Paragraph("Included Evidence", HEADING_STYLE))
    if rename_map:
        for entry in rename_map:
            story.append(Paragraph(
                f"✓ <b>{entry.recommended_filename}</b> "
                f"(original: {entry.original_filename}, confidence: {entry.confidence.value})",
                BODY_STYLE,
            ))
    else:
        story.append(Paragraph("No evidence files included.", BODY_STYLE))

    story.append(Spacer(1, 16))

    # Missing evidence
    story.append(Paragraph("Missing Evidence (Recommended)", HEADING_STYLE))
    if missing_evidence:
        for m in missing_evidence:
            story.append(Paragraph(
                f"✗ <b>{m.item}</b> — {m.reason}",
                BODY_STYLE,
            ))
    else:
        story.append(Paragraph("All expected evidence types are present.", BODY_STYLE))

    doc.build(story)
    return buf.getvalue()


def _text_to_pdf(text: str, title: str) -> bytes:
    """Convert plain text (letter) to a simple PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    # Convert newlines to paragraphs
    for line in text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, BODY_STYLE))
        else:
            story.append(Spacer(1, 8))

    if not story:
        story.append(Paragraph(title, BODY_STYLE))

    doc.build(story)
    return buf.getvalue()


async def build_packet(request: PacketBuildRequest) -> tuple[bytes, list[str]]:
    """
    Build the full submission packet ZIP.

    Returns:
        (zip_bytes, list_of_files_included)
    """
    files_included: list[str] = []
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. CoverSheet.pdf
        cover = _build_cover_sheet(
            user_info=request.user_info,
            disaster_id=request.disaster_id,
            declarations=request.declarations,
            daily_burn=request.daily_burn,
            runway_days=request.runway_days,
            monthly_rent=request.runway.monthly_rent,
            monthly_payroll=request.runway.monthly_payroll,
            cash_on_hand=request.runway.cash_on_hand,
            num_employees=request.runway.num_employees,
            business_type=request.runway.business_type,
        )
        zf.writestr("CoverSheet.pdf", cover)
        files_included.append("CoverSheet.pdf")

        # 2. DamageSummary.pdf
        damage_pdf = _build_damage_summary(request.damage_claims)
        zf.writestr("DamageSummary.pdf", damage_pdf)
        files_included.append("DamageSummary.pdf")

        # 3. ExpenseLedger.csv + ExpenseLedger.pdf
        ledger_csv = _build_expense_ledger_csv(request.expense_items)
        zf.writestr("ExpenseLedger.csv", ledger_csv)
        files_included.append("ExpenseLedger.csv")

        ledger_pdf = _build_expense_ledger_pdf(request.expense_items)
        zf.writestr("ExpenseLedger.pdf", ledger_pdf)
        files_included.append("ExpenseLedger.pdf")

        # 4. EvidenceChecklist.pdf
        checklist = _build_evidence_checklist(
            request.rename_map, request.missing_evidence
        )
        zf.writestr("EvidenceChecklist.pdf", checklist)
        files_included.append("EvidenceChecklist.pdf")

        # 5. Evidence/ folder with standardized filenames
        rename_lookup = {
            entry.original_filename: entry.recommended_filename
            for entry in request.rename_map
        }
        for original_fn, b64_content in request.evidence_files.items():
            try:
                file_bytes = base64.b64decode(b64_content)
                new_name = rename_lookup.get(original_fn, original_fn)
                path = f"Evidence/{new_name}"
                zf.writestr(path, file_bytes)
                files_included.append(path)
            except Exception as e:
                logger.warning(f"Failed to include evidence file {original_fn}: {e}")

        # 6. Letters/ folder
        letter_vars = {
            "business_name": request.user_info.business_name,
            "owner_name": request.user_info.owner_name,
            "business_address": request.user_info.address,
            "phone": request.user_info.phone,
            "email": request.user_info.email,
            "disaster_id": request.disaster_id,
            "declaration_title": (
                request.declarations[0].declaration_title
                if request.declarations
                else "the recent disaster"
            ),
            "days_closed": request.runway.days_closed,
            "num_employees": request.runway.num_employees,
            "monthly_rent": request.runway.monthly_rent,
            "monthly_payroll": request.runway.monthly_payroll,
            "business_type": request.runway.business_type,
        }

        rendered_letters = await render_all_letters(letter_vars)
        for letter_name, letter_text in rendered_letters.items():
            # Save as both .txt and .pdf
            txt_path = f"Letters/{letter_name}.txt"
            zf.writestr(txt_path, letter_text)
            files_included.append(txt_path)

            pdf_bytes = _text_to_pdf(letter_text, letter_name.replace("_", " ").title())
            pdf_path = f"Letters/{letter_name}.pdf"
            zf.writestr(pdf_path, pdf_bytes)
            files_included.append(pdf_path)

    return zip_buf.getvalue(), files_included
