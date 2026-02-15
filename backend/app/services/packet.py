"""Submission Packet Builder — generates PDFs and assembles the ZIP bundle.

Produces:
  - OverallSummary.pdf (holistic overview, steps, resources)
  - CoverSheet.pdf
  - DamageSummary.pdf
  - ExpenseLedger.pdf + ExpenseLedger.csv
  - EvidenceChecklist.pdf
  - Evidence/ folder with standardized filenames
  - Letters/ folder with ready-to-send PDFs
"""

import asyncio
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

from app.models.actions import ChecklistItem
from app.models.evidence import DamageClaim, ExpenseItem, MissingEvidence, RenameEntry
from app.models.inputs import Declaration, PacketBuildRequest, UserInfo
from app.models.outputs import (
    CompletenessScore,
    Deadline,
    DeferrableEstimate,
    DisasterBenchmark,
    PacketFileEntry,
    ResultsSummary,
)
from app.services.benchmarks import fetch_disaster_benchmarks
from app.services.completeness import score_completeness
from app.services.deadlines import compute_deadlines
from app.services.insights import (
    ActionNarrativesResult,
    FinancialBreakdownResult,
    SituationAnalysisResult,
    generate_action_narratives,
    generate_financial_breakdown,
    generate_situation_analysis,
)
from app.services.letters import render_all_letters
from app.services.runway import calculate_runway
from app.services.sequencer import generate_plan

logger = logging.getLogger(__name__)

# Path or prefix -> short description for results UI
FILE_DESCRIPTIONS: dict[str, str] = {
    "OverallSummary.pdf": "Read-first holistic overview, steps, and resources",
    "CoverSheet.pdf": "Business info & disaster ID",
    "DamageSummary.pdf": "Bullet-point damage overview",
    "ExpenseLedger.csv": "Categorized expense data",
    "ExpenseLedger.pdf": "Formatted expense report",
    "EvidenceChecklist.pdf": "What's included & what's missing",
    "Evidence/": "Evidence file",
    "Letters/": "Forbearance or waiver letter",
}

# Curated resource links for the Overall Summary PDF
RESOURCE_LINKS: list[dict[str, str]] = [
    {
        "name": "SBA Disaster Loan Assistance",
        "url": "https://disasterloanassistance.sba.gov/",
        "description": "Apply for SBA disaster loans",
    },
    {
        "name": "FEMA Disaster Assistance",
        "url": "https://www.disasterassistance.gov/",
        "description": "Register for FEMA assistance (or call 1-800-621-3362)",
    },
    {
        "name": "USA.gov disaster recovery",
        "url": "https://www.usa.gov/disaster-recovery",
        "description": "Federal disaster recovery resources",
    },
]


def _description_for_path(path: str) -> str:
    """Return a short description for a packet file path."""
    if path in FILE_DESCRIPTIONS:
        return FILE_DESCRIPTIONS[path]
    for prefix, desc in FILE_DESCRIPTIONS.items():
        if path.startswith(prefix):
            return desc
    return "Packet file"

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
    story.append(Paragraph("Remedy — Submission Cover Sheet", TITLE_STYLE))
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


AI_BODY_STYLE = ParagraphStyle(
    "AIBody",
    parent=BODY_STYLE,
    leftIndent=8,
    borderPadding=4,
    backColor=colors.HexColor("#F8FAFC"),
)


def _add_ai_section(story: list, text: str) -> None:
    """Append an AI-generated text block to the story."""
    if not text:
        return
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para.replace("\n", "<br/>"), AI_BODY_STYLE))
    story.append(Spacer(1, 10))


def _build_overall_summary_pdf(
    request: PacketBuildRequest,
    deferrable_estimates: list[DeferrableEstimate],
    checklist: list[ChecklistItem],
    total_expenses: float,
    *,
    situation: SituationAnalysisResult | None = None,
    financial: FinancialBreakdownResult | None = None,
    narratives: ActionNarrativesResult | None = None,
    deadlines: list[Deadline] | None = None,
    benchmark: DisasterBenchmark | None = None,
    completeness: CompletenessScore | None = None,
) -> bytes:
    """Generate OverallSummary.pdf: holistic overview, AI insights, numbers, steps, resources."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    story = []

    business_name = request.user_info.business_name or "Your business"
    disaster_id = request.disaster_id or "N/A"
    need_2_4_weeks = request.daily_burn * 21 if request.daily_burn else 0

    # 1. Title and overview
    story.append(Paragraph("Overall Summary — Remedy Submission Packet", TITLE_STYLE))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"This packet supports <b>{business_name}</b> for disaster declaration <b>{disaster_id}</b>. "
        "Use it for relief applications and immediate runway actions (forbearance, waivers, SBA, FEMA).",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 12))

    # --- AI: Situation Assessment (injected after overview) ---
    if situation and situation.assessment_text:
        story.append(Paragraph("Situation Assessment", HEADING_STYLE))
        _add_ai_section(story, situation.assessment_text)

    # --- Deadline Countdowns ---
    if deadlines:
        story.append(Paragraph("Critical Filing Deadlines", HEADING_STYLE))
        dl_data = [["Program", "Deadline", "Days Left", "Status"]]
        for dl in deadlines:
            status = "EXPIRED" if dl.is_expired else ("URGENT" if dl.days_remaining <= 7 else "OK")
            dl_data.append([
                dl.program,
                dl.due_date,
                str(dl.days_remaining),
                status,
            ])
        dl_table = Table(dl_data, colWidths=[2.5 * inch, 1.5 * inch, 1 * inch, 1 * inch])
        dl_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        # Color-code expired/urgent rows
        for row_idx, dl in enumerate(deadlines, start=1):
            if dl.is_expired:
                dl_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (2, row_idx), (3, row_idx), colors.HexColor("#DC2626")),
                    ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]))
            elif dl.days_remaining <= 7:
                dl_table.setStyle(TableStyle([
                    ("TEXTCOLOR", (2, row_idx), (3, row_idx), colors.HexColor("#D97706")),
                    ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"),
                ]))
        story.append(dl_table)
        story.append(Spacer(1, 12))

    # --- SBA / FEMA Historical Benchmark ---
    if benchmark and benchmark.available:
        story.append(Paragraph("How Others Fared — FEMA Disaster Data", HEADING_STYLE))
        bench_lines = []
        if benchmark.disaster_title:
            bench_lines.append(f"<b>Disaster:</b> {benchmark.disaster_title}")
        if benchmark.incident_type:
            bench_lines.append(f"<b>Incident type:</b> {benchmark.incident_type}")
        if benchmark.total_amount_ihp_approved is not None:
            bench_lines.append(
                f"<b>Total Individual &amp; Households Assistance approved:</b> "
                f"${benchmark.total_amount_ihp_approved:,.0f}"
            )
        if benchmark.total_amount_ha_approved is not None:
            bench_lines.append(
                f"<b>Total Housing Assistance approved:</b> "
                f"${benchmark.total_amount_ha_approved:,.0f}"
            )
        if benchmark.total_amount_ona_approved is not None:
            bench_lines.append(
                f"<b>Total Other Needs Assistance approved:</b> "
                f"${benchmark.total_amount_ona_approved:,.0f}"
            )
        if benchmark.total_applicants is not None:
            bench_lines.append(
                f"<b>Total approved applicants:</b> {benchmark.total_applicants:,}"
            )
        for line in bench_lines:
            story.append(Paragraph(line, BODY_STYLE))
        story.append(Paragraph(
            "<i>Source: OpenFEMA FemaWebDisasterSummaries — live aggregate data</i>",
            ParagraphStyle("Source", parent=BODY_STYLE, fontSize=8, textColor=colors.grey),
        ))
        story.append(Spacer(1, 12))

    # --- Packet Completeness ---
    if completeness:
        story.append(Paragraph(
            f"Packet Completeness: {completeness.score}%", HEADING_STYLE
        ))
        story.append(Paragraph(completeness.summary, BODY_STYLE))
        story.append(Spacer(1, 4))
        for ci in completeness.items:
            marker = "✓" if ci.present else "✗"
            weight_label = {1: "", 2: " (important)", 3: " (critical)"}.get(ci.weight, "")
            line = f"{marker} <b>{ci.item}</b>{weight_label}"
            if not ci.present and ci.reason:
                line += f" — <i>{ci.reason}</i>"
            story.append(Paragraph(line, BODY_STYLE))
        story.append(Spacer(1, 12))

    # 2. Your numbers
    story.append(Paragraph("Your Numbers", HEADING_STYLE))
    numbers_data = [
        ["Cash on hand", f"${request.runway.cash_on_hand:,.2f}"],
        ["Monthly rent", f"${request.runway.monthly_rent:,.2f}"],
        ["Monthly payroll", f"${request.runway.monthly_payroll:,.2f}"],
        ["Daily burn rate", f"${request.daily_burn:,.2f}"],
        ["Estimated runway", f"{request.runway_days:.1f} days"],
        ["Total documented expenses (from evidence)", f"${total_expenses:,.2f}"],
        ["Damage claims in packet", str(len(request.damage_claims))],
    ]
    if need_2_4_weeks > 0:
        numbers_data.append(["Estimated 2–4 week need (reference)", f"~${need_2_4_weeks:,.0f}"])
    t = Table(numbers_data, colWidths=[3.5 * inch, 3 * inch])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # --- AI: Financial Breakdown (injected after numbers) ---
    if financial:
        if financial.expense_analysis_text:
            story.append(Paragraph("Expense Analysis", HEADING_STYLE))
            _add_ai_section(story, financial.expense_analysis_text)

        if financial.scenarios_text:
            story.append(Paragraph("Runway Scenarios", HEADING_STYLE))
            _add_ai_section(story, financial.scenarios_text)

        if financial.weekly_narrative_text:
            story.append(Paragraph("Week-by-Week Cash Outlook", HEADING_STYLE))
            _add_ai_section(story, financial.weekly_narrative_text)

    # 3. What to ask for
    story.append(Paragraph("What to Ask For", HEADING_STYLE))
    for est in deferrable_estimates:
        story.append(Paragraph(f"<b>{est.category}</b>", BODY_STYLE))
        story.append(Paragraph(f"   {est.description}", BODY_STYLE))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))

    # 4. Steps towards action (with AI-enriched narratives)
    story.append(Paragraph("Steps Towards Action", HEADING_STYLE))
    for item in checklist:
        line = f"<b>{item.step_number}. {item.title}</b> — Time: {item.time_estimate_min} min."
        if item.attached_file:
            line += f" Attach: {item.attached_file}"
        story.append(Paragraph(line, BODY_STYLE))
        # Use AI personalized_why if available, otherwise original why
        if narratives and item.step_number in narratives.personalized_whys:
            story.append(Paragraph(
                f"   Why: {narratives.personalized_whys[item.step_number]}",
                BODY_STYLE,
            ))
        else:
            story.append(Paragraph(f"   Why: {item.why}", BODY_STYLE))
        # AI call/email script
        if narratives and item.step_number in narratives.call_scripts:
            story.append(Paragraph(
                f"   <i>Script: {narratives.call_scripts[item.step_number]}</i>",
                AI_BODY_STYLE,
            ))
        # AI attach reminder
        if narratives and item.step_number in narratives.attach_reminders:
            story.append(Paragraph(
                f"   <i>Files to attach: {narratives.attach_reminders[item.step_number]}</i>",
                AI_BODY_STYLE,
            ))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))

    # 5. Where to send these documents
    story.append(Paragraph("Where to Send These Documents", HEADING_STYLE))
    table_cell_style = ParagraphStyle(
        "TableCell", parent=BODY_STYLE, fontSize=9, leading=12,
    )
    table_header_style = ParagraphStyle(
        "TableHeader", parent=table_cell_style,
        fontName="Helvetica-Bold", textColor=colors.white,
    )
    send_data = [
        [
            Paragraph("Document / Letter", table_header_style),
            Paragraph("Send to", table_header_style),
        ],
        [
            Paragraph("Landlord forbearance letter (Letters/)", table_cell_style),
            Paragraph("Landlord or property manager", table_cell_style),
        ],
        [
            Paragraph("Utility waiver letter (Letters/)", table_cell_style),
            Paragraph("Your utility company", table_cell_style),
        ],
        [
            Paragraph("Lender extension letter (Letters/)", table_cell_style),
            Paragraph("Each vendor or lender", table_cell_style),
        ],
        [
            Paragraph("Cover sheet + Damage summary + Expense ledger + Evidence", table_cell_style),
            Paragraph(f"SBA disaster loan application (reference disaster {disaster_id})", table_cell_style),
        ],
        [
            Paragraph("Same packet elements + FEMA registration", table_cell_style),
            Paragraph("FEMA — disasterassistance.gov or 1-800-621-3362", table_cell_style),
        ],
    ]
    t2 = Table(send_data, colWidths=[3 * inch, 3.5 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 16))

    # 6. Resources and links
    story.append(Paragraph("Resources and Links", HEADING_STYLE))
    for link in RESOURCE_LINKS:
        story.append(Paragraph(f"<b>{link['name']}</b> — {link['description']}", BODY_STYLE))
        story.append(Paragraph(f"   {link['url']}", BODY_STYLE))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This summary is for guidance only; confirm amounts and programs with the agencies.",
        ParagraphStyle("Disclaimer", parent=BODY_STYLE, fontSize=8, textColor=colors.grey),
    ))

    doc.build(story)
    return buf.getvalue()


async def build_packet(
    request: PacketBuildRequest,
) -> tuple[bytes, list[PacketFileEntry], ResultsSummary]:
    """
    Build the full submission packet ZIP.

    Returns:
        (zip_bytes, files_included with descriptions, results_summary)
    """
    files_included_paths: list[str] = []
    zip_buf = io.BytesIO()

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
    letter_count = len(rendered_letters) * 2  # .txt and .pdf per letter

    # Data for OverallSummary.pdf: deferrable estimates, action checklist, total expenses
    runway_result = calculate_runway(
        monthly_rent=request.runway.monthly_rent,
        monthly_payroll=request.runway.monthly_payroll,
        cash_on_hand=request.runway.cash_on_hand,
        days_closed=request.runway.days_closed,
        num_employees=request.runway.num_employees,
    )
    deferrable_estimates = runway_result["deferrable_estimates"]
    action_checklist = generate_plan(
        business_type=request.runway.business_type,
        monthly_rent=request.runway.monthly_rent,
        monthly_payroll=request.runway.monthly_payroll,
        daily_burn=request.daily_burn,
        runway_days=request.runway_days,
        disaster_id=request.disaster_id,
        has_landlord=True,
        has_utilities=True,
        has_lender=True,
        has_insurance=True,
    )
    total_expenses = sum(e.amount for e in request.expense_items)

    # --- AI insight generation (all three run concurrently) ---
    declaration_title = (
        request.declarations[0].declaration_title
        if request.declarations
        else "the recent disaster"
    )
    programs = []
    if request.declarations:
        d0 = request.declarations[0]
        if d0.ih_program:
            programs.append("Individual and Households")
        if d0.ia_program:
            programs.append("Individual Assistance")
        if d0.pa_program:
            programs.append("Public Assistance")
        if d0.hm_program:
            programs.append("Hazard Mitigation")

    situation_task = generate_situation_analysis(
        business_name=request.user_info.business_name or "Your business",
        business_type=request.runway.business_type,
        disaster_id=request.disaster_id,
        declaration_title=declaration_title,
        programs=programs,
        runway_days=request.runway_days,
        daily_burn=request.daily_burn,
        cash_on_hand=request.runway.cash_on_hand,
        monthly_rent=request.runway.monthly_rent,
        monthly_payroll=request.runway.monthly_payroll,
        days_closed=request.runway.days_closed,
        num_employees=request.runway.num_employees,
        expense_total=total_expenses,
        damage_claim_count=len(request.damage_claims),
    )
    financial_task = generate_financial_breakdown(
        expense_items=request.expense_items,
        daily_burn=request.daily_burn,
        runway_days=request.runway_days,
        monthly_rent=request.runway.monthly_rent,
        monthly_payroll=request.runway.monthly_payroll,
        cash_on_hand=request.runway.cash_on_hand,
        deferrable_estimates=deferrable_estimates,
    )
    narratives_task = generate_action_narratives(
        checklist=action_checklist,
        business_name=request.user_info.business_name or "Your business",
        business_type=request.runway.business_type,
        disaster_id=request.disaster_id,
        daily_burn=request.daily_burn,
        runway_days=request.runway_days,
    )

    # --- Deterministic computations (no await needed) ---
    deadline_list = compute_deadlines(request.declarations)

    completeness_result = score_completeness(
        rename_map=request.rename_map,
        missing_evidence=request.missing_evidence,
        has_expense_items=len(request.expense_items) > 0,
        has_damage_claims=len(request.damage_claims) > 0,
        has_letters=True,  # letters are always generated
    )

    # --- Concurrent: AI insights + benchmark API call ---
    benchmark_task = fetch_disaster_benchmarks(request.disaster_id)

    situation_result, financial_result, narratives_result, benchmark_result = (
        await asyncio.gather(
            situation_task, financial_task, narratives_task, benchmark_task,
        )
    )

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 0. OverallSummary.pdf (read this first — now with AI insights)
        overall_pdf = _build_overall_summary_pdf(
            request=request,
            deferrable_estimates=deferrable_estimates,
            checklist=action_checklist,
            total_expenses=total_expenses,
            situation=situation_result,
            financial=financial_result,
            narratives=narratives_result,
            deadlines=deadline_list,
            benchmark=benchmark_result,
            completeness=completeness_result,
        )
        zf.writestr("OverallSummary.pdf", overall_pdf)
        files_included_paths.append("OverallSummary.pdf")

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
        files_included_paths.append("CoverSheet.pdf")

        # 2. DamageSummary.pdf
        damage_pdf = _build_damage_summary(request.damage_claims)
        zf.writestr("DamageSummary.pdf", damage_pdf)
        files_included_paths.append("DamageSummary.pdf")

        # 3. ExpenseLedger.csv + ExpenseLedger.pdf
        ledger_csv = _build_expense_ledger_csv(request.expense_items)
        zf.writestr("ExpenseLedger.csv", ledger_csv)
        files_included_paths.append("ExpenseLedger.csv")

        ledger_pdf = _build_expense_ledger_pdf(request.expense_items)
        zf.writestr("ExpenseLedger.pdf", ledger_pdf)
        files_included_paths.append("ExpenseLedger.pdf")

        # 4. EvidenceChecklist.pdf
        checklist = _build_evidence_checklist(
            request.rename_map, request.missing_evidence
        )
        zf.writestr("EvidenceChecklist.pdf", checklist)
        files_included_paths.append("EvidenceChecklist.pdf")

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
                files_included_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to include evidence file {original_fn}: {e}")

        # 6. Letters/ folder
        for letter_name, letter_text in rendered_letters.items():
            txt_path = f"Letters/{letter_name}.txt"
            zf.writestr(txt_path, letter_text)
            files_included_paths.append(txt_path)

            pdf_bytes = _text_to_pdf(letter_text, letter_name.replace("_", " ").title())
            pdf_path = f"Letters/{letter_name}.pdf"
            zf.writestr(pdf_path, pdf_bytes)
            files_included_paths.append(pdf_path)

    files_included = [
        PacketFileEntry(path=p, description=_description_for_path(p))
        for p in files_included_paths
    ]
    business_name = request.user_info.business_name or "Your business"
    disaster_id = request.disaster_id or "N/A"
    damage_claim_count = len(request.damage_claims)
    expense_count = len(request.expense_items)
    runway_days = request.runway_days
    one_line_summary = (
        f"Submission packet for {business_name} (Disaster {disaster_id}): "
        f"{damage_claim_count} damage claim(s), {expense_count} expense(s), "
        f"{runway_days:.0f} days runway."
    )
    results_summary = ResultsSummary(
        damage_claim_count=damage_claim_count,
        expense_count=expense_count,
        letter_count=letter_count,
        runway_days=runway_days,
        business_name=business_name,
        disaster_id=disaster_id,
        one_line_summary=one_line_summary,
        key_insights=situation_result.key_insights if situation_result else [],
        urgency_level=situation_result.urgency_level if situation_result else "moderate",
        deadlines=deadline_list,
        benchmark=benchmark_result,
        completeness=completeness_result,
    )
    return zip_buf.getvalue(), files_included, results_summary
