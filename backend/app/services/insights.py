"""AI-powered insight generation for the submission packet.

Three functions that call the LLM to produce grounded, personalized analysis
text.  Every function has a deterministic fallback so the OverallSummary.pdf
renders correctly even when the LLM is unavailable.

Guardrails
----------
* Prompts are 100 % grounded in user-supplied numbers — "only reference
  the facts below".
* Strict instructions: "do not invent figures", "do not make legal claims
  or guarantees".
* Each function wraps its LLM call in try/except and returns safe defaults
  on failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pydantic import BaseModel

from app.models.actions import ChecklistItem
from app.models.evidence import ExpenseItem
from app.models.outputs import DeferrableEstimate, InsightUrgency, KeyInsight
from app.services.llm_client import complete_json, complete_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal schemas for structured LLM responses
# ---------------------------------------------------------------------------

class _SituationAnalysisResponse(BaseModel):
    """Schema sent to the LLM for the situation analysis."""

    assessment: str  # 2-3 paragraph plain-language assessment
    urgency_level: str  # "critical" / "urgent" / "moderate"
    key_insights: list[KeyInsight]


class _FinancialBreakdownResponse(BaseModel):
    """Schema sent to the LLM for the financial breakdown."""

    expense_analysis: str  # Category breakdown + prioritization
    scenarios: str  # Three runway scenarios narrative
    weekly_narrative: str  # Week-by-week cash runway narrative


class _ActionNarrativeItem(BaseModel):
    """A single enriched narrative for one checklist step."""

    step_number: int
    personalized_why: str
    call_script: str
    attach_reminder: str


class _ActionNarrativesResponse(BaseModel):
    """Schema sent to the LLM for action narratives."""

    narratives: list[_ActionNarrativeItem]


# ---------------------------------------------------------------------------
# Return containers (plain dataclasses for internal use)
# ---------------------------------------------------------------------------

@dataclass
class SituationAnalysisResult:
    assessment_text: str = ""
    urgency_level: str = "moderate"
    key_insights: list[KeyInsight] = field(default_factory=list)


@dataclass
class FinancialBreakdownResult:
    expense_analysis_text: str = ""
    scenarios_text: str = ""
    weekly_narrative_text: str = ""


@dataclass
class ActionNarrativesResult:
    """Maps step_number -> enriched narrative strings."""

    personalized_whys: dict[int, str] = field(default_factory=dict)
    call_scripts: dict[int, str] = field(default_factory=dict)
    attach_reminders: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Situation Analysis
# ---------------------------------------------------------------------------

async def generate_situation_analysis(
    *,
    business_name: str,
    business_type: str,
    disaster_id: str,
    declaration_title: str,
    programs: list[str],
    runway_days: float,
    daily_burn: float,
    cash_on_hand: float,
    monthly_rent: float,
    monthly_payroll: float,
    days_closed: int,
    num_employees: int,
    expense_total: float,
    damage_claim_count: int,
) -> SituationAnalysisResult:
    """Generate an AI situation assessment with urgency level and key insights."""

    prompt = f"""You are a disaster-relief financial advisor writing for a small business owner.
Using ONLY the facts below, produce a structured JSON response.

STRICT RULES:
- Only reference the facts provided. Do NOT invent figures or statistics.
- Do NOT make legal claims, promises, or use the words "guaranteed" or "entitled".
- Be direct, empathetic, and actionable.
- The assessment should be 2-3 paragraphs of plain-language analysis.
- urgency_level must be exactly one of: "critical", "urgent", "moderate".
  Use "critical" if runway < 14 days, "urgent" if < 30 days, "moderate" otherwise.
- Provide 3-5 key_insights, each with a short title, 1-2 sentence detail, and urgency
  tag ("critical", "action_needed", or "informational").

FACTS:
- Business: {business_name} ({business_type})
- Employees: {num_employees}
- Disaster: {declaration_title} (ID: {disaster_id})
- Eligible programs: {', '.join(programs) if programs else 'None identified'}
- Monthly rent: ${monthly_rent:,.0f}
- Monthly payroll: ${monthly_payroll:,.0f}
- Cash on hand: ${cash_on_hand:,.0f}
- Daily burn rate: ${daily_burn:,.0f}/day
- Estimated runway: {runway_days:.1f} days
- Days closed so far: {days_closed}
- Total documented expenses from evidence: ${expense_total:,.0f}
- Damage claims documented: {damage_claim_count}
- Today's date: {datetime.now().strftime('%B %d, %Y')}

Focus the assessment on:
1. What the financial numbers mean in plain language
2. The most urgent risk (payroll deadline, rent due date, cash exhaustion)
3. Which eligible programs are most relevant and what they provide
4. The single most impactful action to take today"""

    try:
        result = await complete_json(
            schema=_SituationAnalysisResponse,
            prompt=prompt,
            max_retries=1,
        )
        return SituationAnalysisResult(
            assessment_text=result.assessment.strip(),
            urgency_level=result.urgency_level if result.urgency_level in ("critical", "urgent", "moderate") else "moderate",
            key_insights=result.key_insights[:5],
        )
    except Exception as e:
        logger.warning("AI situation analysis failed, using fallback: %s", e)
        return _fallback_situation_analysis(
            business_name=business_name,
            runway_days=runway_days,
            daily_burn=daily_burn,
            days_closed=days_closed,
            num_employees=num_employees,
            monthly_rent=monthly_rent,
            monthly_payroll=monthly_payroll,
            cash_on_hand=cash_on_hand,
        )


def _fallback_situation_analysis(
    *,
    business_name: str,
    runway_days: float,
    daily_burn: float,
    days_closed: int,
    num_employees: int,
    monthly_rent: float,
    monthly_payroll: float,
    cash_on_hand: float,
) -> SituationAnalysisResult:
    """Deterministic fallback when the LLM is unavailable."""

    if runway_days < 14:
        urgency = "critical"
    elif runway_days < 30:
        urgency = "urgent"
    else:
        urgency = "moderate"

    assessment = (
        f"{business_name} currently has approximately ${cash_on_hand:,.0f} in cash "
        f"and a daily burn rate of ${daily_burn:,.0f}, giving an estimated "
        f"{runway_days:.0f} days of runway. The business has been closed for "
        f"{days_closed} days, impacting {num_employees} employees.\n\n"
        f"Monthly obligations include ${monthly_rent:,.0f} in rent and "
        f"${monthly_payroll:,.0f} in payroll. Immediate action on forbearance "
        f"and waiver requests can extend runway significantly."
    )

    insights: list[KeyInsight] = []
    if runway_days < 14:
        insights.append(KeyInsight(
            title=f"Cash runs out in ~{runway_days:.0f} days",
            detail=f"At ${daily_burn:,.0f}/day burn rate, immediate action is required to extend runway.",
            urgency=InsightUrgency.CRITICAL,
        ))
    elif runway_days < 30:
        insights.append(KeyInsight(
            title=f"~{runway_days:.0f} days of runway remaining",
            detail=f"At ${daily_burn:,.0f}/day, runway is limited. Deferring rent or payroll obligations will help.",
            urgency=InsightUrgency.ACTION_NEEDED,
        ))

    if monthly_rent > 0:
        rent_days = monthly_rent / daily_burn if daily_burn > 0 else 0
        insights.append(KeyInsight(
            title="Rent deferral can buy significant time",
            detail=f"A 30-day rent forbearance would free ~${monthly_rent:,.0f}, extending runway by ~{rent_days:.0f} days.",
            urgency=InsightUrgency.ACTION_NEEDED,
        ))

    insights.append(KeyInsight(
        title="Start SBA disaster loan application early",
        detail="SBA loans take 2-3 weeks to process. Starting now ensures funds arrive before runway expires.",
        urgency=InsightUrgency.INFORMATIONAL,
    ))

    return SituationAnalysisResult(
        assessment_text=assessment,
        urgency_level=urgency,
        key_insights=insights[:5],
    )


# ---------------------------------------------------------------------------
# 2. Financial Breakdown
# ---------------------------------------------------------------------------

async def generate_financial_breakdown(
    *,
    expense_items: list[ExpenseItem],
    daily_burn: float,
    runway_days: float,
    monthly_rent: float,
    monthly_payroll: float,
    cash_on_hand: float,
    deferrable_estimates: list[DeferrableEstimate],
) -> FinancialBreakdownResult:
    """Generate an AI financial breakdown with expense analysis and runway scenarios."""

    # Pre-compute category totals for the prompt
    cat_totals: dict[str, float] = {}
    for item in expense_items:
        cat = item.category or "other"
        cat_totals[cat] = cat_totals.get(cat, 0) + item.amount
    total_expenses = sum(cat_totals.values())

    cat_lines = "\n".join(
        f"  - {cat}: ${amt:,.0f} ({amt/total_expenses*100:.0f}%)" if total_expenses > 0
        else f"  - {cat}: ${amt:,.0f}"
        for cat, amt in sorted(cat_totals.items(), key=lambda x: -x[1])
    )

    deferral_lines = "\n".join(
        f"  - {d.category}: saves ~{d.estimated_savings_days:.0f} days — {d.description}"
        for d in deferrable_estimates
    )

    today = datetime.now()
    runway_end = today + timedelta(days=runway_days) if runway_days < 9999 else None
    total_deferral_days = sum(d.estimated_savings_days for d in deferrable_estimates)

    prompt = f"""You are a disaster-relief financial advisor. Using ONLY the facts below, produce a structured JSON response with three fields.

STRICT RULES:
- Only reference the facts provided. Do NOT invent figures.
- Do NOT make legal claims or guarantees.
- Be specific with dollar amounts and timeframes.
- Write in second person ("you", "your business").

FACTS:
- Cash on hand: ${cash_on_hand:,.0f}
- Daily burn rate: ${daily_burn:,.0f}/day
- Monthly rent: ${monthly_rent:,.0f}
- Monthly payroll: ${monthly_payroll:,.0f}
- Runway: {runway_days:.1f} days (cash exhaustion ~{runway_end.strftime('%B %d, %Y') if runway_end else 'not applicable'})
- Documented expenses by category:
{cat_lines if cat_lines else '  (no expenses documented)'}
- Total documented expenses: ${total_expenses:,.0f}
- Available deferrals:
{deferral_lines if deferral_lines else '  (none computed)'}
- Total potential deferral runway gain: ~{total_deferral_days:.0f} days

GENERATE:

1. expense_analysis: A 2-3 paragraph analysis of the expense breakdown. Which categories are largest? Which are deferrable vs critical (payroll is always critical)? What should be prioritized for payment vs. deferred? Be specific with amounts.

2. scenarios: Three short runway scenarios (each 2-3 sentences):
   - Scenario A "Do Nothing": what happens at current burn if no action is taken, with approximate date of cash exhaustion
   - Scenario B "Execute All Deferrals": runway with all forbearance/waivers, approximate new cash-out date
   - Scenario C "Deferrals + SBA Loan": best case with deferrals AND SBA funding (assume 3 months operating expenses), new runway estimate

3. weekly_narrative: A week-by-week breakdown (Week 1, Week 2, Week 3, Week 4) of what the cash position looks like at current burn rate and what deadlines/actions are most important each week. 1-2 sentences per week."""

    try:
        result = await complete_json(
            schema=_FinancialBreakdownResponse,
            prompt=prompt,
            max_retries=1,
        )
        return FinancialBreakdownResult(
            expense_analysis_text=result.expense_analysis.strip(),
            scenarios_text=result.scenarios.strip(),
            weekly_narrative_text=result.weekly_narrative.strip(),
        )
    except Exception as e:
        logger.warning("AI financial breakdown failed, using fallback: %s", e)
        return _fallback_financial_breakdown(
            cat_totals=cat_totals,
            total_expenses=total_expenses,
            daily_burn=daily_burn,
            runway_days=runway_days,
            cash_on_hand=cash_on_hand,
            monthly_rent=monthly_rent,
            monthly_payroll=monthly_payroll,
            total_deferral_days=total_deferral_days,
            runway_end=runway_end,
        )


def _fallback_financial_breakdown(
    *,
    cat_totals: dict[str, float],
    total_expenses: float,
    daily_burn: float,
    runway_days: float,
    cash_on_hand: float,
    monthly_rent: float,
    monthly_payroll: float,
    total_deferral_days: float,
    runway_end: datetime | None,
) -> FinancialBreakdownResult:
    """Deterministic fallback for financial breakdown."""

    # Expense analysis
    if cat_totals:
        sorted_cats = sorted(cat_totals.items(), key=lambda x: -x[1])
        top = sorted_cats[0]
        analysis = (
            f"Your documented expenses total ${total_expenses:,.0f}. "
            f"The largest category is {top[0]} at ${top[1]:,.0f}"
        )
        if total_expenses > 0:
            analysis += f" ({top[1]/total_expenses*100:.0f}% of total)."
        else:
            analysis += "."
        analysis += (
            f" Payroll (${monthly_payroll:,.0f}/month) should be prioritized as it directly "
            f"impacts employee retention. Rent (${monthly_rent:,.0f}/month) is a strong "
            f"candidate for forbearance requests."
        )
    else:
        analysis = (
            f"No expenses have been documented from uploaded evidence. "
            f"Your major monthly obligations are rent (${monthly_rent:,.0f}) and "
            f"payroll (${monthly_payroll:,.0f})."
        )

    # Scenarios
    end_str = runway_end.strftime('%B %d, %Y') if runway_end else "N/A"
    extended_days = runway_days + total_deferral_days
    extended_end = (datetime.now() + timedelta(days=extended_days)).strftime('%B %d, %Y') if extended_days < 9999 else "N/A"
    sba_est = (monthly_rent + monthly_payroll) * 3
    sba_days = sba_est / daily_burn if daily_burn > 0 else 0
    sba_total = extended_days + sba_days

    scenarios = (
        f"Scenario A — Do Nothing: At ${daily_burn:,.0f}/day, cash runs out around {end_str} "
        f"(~{runway_days:.0f} days). No buffer for unexpected costs.\n\n"
        f"Scenario B — Execute All Deferrals: With rent forbearance, utility waivers, and "
        f"vendor extensions, runway extends to ~{extended_days:.0f} days (through ~{extended_end}). "
        f"This buys critical time to apply for longer-term relief.\n\n"
        f"Scenario C — Deferrals + SBA Loan: Adding an SBA disaster loan (~${sba_est:,.0f} in "
        f"operating expenses) brings total runway to ~{sba_total:.0f} days, providing a solid "
        f"bridge to full recovery."
    )

    # Weekly narrative
    week_lines = []
    remaining = cash_on_hand
    for w in range(1, 5):
        spent = daily_burn * 7
        remaining = max(remaining - spent, 0)
        week_lines.append(
            f"Week {w}: Estimated cash remaining ~${remaining:,.0f} "
            f"(after ~${spent:,.0f} in weekly burn)."
        )
        if remaining <= 0:
            week_lines[-1] += " Cash exhausted — immediate relief action required."
            break
    weekly = "\n".join(week_lines)

    return FinancialBreakdownResult(
        expense_analysis_text=analysis,
        scenarios_text=scenarios,
        weekly_narrative_text=weekly,
    )


# ---------------------------------------------------------------------------
# 3. Action Narratives
# ---------------------------------------------------------------------------

async def generate_action_narratives(
    *,
    checklist: list[ChecklistItem],
    business_name: str,
    business_type: str,
    disaster_id: str,
    daily_burn: float,
    runway_days: float,
) -> ActionNarrativesResult:
    """Generate personalized narratives for each action step."""

    steps_block = "\n".join(
        f"  Step {item.step_number}: \"{item.title}\" — "
        f"time: {item.time_estimate_min} min, "
        f"why (generic): {item.why}, "
        f"attached: {item.attached_file or 'none'}"
        for item in checklist
    )

    prompt = f"""You are a disaster-relief advisor personalizing an action plan for a specific business.
Using ONLY the facts below, produce a structured JSON response.

STRICT RULES:
- Only reference the facts provided. Do NOT invent figures or make legal claims.
- Do NOT use the words "guaranteed" or "entitled".
- Write in second person ("you", "your business").
- Each narrative should be specific to THIS business, not generic.

FACTS:
- Business: {business_name} ({business_type})
- Disaster ID: {disaster_id}
- Daily burn: ${daily_burn:,.0f}/day
- Runway: {runway_days:.1f} days
- Today's date: {datetime.now().strftime('%B %d, %Y')}

ACTION STEPS:
{steps_block}

For EACH step, generate a "narratives" array item with:
1. step_number: matching the step number above
2. personalized_why: 1-2 sentences explaining why this step matters specifically for {business_name} as a {business_type}. Reference specific dollar amounts and timeframes from the facts.
3. call_script: A ready-to-use 2-3 sentence phone/email script the business owner can use for this step. Include the disaster ID where relevant.
4. attach_reminder: Which specific files from the submission packet (CoverSheet.pdf, DamageSummary.pdf, ExpenseLedger.pdf, EvidenceChecklist.pdf, or specific letters) should accompany this step."""

    try:
        result = await complete_json(
            schema=_ActionNarrativesResponse,
            prompt=prompt,
            max_retries=1,
        )
        out = ActionNarrativesResult()
        for item in result.narratives:
            out.personalized_whys[item.step_number] = item.personalized_why.strip()
            out.call_scripts[item.step_number] = item.call_script.strip()
            out.attach_reminders[item.step_number] = item.attach_reminder.strip()
        return out
    except Exception as e:
        logger.warning("AI action narratives failed, using fallback: %s", e)
        return _fallback_action_narratives(checklist=checklist)


def _fallback_action_narratives(
    *,
    checklist: list[ChecklistItem],
) -> ActionNarrativesResult:
    """Deterministic fallback — returns empty dicts so the PDF uses existing text."""

    return ActionNarrativesResult()
