## ReliefBridge (MVP) — Agent-Optimized Project Summary

### Mission

Build a **“money-in-hand fastest path” generator** that helps disaster-impacted small businesses/nonprofits **buy 2–4 weeks of runway and cut application/admin friction** by producing a **submission-ready packet (ZIP)** plus a **30-minute action plan**.

### Target users

* **Small business owners** (most critical: payroll + rent pressure)
* **Nonprofits** (cashflow + documentation burden)

### Core problem (post-catastrophe liquidity crunch)

In the first **2–8 weeks after a catastrophe**, many entities fail **not from long-term risk**, but from:

* **insufficient liquidity**
* **admin friction** (evidence gathering, forms, back-and-forth)
* **delayed relief timelines** (SBA/FEMA latency)

ReliefBridge aims to reduce “time-to-submittable” from **days → ~30 minutes** and create immediate runway via **structured forbearance/waiver requests** and **bill prioritization**.

---

## MVP Deliverable

### Output artifacts

1. **Submission Packet ZIP** (the product)

* `CoverSheet.pdf` (who/what/where + disaster ID + key numbers + contact)
* `DamageSummary.pdf` (1 page bullets + totals)
* `ExpenseLedger.csv` + `ExpenseLedger.pdf`
* `EvidenceChecklist.pdf`
* `Evidence/` folder with **standardized filenames**
* `Letters/` folder with **ready-to-send** PDFs + copyable text:

  * landlord forbearance
  * utility late-fee waiver
  * lender/vendor net-terms extension

2. **30-minute Action Plan**

* 3–6 ordered steps
* each step includes: **copyable text**, **attached letter**, **why** (“buys ~14 days runway”, “avoids late fees”), and **time estimate**

---

## Inputs (MVP form)

Required:

* **County + state** (or address → county if you have time)
* Business type
* # employees
* monthly rent
* monthly payroll
* cash on hand
* days closed so far (or “days until reopen”)

Optional:

* Upload **3–10 photos/receipts/screenshots/PDFs**

---

## Data sources (public, hackathon-feasible)

### Eligibility lookup

* **OpenFEMA Disaster Declarations Summaries**: find matching declaration(s), confirm **county inclusion**, extract **disaster declaration ID**, show available **program flags**.

### Optional “expectation bands” (nice-to-have)

* **Public SBA disaster loan dataset (non-PII)** for historical ranges to inform messaging *as a sanity check*, not underwriting.

---

## Key mechanism (non-risk-score)

### 1) Action Sequencer (time-to-cash optimizer)

Rules-based (no ML) ordering to maximize runway gained in the first 48 hours.

**Action scoring**

* `runway_gain_days` (heuristic)
* `success_prob` (heuristic)
* `time_to_execute_min`
* priority rule: “long-latency applications appear in top N”

**Order by**

* `(runway_gain_days * success_prob) / time_to_execute_min`, with hard constraints:

  * Always include “Start SBA application” in top N
  * Always include at least 2 “runway-now” actions (letters)

### 2) Submission Packet Generator

Takes eligibility + user numbers (+ optional evidence extraction results) and builds a consistent bundle that reduces back-and-forth.

---

# Best AI Feature to Implement (24-hour MVP)

## Implement: **Evidence Auto-Builder** (highest ROI, best fit to theme)

This directly attacks the core bottleneck: turning messy evidence into a submission-ready bundle fast.

### AI behavior (constrained)

From uploaded images/PDFs:

* extract structured **expense items** (vendor/date/amount/category/confidence)
* generate **recommended standardized filenames** per upload
* draft **damage narrative bullets** from extracted signals + user context
* detect **missing evidence** (e.g., lease statement missing)

### Why this is the best choice

* **Reduces time-to-submittable** dramatically (your thesis)
* Makes the “packet factory” feel real and valuable in a demo
* Works without requiring risky claims (no underwriting, no approvals)

### Hard guardrails (must-have)

* OCR is the source of truth for amounts/dates.
* LLM can **normalize/categorize**, but **must not invent** values:

  * require each extracted field to reference an OCR span ID or source text snippet
* Strict JSON schema + validation (Pydantic)
* If validation fails: retry once; else degrade to “Needs review” entries.

---

## Secondary AI Feature (if time): **Action Plan Narrator**

Keep sequencing deterministic; use AI only to convert sequencer output into:

* 3–6 step checklist
* call/email scripts
* “attach these files” reminders
  Low hallucination risk because it’s grounded in your own action objects.

---

## Features to keep mostly non-AI (for safety + speed)

### Creditor letters

Use **owned templates** with variables (deterministic), optionally add AI for:

* 1 hardship paragraph
* subject line
* tone adjustment by counterparty
  Run a “letter lint” to block:
* legal promises (“guaranteed”, “entitled”)
* invented program IDs / agency names

If lint fails → fall back to fully templated letter.

---

# Minimal Architecture (implementation-oriented)

### Frontend

* Next.js single-page form
* Upload component
* Results page: “Generate” → download ZIP + checklist UI

### Backend

* FastAPI (recommended for speed + Pydantic)
* Local filesystem storage for hackathon

### Services (suggested module split)

* `services/eligibility.py` → OpenFEMA lookup by county/state, returns declaration + program flags
* `services/runway.py` → deterministic runway calculator
* `services/sequencer.py` → action objects + ordering
* `services/ocr.py` → OCR extraction (Tesseract or cloud OCR if allowed)
* `services/llm_client.py` → `complete_json(schema, inputs)` only
* `services/evidence.py` → OCR → LLM JSON → validate → normalization
* `services/letters.py` → template render (+ optional AI paragraph + lint)
* `services/packet.py` → PDFs + ZIP builder

### Endpoints (MVP)

* `POST /eligibility/lookup` → `{ county, state }` → `{ disaster_id, declarations[], programs[] }`
* `POST /runway/calc` → user numbers → `{ runway_days, deferrable_estimates }`
* `POST /ai/evidence/extract` → uploads + context → `{ expense_items[], rename_map[], missing_evidence[] }`
* `POST /packet/build` → all inputs → returns ZIP bytes/URL
* `POST /plan/generate` → sequencer output → `{ checklist[] }`

---

# Data contracts (agent-friendly)

### Canonical “Action” object

* `id`
* `title`
* `type` (RUNWAY_NOW | LONG_LATENCY | ADMIN)
* `time_to_execute_min`
* `runway_gain_days`
* `success_prob`
* `requires` (files/fields)
* `produces` (letter/pdf/etc.)
* `copy_text` (optional)

### Evidence extraction JSON (strict)

* `expense_items[]: { vendor, date, amount, category, confidence, source_file, source_text }`
* `rename_map[]: { original_filename, recommended_filename, confidence }`
* `damage_claims[]: { label, detail, confidence, source_text }`
* `missing_evidence[]: { item, reason }`

---

# Success criteria (demo + real-world)

* **Eligibility confirmed** for a selected county (disaster ID shown)
* Generate ZIP that contains:

  * cover sheet + damage summary + ledger + letters + evidence folder
* Action plan is:

  * 3–6 steps
  * each step has “why” + time estimate + copyable content
* Evidence uploads produce:

  * categorized ledger lines
  * standardized filenames
  * missing-doc suggestions

---

# Demo Script (2 minutes)

1. Pick a county impacted by a hurricane; ReliefBridge finds FEMA declaration + county coverage.
2. Enter a bakery scenario: 9 employees, rent + payroll, cash on hand → shows runway days.
3. Upload 5 receipts/photos → evidence auto-builder generates ledger + renamed files + damage summary bullets.
4. Click Generate → download ZIP + see “Do this in 30 minutes” list with letters attached.

---

## Recommended build choice (final)

**Ship Evidence Auto-Builder + Packet Generator + Deterministic Sequencer** as the core.
If you have time left, add **Action Plan Narrator** (AI) and keep letter AI minimal (template-first).

If you want, I can also provide:

* a concrete JSON schema + Pydantic models for the evidence extractor,
* 3–4 letter templates (landlord/utility/vendor/lender) with safe variable slots,
* and a small set of sequencer heuristics with default runway-gain assumptions.
