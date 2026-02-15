#  Remedy 
- innovaite-2026-hackathon.vercel.app
- The risk mitigation tool to save communities

---

## Details

- **What problem does this project solve?**  
  Remedy addresses disaster relief concerns by cutting time-to-submittable from days to about 30 minutes. It helps owners quickly create a submission-ready packet (cover sheet, damage summary, expense ledger, letters, evidence) and a 30-minute action plan, so they can immediately request rent forbearance, utility waivers, and lender extensions while applying for SBA/FEMA relief. The goal is to buy 2–4 weeks of runway and reduce friction during the worst period of the liquidity crunch.


- **Did you use any interesting libraries or services?**  
  - CommonStack / Google Gemini – Vision-capable LLM for evidence extraction (OCR + categorization, damage detection)
  - OpenFEMA API – Disaster declarations and county eligibility (no API key)
  - Tesseract (pytesseract) + PyMuPDF – OCR on images and PDFs for amounts/dates before LLM
  - FastAPI + Pydantic – Async API, structured schemas, validation
  - ReportLab – PDF generation for cover sheets, damage summaries, and letters
  - Jinja2 – Letter templates (landlord forbearance, utility waiver, lender extension)
  - Next.js 16 + Radix UI + Framer Motion – Form flow, drag-and-drop evidence upload, download of the generated ZIP

- **What extension type(s) did you build?**  
  - Web app – Next.js frontend + FastAPI backend
  - REST API for eligibility, runway calculation, evidence extraction, packet build, and action plan generation

- **If given longer, what would be the next improvement you would make?**  
  - API performance – Parallelize OCR for evidence extraction and letter rendering; add shared HTTP client and eligibility caching
  - Evidence pipeline – Stream processing so users see partial results while more files are processed
  - Better context - Find APIs and customize AI models for different counties and areas based off environmental conditions 

---

## Set Up Instructions
  - Deployed fullstack: innovaite-2026-hackathon.vercel.app
  
  - Install requirements.txt in backend - venv recommended
  - Setup an .env file with your Commonstack.ai API key
  - You need both the backend and frontend running. The frontend calls the backend API at http://127.0.0.1:8000 by default.

### Prereqs
  - Node.js 18+ (for the frontend)
  - Python 3.11+
  - Tesseract OCR installed on your system (e.g. brew install tesseract on macOS)
  - CommonStack API key (free tier available at commonstack.ai) — or a Google AI (Gemini) API key if you prefer using Gemini directly

### Install
```bash
git clone https://github.com/FireSoc/innovaite-2026-hackathon.git

# backend
cd backend
uvicorn ap

# frontend
cd frontend
npm run dev
