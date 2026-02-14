# Testing ReliefBridge Backend with Postman

**Base URL:** `http://localhost:8000` (start the server with `uvicorn app.main:app --reload` from the `backend` folder)

---

## 1. Health check

| Field   | Value                    |
|--------|---------------------------|
| Method | `GET`                     |
| URL    | `http://localhost:8000/health` |

**Expected response (200):** `{"status":"ok"}`

---

## 2. Eligibility lookup

Find FEMA disaster declarations for a county and state.

| Field   | Value |
|--------|--------|
| Method | `POST` |
| URL    | `http://localhost:8000/eligibility/lookup` |
| Headers | `Content-Type: application/json` (often set automatically when you pick raw JSON) |
| Body   | **raw → JSON** |

**Postman steps (avoids 422 "Field required"):**
1. Click the **Body** tab.
2. Select **raw** (radio button).
3. In the dropdown to the right of "raw", choose **JSON**.
4. In the text area, **paste or type** the body below (do not leave it empty).

**Body (copy this exactly):**
```json
{
  "county": "Harris",
  "state": "TX"
}
```

**Expected response (200):** JSON with `disaster_id`, `declarations[]`, `programs[]`, `county`, `state`.

**If you get 422:** The body was empty or not sent. Ensure the Body tab has **raw** + **JSON** selected and the JSON text is actually in the editor.

---

## 3. Runway calculation

Get runway days and deferrable estimates from business numbers.

| Field   | Value |
|--------|--------|
| Method | `POST` |
| URL    | `http://localhost:8000/runway/calc` |
| Headers | `Content-Type: application/json` |
| Body   | **raw → JSON** |

**Body example:**
```json
{
  "business_type": "bakery",
  "num_employees": 9,
  "monthly_rent": 3500,
  "monthly_payroll": 12000,
  "cash_on_hand": 25000,
  "days_closed": 7
}
```

**Expected response (200):** JSON with `runway_days`, `daily_burn`, `deferrable_estimates[]`.

---

## 4. Evidence extraction (AI)

Extract expense items and damage claims from uploaded files. **OCR-first:** Tesseract extracts text from images/PDFs; the LLM structures and categorizes. Uses CommonStack (or Gemini).

**For evidence extraction with OCR:** Install Tesseract on the host (e.g. `brew install tesseract` on macOS). If Tesseract is missing, the pipeline still runs using the LLM vision path.

| Field   | Value |
|--------|--------|
| Method | `POST` |
| URL    | `http://localhost:8000/ai/evidence/extract` |
| Body   | **form-data** (not raw JSON) |

**Postman steps:**

1. **Method & URL:** Set method to `POST` and URL to `http://localhost:8000/ai/evidence/extract`.
2. **Body tab:** Click **Body**, then select **form-data** (not "raw" and not "x-www-form-urlencoded").
3. **Add the `files` row:**
   - In the Key column, type: `files`
   - Hover over the Key cell; a dropdown appears. Change it from "Text" to **File**.
   - In the Value column, click **Select Files** and choose one or more image or PDF files (e.g. a receipt or damage photo). Max 10 files, 20 MB each.
4. **Add the `context` row:**
   - In the next row, Key: `context` (leave type as **Text**).
   - In Value, paste exactly:  
     `{"business_type":"bakery","county":"Harris","state":"TX","disaster_id":"","declaration_title":""}`
5. **Send:** Click **Send**. You should get 200 with JSON containing `expense_items`, `rename_map`, `damage_claims`, `missing_evidence`.

**Multiple files:** Add another row with Key `files`, type **File**, and choose another file. Postman will send all `files` entries.

**Expected response (200):** JSON with `expense_items[]`, `rename_map[]`, `damage_claims[]`, `missing_evidence[]`.

---

## 5. Plan generate

Get the 30-minute action plan (3–6 steps).

| Field   | Value |
|--------|--------|
| Method | `POST` |
| URL    | `http://localhost:8000/plan/generate` |
| Headers | `Content-Type: application/json` |
| Body   | **raw → JSON** |

**Body example:**
```json
{
  "business_type": "bakery",
  "num_employees": 9,
  "monthly_rent": 3500,
  "monthly_payroll": 12000,
  "cash_on_hand": 25000,
  "days_closed": 7,
  "runway_days": 45.2,
  "daily_burn": 516.67,
  "disaster_id": "4799",
  "has_landlord": true,
  "has_utilities": true,
  "has_lender": true,
  "has_insurance": true
}
```

**Expected response (200):** JSON with `checklist[]` — each item has `step_number`, `title`, `why`, `time_estimate_min`, `copy_text`, `attached_file`.

---

## 6. Packet build (full submission ZIP)

Build the complete submission packet ZIP. Requires a large JSON body that combines eligibility, runway, user info, and (optionally) evidence extraction results.

| Field   | Value |
|--------|--------|
| Method | `POST` |
| URL    | `http://localhost:8000/packet/build` |
| Headers | `Content-Type: application/json` |
| Body   | **raw → JSON** |

**Minimal body example** (no evidence files; you still get CoverSheet, DamageSummary, ledger placeholders, checklist, letters):
```json
{
  "user_info": {
    "business_name": "Demo Bakery",
    "owner_name": "Jane Smith",
    "address": "123 Main St, Houston, TX",
    "phone": "555-123-4567",
    "email": "jane@demobakery.com"
  },
  "eligibility": {
    "county": "Harris",
    "state": "TX"
  },
  "disaster_id": "4799",
  "declarations": [
    {
      "disaster_number": "4799",
      "declaration_title": "Hurricane Helene",
      "incident_type": "Hurricane",
      "ih_program": true,
      "ia_program": true,
      "pa_program": true,
      "hm_program": false,
      "declaration_date": "2024-09-28",
      "incident_begin_date": "2024-09-26",
      "incident_end_date": ""
    }
  ],
  "runway": {
    "business_type": "bakery",
    "num_employees": 9,
    "monthly_rent": 3500,
    "monthly_payroll": 12000,
    "cash_on_hand": 25000,
    "days_closed": 7
  },
  "runway_days": 45.2,
  "daily_burn": 516.67,
  "expense_items": [],
  "damage_claims": [],
  "rename_map": [],
  "missing_evidence": [],
  "evidence_files": {}
}
```

**Expected response (200):** Binary ZIP file. In Postman, use **Send and Download** or ensure response is saved as a file. Headers include `Content-Disposition: attachment; filename="ReliefBridge_..._packet.zip"`.

---

## Sample receipt & damage files (for Evidence extraction)

Use these to test **POST /ai/evidence/extract** with real-looking receipts and disaster images.

### Receipts (direct download)

| Source | What you get | Link |
|--------|----------------|------|
| **ExpressExpense SRD** | 200 receipt images (restaurant style, 600+ px), MIT license. Zip ~19 MB. | [large-receipt-image-dataset-SRD.zip](https://expressexpense.com/large-receipt-image-dataset-SRD.zip) |
| **Humans in the Loop** | 192 receipt images + bounding-box labels (CC0). | [Free Receipt OCR Dataset](https://humansintheloop.org/resources/datasets/free-receipt-ocr-dataset/) (form to request access) |
| **SROIE (Kaggle)** | 973 scanned receipts, English. Good for vendor/date/amount extraction. | [SROIE dataset on Kaggle](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2) (Kaggle account needed) |
| **Hugging Face** | 7,000 invoice images with JSON labels. | [7000_invoice_images_with_json](https://huggingface.co/datasets/Ananthu01/7000_invoice_images_with_json) |

**Quick start:** Download [ExpressExpense SRD](https://expressexpense.com/large-receipt-image-dataset-SRD.zip), unzip, then in Postman pick 1–2 images from the folder as your `files` in the Evidence extract request.

### Damage / disaster images

| Source | What you get | Link |
|--------|----------------|------|
| **FEMA Media Library** | 4,400+ photos (damage, response, recovery). Public domain. Browse and download. | [FEMA Media Library](https://www.fema.gov/fema-media-library) |
| **IEEE DataPort (Hurricane damage)** | Post‑hurricane building damage imagery (e.g. Houston after Harvey), with damage labels. | [Benchmark dataset on IEEE DataPort](https://ieee-dataport.org/open-access/benchmark-dataset-automatic-damaged-building-detection-post-hurricane-remotely-sensed) |

No API key needed for these datasets; download files and upload them in Postman as the `files` in your Evidence extraction request.

---

## Suggested test order

1. **GET /health** — confirm server is up.
2. **POST /eligibility/lookup** — get a real `disaster_id` and `declarations` (e.g. Harris TX).
3. **POST /runway/calc** — get `runway_days` and `daily_burn` for the same scenario.
4. **POST /plan/generate** — use the same numbers + `disaster_id` from step 2.
5. **POST /ai/evidence/extract** — upload 1–2 image files + context; check extracted data.
6. **POST /packet/build** — use the minimal body above first; then try with `expense_items` / `damage_claims` / `rename_map` / `missing_evidence` from step 5 and (optionally) `evidence_files` as base64 if you want files in the ZIP.

---

## Importing into Postman

You can create a new Collection (e.g. "ReliefBridge"), add the 6 requests above, and set a collection variable `base_url` = `http://localhost:8000` so each request URL is `{{base_url}}/eligibility/lookup`, etc.
