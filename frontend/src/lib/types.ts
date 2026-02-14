/* ── Types mirroring backend Pydantic models ──────────────── */

// --- Eligibility ---

export interface EligibilityRequest {
  county: string;
  state: string;
}

export interface Declaration {
  disaster_number: string;
  declaration_title: string;
  incident_type: string;
  ih_program: boolean;
  ia_program: boolean;
  pa_program: boolean;
  hm_program: boolean;
  declaration_date: string;
  incident_begin_date: string;
  incident_end_date: string;
}

export interface EligibilityResponse {
  disaster_id: string;
  declarations: Declaration[];
  programs: string[];
  county: string;
  state: string;
}

// --- Runway ---

export interface RunwayRequest {
  business_type: string;
  num_employees: number;
  monthly_rent: number;
  monthly_payroll: number;
  cash_on_hand: number;
  days_closed: number;
}

export interface DeferrableEstimate {
  category: string;
  estimated_savings_days: number;
  description: string;
}

export interface RunwayResponse {
  runway_days: number;
  daily_burn: number;
  deferrable_estimates: DeferrableEstimate[];
}

// --- Evidence ---

export type ConfidenceLevel = "high" | "medium" | "needs_review";

export interface ExpenseItem {
  vendor: string;
  date: string;
  amount: number;
  category: string;
  confidence: ConfidenceLevel;
  source_file: string;
  source_text: string;
}

export interface RenameEntry {
  original_filename: string;
  recommended_filename: string;
  confidence: ConfidenceLevel;
}

export interface DamageClaim {
  label: string;
  detail: string;
  confidence: ConfidenceLevel;
  source_text: string;
}

export interface MissingEvidence {
  item: string;
  reason: string;
}

export interface EvidenceContext {
  business_type: string;
  county: string;
  state: string;
  disaster_id: string;
  declaration_title: string;
}

export interface EvidenceExtractionResponse {
  expense_items: ExpenseItem[];
  rename_map: RenameEntry[];
  damage_claims: DamageClaim[];
  missing_evidence: MissingEvidence[];
}

// --- Packet ---

export interface UserInfo {
  business_name: string;
  owner_name: string;
  address: string;
  phone: string;
  email: string;
}

export interface PacketBuildRequest {
  user_info: UserInfo;
  eligibility: EligibilityRequest;
  disaster_id: string;
  declarations: Declaration[];
  runway: RunwayRequest;
  runway_days: number;
  daily_burn: number;
  expense_items: ExpenseItem[];
  damage_claims: DamageClaim[];
  rename_map: RenameEntry[];
  missing_evidence: MissingEvidence[];
  evidence_files: Record<string, string>;
}

// --- Plan ---

export interface ChecklistItem {
  step_number: number;
  title: string;
  why: string;
  time_estimate_min: number;
  copy_text: string | null;
  attached_file: string | null;
}

export interface PlanGenerateRequest {
  business_type: string;
  num_employees: number;
  monthly_rent: number;
  monthly_payroll: number;
  cash_on_hand: number;
  days_closed: number;
  runway_days: number;
  daily_burn: number;
  disaster_id: string;
  has_landlord: boolean;
  has_utilities: boolean;
  has_lender: boolean;
  has_insurance: boolean;
}

export interface PlanResponse {
  checklist: ChecklistItem[];
}
