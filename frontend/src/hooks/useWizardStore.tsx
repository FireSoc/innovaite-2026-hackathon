"use client";

import {
  createContext,
  useContext,
  useReducer,
  type Dispatch,
  type ReactNode,
} from "react";
import type {
  EligibilityResponse,
  RunwayResponse,
  RunwayRequest,
  EligibilityRequest,
  EvidenceExtractionResponse,
  PlanResponse,
  UserInfo,
  Declaration,
  ExpenseItem,
  DamageClaim,
  RenameEntry,
  MissingEvidence,
} from "@/lib/types";

/* ── State Shape ───────────────────────────────────────────── */

export interface WizardState {
  currentStep: number;
  direction: 1 | -1; // for animation direction

  // Step 1: Eligibility
  eligibility: EligibilityRequest;
  eligibilityResult: EligibilityResponse | null;

  // Step 2: Financials / Runway input
  runway: RunwayRequest;

  // Step 3: Runway result
  runwayResult: RunwayResponse | null;

  // Step 4: Evidence
  evidenceFiles: File[];
  evidenceResult: EvidenceExtractionResponse | null;

  // Step 5: Contact / User info
  userInfo: UserInfo;

  // Results
  planResult: PlanResponse | null;
  packetBlob: Blob | null;
  packetFilename: string;

  // UI state
  loading: boolean;
  error: string | null;
}

const initialState: WizardState = {
  currentStep: 0,
  direction: 1,

  eligibility: { county: "", state: "" },
  eligibilityResult: null,

  runway: {
    business_type: "",
    num_employees: 0,
    monthly_rent: 0,
    monthly_payroll: 0,
    cash_on_hand: 0,
    days_closed: 0,
  },

  runwayResult: null,

  evidenceFiles: [],
  evidenceResult: null,

  userInfo: {
    business_name: "",
    owner_name: "",
    address: "",
    phone: "",
    email: "",
  },

  planResult: null,
  packetBlob: null,
  packetFilename: "Remedy_packet.zip",

  loading: false,
  error: null,
};

/* ── Actions ───────────────────────────────────────────────── */

type WizardAction =
  | { type: "NEXT_STEP" }
  | { type: "PREV_STEP" }
  | { type: "GO_TO_STEP"; step: number }
  | { type: "SET_ELIGIBILITY"; data: Partial<EligibilityRequest> }
  | { type: "SET_ELIGIBILITY_RESULT"; data: EligibilityResponse }
  | { type: "SET_RUNWAY"; data: Partial<RunwayRequest> }
  | { type: "SET_RUNWAY_RESULT"; data: RunwayResponse }
  | { type: "SET_EVIDENCE_FILES"; files: File[] }
  | { type: "SET_EVIDENCE_RESULT"; data: EvidenceExtractionResponse }
  | { type: "SET_USER_INFO"; data: Partial<UserInfo> }
  | { type: "SET_PLAN_RESULT"; data: PlanResponse }
  | { type: "SET_PACKET"; blob: Blob; filename: string }
  | { type: "SET_LOADING"; loading: boolean }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "RESET" };

function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case "NEXT_STEP":
      return {
        ...state,
        currentStep: Math.min(state.currentStep + 1, 6), // 6 = results
        direction: 1,
        error: null,
      };
    case "PREV_STEP":
      return {
        ...state,
        currentStep: Math.max(state.currentStep - 1, 0),
        direction: -1,
        error: null,
      };
    case "GO_TO_STEP":
      return {
        ...state,
        currentStep: action.step,
        direction: action.step > state.currentStep ? 1 : -1,
        error: null,
      };
    case "SET_ELIGIBILITY":
      return {
        ...state,
        eligibility: { ...state.eligibility, ...action.data },
      };
    case "SET_ELIGIBILITY_RESULT":
      return { ...state, eligibilityResult: action.data };
    case "SET_RUNWAY":
      return {
        ...state,
        runway: { ...state.runway, ...action.data },
      };
    case "SET_RUNWAY_RESULT":
      return { ...state, runwayResult: action.data };
    case "SET_EVIDENCE_FILES":
      return { ...state, evidenceFiles: action.files };
    case "SET_EVIDENCE_RESULT":
      return { ...state, evidenceResult: action.data };
    case "SET_USER_INFO":
      return {
        ...state,
        userInfo: { ...state.userInfo, ...action.data },
      };
    case "SET_PLAN_RESULT":
      return { ...state, planResult: action.data };
    case "SET_PACKET":
      return {
        ...state,
        packetBlob: action.blob,
        packetFilename: action.filename,
      };
    case "SET_LOADING":
      return { ...state, loading: action.loading };
    case "SET_ERROR":
      return { ...state, error: action.error, loading: false };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

/* ── Context ───────────────────────────────────────────────── */

const WizardContext = createContext<{
  state: WizardState;
  dispatch: Dispatch<WizardAction>;
} | null>(null);

export function WizardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(wizardReducer, initialState);

  return (
    <WizardContext.Provider value={{ state, dispatch }}>
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const context = useContext(WizardContext);
  if (!context) {
    throw new Error("useWizard must be used within a WizardProvider");
  }
  return context;
}
