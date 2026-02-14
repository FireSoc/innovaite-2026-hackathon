import type {
  EligibilityResponse,
  RunwayRequest,
  RunwayResponse,
  EvidenceContext,
  EvidenceExtractionResponse,
  PacketBuildRequest,
  PlanGenerateRequest,
  PlanResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new ApiError(body.detail || "An unexpected error occurred", response.status);
  }
  return response.json();
}

// ── Eligibility ──────────────────────────────────────────────

export async function lookupEligibility(
  county: string,
  state: string
): Promise<EligibilityResponse> {
  const res = await fetch(`${API_BASE}/eligibility/lookup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ county, state }),
  });
  return handleResponse<EligibilityResponse>(res);
}

// ── Runway ───────────────────────────────────────────────────

export async function calculateRunway(
  data: RunwayRequest
): Promise<RunwayResponse> {
  const res = await fetch(`${API_BASE}/runway/calc`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse<RunwayResponse>(res);
}

// ── Evidence Extraction ──────────────────────────────────────

export async function extractEvidence(
  files: File[],
  context: EvidenceContext
): Promise<EvidenceExtractionResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("context", JSON.stringify(context));

  const res = await fetch(`${API_BASE}/ai/evidence/extract`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<EvidenceExtractionResponse>(res);
}

// ── Packet Build ─────────────────────────────────────────────

export async function buildPacket(data: PacketBuildRequest): Promise<Blob> {
  const res = await fetch(`${API_BASE}/packet/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(body.detail || "Failed to build packet", res.status);
  }
  return res.blob();
}

// ── Plan Generate ────────────────────────────────────────────

export async function generatePlan(
  data: PlanGenerateRequest
): Promise<PlanResponse> {
  const res = await fetch(`${API_BASE}/plan/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return handleResponse<PlanResponse>(res);
}
