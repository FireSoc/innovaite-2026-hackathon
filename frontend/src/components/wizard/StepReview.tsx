"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { buildPacket, generatePlan } from "@/lib/api";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Separator } from "@/components/ui/separator";
import { ShimmerButton } from "@/components/magicui/ShimmerButton";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  ClipboardCheck,
  ArrowLeft,
  Download,
  MapPin,
  Building2,
  Camera,
  User,
  Phone,
  Mail,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { toast } from "sonner";

export function StepReview() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(",")[1] || result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleGenerate = useCallback(async () => {
    const { userInfo } = state;
    if (!userInfo.business_name.trim() || !userInfo.owner_name.trim()) {
      toast.error("Please fill in at least business name and owner name.");
      return;
    }

    setLoading(true);

    try {
      const evidenceFiles: Record<string, string> = {};
      for (const file of state.evidenceFiles) {
        evidenceFiles[file.name] = await fileToBase64(file);
      }

      const packetRequest = {
        user_info: state.userInfo,
        eligibility: state.eligibility,
        disaster_id: state.eligibilityResult?.disaster_id || "",
        declarations: state.eligibilityResult?.declarations || [],
        runway: state.runway,
        runway_days: state.runwayResult?.runway_days || 0,
        daily_burn: state.runwayResult?.daily_burn || 0,
        expense_items: state.evidenceResult?.expense_items || [],
        damage_claims: state.evidenceResult?.damage_claims || [],
        rename_map: state.evidenceResult?.rename_map || [],
        missing_evidence: state.evidenceResult?.missing_evidence || [],
        evidence_files: evidenceFiles,
      };

      const planRequest = {
        business_type: state.runway.business_type,
        num_employees: state.runway.num_employees,
        monthly_rent: state.runway.monthly_rent,
        monthly_payroll: state.runway.monthly_payroll,
        cash_on_hand: state.runway.cash_on_hand,
        days_closed: state.runway.days_closed,
        runway_days: state.runwayResult?.runway_days || 0,
        daily_burn: state.runwayResult?.daily_burn || 0,
        disaster_id: state.eligibilityResult?.disaster_id || "",
        has_landlord: true,
        has_utilities: true,
        has_lender: true,
        has_insurance: true,
      };

      const [blob, planData] = await Promise.all([
        buildPacket(packetRequest),
        generatePlan(planRequest),
      ]);

      const safeName =
        state.userInfo.business_name
          .replace(/[^a-zA-Z0-9\s-_]/g, "")
          .trim()
          .replace(/\s+/g, "_")
          .slice(0, 50) || "submission";

      dispatch({
        type: "SET_PACKET",
        blob,
        filename: `Remedy_${safeName}_packet.zip`,
      });
      dispatch({ type: "SET_PLAN_RESULT", data: planData });
      dispatch({ type: "NEXT_STEP" });

      toast.success("Your packet is ready!");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to generate packet";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  }, [state, dispatch]);

  const summaryItems = [
    {
      key: "location",
      icon: <MapPin className="h-4 w-4 text-info" />,
      label: "Location",
      value: state.eligibilityResult
        ? `${state.eligibilityResult.county} County, ${state.eligibilityResult.state}`
        : "Not checked",
      detail: state.eligibilityResult
        ? `Disaster ID: ${state.eligibilityResult.disaster_id}`
        : undefined,
      isReady: !!state.eligibilityResult,
    },
    {
      key: "business",
      icon: <Building2 className="h-4 w-4 text-violet-500" />,
      label: "Business",
      value: state.runway.business_type || "Not specified",
      detail: `${state.runway.num_employees} employees · ${state.runway.days_closed} days closed`,
      isReady: !!state.runway.business_type,
    },
    {
      key: "runway",
      icon: <TrendingUp className="h-4 w-4 text-amber" />,
      label: "Runway",
      value: state.runwayResult
        ? `${Math.round(state.runwayResult.runway_days)} days remaining`
        : "Not calculated",
      detail: state.runwayResult
        ? `$${state.runwayResult.daily_burn.toLocaleString()}/day burn rate`
        : undefined,
      isReady: !!state.runwayResult,
    },
    {
      key: "evidence",
      icon: <Camera className="h-4 w-4 text-success" />,
      label: "Evidence",
      value: `${state.evidenceFiles.length} files uploaded`,
      detail: state.evidenceResult
        ? `${state.evidenceResult.expense_items.length} expenses · ${state.evidenceResult.damage_claims.length} claims`
        : undefined,
      isReady: state.evidenceFiles.length > 0,
    },
  ];

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
      {/* Header */}
      <div className="mb-8">
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="flex items-center gap-2 text-amber mb-3"
        >
          <ClipboardCheck className="h-5 w-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">
            Step 5
          </span>
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2"
        >
          Review & generate
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-muted-foreground"
        >
          Add your contact details and review everything before we build your
          submission packet.
        </motion.p>
      </div>

      {/* Contact form */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="mb-6">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
            Contact Information
          </h3>
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Input
                label="Business Name"
                placeholder="e.g. Maria's Bakery"
                value={state.userInfo.business_name}
                onChange={(e) =>
                  dispatch({
                    type: "SET_USER_INFO",
                    data: { business_name: e.target.value },
                  })
                }
                icon={<Building2 className="h-4 w-4" />}
              />
              <Input
                label="Owner Name"
                placeholder="e.g. Maria Rodriguez"
                value={state.userInfo.owner_name}
                onChange={(e) =>
                  dispatch({
                    type: "SET_USER_INFO",
                    data: { owner_name: e.target.value },
                  })
                }
                icon={<User className="h-4 w-4" />}
              />
            </div>
            <Input
              label="Business Address"
              placeholder="e.g. 123 Main St, Houston, TX 77001"
              value={state.userInfo.address}
              onChange={(e) =>
                dispatch({
                  type: "SET_USER_INFO",
                  data: { address: e.target.value },
                })
              }
              icon={<MapPin className="h-4 w-4" />}
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Input
                label="Phone"
                type="tel"
                placeholder="e.g. (713) 555-0123"
                value={state.userInfo.phone}
                onChange={(e) =>
                  dispatch({
                    type: "SET_USER_INFO",
                    data: { phone: e.target.value },
                  })
                }
                icon={<Phone className="h-4 w-4" />}
              />
              <Input
                label="Email"
                type="email"
                placeholder="e.g. maria@bakery.com"
                value={state.userInfo.email}
                onChange={(e) =>
                  dispatch({
                    type: "SET_USER_INFO",
                    data: { email: e.target.value },
                  })
                }
                icon={<Mail className="h-4 w-4" />}
              />
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Summary */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card variant="glass" className="mb-6">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
            Summary
          </h3>
          <div className="space-y-2">
            {summaryItems.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between p-3 rounded-xl hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <span>{item.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">
                        {item.label}
                      </span>
                      <span className="text-xs text-muted-foreground truncate">
                        {item.value}
                      </span>
                    </div>
                    {item.detail && (
                      <span className="text-xs text-muted-foreground">
                        {item.detail}
                      </span>
                    )}
                  </div>
                </div>
                {item.isReady ? (
                  <CheckCircle2 className="h-4 w-4 text-success shrink-0" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-amber shrink-0" />
                )}
              </div>
            ))}
          </div>
        </Card>
      </motion.div>

      {/* Generate button */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="text-center"
      >
        <ShimmerButton
          onClick={handleGenerate}
          disabled={loading}
          className="w-full shadow-[0_0_30px_rgb(245_158_11/0.2)]"
          shimmerDuration="2.5s"
        >
          {loading ? (
            <>
              <span className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Building your packet...
            </>
          ) : (
            <>
              <Download className="h-5 w-5" />
              Generate Submission Packet
            </>
          )}
        </ShimmerButton>
        <p className="text-xs text-muted-foreground text-center mt-3">
          This will generate your complete ZIP packet with cover sheet, damage
          summary, expense ledger, creditor letters, and a 30-minute action
          plan.
        </p>
      </motion.div>

      {/* Back button */}
      <div className="flex items-center justify-start mt-6">
        <Button
          variant="ghost"
          onClick={() => dispatch({ type: "PREV_STEP" })}
          icon={<ArrowLeft className="h-4 w-4" />}
        >
          Back
        </Button>
      </div>
    </div>
  );
}
