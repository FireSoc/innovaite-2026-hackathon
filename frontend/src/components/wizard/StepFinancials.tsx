"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { calculateRunway } from "@/lib/api";
import { BUSINESS_TYPES } from "@/lib/constants";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Separator } from "@/components/ui/separator";
import {
  Building2,
  ArrowRight,
  ArrowLeft,
  Users,
  DollarSign,
  Calendar,
} from "lucide-react";
import { toast } from "sonner";

export function StepFinancials() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const r = state.runway;

  const updateField = (field: string, value: string | number) => {
    dispatch({ type: "SET_RUNWAY", data: { [field]: value } });
    setErrors((prev) => {
      const next = { ...prev };
      delete next[field];
      return next;
    });
  };

  const validate = () => {
    const errs: Record<string, string> = {};
    if (!r.business_type) errs.business_type = "Required";
    if (r.monthly_rent <= 0) errs.monthly_rent = "Enter a value greater than 0";
    if (r.monthly_payroll <= 0) errs.monthly_payroll = "Enter a value greater than 0";
    if (r.cash_on_hand <= 0) errs.cash_on_hand = "Enter a value greater than 0";
    if (r.days_closed <= 0) errs.days_closed = "Enter a value greater than 0";
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async () => {
    if (!validate()) return;
    setLoading(true);

    try {
      const data = await calculateRunway(r);
      dispatch({ type: "SET_RUNWAY_RESULT", data });
      dispatch({ type: "NEXT_STEP" });
      toast.success("Runway calculated!");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to calculate runway";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

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
          <Building2 className="h-5 w-5" />
          <span className="text-sm font-semibold uppercase-label">
            Step 2
          </span>
          <span className="text-muted-foreground/70 text-xs font-medium">· Remedy</span>
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2"
        >
          Tell us about your business
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-muted-foreground"
        >
          This information helps us calculate your financial runway and
          prioritize the most impactful relief actions.
        </motion.p>
      </div>

      {/* Form */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="space-y-5">
          <Select
            label="Business Type"
            placeholder="Select your business type..."
            value={r.business_type}
            onChange={(e) => updateField("business_type", e.target.value)}
            options={BUSINESS_TYPES.map((t) => ({ value: t, label: t }))}
            error={errors.business_type}
          />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Input
              label="Number of Employees"
              type="number"
              min={0}
              placeholder="e.g. 9"
              value={r.num_employees || ""}
              onChange={(e) =>
                updateField("num_employees", parseInt(e.target.value) || 0)
              }
              icon={<Users className="h-4 w-4" />}
            />
            <Input
              label="Days Closed / Until Reopen"
              type="number"
              min={0}
              placeholder="e.g. 14"
              value={r.days_closed || ""}
              onChange={(e) =>
                updateField("days_closed", parseInt(e.target.value) || 0)
              }
              icon={<Calendar className="h-4 w-4" />}
              error={errors.days_closed}
            />
          </div>

          <Separator />

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Input
              label="Monthly Rent"
              type="number"
              min={0}
              step={100}
              placeholder="e.g. 3500"
              value={r.monthly_rent || ""}
              onChange={(e) =>
                updateField("monthly_rent", parseFloat(e.target.value) || 0)
              }
              prefix="$"
              error={errors.monthly_rent}
            />
            <Input
              label="Monthly Payroll"
              type="number"
              min={0}
              step={100}
              placeholder="e.g. 28000"
              value={r.monthly_payroll || ""}
              onChange={(e) =>
                updateField("monthly_payroll", parseFloat(e.target.value) || 0)
              }
              prefix="$"
              error={errors.monthly_payroll}
            />
          </div>

          <Input
            label="Cash on Hand"
            type="number"
            min={0}
            step={100}
            placeholder="e.g. 15000"
            value={r.cash_on_hand || ""}
            onChange={(e) =>
              updateField("cash_on_hand", parseFloat(e.target.value) || 0)
            }
            prefix="$"
            icon={<DollarSign className="h-4 w-4" />}
            hint="Total available cash / liquid assets right now"
            error={errors.cash_on_hand}
          />
        </Card>
      </motion.div>

      {/* Navigation */}
      <div className="flex items-center justify-between mt-8">
        <Button
          variant="ghost"
          onClick={() => dispatch({ type: "PREV_STEP" })}
          icon={<ArrowLeft className="h-4 w-4" />}
        >
          Back
        </Button>
        <Button
          onClick={handleSubmit}
          loading={loading}
          iconRight={<ArrowRight className="h-4 w-4" />}
        >
          Calculate Runway
        </Button>
      </div>
    </div>
  );
}
