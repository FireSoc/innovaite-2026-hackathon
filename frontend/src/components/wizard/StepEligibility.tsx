"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { lookupEligibility } from "@/lib/api";
import { US_STATES } from "@/lib/constants";
import { Input } from "@/components/ui/Input";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Separator } from "@/components/ui/separator";
import {
  MapPin,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";

export function StepEligibility() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { county, state: stateCode } = state.eligibility;
  const result = state.eligibilityResult;

  const handleLookup = async () => {
    if (!county.trim() || !stateCode) {
      setError("Please enter both county and state.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const data = await lookupEligibility(county.trim(), stateCode);
      dispatch({ type: "SET_ELIGIBILITY_RESULT", data });
      toast.success("Disaster declaration found!");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to lookup eligibility";
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const canProceed = !!result;

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
          <MapPin className="h-5 w-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">
            Step 1
          </span>
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2"
        >
          Verify your location
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-muted-foreground"
        >
          We&apos;ll check FEMA disaster declarations for your county to confirm
          available relief programs.
        </motion.p>
      </div>

      {/* Form */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="mb-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <Input
              label="County"
              placeholder="e.g. Harris"
              value={county}
              onChange={(e) =>
                dispatch({
                  type: "SET_ELIGIBILITY",
                  data: { county: e.target.value },
                })
              }
              icon={<MapPin className="h-4 w-4" />}
              error={error && !county.trim() ? "Required" : undefined}
            />
            <Select
              label="State"
              placeholder="Select state..."
              value={stateCode}
              onChange={(e) =>
                dispatch({
                  type: "SET_ELIGIBILITY",
                  data: { state: e.target.value },
                })
              }
              options={US_STATES.map((s) => ({
                value: s.value,
                label: s.label,
              }))}
              error={error && !stateCode ? "Required" : undefined}
            />
          </div>

          <Button
            onClick={handleLookup}
            loading={loading}
            icon={<ShieldCheck className="h-4 w-4" />}
            className="w-full"
          >
            Check Eligibility
          </Button>

          {error && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-2 mt-4 p-3 bg-danger-light rounded-xl"
            >
              <AlertTriangle className="h-4 w-4 text-danger shrink-0 mt-0.5" />
              <p className="text-sm text-danger">{error}</p>
            </motion.div>
          )}
        </Card>
      </motion.div>

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <Card variant="default" className="border-success/30 bg-success-light/20 relative overflow-hidden">
            {/* Subtle success gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-success/5 to-transparent pointer-events-none" />

            <div className="relative">
              <div className="flex items-start gap-3 mb-4">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: "spring", stiffness: 300, delay: 0.2 }}
                  className="h-10 w-10 rounded-xl bg-success/10 flex items-center justify-center shrink-0"
                >
                  <CheckCircle2 className="h-5 w-5 text-success" />
                </motion.div>
                <div>
                  <h3 className="font-semibold text-foreground mb-0.5">
                    Disaster Declaration Confirmed
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {result.county} County, {result.state} is covered under FEMA
                    declaration{result.declarations.length > 1 ? "s" : ""}.
                  </p>
                </div>
              </div>

              {/* Declarations */}
              <div className="space-y-3 mb-4">
                {result.declarations.map((decl, i) => (
                  <motion.div
                    key={decl.disaster_number}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + i * 0.1 }}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 bg-card rounded-xl border border-border hover:border-success/30 transition-colors"
                  >
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono text-sm font-semibold text-foreground">
                          {decl.disaster_number}
                        </span>
                        <Badge variant="info" size="sm">
                          {decl.incident_type}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {decl.declaration_title}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {decl.ih_program && (
                        <Badge variant="success" size="sm">IH</Badge>
                      )}
                      {decl.ia_program && (
                        <Badge variant="success" size="sm">IA</Badge>
                      )}
                      {decl.pa_program && (
                        <Badge variant="success" size="sm">PA</Badge>
                      )}
                      {decl.hm_program && (
                        <Badge variant="success" size="sm">HM</Badge>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>

              {/* Programs */}
              {result.programs.length > 0 && (
                <div>
                  <Separator className="mb-3" />
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
                    Available Programs
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {result.programs.map((prog) => (
                      <Badge key={prog} variant="outline" size="sm">
                        {prog}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        </motion.div>
      )}

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
          onClick={() => dispatch({ type: "NEXT_STEP" })}
          disabled={!canProceed}
          iconRight={<ArrowRight className="h-4 w-4" />}
        >
          Continue
        </Button>
      </div>
    </div>
  );
}
