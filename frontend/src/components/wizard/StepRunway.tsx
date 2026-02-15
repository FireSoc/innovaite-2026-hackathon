"use client";

import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ProgressRing } from "@/components/ui/ProgressRing";
import { NumberTicker } from "@/components/magicui/NumberTicker";
import {
  TrendingUp,
  ArrowRight,
  ArrowLeft,
  Clock,
  Flame,
  Plus,
  AlertTriangle,
} from "lucide-react";

export function StepRunway() {
  const { state, dispatch } = useWizard();
  const result = state.runwayResult;

  if (!result) {
    return (
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-12 text-center">
        <p className="text-muted-foreground">No runway data. Please go back and fill in financials.</p>
        <Button
          variant="ghost"
          onClick={() => dispatch({ type: "PREV_STEP" })}
          className="mt-4"
          icon={<ArrowLeft className="h-4 w-4" />}
        >
          Go Back
        </Button>
      </div>
    );
  }

  const { runway_days, daily_burn, deferrable_estimates } = result;

  // Calculate ring percentage: 30 days = healthy (100%), 0 = critical (0%)
  const maxDays = 60;
  const ringPercent = Math.min(100, (runway_days / maxDays) * 100);
  const variant =
    runway_days < 7 ? "danger" : runway_days < 21 ? "amber" : "success";

  // Total potential runway gain from deferrals
  const totalGain = deferrable_estimates.reduce(
    (sum, d) => sum + d.estimated_savings_days,
    0
  );

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
          <TrendingUp className="h-5 w-5" />
          <span className="text-sm font-semibold uppercase-label">
            Step 3
          </span>
          <span className="text-muted-foreground/70 text-xs font-medium">· Remedy</span>
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2"
        >
          Your financial runway
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-muted-foreground"
        >
          Based on your numbers, here&apos;s how much time you have — and how
          relief actions can extend it.
        </motion.p>
      </div>

      {/* Runway gauge */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="flex flex-col items-center py-10 mb-6 relative overflow-hidden">
          {/* Subtle gradient background */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-amber/3 pointer-events-none" />

          <div className="relative">
            <ProgressRing
              value={ringPercent}
              size={220}
              strokeWidth={14}
              variant={variant}
              label={`${Math.round(runway_days)}`}
              sublabel="days remaining"
            />
          </div>

          <div className="flex items-center gap-6 mt-8 text-sm relative">
            <div className="flex items-center gap-2">
              <Flame className="h-4 w-4 text-danger" />
              <span className="text-muted-foreground">
                Daily burn:{" "}
                <NumberTicker
                  value={daily_burn}
                  prefix="$"
                  className="font-mono font-semibold text-foreground"
                  decimalPlaces={0}
                  delay={0.3}
                />
              </span>
            </div>
            <div className="h-4 w-px bg-border" />
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-amber" />
              <span className="text-muted-foreground">
                Potential gain:{" "}
                <NumberTicker
                  value={Math.round(totalGain)}
                  prefix="+"
                  suffix=" days"
                  className="font-mono font-semibold text-success"
                  delay={0.5}
                />
              </span>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Deferrable estimates */}
      {deferrable_estimates.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
            Runway Extension Opportunities
          </h3>
          <div className="space-y-3">
            {deferrable_estimates.map((d, i) => (
              <motion.div
                key={d.category}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
              >
                <Card
                  padding="sm"
                  className="flex items-center justify-between gap-4 hover:border-success/30 transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <Plus className="h-3.5 w-3.5 text-success" />
                      <span className="text-sm font-semibold text-foreground">
                        {d.category}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground pl-5.5">
                      {d.description}
                    </p>
                  </div>
                  <Badge variant="success" size="md">
                    +{Math.round(d.estimated_savings_days)}d
                  </Badge>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      )}

      {/* Urgency message */}
      {runway_days < 14 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="p-4 rounded-2xl bg-danger-light/50 border border-danger/20 mb-6 flex items-start gap-3"
        >
          <AlertTriangle className="h-5 w-5 text-danger shrink-0 mt-0.5" />
          <p className="text-sm text-danger font-medium">
            With less than 2 weeks of runway, immediate action on forbearance
            letters and SBA applications is critical. The next steps will help
            you maximize your time.
          </p>
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
          iconRight={<ArrowRight className="h-4 w-4" />}
        >
          Upload Evidence
        </Button>
      </div>
    </div>
  );
}
