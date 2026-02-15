"use client";

import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { ActionPlan } from "./ActionPlan";
import { BenchmarkCard } from "./BenchmarkCard";
import { CompletenessRing } from "./CompletenessRing";
import { DeadlineCountdowns } from "./DeadlineCountdowns";
import { KeyInsights } from "./KeyInsights";
import { PacketSummary } from "./PacketSummary";
import { ScenarioModeler } from "./ScenarioModeler";
import { Button } from "@/components/ui/Button";
import { Particles } from "@/components/magicui/Particles";
import { RotateCcw, PartyPopper } from "lucide-react";
import { RESULTS_HEADLINE, RESULTS_SUBTITLE } from "@/lib/copy";
import { Footer } from "@/components/Footer";

export function ResultsView() {
  const { state, dispatch } = useWizard();
  const { planResult, packetBlob, packetFilename, resultsSummary, filesIncluded } = state;

  const urgencyLevel = resultsSummary?.urgency_level ?? "moderate";
  const urgencyConfig: Record<string, { label: string; color: string }> = {
    critical: { label: "Critical", color: "text-danger" },
    urgent: { label: "Urgent", color: "text-amber-dark" },
    moderate: { label: "Moderate", color: "text-success" },
  };
  const urgency = urgencyConfig[urgencyLevel] ?? urgencyConfig.moderate;

  return (
    <div className="min-h-screen bg-muted/30 relative">
      {/* Celebration header */}
      <div className="bg-card border-b border-border relative overflow-hidden">
        {/* Subtle particle effect in the header */}
        <Particles
          className="absolute inset-0"
          quantity={30}
          staticity={50}
          ease={100}
          color="#10B981"
          size={0.3}
        />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 sm:py-14 text-center relative">
          <div className="inline-flex items-center justify-center gap-5">
            <motion.div
              initial={{ opacity: 0, scale: 0.5, rotate: -10 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              transition={{ type: "spring", stiffness: 200, damping: 12 }}
              className="inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-success-light border border-success/20 flex-shrink-0"
            >
              <PartyPopper className="h-7 w-7 text-success" />
            </motion.div>
            <div className="text-left">
              <motion.h1
                initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight"
              >
                {RESULTS_HEADLINE}
                {resultsSummary?.urgency_level && (
                  <span className={`ml-3 text-base font-semibold ${urgency.color}`}>
                    ({urgency.label} Priority)
                  </span>
                )}
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                className="text-muted-foreground text-lg mt-1"
              >
                {resultsSummary?.one_line_summary ?? RESULTS_SUBTITLE}
              </motion.p>
            </div>
          </div>
        </div>
      </div>

      {/* Two-column: Packet (left) + Action Plan (right) */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 sm:pt-10">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Left column: Action Plan */}
          <motion.div
            initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.25, duration: 0.6 }}
            className="lg:col-span-3"
          >
            {planResult && <ActionPlan checklist={planResult.checklist} />}
          </motion.div>

          {/* Right column: Packet download + included files */}
          <motion.div
            initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.35, duration: 0.6 }}
            className="lg:col-span-2"
          >
            <div className="lg:sticky lg:top-8">
              <PacketSummary
                blob={packetBlob}
                filename={packetFilename}
                runwayDays={state.runwayResult?.runway_days}
                businessName={state.userInfo.business_name}
                disasterId={state.eligibilityResult?.disaster_id}
                resultsSummary={resultsSummary}
                filesIncluded={filesIncluded}
              />
            </div>
          </motion.div>
        </div>
      </div>

      {/* Key Insights */}
      {resultsSummary?.key_insights && resultsSummary.key_insights.length > 0 && (
        <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-8 sm:pt-10">
          <KeyInsights insights={resultsSummary.key_insights} />
        </div>
      )}

      {/* Data-driven feature panels */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-6 sm:pt-8 pb-8 sm:pb-12 space-y-6">
        {/* Row 1: Deadlines + Benchmark side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {resultsSummary?.deadlines && resultsSummary.deadlines.length > 0 && (
            <DeadlineCountdowns deadlines={resultsSummary.deadlines} />
          )}
          {resultsSummary?.benchmark && resultsSummary.benchmark.available && (
            <BenchmarkCard benchmark={resultsSummary.benchmark} />
          )}
        </div>

        {/* Row 2: Completeness + Scenario Modeler side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {resultsSummary?.completeness && (
            <CompletenessRing completeness={resultsSummary.completeness} />
          )}
          {state.runwayResult && (
            <ScenarioModeler
              baselineRunwayDays={state.runwayResult.runway_days}
              dailyBurn={state.runwayResult.daily_burn}
              cashOnHand={state.runway.cash_on_hand}
              deferrableEstimates={state.runwayResult.deferrable_estimates}
            />
          )}
        </div>

      </div>

      {/* Start over — full-width banner */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 mt-[-4] mb-10">
        <Button
          variant="secondary"
          size="lg"
          onClick={() => dispatch({ type: "RESET" })}
          icon={<RotateCcw className="h-5 w-5" />}
          className="w-full text-base font-semibold"
        >
          Start Over with a New Case
        </Button>
      </div>

      <Footer />
    </div>
  );
}
