"use client";

import { motion } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { ActionPlan } from "./ActionPlan";
import { PacketSummary } from "./PacketSummary";
import { Button } from "@/components/ui/Button";
import { Particles } from "@/components/magicui/Particles";
import { RotateCcw, PartyPopper, Sparkles } from "lucide-react";

export function ResultsView() {
  const { state, dispatch } = useWizard();
  const { planResult, packetBlob, packetFilename } = state;

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
          <motion.div
            initial={{ opacity: 0, scale: 0.5, rotate: -10 }}
            animate={{ opacity: 1, scale: 1, rotate: 0 }}
            transition={{ type: "spring", stiffness: 200, damping: 12 }}
            className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-success-light border border-success/20 mb-5"
          >
            <PartyPopper className="h-8 w-8 text-success" />
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 12, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.2, duration: 0.6 }}
            className="text-3xl sm:text-4xl font-bold text-foreground tracking-tight mb-3"
          >
            Your relief packet is ready!
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-muted-foreground max-w-lg mx-auto text-lg"
          >
            Download your submission-ready ZIP and follow the action plan below
            to maximize your runway in the next 30 minutes.
          </motion.p>
        </div>
      </div>

      {/* Two-column content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Left column: Action Plan (wider) */}
          <motion.div
            initial={{ opacity: 0, x: -20, filter: "blur(4px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="lg:col-span-3"
          >
            {planResult && <ActionPlan checklist={planResult.checklist} />}
          </motion.div>

          {/* Right column: Packet Summary */}
          <motion.div
            initial={{ opacity: 0, x: 20, filter: "blur(4px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="lg:col-span-2"
          >
            <div className="lg:sticky lg:top-8 space-y-6">
              <PacketSummary
                blob={packetBlob}
                filename={packetFilename}
                runwayDays={state.runwayResult?.runway_days}
                businessName={state.userInfo.business_name}
                disasterId={state.eligibilityResult?.disaster_id}
              />

              {/* Start over */}
              <div className="text-center">
                <Button
                  variant="ghost"
                  onClick={() => dispatch({ type: "RESET" })}
                  icon={<RotateCcw className="h-4 w-4" />}
                  className="text-muted-foreground"
                >
                  Start Over
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
