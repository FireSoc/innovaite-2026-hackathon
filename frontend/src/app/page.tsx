"use client";

import { useEffect } from "react";
import { WizardProvider, useWizard } from "@/hooks/useWizardStore";
import { WizardShell } from "@/components/wizard/WizardShell";
import { StepHero } from "@/components/wizard/StepHero";
import { StepEligibility } from "@/components/wizard/StepEligibility";
import { StepFinancials } from "@/components/wizard/StepFinancials";
import { StepRunway } from "@/components/wizard/StepRunway";
import { StepEvidence } from "@/components/wizard/StepEvidence";
import { StepReview } from "@/components/wizard/StepReview";
import { ResultsView } from "@/components/results/ResultsView";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";

function WizardContent() {
  const { state } = useWizard();
  const { currentStep } = state;

  // Scroll to top on step change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [currentStep]);

  // Results page (step 6) gets its own full-page layout
  if (currentStep === 6) {
    return <ResultsView />;
  }

  // All wizard steps use the WizardShell wrapper
  return (
    <WizardShell>
      {currentStep === 0 && <StepHero />}
      {currentStep === 1 && <StepEligibility />}
      {currentStep === 2 && <StepFinancials />}
      {currentStep === 3 && <StepRunway />}
      {currentStep === 4 && <StepEvidence />}
      {currentStep === 5 && <StepReview />}
    </WizardShell>
  );
}

export default function Home() {
  return (
    <ErrorBoundary>
      <WizardProvider>
        <WizardContent />
      </WizardProvider>
    </ErrorBoundary>
  );
}
