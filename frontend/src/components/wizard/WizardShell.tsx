"use client";

import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useWizard } from "@/hooks/useWizardStore";
import { WIZARD_STEPS } from "@/lib/constants";
import { Check } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import Image from "next/image";
import { Footer } from "@/components/Footer";

interface WizardShellProps {
  children: React.ReactNode;
}

const slideVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 40 : -40,
    opacity: 0,
    filter: "blur(4px)",
  }),
  center: {
    x: 0,
    opacity: 1,
    filter: "blur(0px)",
  },
  exit: (direction: number) => ({
    x: direction > 0 ? -40 : 40,
    opacity: 0,
    filter: "blur(4px)",
  }),
};

export function WizardShell({ children }: WizardShellProps) {
  const { state } = useWizard();
  const { currentStep, direction } = state;

  // Don't show step indicator on hero (step 0) or results (step 6)
  const showStepper = currentStep > 0 && currentStep <= 5;

  return (
    <div className="min-h-screen flex flex-col">
      {/* Step indicator */}
      {showStepper && (
        <motion.nav
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="sticky top-0 z-40 bg-card/80 backdrop-blur-xl border-b border-border"
        >
          <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3">
            <div className="flex items-center justify-between">
              {/* Logo — Remedy prominence */}
              <div className="flex items-center gap-2.5 mr-6 shrink-0">
                <Image src="/logo.png" alt="Remedy" width={32} height={32} className="h-8 w-8 object-contain" />
                <div className="flex flex-col">
                  <span className="text-base font-bold text-foreground leading-tight">
                    Remedy
                  </span>
                  <span className="hidden lg:inline text-[11px] text-muted-foreground font-medium uppercase-label">
                    Disaster Relief Made Simple
                  </span>
                </div>
              </div>

              {/* Steps */}
              <div className="flex items-center gap-1 sm:gap-2 flex-1">
                {WIZARD_STEPS.slice(1).map((step, index) => {
                  const stepNum = index + 1;
                  const isActive = currentStep === stepNum;
                  const isCompleted = currentStep > stepNum;

                  return (
                    <div key={step.id} className="flex items-center flex-1 last:flex-none">
                      {/* Step pill */}
                      <div className="flex items-center gap-1.5">
                        <motion.div
                          layout
                          className={cn(
                            "flex items-center justify-center h-7 w-7 rounded-full text-xs font-semibold transition-all duration-300 shrink-0",
                            isCompleted
                              ? "bg-success text-white shadow-sm shadow-success/20"
                              : isActive
                              ? "bg-primary text-primary-foreground shadow-md shadow-amber/20"
                              : "bg-muted text-muted-foreground border border-border"
                          )}
                        >
                          {isCompleted ? (
                            <motion.span
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ type: "spring", stiffness: 300 }}
                            >
                              <Check className="h-3.5 w-3.5" />
                            </motion.span>
                          ) : (
                            stepNum
                          )}
                        </motion.div>
                        <span
                          className={cn(
                            "text-xs font-medium transition-colors hidden sm:block",
                            isActive
                              ? "text-foreground"
                              : isCompleted
                              ? "text-success"
                              : "text-muted-foreground"
                          )}
                        >
                          {step.label}
                        </span>
                      </div>

                      {/* Connector */}
                      {index < WIZARD_STEPS.length - 2 && (
                        <div className="flex-1 mx-2 h-0.5 bg-border/50 relative overflow-hidden rounded-full">
                          <motion.div
                            className="absolute inset-y-0 left-0 bg-gradient-to-r from-success to-success/80 rounded-full"
                            initial={{ width: "0%" }}
                            animate={{
                              width: isCompleted ? "100%" : "0%",
                            }}
                            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Progress bar */}
            <div className="mt-2 sm:hidden">
              <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-amber to-success rounded-full"
                  animate={{ width: `${((currentStep - 1) / 4) * 100}%` }}
                  transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
            </div>
          </div>
        </motion.nav>
      )}

      {/* Content area */}
      <div className="flex-1 relative">
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={currentStep}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              duration: 0.35,
              ease: [0.16, 1, 0.3, 1],
            }}
            className="w-full"
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </div>

      <Footer />
    </div>
  );
}
