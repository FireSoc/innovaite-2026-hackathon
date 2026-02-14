"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import type { ChecklistItem } from "@/lib/types";
import {
  Clock,
  Copy,
  Check,
  FileText,
  Lightbulb,
  Zap,
} from "lucide-react";

interface ActionPlanProps {
  checklist: ChecklistItem[];
}

export function ActionPlan({ checklist }: ActionPlanProps) {
  const [copiedStep, setCopiedStep] = useState<number | null>(null);

  const totalTime = checklist.reduce(
    (sum, item) => sum + item.time_estimate_min,
    0
  );

  const handleCopy = async (text: string, stepNum: number) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStep(stepNum);
      setTimeout(() => setCopiedStep(null), 2000);
    } catch {
      // Fallback
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-amber/10 flex items-center justify-center">
            <Zap className="h-5 w-5 text-amber" />
          </div>
          <h2 className="text-xl font-bold text-foreground tracking-tight">
            30-Minute Action Plan
          </h2>
        </div>
        <Badge variant="info" size="md">
          <Clock className="h-3.5 w-3.5 mr-1" />
          ~{totalTime} min
        </Badge>
      </div>

      {/* Steps */}
      <Accordion type="single" collapsible className="space-y-3">
        {checklist.map((item, index) => {
          const isCopied = copiedStep === item.step_number;

          return (
            <motion.div
              key={item.step_number}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.08, duration: 0.4 }}
            >
              <AccordionItem
                value={`step-${item.step_number}`}
                className="border-none"
              >
                <Card padding="none" className="overflow-hidden transition-all duration-200 data-[state=open]:ring-1 data-[state=open]:ring-amber/20">
                  {/* Step header */}
                  <AccordionTrigger className="px-4 py-3.5 hover:no-underline hover:bg-accent/50 transition-colors [&[data-state=open]>svg]:rotate-180">
                    <div className="flex items-start gap-4 w-full text-left">
                      {/* Step number */}
                      <div className="flex items-center justify-center h-8 w-8 rounded-full bg-gradient-to-br from-amber/15 to-amber/5 text-amber font-bold text-sm shrink-0 border border-amber/10">
                        {item.step_number}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-sm font-semibold text-foreground">
                            {item.title}
                          </h3>
                          <Badge variant="outline" size="sm">
                            {item.time_estimate_min} min
                          </Badge>
                        </div>
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <Lightbulb className="h-3 w-3 text-amber shrink-0" />
                          <span>{item.why}</span>
                        </div>
                      </div>
                    </div>
                  </AccordionTrigger>

                  {/* Expanded content */}
                  <AccordionContent>
                    <div className="px-4 pb-4 pl-16 space-y-3">
                      {/* Copy text */}
                      {item.copy_text && (
                        <div className="relative group">
                          <div className="p-3 bg-muted/50 rounded-xl text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap font-mono text-xs border border-border/50">
                            {item.copy_text}
                          </div>
                          <button
                            onClick={() =>
                              handleCopy(item.copy_text!, item.step_number)
                            }
                            className="absolute top-2 right-2 flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-border rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:border-ring/50 transition-all focus-ring cursor-pointer shadow-sm"
                          >
                            {isCopied ? (
                              <>
                                <Check className="h-3 w-3 text-success" />
                                Copied!
                              </>
                            ) : (
                              <>
                                <Copy className="h-3 w-3" />
                                Copy
                              </>
                            )}
                          </button>
                        </div>
                      )}

                      {/* Attached file */}
                      {item.attached_file && (
                        <div className="flex items-center gap-2 p-2.5 bg-info-light/30 border border-info/15 rounded-xl">
                          <FileText className="h-4 w-4 text-info shrink-0" />
                          <span className="text-xs font-medium text-info">
                            Attached: {item.attached_file}
                          </span>
                        </div>
                      )}
                    </div>
                  </AccordionContent>
                </Card>
              </AccordionItem>
            </motion.div>
          );
        })}
      </Accordion>
    </div>
  );
}
