"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useWizard } from "@/hooks/useWizardStore";
import { extractEvidence } from "@/lib/api";
import { FileUpload } from "@/components/ui/FileUpload";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Camera,
  ArrowRight,
  ArrowLeft,
  Sparkles,
  AlertTriangle,
  FileText,
  CheckCircle2,
} from "lucide-react";
import { toast } from "sonner";
import type { ConfidenceLevel } from "@/lib/types";

const confidenceBadge = (level: ConfidenceLevel) => {
  switch (level) {
    case "high":
      return <Badge variant="success" size="sm">High</Badge>;
    case "medium":
      return <Badge variant="warning" size="sm">Medium</Badge>;
    case "needs_review":
      return <Badge variant="danger" size="sm">Review</Badge>;
  }
};

export function StepEvidence() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);

  const result = state.evidenceResult;

  const handleFilesChange = (files: File[]) => {
    dispatch({ type: "SET_EVIDENCE_FILES", files });
  };

  const handleExtract = async () => {
    if (state.evidenceFiles.length === 0) {
      toast.error("Please upload at least one file.");
      return;
    }
    setLoading(true);

    try {
      const context = {
        business_type: state.runway.business_type,
        county: state.eligibility.county,
        state: state.eligibility.state,
        disaster_id: state.eligibilityResult?.disaster_id || "",
        declaration_title:
          state.eligibilityResult?.declarations[0]?.declaration_title || "",
      };
      const data = await extractEvidence(state.evidenceFiles, context);
      dispatch({ type: "SET_EVIDENCE_RESULT", data });
      toast.success(
        `Extracted ${data.expense_items.length} expenses and ${data.damage_claims.length} damage claims!`
      );
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Evidence extraction failed";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
      {/* Header */}
      <div className="mb-8">
        <motion.div
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
          className="flex items-center gap-2 text-amber mb-3"
        >
          <Camera className="h-5 w-5" />
          <span className="text-sm font-semibold uppercase tracking-wider">
            Step 4
          </span>
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-2xl sm:text-3xl font-bold text-foreground tracking-tight mb-2"
        >
          Upload your evidence
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-muted-foreground"
        >
          Upload receipts, photos of damage, invoices, or any supporting
          documents. Our AI will extract and categorize everything automatically.
        </motion.p>
      </div>

      {/* Upload zone */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="mb-6">
          <FileUpload
            onFilesChange={handleFilesChange}
            maxFiles={10}
            maxSizeMB={20}
          />

          {state.evidenceFiles.length > 0 && (
            <div className="mt-6">
              <Button
                onClick={handleExtract}
                loading={loading}
                icon={<Sparkles className="h-4 w-4" />}
                className="w-full"
              >
                {loading
                  ? "Analyzing with AI..."
                  : `Analyze ${state.evidenceFiles.length} file${state.evidenceFiles.length > 1 ? "s" : ""} with AI`}
              </Button>
            </div>
          )}
        </Card>
      </motion.div>

      {/* Results */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 16, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="space-y-4"
          >
            {/* Summary bar */}
            <div className="flex items-center gap-3 p-4 rounded-2xl bg-success-light/30 border border-success/20">
              <CheckCircle2 className="h-5 w-5 text-success shrink-0" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">AI extraction complete.</span>{" "}
                Found {result.expense_items.length} expenses,{" "}
                {result.damage_claims.length} damage claims, and{" "}
                {result.rename_map.length} file renames.
                {result.missing_evidence.length > 0 && (
                  <span className="text-danger font-medium">
                    {" "}
                    {result.missing_evidence.length} missing item
                    {result.missing_evidence.length > 1 ? "s" : ""} detected.
                  </span>
                )}
              </p>
            </div>

            {/* Accordion for results */}
            <Accordion type="multiple" defaultValue={["expenses", "damage", "missing"]} className="space-y-3">
              {/* Expense items */}
              {result.expense_items.length > 0 && (
                <AccordionItem value="expenses" className="border-none">
                  <Card padding="none" className="overflow-hidden">
                    <AccordionTrigger className="px-4 py-3 hover:no-underline hover:bg-accent/50 transition-colors [&[data-state=open]>svg]:rotate-180">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-amber" />
                        <span className="font-semibold text-foreground text-sm">
                          Expense Items ({result.expense_items.length})
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-0">
                      <div className="px-4 pb-4 overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow className="border-border">
                              <TableHead className="text-xs uppercase font-medium text-muted-foreground">Vendor</TableHead>
                              <TableHead className="text-xs uppercase font-medium text-muted-foreground">Date</TableHead>
                              <TableHead className="text-right text-xs uppercase font-medium text-muted-foreground">Amount</TableHead>
                              <TableHead className="text-xs uppercase font-medium text-muted-foreground">Category</TableHead>
                              <TableHead className="text-center text-xs uppercase font-medium text-muted-foreground">Confidence</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.expense_items.map((item, i) => (
                              <TableRow key={i} className="border-border/50">
                                <TableCell className="font-medium text-foreground">
                                  {item.vendor}
                                </TableCell>
                                <TableCell className="text-muted-foreground font-mono text-xs">
                                  {item.date}
                                </TableCell>
                                <TableCell className="text-right font-mono font-semibold text-foreground">
                                  ${item.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </TableCell>
                                <TableCell>
                                  <Badge variant="outline" size="sm">
                                    {item.category}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-center">
                                  {confidenceBadge(item.confidence)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </AccordionContent>
                  </Card>
                </AccordionItem>
              )}

              {/* Damage claims */}
              {result.damage_claims.length > 0 && (
                <AccordionItem value="damage" className="border-none">
                  <Card padding="none" className="overflow-hidden">
                    <AccordionTrigger className="px-4 py-3 hover:no-underline hover:bg-accent/50 transition-colors [&[data-state=open]>svg]:rotate-180">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-danger" />
                        <span className="font-semibold text-foreground text-sm">
                          Damage Claims ({result.damage_claims.length})
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-0">
                      <div className="px-4 pb-4 space-y-2">
                        {result.damage_claims.map((claim, i) => (
                          <div key={i} className="p-3 bg-muted/50 rounded-xl">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm font-semibold text-foreground">
                                {claim.label}
                              </span>
                              {confidenceBadge(claim.confidence)}
                            </div>
                            <p className="text-xs text-muted-foreground">
                              {claim.detail}
                            </p>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </Card>
                </AccordionItem>
              )}

              {/* Missing evidence */}
              {result.missing_evidence.length > 0 && (
                <AccordionItem value="missing" className="border-none">
                  <Card padding="none" className="overflow-hidden">
                    <AccordionTrigger className="px-4 py-3 hover:no-underline hover:bg-accent/50 transition-colors [&[data-state=open]>svg]:rotate-180">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-amber" />
                        <span className="font-semibold text-foreground text-sm">
                          Missing Evidence ({result.missing_evidence.length})
                        </span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-0">
                      <div className="px-4 pb-4 space-y-2">
                        {result.missing_evidence.map((item, i) => (
                          <div
                            key={i}
                            className="flex items-start gap-3 p-3 bg-amber-light/20 border border-amber/15 rounded-xl"
                          >
                            <AlertTriangle className="h-4 w-4 text-amber shrink-0 mt-0.5" />
                            <div>
                              <p className="text-sm font-medium text-foreground">
                                {item.item}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {item.reason}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </Card>
                </AccordionItem>
              )}
            </Accordion>
          </motion.div>
        )}
      </AnimatePresence>

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
          {state.evidenceFiles.length === 0 ? "Skip & Continue" : "Continue"}
        </Button>
      </div>
    </div>
  );
}
