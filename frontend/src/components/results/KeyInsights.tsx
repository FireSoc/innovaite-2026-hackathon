"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import {
  AlertTriangle,
  ArrowRight,
  Info,
  Sparkles,
} from "lucide-react";
import type { KeyInsight, InsightUrgency } from "@/lib/types";

const urgencyConfig: Record<
  InsightUrgency,
  {
    borderColor: string;
    bgColor: string;
    iconColor: string;
    badgeBg: string;
    badgeText: string;
    label: string;
    Icon: typeof AlertTriangle;
  }
> = {
  critical: {
    borderColor: "border-l-danger",
    bgColor: "bg-danger/5",
    iconColor: "text-danger",
    badgeBg: "bg-danger/10",
    badgeText: "text-danger",
    label: "Critical",
    Icon: AlertTriangle,
  },
  action_needed: {
    borderColor: "border-l-amber",
    bgColor: "bg-amber/5",
    iconColor: "text-amber-dark",
    badgeBg: "bg-amber/10",
    badgeText: "text-amber-dark",
    label: "Action Needed",
    Icon: ArrowRight,
  },
  informational: {
    borderColor: "border-l-info",
    bgColor: "bg-info/5",
    iconColor: "text-info",
    badgeBg: "bg-info/10",
    badgeText: "text-info",
    label: "Info",
    Icon: Info,
  },
};

interface KeyInsightsProps {
  insights: KeyInsight[];
}

export function KeyInsights({ insights }: KeyInsightsProps) {
  if (!insights || insights.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.35, duration: 0.5 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="h-5 w-5 text-amber" />
        <h2 className="text-lg font-bold text-foreground tracking-tight">
          Key Insights
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {insights.map((insight, i) => {
          const config =
            urgencyConfig[insight.urgency] ?? urgencyConfig.informational;
          const { Icon } = config;

          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + i * 0.08, duration: 0.4 }}
            >
              <Card
                padding="sm"
                className={`border-l-4 ${config.borderColor} ${config.bgColor} h-full`}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`h-8 w-8 rounded-lg ${config.badgeBg} flex items-center justify-center shrink-0 mt-0.5`}
                  >
                    <Icon className={`h-4 w-4 ${config.iconColor}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-sm font-semibold text-foreground leading-snug">
                        {insight.title}
                      </h3>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {insight.detail}
                    </p>
                    <span
                      className={`inline-block mt-2 text-[10px] font-medium px-2 py-0.5 rounded-full ${config.badgeBg} ${config.badgeText}`}
                    >
                      {config.label}
                    </span>
                  </div>
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
